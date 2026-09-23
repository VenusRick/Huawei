# -*- coding: utf-8 -*-
"""Scenario A (2026-09-22): mixed 16-class four-metric push, richer features.

Contract (user, 2026-09-22):
- The fresh split v2 (output/mixed_dataset_full_pass_20260921/fresh_split.csv,
  leakage-audited, Test still sealed) is reused UNCHANGED. No re-split.
- The shared-shape discipline of 20260921 (no port/TLS/QUIC/protocol fields)
  is RELAXED for a separate, clearly labeled competition-oriented track:
  inference-visible handshake/infra fields (ports, protocol, TLS handshake
  statistics, QUIC version/token, TCP window/flags) are allowed because a real
  DPI engine sees them at inference time. This is NOT the same claim as the
  shared-shape generalization experiment and is labeled separately everywhere.
- Raw SNI strings are NOT used (they are absent from the feature cache anyway;
  only tls_has_sni/tls_sni_length). JA3/JA4 hash-prefix columns are tried in
  one explicitly labeled sub-pool (hash-coded, 15/30 distinct values).
- Train/Validation search only; the sealed Test opens exactly once after
  frozen_config.json exists (validation four-metric gate passed single-fit
  AND 3-seed stability mean).

Stages:
  search — build pools, leaderboard on validation, stability, gate decision
  test   — one-shot open of the sealed fresh Test (refit train+validation)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path('/workspace/Huawei')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_mixed_dataset_full_pass_20260921 import (  # noqa: E402
    DATASETS, SEED, TARGETS, build_score_table, clean, evaluate,
    fit_predict_proba, load_split, make_model, official_metrics, rank_variant,
    slack)

OUT = ROOT / 'output/all_scenarios_metric_push_20260922/mixed16_rich'
THREADS = int(os.environ.get('TRAIN_THREADS', '8'))

TLS_FIELDS = [
    'tls_version', 'tls_has_sni', 'tls_sni_length', 'tls_num_cipher_suites',
    'tls_num_extensions', 'tls_num_supported_versions', 'tls_num_supported_groups',
    'tls_num_signature_algorithms', 'tls_has_h2', 'tls_has_http11',
    'tls_has_tls13_ciphers', 'tls_num_certificates', 'tls_handshake_bytes',
    'tls_has_encrypted_extensions',
]
QUIC_FIELDS = ['quic_is_present', 'quic_version', 'quic_is_v1', 'quic_has_token',
               'quic_token_length']
PORT_FIELDS = ['src_port', 'dst_port', 'is_well_known_src_port',
               'is_well_known_dst_port']
FLAG_FIELDS = [
    'syn_flag_count', 'ack_flag_count', 'fin_flag_count', 'rst_flag_count',
    'psh_flag_count', 'urg_flag_count', 'ece_flag_count', 'cwr_flag_count',
    'fwd_syn_flag_count', 'fwd_ack_flag_count', 'fwd_fin_flag_count',
    'fwd_rst_flag_count', 'fwd_psh_flag_count', 'fwd_urg_flag_count',
    'fwd_ece_flag_count', 'fwd_cwr_flag_count',
    'bwd_syn_flag_count', 'bwd_ack_flag_count', 'bwd_fin_flag_count',
    'bwd_rst_flag_count', 'bwd_psh_flag_count', 'bwd_urg_flag_count',
    'bwd_ece_flag_count', 'bwd_cwr_flag_count',
    'syn_ack_rtt', 'num_retransmissions', 'retransmission_ratio',
    'tcp_window_max', 'tcp_window_min', 'tcp_window_mean', 'tcp_window_std',
    'tcp_window_sum', 'tcp_window_median',
]
JA_FIELDS = ['tls_ja3_hash_prefix', 'tls_ja4_hash_prefix']


def build_pools(cols: Sequence[str]) -> Dict[str, List[str]]:
    present = set(cols)
    seq = [c for c in cols if c.startswith('seq_')]
    infra = [f for f in TLS_FIELDS + QUIC_FIELDS + PORT_FIELDS + FLAG_FIELDS
             + ['protocol_tcp', 'protocol_udp'] if f in present]
    pools: Dict[str, List[str]] = {
        'seq197_baseline': seq,
        'hs_all': seq + infra,
        'hs_noport': seq + [f for f in infra if f not in
                            ('src_port', 'dst_port')],
        'hs_ja34': seq + infra + [f for f in JA_FIELDS if f in present],
    }
    return pools


# ---------------------------------------------------------------------------
# OOF per-class bias tuning (fit on train only; validation stays untouched)
# ---------------------------------------------------------------------------
def oof_proba(train: pd.DataFrame, fs: Sequence[str], model: str,
              labels: Sequence[str], seed: int) -> np.ndarray:
    lid = {c: i for i, c in enumerate(labels)}
    y = train['_label'].map(lid).to_numpy()
    oof = np.zeros((len(train), len(labels)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        p = fit_predict_proba(train.iloc[tr], train.iloc[te], fs, model,
                              seed + 1000 + k, labels)
        oof[te] = p
    return oof


def tune_bias(y_true: np.ndarray, oof: np.ndarray, labels: Sequence[str],
              margin_p: float = 0.005, margin_fpr: float = 0.005,
              step: float = 0.15) -> np.ndarray:
    """Coordinate ascent on OOF macro recall, constrained to keep macro
    precision >= TARGET + margin and max per-class FPR <= TARGET - margin.
    Returns a per-class additive log-bias vector."""
    k = oof.shape[1]
    logit = np.log(np.maximum(oof, 1e-12))
    b = np.zeros(k)
    base = official_metrics(y_true, _pred(oof, b, labels), labels)

    def _score(m: Dict[str, object]) -> float:
        ok = (m['macro_precision'] >= TARGETS['macro_precision'] + margin_p
              and m['max_per_class_fpr'] <= TARGETS['max_per_class_fpr'] - margin_fpr)
        return m['macro_recall'] if ok else -1.0

    best_s = _score(base)
    improved = True
    rounds = 0
    while improved and rounds < 6:
        improved = False
        rounds += 1
        for c in range(k):
            for direction in (+1, -1):
                nb = b.copy()
                nb[c] += direction * step
                m = official_metrics(y_true, _pred(oof, nb, labels), labels)
                s = _score(m)
                if s > best_s + 1e-9:
                    best_s, b = s, nb
                    improved = True
    return b


def _pred(oof: np.ndarray, bias: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    logit = np.log(np.maximum(oof, 1e-12)) + bias
    return np.array([labels[i] for i in np.argmax(logit, axis=1)])


def predict_with_bias(train: pd.DataFrame, eval_df: pd.DataFrame, fs: Sequence[str],
                      model: str, seed: int, labels: Sequence[str],
                      bias: np.ndarray) -> np.ndarray:
    p = fit_predict_proba(train, eval_df, fs, model, seed, labels)
    return _pred(p, bias, labels)


def evaluate_bias(train: pd.DataFrame, eval_df: pd.DataFrame, fs: Sequence[str],
                  model: str, seed: int, labels: Sequence[str]) -> Tuple[Dict, np.ndarray]:
    ytr = train['_label'].to_numpy()  # label strings for official_metrics
    oof = oof_proba(train, fs, model, labels, seed)
    bias = tune_bias(ytr, oof, labels)
    yhat = predict_with_bias(train, eval_df, fs, model, seed, labels, bias)
    m = official_metrics(eval_df['_label'].to_numpy(), yhat, labels)
    return m, bias


# ---------------------------------------------------------------------------
def stage_search() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_split()
    train = df[df['_split'] == 'train_pool'].reset_index(drop=True)
    val = df[df['_split'] == 'validation'].reset_index(drop=True)
    labels = sorted(df['_label'].unique())
    assert len(labels) == 16, len(labels)

    pools = build_pools(df.columns)
    subsets: Dict[str, Tuple[str, List[str]]] = {}
    for pname, pool in pools.items():
        subsets[f'{pname}__full'] = (pname, list(pool))
        scores_path = OUT / f'feature_scores_{pname}.csv'
        if scores_path.exists():
            tab = pd.read_csv(scores_path)
        else:
            tab = build_score_table(train, list(pool))
            tab.to_csv(scores_path, index=False)
        subsets[f'{pname}__k64'] = (pname, rank_variant(tab, 'robust_stable')[:64])
        subsets[f'{pname}__k128'] = (pname, rank_variant(tab, 'pooled')[:128])

    models = ['xgb', 'extra', 'hgb', 'ens_xe', 'ens_all']
    rows, biases = [], {}
    base_path = OUT / 'validation_leaderboard_base.csv'
    if base_path.exists():
        rows = pd.read_csv(base_path).to_dict('records')
        print(f'RESUME base leaderboard rows={len(rows)}', flush=True)
    else:
        for sname, (pname, fs) in subsets.items():
            for model in models:
                t0 = time.time()
                m = evaluate(train, val, fs, model, SEED + 10, labels)
                rows.append({'subset': sname, 'pool': pname, 'k': len(fs),
                             'model': model, **{k: m[k] for k in
                             ('accuracy', 'macro_precision', 'macro_recall',
                              'macro_f1', 'max_per_class_fpr', 'errors')},
                             'slack': slack(m),
                         **{f'acc_{d}': m.get('per_dataset', {}).get(d, {}).get('accuracy')
                            for d in DATASETS}})
                print('VAL', rows[-1], f'{time.time()-t0:.1f}s', flush=True)
        pd.DataFrame(rows).sort_values('slack', ascending=False).reset_index(
            drop=True).to_csv(base_path, index=False)
    lb = pd.DataFrame(rows).sort_values('slack', ascending=False).reset_index(drop=True)
    lb.to_csv(OUT / 'validation_leaderboard.csv', index=False)

    # OOF-bias variants for the best (pool, model) combos of each pool: take the
    # top config per pool by slack, retune operating point on train OOF only.
    cand = []
    for pname in pools:
        sub = lb[lb['pool'] == pname]
        if len(sub):
            cand.append(sub.iloc[0])
    for _, r in pd.DataFrame(cand).iterrows():
        pname, fs = subsets[r['subset']]
        t0 = time.time()
        m, bias = evaluate_bias(train, val, fs, r['model'], SEED + 10, labels)
        sname = r['subset'] + '+oofbias'
        biases[sname] = [float(x) for x in bias]
        rows.append({'subset': sname, 'pool': pname, 'k': len(fs),
                     'model': r['model'] + '+oofbias',
                     **{k: m[k] for k in ('accuracy', 'macro_precision',
                                          'macro_recall', 'macro_f1',
                                          'max_per_class_fpr', 'errors')},
                     'slack': slack(m),
                     **{f'acc_{d}': m.get('per_dataset', {}).get(d, {}).get('accuracy')
                        for d in DATASETS}})
        print('VAL-BIAS', rows[-1], f'{time.time()-t0:.1f}s', flush=True)
    lb = pd.DataFrame(rows).sort_values('slack', ascending=False).reset_index(drop=True)
    lb.to_csv(OUT / 'validation_leaderboard.csv', index=False)
    (OUT / 'oof_bias_vectors.json').write_text(json.dumps(biases, indent=2))

    # 3-seed stability on top-3 by slack.
    top3 = lb.head(3)
    stab = []
    for _, r in top3.iterrows():
        pname, fs = subsets[r['subset']]
        is_bias = r['model'].endswith('+oofbias')
        base_model = r['model'].replace('+oofbias', '')
        ms = []
        for i in range(3):
            sd = SEED + 200 + 40 * i
            if is_bias:
                ytr = train['_label'].to_numpy()
                oof = oof_proba(train, fs, base_model, labels, sd)
                b = tune_bias(ytr, oof, labels)
                yhat = predict_with_bias(train, val, fs, base_model, sd, labels, b)
                ms.append(official_metrics(val['_label'].to_numpy(), yhat, labels))
            else:
                ms.append(evaluate(train, val, fs, base_model, sd, labels))
        stab.append({'subset': r['subset'], 'model': r['model'],
                     'macro_recall_mean': float(np.mean([m['macro_recall'] for m in ms])),
                     'macro_recall_min': float(np.min([m['macro_recall'] for m in ms])),
                     'accuracy_mean': float(np.mean([m['accuracy'] for m in ms])),
                     'macro_precision_mean': float(np.mean([m['macro_precision'] for m in ms])),
                     'macro_precision_min': float(np.min([m['macro_precision'] for m in ms])),
                     'max_fpr_max': float(np.max([m['max_per_class_fpr'] for m in ms])),
                     'slack_mean': float(np.mean([slack(m) for m in ms]))})
        print('STAB', stab[-1], flush=True)
    pd.DataFrame(stab).to_csv(OUT / 'validation_stability.csv', index=False)

    def _passes(m: Dict[str, float]) -> bool:
        return (m['accuracy'] >= TARGETS['accuracy']
                and m['macro_precision'] >= TARGETS['macro_precision']
                and m['macro_recall'] >= TARGETS['macro_recall']
                and m['max_per_class_fpr'] <= TARGETS['max_per_class_fpr'])

    best = max(stab, key=lambda r: (r['slack_mean'], r['macro_recall_mean']))
    best_row = lb.iloc[0]
    single_pass = bool(best_row['slack'] >= 0)
    stab_pass = (best['accuracy_mean'] >= TARGETS['accuracy']
                 and best['macro_precision_mean'] >= TARGETS['macro_precision']
                 and best['macro_recall_mean'] >= TARGETS['macro_recall']
                 and best['max_fpr_max'] <= TARGETS['max_per_class_fpr'])
    pname, best_fs = subsets[best['subset']]
    outcome = {
        'experiment': 'all_scenarios_metric_push_20260922/mixed16_rich',
        'scenario': 'A_mixed16_competition_rich',
        'blinding': 'fresh split v2 (20260921, leakage-audited) reused unchanged; '
                    'Test first opened only after this gate',
        'feature_space': ('competition-oriented: generic seq197 + inference-visible '
                          'handshake/infra fields of the pool %s (see pools.json); '
                          'raw SNI strings not used' % pname),
        'pool': pname, 'subset': best['subset'], 'selected_features': best_fs,
        'selected_k': len(best_fs),
        'model': best['model'],
        'fit_policy': 'refit on train+validation with frozen features/model, fixed seed',
        'refit_seed': SEED + 900,
        'targets': TARGETS,
        'validation_snapshot': best,
        'validation_best_single': {k: float(best_row[k]) for k in
                                   ('accuracy', 'macro_precision', 'macro_recall',
                                    'max_per_class_fpr', 'slack')},
        'pools': {k: len(v) for k, v in pools.items()},
    }
    (OUT / 'pools.json').write_text(json.dumps(
        {k: v for k, v in pools.items()}, indent=2))
    if single_pass and stab_pass:
        outcome['fresh_test_policy'] = ('test rows opened exactly once after this file '
                                        'is written; no retuning if test fails')
        fp = OUT / 'frozen_config.json'
        fp.write_text(json.dumps(outcome, indent=2, ensure_ascii=False))
        print('FROZEN (validation four-metric gate passed)', best, flush=True)
        print('frozen_config_sha256',
              hashlib.sha256(fp.read_bytes()).hexdigest(), flush=True)
    else:
        outcome['status'] = 'validation_not_passed'
        outcome['reason'] = ('best configuration does not simultaneously meet the four '
                             'targets on validation (single-fit pass=%s, 3-seed-mean '
                             'pass=%s); fresh test stays sealed'
                             % (single_pass, stab_pass))
        (OUT / 'search_outcome.json').write_text(
            json.dumps(outcome, indent=2, ensure_ascii=False))
        print('NOT FROZEN:', outcome['reason'], flush=True)


def stage_test() -> None:
    tm = OUT / 'test_metrics.json'
    if tm.exists():
        raise RuntimeError('test_metrics.json already exists; fresh holdout may only '
                           'be opened once (refusing to overwrite)')
    if not (OUT / 'frozen_config.json').exists():
        raise RuntimeError('frozen_config.json missing: validation gate not passed, '
                           'fresh test must stay sealed')
    frozen = json.loads((OUT / 'frozen_config.json').read_text())
    df = load_split()
    labels = sorted(df['_label'].unique())
    dev = df[df['_split'].isin(['train_pool', 'validation'])].reset_index(drop=True)
    test = df[df['_split'] == 'test'].reset_index(drop=True)
    fs = frozen['selected_features']
    model = frozen['model']
    is_bias = model.endswith('+oofbias')
    base = model.replace('+oofbias', '')
    if is_bias:
        ydev = dev['_label'].to_numpy()
        oof = oof_proba(dev, fs, base, labels, frozen['refit_seed'])
        bias = tune_bias(ydev, oof, labels)
        yhat = predict_with_bias(dev, test, fs, base, frozen['refit_seed'],
                                 labels, bias)
        m = official_metrics(test['_label'].to_numpy(), yhat, labels)
        m['per_dataset'] = None
        ds = test['_dataset'].to_numpy()
        yt = test['_label'].to_numpy()
        per_ds = {}
        for d in DATASETS:
            mask = ds == d
            if not mask.any():
                continue
            rec = [float((yhat[mask & (yt == c)] == c).mean())
                   for c in pd.unique(yt[mask])]
            per_ds[d] = {'accuracy': float((yhat[mask] == yt[mask]).mean()),
                         'macro_recall': float(np.mean(rec)), 'n': int(mask.sum())}
        m['per_dataset'] = per_ds
        m['worst_dataset_accuracy'] = min(v['accuracy'] for v in per_ds.values())
    else:
        m = evaluate(dev, test, fs, base, frozen['refit_seed'], labels)
    per_rows = [{'label': c, **v} for c, v in m['per_class'].items()]
    pd.DataFrame(per_rows).to_csv(OUT / 'per_class_metrics.csv', index=False)
    pd.DataFrame(m['confusion'], index=labels, columns=labels).to_csv(
        OUT / 'confusion.csv')
    passes = {'accuracy': m['accuracy'] >= TARGETS['accuracy'],
              'macro_precision': m['macro_precision'] >= TARGETS['macro_precision'],
              'macro_recall': m['macro_recall'] >= TARGETS['macro_recall'],
              'max_per_class_fpr': m['max_per_class_fpr'] <= TARGETS['max_per_class_fpr']}
    result = {
        'frozen_config_sha256': hashlib.sha256(
            (OUT / 'frozen_config.json').read_bytes()).hexdigest(),
        'split_sha256': json.loads((ROOT / 'output/mixed_dataset_full_pass_20260921'
                                    '/fresh_split_spec.json').read_text())['split_sha256'],
        'metrics': {k: m[k] for k in ('accuracy', 'macro_precision', 'macro_recall',
                                      'macro_f1', 'max_per_class_fpr', 'errors', 'n')},
        'targets': TARGETS, 'pass': passes, 'all_pass': all(passes.values()),
        'per_dataset': m.get('per_dataset'),
        'opened_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'one_shot': True,
        'per_class': m['per_class'],
    }
    tm.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: result[k] for k in ('metrics', 'pass', 'all_pass')},
                     indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['search', 'test'])
    args = ap.parse_args()
    if args.stage == 'search':
        stage_search()
    else:
        stage_test()


if __name__ == '__main__':
    main()
