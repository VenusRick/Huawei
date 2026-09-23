# -*- coding: utf-8 -*-
"""Final closure round (2026-09-22): mixed16 identifier-assisted track with
capture-scoped endpoint aggregation.

Prior state (ledger 31.4): mixed16 fresh split v2 (leakage-audited, sealed
test never opened) plateaued at Macro-R ~95.2 (shared-shape), ~96.3
(handshake-visible), ~97.17 (IP feed + OOF bias). Residual errors concentrate
in CrossPlatform app-pair confusions (baidu.BaiduMap <-> autonavi.minimap <->
bubei.tingshu, com.aikan), many of them LONG sessions -- not only the 3-14
packet short sessions.

Root-cause treatment tested here: sessions of the same capture that talk to
the SAME remote endpoint almost always belong to the same application; a real
DPI engine processing a capture can therefore pool the per-session evidence
per (capture-file, remote /24) before deciding. This is label-free,
inference-visible (remote IP is part of the packet), transductive within the
input unit only -- grouped with the identifier-assisted (IP feed) track, NOT
the shared-shape generalization claim.

Protocol (unchanged split, no new splits, no test peeking):
- reuse output/mixed_dataset_full_pass_20260921/fresh_split.csv (seed
  20260922, SHA d3524c4c...) and the ipmap built in
  all_scenarios_metric_push_20260922/mixed16_ipfeed;
- search on train_pool -> validation only: feature subsets x models x
  grouping {none, file+remote_ip, file+remote_/24};
- gate = validation four metrics single-fit AND 3-seed stability mean/min;
- if and only if the gate passes: frozen_config.json -> open the sealed fresh
  test EXACTLY ONCE (same aggregation applied within each test file; this
  consumes the one shared mixed16 test opening for both the rich and ipfeed
  tracks).

Stages: search | test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/workspace/Huawei')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_mixed_dataset_full_pass_20260921 import (  # noqa: E402
    SEED, TARGETS, build_score_table, official_metrics, rank_variant, slack,
    fit_predict_proba)
import run_scenario_mixed16_rich_20260922 as rich  # noqa: E402
import run_scenario_mixed16_ipfeed_20260922 as ipf  # noqa: E402

OUT = ROOT / 'output/final_experiment_closure_20260922/mixed16_endpoint_agg'
LEN_FEATS = ['_full_packets', '_full_bytes']


def group_key(eval_df: pd.DataFrame, grouping: str) -> np.ndarray:
    if grouping == 'file_ip':
        return (eval_df['_source_file'].astype(str) + '|'
                + eval_df['_remote_ip'].astype(str)).to_numpy()
    if grouping == 'file_slash24':
        return (eval_df['_source_file'].astype(str) + '|'
                + eval_df['_remote_slash24'].astype(str)).to_numpy()
    raise ValueError(grouping)


def aggregate(proba: np.ndarray, eval_df: pd.DataFrame,
              grouping: str) -> np.ndarray:
    """Capture-scoped endpoint pooling of per-session class probabilities.

    Label-free: only the capture-file identity and the remote endpoint of
    each session are used. Groups never cross files.
    """
    if grouping == 'none':
        return proba
    key = group_key(eval_df, grouping)
    out = proba.copy()
    for k in np.unique(key):
        idxs = np.where(key == k)[0]
        out[idxs] = proba[idxs].mean(axis=0)
    return out


def predict(train: pd.DataFrame, eval_df: pd.DataFrame, fs, model: str,
            seed: int, labels, grouping: str):
    proba = fit_predict_proba(train, eval_df, fs, model, seed, labels)
    proba = aggregate(proba, eval_df, grouping)
    pred = np.array([labels[i] for i in np.argmax(proba, axis=1)])
    return official_metrics(eval_df['_label'].to_numpy(), pred, labels), proba


def load_dev_val():
    df = ipf.load_with_ip()
    train = df[df['_split'] == 'train_pool'].reset_index(drop=True)
    val = df[df['_split'] == 'validation'].reset_index(drop=True)
    labels = sorted(df['_label'].unique())
    assert len(labels) == 16
    return df, train, val, labels


def build_subsets(df, train):
    pools = rich.build_pools(df.columns)
    subsets = {
        'hs_all_ip__full': pools['hs_all'] + ipf.IP_FEATS,
        'hs_ja34_ip__full': pools['hs_ja34'] + ipf.IP_FEATS,
        'hs_all_ip_len__full': pools['hs_all'] + ipf.IP_FEATS + LEN_FEATS,
    }
    for pname, base in (('hs_all_ip', pools['hs_all'] + ipf.IP_FEATS),
                        ('hs_ja34_ip', pools['hs_ja34'] + ipf.IP_FEATS)):
        tab = build_score_table(train, list(base))
        subsets[f'{pname}__k64'] = rank_variant(tab, 'robust_stable')[:64]
        subsets[f'{pname}__k128'] = rank_variant(tab, 'pooled')[:128]
        tab.to_csv(OUT / f'feature_scores_{pname}.csv', index=False)
    return subsets


def stage_search() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, train, val, labels = load_dev_val()
    subsets = build_subsets(df, train)
    rows = []
    for sname, fs in subsets.items():
        for model in ('hgb', 'xgb', 'ens_xe', 'ens_all'):
            for grouping in ('none', 'file_ip', 'file_slash24'):
                m, _ = predict(train, val, fs, model, SEED + 10, labels,
                               grouping)
                rows.append({'subset': sname, 'k': len(fs), 'model': model,
                             'grouping': grouping,
                             **{k: m[k] for k in (
                                 'accuracy', 'macro_precision', 'macro_recall',
                                 'macro_f1', 'max_per_class_fpr', 'errors')},
                             'slack': slack(m),
                             **{f'acc_{d}': m.get('per_dataset', {}).get(d, {})
                                .get('accuracy') for d in
                                ('CSTNET', 'CrossPlatform', 'USTC', 'VisQUIC')}})
                print('VAL', rows[-1], flush=True)
    lb = pd.DataFrame(rows).sort_values('slack', ascending=False).reset_index(
        drop=True)
    lb.to_csv(OUT / 'validation_leaderboard.csv', index=False)

    top = lb.iloc[0]
    fs = subsets[top['subset']]
    grouping, model = top['grouping'], top['model']
    ms = []
    for i in range(3):
        m, _ = predict(train, val, fs, model, SEED + 200 + 40 * i, labels,
                       grouping)
        ms.append(m)
        print('STAB', i, {k: round(m[k], 4) for k in (
            'accuracy', 'macro_precision', 'macro_recall',
            'max_per_class_fpr')}, flush=True)
    stab = {
        'subset': top['subset'], 'model': model, 'grouping': grouping,
        'accuracy_mean': float(np.mean([m['accuracy'] for m in ms])),
        'macro_precision_min': float(np.min([m['macro_precision'] for m in ms])),
        'macro_recall_mean': float(np.mean([m['macro_recall'] for m in ms])),
        'macro_recall_min': float(np.min([m['macro_recall'] for m in ms])),
        'max_fpr_max': float(np.max([m['max_per_class_fpr'] for m in ms])),
        'all_pass_seeds': int(sum(
            m['accuracy'] >= TARGETS['accuracy']
            and m['macro_precision'] >= TARGETS['macro_precision']
            and m['macro_recall'] >= TARGETS['macro_recall']
            and m['max_per_class_fpr'] <= TARGETS['max_per_class_fpr']
            for m in ms)),
    }
    pd.DataFrame([stab]).to_csv(OUT / 'validation_stability.csv', index=False)
    single_pass = bool(top['slack'] >= 0)
    stab_pass = (stab['accuracy_mean'] >= TARGETS['accuracy']
                 and stab['macro_precision_min'] >= TARGETS['macro_precision']
                 and stab['macro_recall_mean'] >= TARGETS['macro_recall']
                 and stab['max_fpr_max'] <= TARGETS['max_per_class_fpr'])
    outcome = {
        'experiment': 'final_experiment_closure_20260922/mixed16_endpoint_agg',
        'scenario': 'A2+_mixed16_ipfeed_capture_scoped_endpoint_aggregation',
        'track': ('IDENTIFIER-ASSISTED (remote server IP feed) + label-free '
                  'transductive pooling of per-session probabilities per '
                  '(capture-file, remote /24). NOT the shared-shape '
                  'generalization claim.'),
        'split_reused': 'mixed_dataset_full_pass_20260921/fresh_split.csv '
                        '(seed 20260922, leakage-audited, unchanged)',
        'test_opening_rule': ('this file consumes the ONE shared sealed-test '
                             'opening of the mixed16 fresh split v2 (both the '
                             'rich and ipfeed tracks defer to it)'),
        'subset': top['subset'], 'selected_features': fs, 'k': len(fs),
        'model': model, 'grouping': grouping,
        'fit_policy': 'refit on train+validation with frozen features/model/'
                      'grouping, fixed seed', 'refit_seed': SEED + 900,
        'targets': TARGETS,
        'validation_best_single': {k: float(top[k]) for k in (
            'accuracy', 'macro_precision', 'macro_recall',
            'max_per_class_fpr', 'slack')},
        'validation_stability': stab,
    }
    if single_pass and stab_pass:
        outcome['fresh_test_policy'] = ('test rows opened exactly once after '
                                        'this file is written; no retuning')
        (OUT / 'frozen_config.json').write_text(
            json.dumps(outcome, indent=2, ensure_ascii=False))
        print('FROZEN', json.dumps(stab), flush=True)
    else:
        outcome['status'] = 'validation_not_passed'
        outcome['reason'] = (f'best single pass={single_pass}, stability pass='
                             f'{stab_pass}; sealed test stays closed')
        (OUT / 'search_outcome.json').write_text(json.dumps(outcome, indent=2))
        print('NOT FROZEN', outcome['reason'], flush=True)


def stage_test() -> None:
    tm = OUT / 'test_metrics.json'
    if tm.exists():
        raise RuntimeError('test_metrics.json already exists: one-shot rule')
    fp = OUT / 'frozen_config.json'
    if not fp.exists():
        raise RuntimeError('frozen_config.json missing: gate not passed')
    frozen = json.loads(fp.read_text())
    df = ipf.load_with_ip()
    labels = sorted(df['_label'].unique())
    dev = df[df['_split'].isin(['train_pool', 'validation'])].reset_index(
        drop=True)
    test = df[df['_split'] == 'test'].reset_index(drop=True)
    m, _ = predict(dev, test, frozen['selected_features'], frozen['model'],
                   frozen['refit_seed'], labels, frozen['grouping'])
    passes = {'accuracy': m['accuracy'] >= TARGETS['accuracy'],
              'macro_precision': m['macro_precision'] >= TARGETS['macro_precision'],
              'macro_recall': m['macro_recall'] >= TARGETS['macro_recall'],
              'max_per_class_fpr': m['max_per_class_fpr'] <= TARGETS['max_per_class_fpr']}
    result = {
        'frozen_config_sha256': hashlib.sha256(fp.read_bytes()).hexdigest(),
        'identifier_assisted': True,
        'transductive_grouping': frozen['grouping'],
        'metrics': {k: m[k] for k in ('accuracy', 'macro_precision',
                                      'macro_recall', 'macro_f1',
                                      'max_per_class_fpr', 'errors', 'n')},
        'targets': TARGETS, 'pass': passes, 'all_pass': all(passes.values()),
        'per_dataset': m.get('per_dataset'), 'per_class': m['per_class'],
        'opened_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'one_shot': True}
    tm.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    pd.DataFrame([{'label': c, **v} for c, v in m['per_class'].items()]).to_csv(
        OUT / 'per_class_metrics.csv', index=False)
    pd.DataFrame(m['confusion'], index=labels, columns=labels).to_csv(
        OUT / 'confusion.csv')
    print(json.dumps({k: result[k] for k in ('metrics', 'pass', 'all_pass')},
                     indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['search', 'test'])
    args = ap.parse_args()
    {'search': stage_search, 'test': stage_test}[args.stage]()


if __name__ == '__main__':
    main()
