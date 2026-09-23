# -*- coding: utf-8 -*-
"""Scenario A2 (2026-09-22): mixed 16-class, identifier-assisted IP-feed track.

Explicitly labeled IDENTIFIER-ASSISTED / competition-oriented. This is NOT the
shared-shape generalization claim and NOT the same claim as the
hs_all/handshake-visible track in mixed16_rich:
- the CSTNET-TLS1.3 captures are server-side captures without Client Hello,
  so SNI is structurally unavailable; the sanctioned identifier is the remote
  (server) IP. Matching traffic against a server-IP feed is classic DPI;
- per observation we add ONLY the remote endpoint identity: remote IP as u32,
  /24 prefix u32, low-port flag. No label/filename/path/dataset fields. The
  IP->class association is LEARNED from train files only, so the sealed fresh
  test (held files) still measures whether the mapping generalizes to unseen
  captures;
- everything else (split reuse, model grid, four-metric gate, one-shot test)
  is identical to mixed16_rich.

Stages: ipmap | search | test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path('/workspace/Huawei')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_mixed_dataset_full_pass_20260921 import (  # noqa: E402
    DATASETS, READ_CAPS, dataset_files, SEED, TARGETS, build_score_table,
    evaluate, load_split, official_metrics, rank_variant, slack)
import run_scenario_mixed16_rich_20260922 as rich  # noqa: E402

from src.parser.pcap_reader import PCAPReader  # noqa: E402
from src.parser.session.session_manager import SessionManager  # noqa: E402

OUT = ROOT / 'output/all_scenarios_metric_push_20260922/mixed16_ipfeed'
IPMAP = OUT / 'ipmap.csv'


def _u32(ip: str) -> int:
    try:
        a, b, c, d = (int(x) for x in ip.split('.'))
        return (a << 24) | (b << 16) | (c << 8) | d
    except Exception:  # noqa: BLE001
        return 0


def remote_ip_of(session) -> Tuple[str, int]:
    """Remote (server-ish) endpoint: the side with the low port; ties/mixed ->
    the side with more bytes; fallback dst."""
    sp, dp = session.src_port, session.dst_port
    src_bytes = session.total_fwd_bytes
    dst_bytes = session.total_bwd_bytes
    if sp and dp:
        if dp < 1024 and sp >= 1024:
            return session.dst_ip, dp
        if sp < 1024 and dp >= 1024:
            return session.src_ip, sp
    return (session.dst_ip if dst_bytes >= src_bytes else session.src_ip), dp


def stage_ipmap() -> None:
    """One read pass per split source file: observation_id -> remote ip/port.
    Session enumeration order matches the original cache extraction (the
    observation id encodes `filename:session_index`)."""
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_split()
    need: Dict[str, set] = defaultdict(set)
    for oid in df['_observation_id']:
        fname, idx = oid.rsplit(':', 1)
        need[fname].add(int(idx))
    # map basename -> (dataset, class, path, cap)
    configs = dataset_files()
    file_cfg = {}
    for dname, classes in configs.items():
        for cname, files in classes.items():
            for f in files:
                file_cfg[f.name] = (dname, cname, str(f), READ_CAPS[dname])
    rows = []
    t0 = time.time()
    for fname, idxs in sorted(need.items()):
        if fname not in file_cfg:
            raise RuntimeError(f'split file unknown to dataset_files: {fname}')
        dname, cname, path, cap = file_cfg[fname]
        sm = SessionManager()
        reader = PCAPReader(sm)
        got = 0
        for idx, session in enumerate(reader.read_pcap_generator(path,
                                                                max_read_packets=cap)):
            if idx not in idxs:
                continue
            ip, port = remote_ip_of(session)
            rows.append({'_observation_id': f'{fname}:{idx}',
                         '_remote_ip': ip, '_remote_port': int(port),
                         '_remote_u32': _u32(ip),
                         '_remote_slash24': _u32(ip) & 0xFFFFFF00,
                         '_remote_lowport': int(port < 1024)})
            got += 1
            if got == len(idxs):
                break
        print(f'IPMAP {fname} matched {got}/{len(idxs)} '
              f'({time.time()-t0:.0f}s)', flush=True)
    m = pd.DataFrame(rows)
    if m['_observation_id'].duplicated().any():
        raise RuntimeError('duplicate ipmap rows')
    m.to_csv(IPMAP, index=False)
    print('TOTAL', len(m), flush=True)


def load_with_ip() -> pd.DataFrame:
    df = load_split()
    ip = pd.read_csv(IPMAP)
    df = df.merge(ip, on='_observation_id', how='left')
    missing = int(df['_remote_ip'].isna().sum())
    if missing:
        raise RuntimeError(f'{missing} observations without ipmap rows')
    return df


IP_FEATS = ['_remote_u32', '_remote_slash24', '_remote_lowport', '_remote_port']


def stage_search() -> None:
    df = load_with_ip()
    train = df[df['_split'] == 'train_pool'].reset_index(drop=True)
    val = df[df['_split'] == 'validation'].reset_index(drop=True)
    labels = sorted(df['_label'].unique())
    assert len(labels) == 16
    pools = rich.build_pools(df.columns)
    subsets = {}
    subsets['hs_all_ip__full'] = pools['hs_all'] + IP_FEATS
    subsets['hs_all_ip24__full'] = pools['hs_all'] + IP_FEATS[:2] + IP_FEATS[2:]
    subsets['ip_only__full'] = list(IP_FEATS)
    subsets['hs_ja34_ip__full'] = pools['hs_ja34'] + IP_FEATS
    for pname, base in (('hs_all', pools['hs_all']),
                        ('hs_ja34_ip', pools['hs_ja34'] + IP_FEATS)):
        tab = build_score_table(train, list(base))
        subsets[f'{pname}__k64'] = rank_variant(tab, 'robust_stable')[:64]
        subsets[f'{pname}__k128'] = rank_variant(tab, 'pooled')[:128]
        tab.to_csv(OUT / f'feature_scores_{pname}.csv', index=False)

    rows = []
    for sname, fs in subsets.items():
        for model in ('xgb', 'hgb', 'ens_xe', 'ens_all'):
            m = evaluate(train, val, fs, model, SEED + 10, labels)
            rows.append({'subset': sname, 'k': len(fs), 'model': model,
                         **{k: m[k] for k in ('accuracy', 'macro_precision',
                                              'macro_recall', 'macro_f1',
                                              'max_per_class_fpr', 'errors')},
                         'slack': slack(m),
                         **{f'acc_{d}': m.get('per_dataset', {}).get(d, {}).get('accuracy')
                            for d in DATASETS}})
            print('VAL', rows[-1], flush=True)
    lb = pd.DataFrame(rows).sort_values('slack', ascending=False).reset_index(drop=True)
    lb.to_csv(OUT / 'validation_leaderboard.csv', index=False)

    # 3-seed stability of the top config
    top = lb.iloc[0]
    fs = subsets[top['subset']]
    from run_mixed_dataset_full_pass_20260921 import fit_predict_proba, \
        official_metrics
    ms = [official_metrics(val['_label'].to_numpy(),
                           np.array([labels[i] for i in np.argmax(
                               fit_predict_proba(train, val, fs, top['model'],
                                                 SEED + 200 + 40 * i, labels),
                               axis=1)]), labels) for i in range(3)]
    stab = {'subset': top['subset'], 'model': top['model'],
            'macro_recall_mean': float(np.mean([m['macro_recall'] for m in ms])),
            'macro_recall_min': float(np.min([m['macro_recall'] for m in ms])),
            'accuracy_mean': float(np.mean([m['accuracy'] for m in ms])),
            'macro_precision_min': float(np.min([m['macro_precision'] for m in ms])),
            'max_fpr_max': float(np.max([m['max_per_class_fpr'] for m in ms])),
            'slack_mean': float(np.mean([slack(m) for m in ms]))}
    pd.DataFrame([stab]).to_csv(OUT / 'validation_stability.csv', index=False)
    single_pass = bool(top['slack'] >= 0)
    stab_pass = (stab['accuracy_mean'] >= TARGETS['accuracy']
                 and stab['macro_precision_min'] >= TARGETS['macro_precision']
                 and stab['macro_recall_mean'] >= TARGETS['macro_recall']
                 and stab['max_fpr_max'] <= TARGETS['max_per_class_fpr'])
    outcome = {
        'experiment': 'all_scenarios_metric_push_20260922/mixed16_ipfeed',
        'scenario': 'A2_mixed16_identifier_assisted_ipfeed',
        'feature_space': ('IDENTIFIER-ASSISTED: inference-visible remote server '
                          'IP (u32, /24, low-port flag) + handshake/infra pool; '
                          'IP->class association learned from train files only; '
                          'NOT the shared-shape generalization claim'),
        'blinding': 'fresh split v2 (20260921, leakage-audited) reused '
                    'unchanged; this track opens the SAME sealed test exactly '
                    'once if its own validation gate passes',
        'subset': top['subset'], 'selected_features': fs, 'k': len(fs),
        'model': top['model'],
        'fit_policy': 'refit on train+validation with frozen features/model, '
                      'fixed seed', 'refit_seed': SEED + 900,
        'targets': TARGETS, 'validation_snapshot': stab,
        'validation_best_single': {k: float(top[k]) for k in
                                   ('accuracy', 'macro_precision', 'macro_recall',
                                    'max_per_class_fpr', 'slack')},
    }
    if single_pass and stab_pass:
        outcome['fresh_test_policy'] = ('test rows opened exactly once after '
                                        'this file is written; no retuning')
        fp = OUT / 'frozen_config.json'
        fp.write_text(json.dumps(outcome, indent=2, ensure_ascii=False))
        print('FROZEN (identifier-assisted validation gate passed)', stab, flush=True)
    else:
        outcome['status'] = 'validation_not_passed'
        outcome['reason'] = (f'best single pass={single_pass}, stability pass='
                             f'{stab_pass}; sealed test stays closed')
        (OUT / 'search_outcome.json').write_text(json.dumps(outcome, indent=2))
        print('NOT FROZEN', outcome['reason'], flush=True)


def stage_search_bias() -> None:
    """Second validation-side iteration for the IP-feed track: OOF per-class
    bias tuning (fit on train only via rich.evaluate_bias) on the top-3
    configurations of the existing leaderboard. The sealed test is untouched
    unless this gate passes."""
    lb = pd.read_csv(OUT / 'validation_leaderboard.csv')
    if (OUT / 'validation_leaderboard_bias.csv').exists():
        lb = pd.concat([lb, pd.read_csv(OUT / 'validation_leaderboard_bias.csv')],
                       ignore_index=True)
    df = load_with_ip()
    train = df[df['_split'] == 'train_pool'].reset_index(drop=True)
    val = df[df['_split'] == 'validation'].reset_index(drop=True)
    labels = sorted(df['_label'].unique())
    pools = rich.build_pools(df.columns)
    subsets = {'hs_all_ip__full': pools['hs_all'] + IP_FEATS,
               'hs_ja34_ip__full': pools['hs_ja34'] + IP_FEATS,
               'ip_only__full': list(IP_FEATS)}
    for pname, base in (('hs_all', pools['hs_all']),
                        ('hs_ja34_ip', pools['hs_ja34'] + IP_FEATS)):
        tab_path = OUT / f'feature_scores_{pname}.csv'
        if tab_path.exists():
            tab = pd.read_csv(tab_path)
            from run_mixed_dataset_full_pass_20260921 import rank_variant as _rv
            subsets[f'{pname}__k64'] = _rv(tab, 'robust_stable')[:64]
            subsets[f'{pname}__k128'] = _rv(tab, 'pooled')[:128]
    rows, biases = [], {}
    top3 = lb[~lb.model.str.contains('oofbias')].sort_values(
        'slack', ascending=False).head(3)
    for _, r in top3.iterrows():
        if r['subset'] not in subsets:
            continue
        fs = subsets[r['subset']]
        m, bias = rich.evaluate_bias(train, val, fs, r['model'], SEED + 10, labels)
        sname = r['subset'] + '+oofbias'
        biases[sname] = [float(x) for x in bias]
        rows.append({'subset': sname, 'k': len(fs), 'model': r['model'] + '+oofbias',
                     **{k: m[k] for k in ('accuracy', 'macro_precision',
                                          'macro_recall', 'macro_f1',
                                          'max_per_class_fpr', 'errors')},
                     'slack': slack(m)})
        print('VAL-BIAS', rows[-1], flush=True)
    if not rows:
        print('NOT FROZEN (no bias candidates)', flush=True)
        return
    pd.DataFrame(rows).to_csv(OUT / 'validation_leaderboard_bias.csv', index=False)
    (OUT / 'oof_bias_vectors.json').write_text(json.dumps(biases, indent=2))
    best = max(rows, key=lambda r: r['slack'])
    fs = subsets[best['subset'].replace('+oofbias', '')]
    bias = np.array(biases[best['subset']])
    ms = [official_metrics(val['_label'].to_numpy(),
                           rich.predict_with_bias(train, val, fs,
                                                  best['model'].replace('+oofbias', ''),
                                                  SEED + 200 + 40 * i, labels,
                                                  bias), labels)
          for i in range(3)]
    # NOTE: stability here reuses the tuned bias of the primary seed; the
    # members are refit per seed which is the dominant variance source.
    stab_pass = (np.mean([m['accuracy'] for m in ms]) >= TARGETS['accuracy']
                 and np.min([m['macro_precision'] for m in ms]) >= TARGETS['macro_precision']
                 and np.mean([m['macro_recall'] for m in ms]) >= TARGETS['macro_recall']
                 and np.max([m['max_per_class_fpr'] for m in ms]) <= TARGETS['max_per_class_fpr'])
    single_pass = best['slack'] >= 0
    outcome = {
        'experiment': 'all_scenarios_metric_push_20260922/mixed16_ipfeed',
        'scenario': 'A2_mixed16_identifier_assisted_ipfeed_bias',
        'feature_space': ('IDENTIFIER-ASSISTED: remote server IP (u32, /24, '
                          'low-port) + handshake/infra pool + OOF per-class '
                          'bias tuned on train OOF only'),
        'subset': best['subset'], 'selected_features': fs, 'k': len(fs),
        'model': best['model'],
        'fit_policy': 'refit on train+validation with frozen features/model/'
                      'bias, fixed seed', 'refit_seed': SEED + 900,
        'bias': biases[best['subset']], 'targets': TARGETS,
        'validation_best_single': {k: best[k] for k in
                                   ('accuracy', 'macro_precision',
                                    'macro_recall', 'max_per_class_fpr', 'slack')},
    }
    if single_pass and stab_pass:
        outcome['fresh_test_policy'] = ('test rows opened exactly once after '
                                        'this file is written; no retuning')
        fp = OUT / 'frozen_config.json'
        fp.write_text(json.dumps(outcome, indent=2, ensure_ascii=False))
        print('FROZEN (ipfeed+bias gate passed)', flush=True)
    else:
        outcome['status'] = 'validation_not_passed'
        outcome['reason'] = (f'best single pass={single_pass}, stability pass='
                             f'{stab_pass}; sealed test stays closed')
        (OUT / 'search_outcome_bias.json').write_text(json.dumps(outcome, indent=2))
        print('NOT FROZEN', outcome['reason'], flush=True)


def stage_test() -> None:
    tm = OUT / 'test_metrics.json'
    if tm.exists():
        raise RuntimeError('test_metrics.json already exists: one-shot rule')
    fp = OUT / 'frozen_config.json'
    if not fp.exists():
        raise RuntimeError('frozen_config.json missing: gate not passed')
    frozen = json.loads(fp.read_text())
    df = load_with_ip()
    labels = sorted(df['_label'].unique())
    dev = df[df['_split'].isin(['train_pool', 'validation'])].reset_index(drop=True)
    test = df[df['_split'] == 'test'].reset_index(drop=True)
    model = frozen['model']
    if model.endswith('+oofbias'):
        base_model = model.replace('+oofbias', '')
        bias = np.array(frozen['bias'])
        yhat = rich.predict_with_bias(dev, test, frozen['selected_features'],
                                      base_model, frozen['refit_seed'], labels, bias)
        m = official_metrics(test['_label'].to_numpy(), yhat, labels)
        m.setdefault('per_dataset', None)
    else:
        m = evaluate(dev, test, frozen['selected_features'], model,
                     frozen['refit_seed'], labels)
    per_rows = [{'label': c, **v} for c, v in m['per_class'].items()]
    pd.DataFrame(per_rows).to_csv(OUT / 'per_class_metrics.csv', index=False)
    pd.DataFrame(m['confusion'], index=labels, columns=labels).to_csv(
        OUT / 'confusion.csv')
    passes = {'accuracy': m['accuracy'] >= TARGETS['accuracy'],
              'macro_precision': m['macro_precision'] >= TARGETS['macro_precision'],
              'macro_recall': m['macro_recall'] >= TARGETS['macro_recall'],
              'max_per_class_fpr': m['max_per_class_fpr'] <= TARGETS['max_per_class_fpr']}
    result = {
        'frozen_config_sha256': hashlib.sha256(fp.read_bytes()).hexdigest(),
        'identifier_assisted': True,
        'metrics': {k: m[k] for k in ('accuracy', 'macro_precision',
                                      'macro_recall', 'macro_f1',
                                      'max_per_class_fpr', 'errors', 'n')},
        'targets': TARGETS, 'pass': passes, 'all_pass': all(passes.values()),
        'per_dataset': m.get('per_dataset'), 'per_class': m['per_class'],
        'opened_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'one_shot': True}
    tm.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: result[k] for k in ('metrics', 'pass', 'all_pass')},
                     indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['ipmap', 'search', 'search_bias', 'test'])
    args = ap.parse_args()
    {'ipmap': stage_ipmap, 'search': stage_search,
     'search_bias': stage_search_bias, 'test': stage_test}[args.stage]()


if __name__ == '__main__':
    main()
