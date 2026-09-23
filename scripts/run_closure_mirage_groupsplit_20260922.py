# -*- coding: utf-8 -*-
"""Final closure round (2026-09-22): MIRAGE activity, flow-length-grouped
representation.

Prior state (ledger 31.5): on the pre-registered fresh grouped split (seed
20260922, 569 pure single-activity captures of 5 apps; test NEVER opened),
merged-stream plateaued at 96.17 Macro-R and the 20260919 hybrid recipe at
97.28 Macro-R (3 errors / 134 validation captures). The 20260919 sealed test
of the older 4-app pool (R 96.75) was consumed and is untouched.

This final round targets the residual root cause with the sanctioned
observation-unit lever -- short/long flow grouping then aggregation:
- per capture, target-app flows are split by BF_num_packets (full flow
  length, no prefix involved) into short (< T) and long (>= T) groups;
- per group: flow counts, short-group byte/packet shares, and distribution
  statistics (mean/std/min/max/median/quartiles) of per-flow scalars
  (packets, bytes, duration, mean/max packet size, mean IAT, backward
  fraction);
- these group features are CONCATENATED to the existing hybrid pool (prefix
  aggregates + fullflow aggregates + merged stream + cross-flow
  distributions); the leaderboard keeps a hybrid-only arm so the incremental
  value of grouping is measured, not assumed.

Gate identical: validation four metrics single-fit AND 3-seed stability ->
frozen_config -> open this split's test exactly once. Otherwise FINAL_FAIL:
~97.3 ceiling confirmed by two independent representations + an older sealed
test, no further same-family rounds.

Stages: extract | search | test
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path('/workspace/Huawei')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_scenario_mirage_activity_20260922 import (  # noqa: E402
    ACTS, APPS, SEED, TARGETS, pure_captures)
from run_scenario_mirage_hybrid_20260922 import finite, fit_predict, met  # noqa: E402

SRC = ROOT / 'output/all_scenarios_metric_push_20260922/mirage_activity'
OUT = ROOT / 'output/final_experiment_closure_20260922/mirage_groupsplit'
CACHE = OUT / 'group_features.csv'
LEN_THRESHOLD = 10  # packets; full-flow BF_num_packets, no prefix involved


def per_flow_scalars(flow):
    md = flow.get('flow_metadata', {})
    p = flow.get('packet_data', {})
    ip = np.asarray(p.get('IP_packet_bytes', []), dtype=float)
    iat = np.asarray(p.get('iat', []), dtype=float)
    dirs = np.asarray(p.get('packet_dir', []), dtype=float)
    n = len(ip)
    dur = max(finite(md.get('BF_duration', 0)), 1e-9)
    bwd = float(np.mean(dirs < 0)) if n else 0.0
    return [finite(md.get('BF_num_packets', n)),
            finite(md.get('BF_IP_packet_bytes', ip.sum())),
            dur,
            float(ip.mean()) if n else 0.0,
            float(ip.max()) if n else 0.0,
            float(iat.mean()) if len(iat) else 0.0,
            bwd]


def stat8(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return [0.0] * 8
    return [float(np.mean(a)), float(np.std(a)), float(np.min(a)),
            float(np.max(a)), float(np.median(a)), float(np.quantile(a, .25)),
            float(np.quantile(a, .75)), float(np.quantile(a, .90))]


def stage_extract() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in pure_captures():
        try:
            d = json.loads(Path(rec['file']).read_text())
        except Exception:  # noqa: BLE001
            continue
        flows = []
        for flow_id, flow in d.items():
            md = flow.get('flow_metadata', {})
            if md.get('BF_label') != rec['app'] or \
                    md.get('BF_activity') != rec['activity']:
                continue
            ts = flow.get('packet_data', {}).get('timestamp', [])
            flows.append((finite(ts[0]) if ts else 0.0, flow_id, flow))
        flows.sort(key=lambda z: (z[0], z[1]))
        if not flows:
            continue
        scal = np.asarray([per_flow_scalars(f) for _, _, f in flows],
                          dtype=float)
        short = scal[:, 0] < LEN_THRESHOLD
        row = {'_observation_id': Path(rec['file']).name,
               '_label': rec['activity'],
               '_label_id': ACTS.index(rec['activity']),
               '_n_flows': float(len(flows)),
               '_n_short': float(short.sum()),
               '_n_long': float((~short).sum())}
        tot_b = max(scal[:, 1].sum(), 1.0)
        tot_p = max(scal[:, 0].sum(), 1.0)
        row['_short_byte_share'] = float(scal[short, 1].sum() / tot_b)
        row['_short_pkt_share'] = float(scal[short, 0].sum() / tot_p)
        for gname, mask in (('s', short), ('l', ~short)):
            G = scal[mask]
            for j in range(scal.shape[1]):
                for i, v in enumerate(stat8(G[:, j])):
                    row[f'grp_{gname}{j}__q{i}'] = v
        rows.append(row)
        if len(rows) % 100 == 0:
            print('groups extracted', len(rows), flush=True)
    pd.DataFrame(rows).to_csv(CACHE, index=False)
    print('TOTAL captures with group features', len(rows), flush=True)


def load_merged() -> pd.DataFrame:
    merged = pd.read_csv(SRC / 'capture_features.csv')
    hyb = pd.read_csv(SRC / 'hybrid_capture_features.csv')
    grp = pd.read_csv(CACHE)
    df = merged.merge(hyb, on='_observation_id', suffixes=('', '_hyb'))
    df = df.merge(grp, on='_observation_id', suffixes=('', '_grp'))
    n_before = len(df)
    sp = pd.read_csv(SRC / 'fresh_split.csv')
    df = df.merge(sp[['_observation_id', '_split']], on='_observation_id')
    if len(df) != n_before or len(df) != len(sp):
        raise RuntimeError(f'capture coverage mismatch: hybrid={n_before} '
                           f'merged={len(df)} split={len(sp)}')
    if (df['_label_grp'] != df['_label']).any():
        raise RuntimeError('label mismatch between caches')
    return df


def stage_search() -> None:
    df = load_merged()
    tr = df[df._split == 'train'].reset_index(drop=True)
    va = df[df._split == 'validation'].reset_index(drop=True)
    all_cols = [c for c in df.columns if not c.startswith('_')]
    all_cols = [f for f in all_cols
                if tr[f].replace([np.inf, -np.inf], np.nan).fillna(0).std() > 0]
    grp_cols = [c for c in all_cols if c.startswith('grp_')]
    hybrid_cols = [c for c in all_cols if c not in grp_cols]
    # meta group features (counts/shares) carry the '_' prefix and are excluded
    # from the pool by construction; the grp_* q-statistics carry them.

    def rank(pool):
        X = tr[pool].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = tr['_label_id'].astype(int).to_numpy()
        r = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=.05,
                          subsample=.9, colsample_bytree=.9, reg_lambda=1.5,
                          random_state=SEED, n_jobs=8, eval_metric='mlogloss',
                          tree_method='hist')
        r.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
        order = np.argsort(r.feature_importances_)[::-1]
        return [pool[i] for i in order]

    rank_hybrid = rank(hybrid_cols)
    rank_full = rank(all_cols)
    pd.DataFrame({'feature': rank_full}).to_csv(
        OUT / 'train_feature_ranking.csv', index=False)
    rows = []
    arms = {
        'hybrid_only': lambda k: rank_hybrid[:k] if k else list(hybrid_cols),
        'hybrid_groups': lambda k: rank_full[:k] if k else list(all_cols),
        'groups_only': lambda k: grp_cols,
    }
    for aname, fsel in arms.items():
        for k in (48, 64, 128, 0):
            fs = fsel(k)
            if not fs:
                continue
            for model in ('xgb', 'extra', 'hgb', 'ens_xe'):
                p = fit_predict(tr, va, fs, model, SEED + 10)
                m = met(va['_label_id'].astype(int).to_numpy(), p)
                rows.append({'arm': aname, 'k': k if k else len(fs),
                             'model': model,
                             **{x: m[x] for x in (
                                 'accuracy', 'macro_precision', 'macro_recall',
                                 'macro_f1', 'max_class_fpr', 'errors')},
                             'four_pass': m['four_pass']})
                print('VAL', rows[-1], flush=True)
    lb = pd.DataFrame(rows)
    lb.to_csv(OUT / 'validation_leaderboard.csv', index=False)
    top = lb.sort_values(['four_pass', 'macro_recall', 'accuracy'],
                         ascending=False).iloc[0]
    aname, k, model = top['arm'], int(top['k']), top['model']
    fs = arms[aname](k if k in (48, 64, 128) else 0)
    ms = [met(va['_label_id'].astype(int).to_numpy(),
              fit_predict(tr, va, fs, model, SEED + 200 + 40 * i))
          for i in range(3)]
    stab_pass = (np.mean([m['accuracy'] for m in ms]) >= TARGETS['accuracy']
                 and np.min([m['macro_precision'] for m in ms]) >= TARGETS['macro_precision']
                 and np.mean([m['macro_recall'] for m in ms]) >= TARGETS['macro_recall']
                 and np.max([m['max_class_fpr'] for m in ms]) <= TARGETS['max_per_class_fpr'])
    outcome = {
        'experiment': 'final_experiment_closure_20260922/mirage_groupsplit',
        'scenario': 'D3_mirage_flow_length_grouped',
        'split_reused': ('all_scenarios_metric_push_20260922/mirage_activity/'
                         'fresh_split.csv (seed 20260922, registered before '
                         'any search; test never opened)'),
        'len_threshold_packets': LEN_THRESHOLD,
        'representation': ('hybrid pool + short/long flow group distributions '
                           '(counts/shares are recorded as meta, q-statistics '
                           'as features)'),
        'arm': aname, 'selected_features': fs, 'k': len(fs), 'model': model,
        'refit_seed': SEED + 900, 'targets': TARGETS,
        'validation_best': {x: float(top[x]) for x in (
            'accuracy', 'macro_precision', 'macro_recall', 'max_class_fpr')},
        'stability': {'macro_recall_mean': float(np.mean(
            [m['macro_recall'] for m in ms])),
            'macro_recall_min': float(np.min([m['macro_recall'] for m in ms])),
            'all_pass_seeds': int(sum(m['four_pass'] for m in ms))},
    }
    if bool(top['four_pass']) and stab_pass:
        if (SRC / 'test_metrics.json').exists() or \
                (SRC / 'hybrid_test_metrics.json').exists():
            outcome['status'] = 'test_already_consumed'
            (OUT / 'search_outcome.json').write_text(json.dumps(outcome, indent=2))
            print('NOT FROZEN (test consumed)', flush=True)
            return
        outcome['fresh_test_policy'] = ('this split\'s test opened exactly once '
                                        'after this file is written; no retuning')
        (OUT / 'frozen_config.json').write_text(
            json.dumps(outcome, indent=2, ensure_ascii=False))
        print('FROZEN', outcome['stability'], flush=True)
    else:
        outcome['status'] = 'validation_not_passed'
        outcome['reason'] = ('four-metric gate not passed (best Macro-R '
                             f'{top["macro_recall"]:.4f}); test stays sealed '
                             '-> FINAL_FAIL of this round')
        (OUT / 'search_outcome.json').write_text(json.dumps(outcome, indent=2))
        print('NOT FROZEN', outcome['reason'], flush=True)


def stage_test() -> None:
    tm = OUT / 'test_metrics.json'
    if tm.exists():
        raise RuntimeError('test already opened: one-shot rule')
    fp = OUT / 'frozen_config.json'
    if not fp.exists():
        raise RuntimeError('frozen_config.json missing: gate not passed')
    frozen = json.loads(fp.read_text())
    df = load_merged()
    dev = df[df._split.isin(['train', 'validation'])].reset_index(drop=True)
    te = df[df._split == 'test'].reset_index(drop=True)
    p = fit_predict(dev, te, frozen['selected_features'], frozen['model'],
                    frozen['refit_seed'])
    m = met(te['_label_id'].astype(int).to_numpy(), p)
    passes = {'accuracy': m['accuracy'] >= TARGETS['accuracy'],
              'macro_precision': m['macro_precision'] >= TARGETS['macro_precision'],
              'macro_recall': m['macro_recall'] >= TARGETS['macro_recall'],
              'max_per_class_fpr': m['max_class_fpr'] <= TARGETS['max_per_class_fpr']}
    result = {
        'frozen_config_sha256': hashlib.sha256(fp.read_bytes()).hexdigest(),
        'metrics': {x: m[x] for x in ('accuracy', 'macro_precision',
                                      'macro_recall', 'macro_f1',
                                      'max_class_fpr', 'errors')},
        'n_test': int(len(te)), 'targets': TARGETS, 'pass': passes,
        'all_pass': all(passes.values()),
        'opened_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'one_shot': True}
    tm.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result['metrics'], indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['extract', 'search', 'test'])
    args = ap.parse_args()
    {'extract': stage_extract, 'search': stage_search,
     'test': stage_test}[args.stage]()


if __name__ == '__main__':
    main()
