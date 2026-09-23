# -*- coding: utf-8 -*-
"""Scenario D2 (2026-09-22): MIRAGE hybrid representation on the SAME fresh
split registered in mirage_activity/ (split_sha b92ffc76..., seed 20260922).

Rationale: the merged-stream-only representation (scenario D) plateaued at
~94.7% validation accuracy, below the 20260919 hybrid representation (~97.2%
on its own, consumed split). This round combines, per capture:
- the 20260919 hybrid representation, recomputed for all 5 apps:
  prefix{32,64,128} aggregated over target flows + fullflow metadata
  aggregated over all target flows (mean/std/median/sum/count);
- the merged-stream sequence family + cross-flow distributions (scenario D).

The split file is REUSED UNCHANGED (feature work never touches the split).
Gate identical: validation four metrics (single + 3-seed stability) ->
frozen_config_hybrid.json -> open this split's test exactly once. If scenario
D already opened its test (frozen_config.json exists), this hybrid may not
claim a fresh test and will stop before stage test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path('/workspace/Huawei')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_scenario_mirage_activity_20260922 import (ACTS, APPS, SEED, TARGETS,  # noqa: E402
                                                    pure_captures)
import run_scenario_mirage_activity_20260922 as base_scen  # noqa: E402

DS = Path('/workspace/datasets/MIRAGE-AppAct-2024')
OUT = ROOT / 'output/all_scenarios_metric_push_20260922/mirage_activity'
CACHE = OUT / 'hybrid_capture_features.csv'


def finite(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def stat8(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return [0.0] * 8
    return [float(np.mean(a)), float(np.std(a)), float(np.min(a)),
            float(np.max(a)), float(np.median(a)), float(np.quantile(a, .25)),
            float(np.quantile(a, .75)), float(np.quantile(a, .90))]


def prefix64(flow, budget):
    p = flow.get('packet_data', {})
    ip = np.asarray(p.get('IP_packet_bytes', [])[:budget], dtype=float)
    payload = np.asarray(p.get('L4_payload_bytes', [])[:budget], dtype=float)
    iat = np.asarray(p.get('iat', [])[:budget], dtype=float)
    dirs = np.asarray(p.get('packet_dir', [])[:budget], dtype=int)
    iph = np.asarray(p.get('IP_header_bytes', [])[:budget], dtype=float)
    l4h = np.asarray(p.get('L4_header_bytes', [])[:budget], dtype=float)
    ts = np.asarray(p.get('timestamp', [])[:budget], dtype=float)
    n = min(len(ip), len(payload), len(iat), len(dirs))
    if n < 3:
        return None
    ip, payload, iat, dirs = ip[:n], payload[:n], iat[:n], dirs[:n]
    iph = iph[:n] if len(iph) >= n else np.zeros(n)
    l4h = l4h[:n] if len(l4h) >= n else np.zeros(n)
    header = iph + l4h
    duration = max(float(ts[n - 1] - ts[0]), 1e-9) if len(ts) >= n \
        else max(float(np.sum(np.maximum(iat, 0))), 1e-9)
    u = set(np.unique(dirs).tolist())
    if u.issubset({0, 1}):
        f = (dirs == 0)
        b = (dirs == 1)
    else:
        f = (dirs > 0)
        b = (dirs < 0)
    fc, bc = int(f.sum()), int(b.sum())
    fb, bb = float(ip[f].sum()), float(ip[b].sum())
    changes = int(np.sum(dirs[1:] != dirs[:-1])) if n > 1 else 0
    vals = [float(n), duration, float(ip.sum()), float(payload.sum()), float(fc),
            float(bc), fb, bb, float(n / duration),
            float(ip.sum() / duration),
            float(payload.sum() / max(ip.sum(), 1.0)), float(fc / max(bc, 1)),
            float(fb / max(bb, 1.0)), float(changes), float(changes / max(n - 1, 1))]
    vals += stat8(ip) + stat8(payload) + stat8(iat)
    for arr in [ip[f], ip[b]]:
        q = stat8(arr)
        vals += q[:5]
    for arr in [iat[f], iat[b]]:
        q = stat8(arr)
        vals += [q[0], q[1], q[3], q[4]]
    vals += stat8(header)[:3]
    den = max(n - 1, 1)
    if u.issubset({0, 1}):
        pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    else:
        pairs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    vals += [float(np.sum((dirs[:-1] == x) & (dirs[1:] == y)) / den)
             for x, y in pairs]
    return np.asarray(vals, dtype=float)


FULLFLOW_KEYS = [
    'BF_num_packets', 'BF_IP_packet_bytes', 'BF_L4_payload_bytes', 'BF_duration',
    'UF_num_packets', 'UF_IP_packet_bytes', 'UF_L4_payload_bytes', 'UF_duration',
    'UF_MSS', 'UF_WS', 'DF_num_packets', 'DF_IP_packet_bytes',
    'DF_L4_payload_bytes', 'DF_duration', 'DF_MSS', 'DF_WS',
    'up_down_packets_ratio', 'up_down_bytes_ratio', 'payload_ratio',
    'packet_rate', 'byte_rate',
]


def fullflow_row(flow) -> List[float]:
    md = flow.get('flow_metadata', {})
    out = [finite(md.get(k, 0)) for k in FULLFLOW_KEYS[:16]]
    ff = flow.get('flow_features', {})
    for fam in ('packet_length', 'iat'):
        obj = ff.get(fam, {})
        for direction in ('biflow', 'upstream_flow', 'downstream_flow'):
            st = obj.get(direction, {})
            for stat in ('min', 'max', 'mean', 'std', 'skew', 'kurtosis',
                         '10_percentile', '50_percentile', '90_percentile'):
                out.append(finite(st.get(stat, 0)))
    down_pkts = max(out[10], 1.0)   # DF_num_packets
    out += [out[5] / down_pkts, out[6] / max(out[11], 1.0),
            out[2] / max(out[1], 1.0), out[0] / max(out[3], 1e-9),
            out[1] / max(out[3], 1e-9)]
    return out


def aggregate_matrix(V):
    V = np.asarray(V, dtype=float)
    return np.concatenate([V.mean(0), V.std(0), np.median(V, axis=0), V.sum(0),
                           [len(V)]])


def stage_extract() -> None:
    base_scen  # noqa: B018 - imported for shared constants
    rows = []
    for rec in pure_captures():
        try:
            d = json.loads(Path(rec['file']).read_text())
        except Exception:  # noqa: BLE001
            continue
        flows = []
        for flow_id, flow in d.items():
            md = flow.get('flow_metadata', {})
            if md.get('BF_label') != rec['app'] or md.get('BF_activity') != rec['activity']:
                continue
            ts = flow.get('packet_data', {}).get('timestamp', [])
            flows.append((finite(ts[0]) if ts else 0.0, flow_id, flow))
        flows.sort(key=lambda z: (z[0], z[1]))
        if not flows:
            continue
        row = {'_observation_id': Path(rec['file']).name,
               '_label': rec['activity'],
               '_label_id': ACTS.index(rec['activity']),
               '_n_target_flows': float(len(flows))}
        for budget in (32, 64, 128):
            vv = [prefix64(f, budget) for _, _, f in flows]
            vv = [z for z in vv if z is not None]
            if vv:
                agg = aggregate_matrix(vv)
                for i, v in enumerate(agg):
                    row[f'pfx{budget}__f{i:04d}'] = float(v)
        V = np.asarray([fullflow_row(f) for _, _, f in flows], dtype=float)
        agg = aggregate_matrix(V)
        for i, v in enumerate(agg):
            row[f'full__f{i:04d}'] = float(v)
        rows.append(row)
        if len(rows) % 50 == 0:
            print('hybrid extracted', len(rows), flush=True)
    pd.DataFrame(rows).to_csv(CACHE, index=False)
    print('TOTAL hybrid captures', len(rows), flush=True)


def met(y, p):
    P, R, F1, s = precision_recall_fscore_support(y, p, labels=[0, 1, 2],
                                                  zero_division=0)
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    fprs = []
    for i in range(3):
        fp = cm[:, i].sum() - cm[i, i]
        neg = cm.sum() - cm[i, :].sum()
        fprs.append(fp / neg if neg else 0.0)
    return {'accuracy': float(accuracy_score(y, p)),
            'macro_precision': float(P.mean()), 'macro_recall': float(R.mean()),
            'macro_f1': float(F1.mean()), 'max_class_fpr': float(max(fprs)),
            'errors': int((p != y).sum()),
            'four_pass': bool(accuracy_score(y, p) >= TARGETS['accuracy']
                              and P.mean() >= TARGETS['macro_precision']
                              and R.mean() >= TARGETS['macro_recall']
                              and max(fprs) <= TARGETS['max_per_class_fpr'])}


def make_model(name, seed):
    if name == 'xgb':
        return XGBClassifier(n_estimators=320, max_depth=6, learning_rate=.035,
                             subsample=.9, colsample_bytree=.9, reg_lambda=1.8,
                             random_state=seed, n_jobs=8, eval_metric='mlogloss',
                             tree_method='hist')
    if name == 'extra':
        return ExtraTreesClassifier(n_estimators=800, max_features='sqrt',
                                    class_weight='balanced', random_state=seed,
                                    n_jobs=8)
    if name == 'hgb':
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=.06,
                                              l2_regularization=1.0,
                                              class_weight='balanced',
                                              random_state=seed)
    raise ValueError(name)


def fit_predict(train, eval_df, fs, model, seed):
    Xtr = train[list(fs)].replace([np.inf, -np.inf], np.nan).fillna(0)
    ytr = train['_label_id'].astype(int).to_numpy()
    Xe = eval_df[list(fs)].replace([np.inf, -np.inf], np.nan).fillna(0)
    members = ['xgb', 'extra'] if model == 'ens_xe' else [model]
    probs = []
    for j, mem in enumerate(members):
        m = make_model(mem, seed + j)
        if mem == 'xgb':
            m.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
        else:
            m.fit(Xtr, ytr)
        p = np.zeros((len(Xe), 3))
        p[:, m.classes_] = m.predict_proba(Xe)
        probs.append(p)
    return np.argmax(np.mean(probs, axis=0), axis=1)


def stage_search() -> None:
    merged = pd.read_csv(OUT / 'capture_features.csv')
    hyb = pd.read_csv(CACHE)
    df = merged.merge(hyb, on='_observation_id', suffixes=('', '_hyb'))
    sp = pd.read_csv(OUT / 'fresh_split.csv')
    df = df.merge(sp[['_observation_id', '_split']], on='_observation_id')
    tr = df[df._split == 'train'].reset_index(drop=True)
    va = df[df._split == 'validation'].reset_index(drop=True)
    fs_all = [c for c in df.columns if not c.startswith('_')]
    fs_all = [f for f in fs_all
              if tr[f].replace([np.inf, -np.inf], np.nan).fillna(0).std() > 0]
    X = tr[fs_all].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = tr['_label_id'].astype(int).to_numpy()
    ranker = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=.05,
                           subsample=.9, colsample_bytree=.9, reg_lambda=1.5,
                           random_state=SEED, n_jobs=8, eval_metric='mlogloss',
                           tree_method='hist')
    ranker.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    order = np.argsort(ranker.feature_importances_)[::-1]
    rank = [fs_all[i] for i in order]
    pd.DataFrame({'feature': rank,
                  'importance': [float(ranker.feature_importances_[i])
                                 for i in order]}).to_csv(
        OUT / 'hybrid_train_feature_ranking.csv', index=False)
    rows = []
    for k in (48, 64, 128, 0):
        fs = rank[:k] if k else list(fs_all)
        for model in ('xgb', 'extra', 'hgb', 'ens_xe'):
            p = fit_predict(tr, va, fs, model, SEED + 10)
            m = met(va['_label_id'].astype(int).to_numpy(), p)
            rows.append({'k': k if k else len(fs), 'model': model,
                         **{x: m[x] for x in ('accuracy', 'macro_precision',
                                              'macro_recall', 'macro_f1',
                                              'max_class_fpr', 'errors')},
                         'four_pass': m['four_pass']})
            print('VAL', rows[-1], flush=True)
    lb = pd.DataFrame(rows)
    lb.to_csv(OUT / 'hybrid_validation_leaderboard.csv', index=False)
    top = lb.sort_values(['four_pass', 'macro_recall', 'accuracy'],
                         ascending=False).iloc[0]
    k, model = int(top['k']), top['model']
    fs = rank[:k] if top['k'] in (48, 64, 128) else list(fs_all)
    ms = [met(va['_label_id'].astype(int).to_numpy(),
              fit_predict(tr, va, fs, model, SEED + 200 + 40 * i))
          for i in range(3)]
    stab_pass = (np.mean([m['accuracy'] for m in ms]) >= TARGETS['accuracy']
                 and np.min([m['macro_precision'] for m in ms]) >= TARGETS['macro_precision']
                 and np.mean([m['macro_recall'] for m in ms]) >= TARGETS['macro_recall']
                 and np.max([m['max_class_fpr'] for m in ms]) <= TARGETS['max_per_class_fpr'])
    outcome = {'scenario': 'D2_mirage_hybrid_on_fresh_split',
               'split_reused_from': 'mirage_activity/fresh_split.csv '
                                    '(sha b92ffc76fc37..., registered before '
                                    'any search)',
               'representation': 'prefix{32,64,128}+fullflow aggregates '
                                 '(20260919 recipe, recomputed for 5 apps) + '
                                 'merged-stream + cross-flow distributions',
               'selected_features': fs, 'k': len(fs), 'model': model,
               'refit_seed': SEED + 900, 'targets': TARGETS,
               'validation_best': {x: float(top[x]) for x in
                                   ('accuracy', 'macro_precision',
                                    'macro_recall', 'max_class_fpr')},
               'stability': {'macro_recall_mean': float(np.mean(
                   [m['macro_recall'] for m in ms])),
                   'macro_recall_min': float(np.min(
                       [m['macro_recall'] for m in ms])),
                   'all_pass_seeds': int(sum(m['four_pass'] for m in ms))}}
    if bool(top['four_pass']) and stab_pass:
        if (OUT / 'test_metrics.json').exists():
            outcome['status'] = 'test_already_opened_by_scenario_D'
            outcome['reason'] = ('scenario D already consumed the one-shot '
                                 'test; hybrid records validation-only')
            (OUT / 'hybrid_search_outcome.json').write_text(
                json.dumps(outcome, indent=2))
            print('NOT FROZEN (test consumed by scenario D)', flush=True)
            return
        (OUT / 'frozen_config_hybrid.json').write_text(
            json.dumps(outcome, indent=2, ensure_ascii=False))
        print('FROZEN hybrid', outcome['stability'], flush=True)
    else:
        outcome['status'] = 'validation_not_passed'
        (OUT / 'hybrid_search_outcome.json').write_text(json.dumps(outcome, indent=2))
        print('NOT FROZEN hybrid', flush=True)


def stage_test() -> None:
    tm = OUT / 'hybrid_test_metrics.json'
    if tm.exists():
        raise RuntimeError('hybrid test already opened: one-shot rule')
    fp = OUT / 'frozen_config_hybrid.json'
    if not fp.exists():
        raise RuntimeError('frozen_config_hybrid.json missing: gate not passed '
                           'or test consumed by scenario D')
    frozen = json.loads(fp.read_text())
    merged = pd.read_csv(OUT / 'capture_features.csv')
    hyb = pd.read_csv(CACHE)
    df = merged.merge(hyb, on='_observation_id', suffixes=('', '_hyb'))
    sp = pd.read_csv(OUT / 'fresh_split.csv')
    df = df.merge(sp[['_observation_id', '_split']], on='_observation_id')
    dev = df[df._split.isin(['train', 'validation'])].reset_index(drop=True)
    te = df[df._split == 'test'].reset_index(drop=True)
    p = fit_predict(dev, te, frozen['selected_features'], frozen['model'],
                    frozen['refit_seed'])
    m = met(te['_label_id'].astype(int).to_numpy(), p)
    passes = {'accuracy': m['accuracy'] >= TARGETS['accuracy'],
              'macro_precision': m['macro_precision'] >= TARGETS['macro_precision'],
              'macro_recall': m['macro_recall'] >= TARGETS['macro_recall'],
              'max_per_class_fpr': m['max_class_fpr'] <= TARGETS['max_per_class_fpr']}
    result = {'frozen_config_sha256': hashlib.sha256(fp.read_bytes()).hexdigest(),
              'metrics': {x: m[x] for x in ('accuracy', 'macro_precision',
                                            'macro_recall', 'macro_f1',
                                            'max_class_fpr', 'errors')},
              'n_test': int(len(te)), 'targets': TARGETS, 'pass': passes,
              'all_pass': all(passes.values()), 'per_class': None,
              'opened_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'one_shot': True}
    tm.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: result[k] for k in ('metrics', 'pass', 'all_pass')},
                     indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['extract', 'search', 'test'])
    args = ap.parse_args()
    {'extract': stage_extract, 'search': stage_search,
     'test': stage_test}[args.stage]()


if __name__ == '__main__':
    main()
