# -*- coding: utf-8 -*-
"""Final closure round (2026-09-22): Behavior15 raw-PCAP capture-disjoint.

Prior state (ledger 31.2): multi-flow aggregate windows PASS session-disjoint
(98.55/97.78/99.31/1.82 fresh Test) but FAIL capture-disjoint validation at
57.27/77.50/66.89/84.21 -- WORSE than the old single-flow file-disjoint
(84.75), evidence that aggregate-window features carry capture-level
background fingerprints (the composition of background flows is
capture-specific).

This is the final targeted round on the SAME pre-registered capture-disjoint
split (seed 20260922, output/all_scenarios_metric_push_20260922/
behavior_rawpcap/splits). Root-cause treatments, all label-free and
inference-visible:

1. observation variants: aggregate stream of ALL active sessions (as before)
   vs top-1 / top-2 dominant flows per window (background suppression at the
   observation level);
2. capture-relative normalization (transductive within the input capture,
   never crossing splits because capture-disjoint puts a whole file in one
   split, never using labels):
   - raw: no normalization (reproduces the failing baseline arm)
   - capz: each feature z-scored by its capture's window mean/std
   - caprank: each feature replaced by its within-capture quantile rank

Gate identical to the scenario push: validation four metrics single-fit AND
3-seed stability -> frozen_config -> open the capture-disjoint test exactly
once (never opened before). If the gate still fails -> FINAL_FAIL closure
with root-cause evidence; no further rounds of the same family.

Stages: windows | search | test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path('/workspace/Huawei')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import run_scenario_behavior_agg_20260922 as base  # noqa: E402
from run_scenario_behavior_agg_20260922 import (  # noqa: E402
    CACHE, FILES, LABELS, LID, MIN_WINDOW_PACKETS, SEED, TARGETS,
    WINDOW_CONFIGS, _P, metric, sequence_features)

SRC = ROOT / 'output/all_scenarios_metric_push_20260922/behavior_rawpcap'
OUT = ROOT / 'output/final_experiment_closure_20260922/behavior_capture_disjoint'
WDIR = OUT / 'windows'
OBS_VARIANTS = {  # name -> top-K flows by in-window bytes (0 = all)
    'all': 0, 'top1': 1, 'top2': 2,
}
NORMS = ('raw', 'capz', 'caprank')


def build_windows_topk(cfg: str, topk: int) -> pd.DataFrame:
    with open(CACHE, 'rb') as f:
        data = pickle.load(f)
    W, step_frac = WINDOW_CONFIGS[cfg]
    step = W * step_frac
    rows = []
    for fname, rec in data.items():
        lab = rec['label']
        all_ts = np.array([t for s in rec['sessions'] for (t, _, _) in s['pkts']])
        if len(all_ts) == 0:
            continue
        t0, t_end = float(all_ts.min()), float(all_ts.max())
        n_win = int(np.ceil((t_end - t0 - W) / step)) + 1 if t_end - t0 > W else 1
        for i in range(max(1, n_win)):
            lo, hi = t0 + i * step, t0 + i * step + W
            flow_pk: Dict[str, int] = {}
            flow_by: Dict[str, int] = {}
            sess_pkts = {}
            for s in rec['sessions']:
                pk = [(t, L, d) for (t, L, d) in s['pkts'] if lo <= t < hi]
                if pk:
                    sess_pkts[s['sid']] = pk
                    flow_pk[s['sid']] = len(pk)
                    flow_by[s['sid']] = sum(L for _, L, _ in pk)
            if not sess_pkts:
                continue
            if topk:
                keep = set(sorted(flow_by, key=lambda k: (-flow_by[k], k))[:topk])
                sess_pkts = {k: v for k, v in sess_pkts.items() if k in keep}
            agg = [p for pk in sess_pkts.values() for p in pk]
            if len(agg) < MIN_WINDOW_PACKETS:
                continue
            agg.sort(key=lambda p: p[0])
            if len(agg) > 4096:
                agg = agg[:4096]
            row = {'_observation_id': f'{fname}:{cfg}:{i}',
                   '_source_file': fname, '_label': lab,
                   '_label_id': LID[lab], '_n_packets': len(agg),
                   '_n_flows': len(sess_pkts)}
            row.update(sequence_features([_P(t, L, d) for t, L, d in agg]))
            rows.append(row)
    return pd.DataFrame(rows)


def stage_windows() -> None:
    WDIR.mkdir(parents=True, exist_ok=True)
    meta = {}
    for cfg in ('w30', 'w45', 'w60', 'w45ov'):
        for name, topk in OBS_VARIANTS.items():
            df = build_windows_topk(cfg, topk)
            path = WDIR / f'windows_{cfg}__{name}.csv'
            df.to_csv(path, index=False)
            meta[f'{cfg}__{name}'] = {
                'windows': int(len(df)),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            print('WINDOWS', cfg, name, len(df), flush=True)
    (OUT / 'windows_manifest.json').write_text(json.dumps(meta, indent=2))


def load_variant(cfg: str, obs: str):
    df = pd.read_csv(WDIR / f'windows_{cfg}__{obs}.csv')
    # registered capture-disjoint split (file -> split), unchanged
    sp = pd.read_csv(SRC / 'splits' / f'split_{cfg}.csv')
    cap = sp[['_observation_id', '_split_cap']].drop_duplicates()
    df = df.merge(cap, on='_observation_id', how='left')
    if df['_split_cap'].isna().any():
        raise RuntimeError(f'{int(df["_split_cap"].isna().sum())} windows '
                           'without registered capture split')
    df['_split'] = df['_split_cap']
    return df


def apply_norm(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame,
               fs: List[str], norm: str):
    """Capture-relative normalization. A whole capture lives in exactly one
    split (capture-disjoint), so normalizing within a capture never mixes
    splits and never touches labels."""
    if norm == 'raw':
        return tr, va, te

    def transform(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        if norm == 'capz':
            g = d.groupby('_source_file')[fs]
            mu, sd = g.transform('mean'), g.transform('std').fillna(0)
            sd = sd.replace(0, np.nan)
            d[fs] = ((d[fs] - mu) / sd).fillna(0)
        elif norm == 'caprank':
            d[fs] = d.groupby('_source_file')[fs].rank(pct=True).fillna(0.5)
        else:
            raise ValueError(norm)
        return d

    return transform(tr), transform(va), transform(te)


def make_model(name: str, seed: int):
    if name == 'xgb':
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=.04,
                             min_child_weight=1, subsample=.9, colsample_bytree=.9,
                             reg_lambda=1.5, random_state=seed, n_jobs=8,
                             eval_metric='mlogloss', tree_method='hist')
    if name == 'extra':
        return ExtraTreesClassifier(n_estimators=600, max_features='sqrt',
                                    class_weight='balanced', random_state=seed,
                                    n_jobs=8)
    raise ValueError(name)


def fit_predict(tr, va, fs, model, seed):
    Xtr = tr[list(fs)].replace([np.inf, -np.inf], np.nan).fillna(0)
    ytr = tr['_label_id'].astype(int).to_numpy()
    Xe = va[list(fs)].replace([np.inf, -np.inf], np.nan).fillna(0)
    probs = []
    for j, mem in enumerate((['xgb', 'extra'] if model == 'ens_xe'
                             else [model])):
        m = make_model(mem, seed + j)
        if mem == 'xgb':
            m.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
        else:
            m.fit(Xtr, ytr)
        p = np.zeros((len(Xe), 3))
        p[:, m.classes_] = m.predict_proba(Xe)
        probs.append(p)
    return np.argmax(np.mean(probs, axis=0), axis=1)


def rank_features(tr, fs):
    X = tr[list(fs)].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = tr['_label_id'].astype(int).to_numpy()
    m = XGBClassifier(n_estimators=160, max_depth=5, learning_rate=.06,
                      subsample=.9, colsample_bytree=.9, reg_lambda=1.5,
                      random_state=SEED, n_jobs=8, eval_metric='mlogloss',
                      tree_method='hist')
    m.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    order = np.argsort(m.feature_importances_)[::-1]
    return [fs[i] for i in order]


def eval_config(cfg, obs, norm, k, model, seed, return_metric=True):
    df = load_variant(cfg, obs)
    fs_all = [c for c in df.columns if not c.startswith('_')]
    tr = df[df._split == 'train'].reset_index(drop=True)
    va = df[df._split == 'validation'].reset_index(drop=True)
    te = df[df._split == 'test'].reset_index(drop=True)
    if len(tr) < 20 or len(va) < 8:
        return None
    rank = rank_features(tr, fs_all)
    fs = rank[:k] if k else list(fs_all)
    tr2, va2, te2 = apply_norm(tr, va, te, fs, norm)
    p = fit_predict(tr2, va2, fs, model, seed)
    return metric(va['_label_id'].astype(int).to_numpy(), p)


def stage_search() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in ('w30', 'w45', 'w60', 'w45ov'):
        for obs in OBS_VARIANTS:
            for norm in NORMS:
                for k in (32, 0):
                    for model in ('xgb', 'extra', 'ens_xe'):
                        m = eval_config(cfg, obs, norm, k, model, SEED + 10)
                        if m is None:
                            continue
                        rows.append({'win_cfg': cfg, 'obs': obs, 'norm': norm,
                                     'k': k or 197, 'model': model,
                                     **{x: m[x] for x in (
                                         'accuracy', 'macro_precision',
                                         'macro_recall', 'macro_f1',
                                         'max_class_fpr', 'errors')},
                                     'target_pass': m['target_pass']})
                        print('VAL', rows[-1], flush=True)
    lb = pd.DataFrame(rows)
    lb.to_csv(OUT / 'validation_leaderboard.csv', index=False)
    if not len(lb):
        (OUT / 'search_outcome.json').write_text(json.dumps(
            {'status': 'no_evaluable_configuration'}, indent=2))
        return
    top = lb.sort_values(['target_pass', 'macro_recall', 'accuracy'],
                         ascending=False).iloc[0]
    cfg, obs, norm = top['win_cfg'], top['obs'], top['norm']
    k = 0 if top['k'] == 197 else int(top['k'])
    model = top['model']
    ms = [eval_config(cfg, obs, norm, k, model, SEED + 200 + 40 * i)
          for i in range(3)]
    stab = {'win_cfg': cfg, 'obs': obs, 'norm': norm, 'k': int(top['k']),
            'model': model,
            'macro_recall_mean': float(np.mean([m['macro_recall'] for m in ms])),
            'accuracy_mean': float(np.mean([m['accuracy'] for m in ms])),
            'macro_precision_min': float(np.min([m['macro_precision'] for m in ms])),
            'max_fpr_max': float(np.max([m['max_class_fpr'] for m in ms])),
            'all_pass_seeds': int(sum(m['target_pass'] for m in ms))}
    pd.DataFrame([stab]).to_csv(OUT / 'stability.csv', index=False)
    single_pass = bool(top['target_pass'])
    stab_pass = (stab['accuracy_mean'] >= TARGETS['accuracy']
                 and stab['macro_precision_min'] >= TARGETS['macro_precision']
                 and stab['macro_recall_mean'] >= TARGETS['macro_recall']
                 and stab['max_fpr_max'] <= TARGETS['max_per_class_fpr'])
    # diagnostic: per-class validation metrics of the top config
    diag = eval_config(cfg, obs, norm, k, model, SEED + 10)
    outcome = {
        'experiment': 'final_experiment_closure_20260922/behavior_capture_disjoint',
        'scenario': 'B_behavior_rawpcap_multiflow_capture_disjoint',
        'split_reused': ('all_scenarios_metric_push_20260922/behavior_rawpcap/'
                         'splits (seed 20260922, registered before search); '
                         'capture-disjoint test never opened'),
        'treatments': {'obs_variants': OBS_VARIANTS, 'norms': NORMS},
        'win_cfg': cfg, 'obs': obs, 'norm': norm, 'k': int(top['k']),
        'model': model, 'refit_seed': SEED + 900,
        'validation_best': {x: float(top[x]) for x in (
            'accuracy', 'macro_precision', 'macro_recall', 'max_class_fpr')},
        'validation_per_class': diag['per_class'],
        'validation_stability': stab, 'targets': TARGETS,
    }
    if single_pass and stab_pass:
        outcome['fresh_test_policy'] = ('capture-disjoint test captures opened '
                                        'exactly once after this file is '
                                        'written; no retuning')
        (OUT / 'frozen_config.json').write_text(
            json.dumps(outcome, indent=2, ensure_ascii=False))
        print('FROZEN', json.dumps(stab), flush=True)
    else:
        outcome['status'] = 'validation_not_passed'
        outcome['reason'] = (f'best single pass={single_pass}, stability pass='
                             f'{stab_pass}; capture-disjoint test stays sealed '
                             '-> FINAL_FAIL of this round')
        (OUT / 'search_outcome.json').write_text(
            json.dumps(outcome, indent=2, ensure_ascii=False))
        print('NOT FROZEN', outcome['reason'], flush=True)


def stage_test() -> None:
    tm = OUT / 'test_metrics.json'
    if tm.exists():
        raise RuntimeError('test_metrics.json already exists: one-shot rule')
    fp = OUT / 'frozen_config.json'
    if not fp.exists():
        raise RuntimeError('frozen_config.json missing: gate not passed')
    frozen = json.loads(fp.read_text())
    cfg, obs, norm = frozen['win_cfg'], frozen['obs'], frozen['norm']
    df = load_variant(cfg, obs)
    fs_all = [c for c in df.columns if not c.startswith('_')]
    dev = df[df._split.isin(['train', 'validation'])].reset_index(drop=True)
    te = df[df._split == 'test'].reset_index(drop=True)
    # rank on dev, then normalize with the frozen k
    rank = rank_features(dev, fs_all)
    fs = rank[:frozen['k']] if frozen['k'] < len(rank) else list(fs_all)
    # capture-relative normalization needs per-split frames; dev captures and
    # test captures are disjoint files
    dev2, te2, _ = apply_norm(dev, te, dev.iloc[0:0], fs, norm)
    p = fit_predict(dev2, te2, fs, frozen['model'], frozen['refit_seed'])
    m = metric(te['_label_id'].astype(int).to_numpy(), p)
    passes = {'accuracy': m['accuracy'] >= TARGETS['accuracy'],
              'macro_precision': m['macro_precision'] >= TARGETS['macro_precision'],
              'macro_recall': m['macro_recall'] >= TARGETS['macro_recall'],
              'max_per_class_fpr': m['max_class_fpr'] <= TARGETS['max_per_class_fpr']}
    result = {
        'frozen_config_sha256': hashlib.sha256(fp.read_bytes()).hexdigest(),
        'granularity': 'capture_disjoint', 'win_cfg': cfg, 'obs': obs,
        'norm': norm,
        'metrics': {x: m[x] for x in ('accuracy', 'macro_precision',
                                      'macro_recall', 'macro_f1',
                                      'max_class_fpr', 'errors')},
        'n_test': int(len(te)), 'n_test_captures': int(
            te['_source_file'].nunique()),
        'targets': TARGETS, 'pass': passes, 'all_pass': all(passes.values()),
        'per_class': m['per_class'], 'cm': m['cm'],
        'opened_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'one_shot': True}
    tm.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    pd.DataFrame([{'label': c, **v} for c, v in m['per_class'].items()]).to_csv(
        OUT / 'per_class_metrics.csv', index=False)
    print(json.dumps({k: result[k] for k in ('metrics', 'pass', 'all_pass')},
                     indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['windows', 'search', 'test'])
    args = ap.parse_args()
    {'windows': stage_windows, 'search': stage_search,
     'test': stage_test}[args.stage]()


if __name__ == '__main__':
    main()
