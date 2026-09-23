# -*- coding: utf-8 -*-
"""Scenario D (2026-09-22): MIRAGE activity classification, fresh round.

Prior state (ledger / output/mirage_final_20260919): 4-app pooled 3-class
(chat/audiocall/videocall) capture-level classifier, sealed Test opened once:
Acc 97.22 / Macro-P 97.60 / Macro-R 96.75 / MaxFPR 3.13 — Macro Recall below
98 (chat R=93.1). That test is consumed; this round never reuses it as fresh.

New protocol:
- grouped fresh split (seed 20260922) over PURE single-activity captures of
  the 5 MIRAGE apps that have all three activities (Telegram/Skype/Discord/
  Zoom/Teams; Meet & Webex lack chat, WhatsApp/Signal only videocall, and are
  excluded to keep the class set well-defined), file-disjoint, 60/20/20 per
  activity, WRITTEN AND HASHED BEFORE ANY MODELING;
- observation = one capture (full activity context): merged packet stream of
  all target-app activity flows -> the generic 197-dim sequence family (same
  feature code as the mixed-16 and behavior scenarios), plus cross-flow
  distribution features (per-flow scalars aggregated mean/std/quantiles over
  flows), plus capture-level counts. No app name, no ports/SNI payload
  strings; packet shape/timing only;
- models: xgb / extra / hgb / ens_xe on top32/top64/full pools;
- gate: validation four metrics (single fit AND 3-seed stability) -> frozen
  config -> open this split's test exactly once.

Blindness: captures were parsed in earlier rounds (feature style differs);
labeled "fresh split over previously-explored captures".

Stages: extract | split | search | test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

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

from run_mixed_dataset_full_pass_20260921 import sequence_features  # noqa: E402

DS = Path('/workspace/datasets/MIRAGE-AppAct-2024')
OUT = ROOT / 'output/all_scenarios_metric_push_20260922/mirage_activity'
CACHE = OUT / 'capture_features.csv'
SEED = 20260922
APPS = ['Telegram', 'Skype', 'Discord', 'Zoom', 'Teams']
ACTS = ['chat', 'audiocall', 'videocall']
LID = {a: i for i, a in enumerate(ACTS)}
TARGETS = {'accuracy': .95, 'macro_precision': .95, 'macro_recall': .98,
           'max_per_class_fpr': .05}
MAX_STREAM = 4096
CATALOG = ROOT / 'output/mirage_final_20260919/activity_catalog.json'


class _P:
    __slots__ = ('timestamp', 'length', 'direction')

    def __init__(self, t: float, L: int, d: int):
        self.timestamp = t
        self.length = L
        self.direction = d


def pure_captures() -> List[Dict[str, str]]:
    cat = json.loads(CATALOG.read_text())
    out = []
    for r in cat:
        if r['app'] not in APPS:
            continue
        act = [a for a in ACTS if r['counts'].get(a, 0) > 0]
        if len(act) == 1:
            out.append({'app': r['app'], 'file': r['file'], 'activity': act[0]})
    return out


def _q(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p)) if len(a) else 0.0


def capture_features(rec: Dict[str, str]) -> Dict[str, float] | None:
    d = json.loads(Path(rec['file']).read_text())
    app_pkg = None
    # identify the foreground app package from the capture directory + labels
    flows = []
    for key, f in d.items():
        md = f.get('flow_metadata', {})
        act = md.get('BF_activity', '')
        lab = md.get('BF_label', '')
        if act != rec['activity'] or not lab:
            continue
        if app_pkg is None and md.get('BF_labeling_type') == 'exact' \
                and '.' in lab and ' ' not in lab:
            app_pkg = lab
        flows.append(f)
    if app_pkg is not None:
        flows = [f for f in flows
                 if f['flow_metadata'].get('BF_label') == app_pkg]
    if not flows:
        return None
    pkts = []
    per_flow = []
    endpoints = set()
    for f in flows:
        pdta = f.get('packet_data', {})
        ts = pdta.get('timestamp', [])
        byt = pdta.get('IP_packet_bytes', [])
        dirs = pdta.get('packet_dir', [])
        n = min(len(ts), len(byt), len(dirs))
        if n == 0:
            continue
        parts = f.get('flow_metadata', {}).get('BF_flow', key) if False else None
        for i in range(n):
            pkts.append(_P(float(ts[i]), int(byt[i]), int(dirs[i])))
        arr = np.asarray(byt[:n], float)
        dur = max(float(ts[n - 1] - ts[0]), 1e-9)
        up = np.asarray(dirs[:n], int) > 0
        per_flow.append({
            'n': float(n), 'bytes': float(arr.sum()), 'dur': dur,
            'rate': float(n / dur), 'up_frac': float(up.mean()),
            'size_mean': float(arr.mean()), 'size_max': float(arr.max()),
            'size_std': float(arr.std()) if n > 1 else 0.0,
        })
    if not per_flow or not pkts:
        return None
    pkts.sort(key=lambda p: p.timestamp)
    pkts = pkts[:MAX_STREAM]
    row: Dict[str, float] = {
        '_observation_id': Path(rec['file']).name,
        '_app': rec['app'], '_label': rec['activity'],
        '_label_id': LID[rec['activity']],
        '_n_flows': float(len(per_flow)),
        '_n_packets': float(len(pkts)),
        '_capture_bytes': float(sum(p.length for p in pkts)),
        '_capture_span': float(pkts[-1].timestamp - pkts[0].timestamp),
    }
    # cross-flow distributions
    for k in ('n', 'bytes', 'dur', 'rate', 'up_frac', 'size_mean', 'size_max',
              'size_std'):
        a = np.array([pf[k] for pf in per_flow], float)
        for stat, v in (('mean', a.mean()), ('std', a.std()),
                        ('q25', _q(a, 25)), ('q50', _q(a, 50)),
                        ('q75', _q(a, 75)), ('max', a.max())):
            row[f'xflow_{k}_{stat}'] = float(v)
    row.update(sequence_features(pkts))
    return row


def stage_extract() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in pure_captures():
        try:
            r = capture_features(rec)
        except Exception:  # noqa: BLE001
            r = None
        if r is not None:
            rows.append(r)
        if len(rows) % 50 == 0 and rows:
            print('extracted', len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    print('TOTAL captures', len(df), df.groupby('_label').size().to_dict(),
          flush=True)


def stage_split() -> None:
    df = pd.read_csv(CACHE)
    assign = {}
    for act in ACTS:
        ids = df[df._label == act]._observation_id.tolist()
        rng = random.Random(f'{SEED}:{act}')
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        nt = max(1, int(round(n * .6)))
        nv = max(1, int(round(n * .2)))
        for x in ids[:nt]:
            assign[x] = 'train'
        for x in ids[nt:nt + nv]:
            assign[x] = 'validation'
        for x in ids[nt + nv:]:
            assign[x] = 'test'
    sdf = pd.DataFrame([{'_observation_id': k, '_split': v,
                         '_label': df.set_index('_observation_id').loc[k, '_label']}
                        for k, v in assign.items()])
    sp = OUT / 'fresh_split.csv'
    sdf.to_csv(sp, index=False)
    sha = hashlib.sha256(sp.read_bytes()).hexdigest()
    (OUT / 'split_manifest.json').write_text(json.dumps({
        'seed': SEED, 'registered_before_search': True,
        'registered_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'unit': 'whole capture (file-disjoint, grouped by activity)',
        'apps': APPS, 'labels': ACTS, 'counts': sdf.groupby(
            ['_label', '_split']).size().unstack(fill_value=0).to_dict('index'),
        'split_sha256': sha}, indent=2))
    print('SPLIT', sdf.groupby(['_label', '_split']).size().to_dict(),
          'sha', sha[:12], flush=True)


def load_split() -> pd.DataFrame:
    df = pd.read_csv(CACHE)
    sp = pd.read_csv(OUT / 'fresh_split.csv')
    df = df.merge(sp[['_observation_id', '_split']], on='_observation_id')
    if df['_split'].isna().any():
        raise RuntimeError('unassigned captures')
    return df


def met(y: np.ndarray, p: np.ndarray) -> Dict[str, object]:
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
            'per_class': {ACTS[i]: {'P': float(P[i]), 'R': float(R[i]),
                                    'FPR': float(fprs[i]), 'N': int(s[i])}
                          for i in range(3)},
            'four_pass': bool(accuracy_score(y, p) >= TARGETS['accuracy']
                              and P.mean() >= TARGETS['macro_precision']
                              and R.mean() >= TARGETS['macro_recall']
                              and max(fprs) <= TARGETS['max_per_class_fpr'])}


def make_model(name: str, seed: int):
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


def fit_predict(train: pd.DataFrame, eval_df: pd.DataFrame, fs: Sequence[str],
                model: str, seed: int) -> np.ndarray:
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


def rank_features(train: pd.DataFrame, fs: Sequence[str]) -> List[str]:
    X = train[list(fs)].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train['_label_id'].astype(int).to_numpy()
    m = XGBClassifier(n_estimators=160, max_depth=5, learning_rate=.06,
                      subsample=.9, colsample_bytree=.9, reg_lambda=1.5,
                      random_state=SEED, n_jobs=8, eval_metric='mlogloss',
                      tree_method='hist')
    m.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    order = np.argsort(m.feature_importances_)[::-1]
    return [fs[i] for i in order]


def stage_search() -> None:
    df = load_split()
    tr = df[df._split == 'train'].reset_index(drop=True)
    va = df[df._split == 'validation'].reset_index(drop=True)
    fs_all = [c for c in df.columns if not c.startswith('_')]
    fs_all = [f for f in fs_all
              if tr[f].replace([np.inf, -np.inf], np.nan).fillna(0).std() > 0]
    rank = rank_features(tr, fs_all)
    rows = []
    for k in (32, 64, 0):
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
    lb.to_csv(OUT / 'validation_leaderboard.csv', index=False)
    top = lb.sort_values(['four_pass', 'macro_recall', 'accuracy'],
                         ascending=False).iloc[0]
    k, model = int(top['k']), top['model']
    fs = rank[:k] if top['k'] in (32, 64) else list(fs_all)
    ms = [met(va['_label_id'].astype(int).to_numpy(),
              fit_predict(tr, va, fs, model, SEED + 200 + 40 * i))
          for i in range(3)]
    stab = {'k': len(fs), 'model': model,
            'macro_recall_mean': float(np.mean([m['macro_recall'] for m in ms])),
            'macro_recall_min': float(np.min([m['macro_recall'] for m in ms])),
            'accuracy_mean': float(np.mean([m['accuracy'] for m in ms])),
            'macro_precision_min': float(np.min([m['macro_precision'] for m in ms])),
            'max_fpr_max': float(np.max([m['max_class_fpr'] for m in ms])),
            'all_pass_seeds': int(sum(m['four_pass'] for m in ms))}
    pd.DataFrame([stab]).to_csv(OUT / 'stability.csv', index=False)
    stab_pass = (stab['accuracy_mean'] >= TARGETS['accuracy']
                 and stab['macro_precision_min'] >= TARGETS['macro_precision']
                 and stab['macro_recall_mean'] >= TARGETS['macro_recall']
                 and stab['max_fpr_max'] <= TARGETS['max_per_class_fpr'])
    outcome = {
        'scenario': 'D_mirage_activity_fresh_v2',
        'blinding': 'fresh pre-registered split (seed 20260922) over previously '
                    'parsed MIRAGE captures; earlier sealed test not reused',
        'representation': 'merged-stream sequence family + cross-flow '
                          'distributions (full activity context, no app id)',
        'selected_features': fs, 'k': len(fs), 'model': model,
        'refit_seed': SEED + 900, 'targets': TARGETS,
        'validation_best': {x: float(top[x]) for x in
                            ('accuracy', 'macro_precision', 'macro_recall',
                             'max_class_fpr')},
        'validation_stability': stab,
    }
    if bool(top['four_pass']) and stab_pass:
        fp = OUT / 'frozen_config.json'
        outcome['fresh_test_policy'] = ('test rows open exactly once next, '
                                        'refit on train+validation')
        fp.write_text(json.dumps(outcome, indent=2, ensure_ascii=False))
        print('FROZEN', stab, flush=True)
    else:
        outcome['status'] = 'validation_not_passed'
        outcome['reason'] = (f'best single pass={bool(top["four_pass"])}, '
                             f'stability pass={stab_pass}; test stays sealed')
        (OUT / 'search_outcome.json').write_text(json.dumps(outcome, indent=2))
        print('NOT FROZEN', outcome['reason'], flush=True)


def stage_test() -> None:
    tm = OUT / 'test_metrics.json'
    if tm.exists():
        raise RuntimeError('test_metrics.json exists: one-shot rule')
    fp = OUT / 'frozen_config.json'
    if not fp.exists():
        raise RuntimeError('frozen_config.json missing: gate not passed')
    frozen = json.loads(fp.read_text())
    df = load_split()
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
        'all_pass': all(passes.values()), 'per_class': m['per_class'],
        'cm': m['cm'], 'opened_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'one_shot': True}
    tm.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    pd.DataFrame([{'label': c, **v} for c, v in m['per_class'].items()]).to_csv(
        OUT / 'per_class_metrics.csv', index=False)
    print(json.dumps({k: result[k] for k in ('metrics', 'pass', 'all_pass')},
                     indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['extract', 'split', 'search', 'test'])
    args = ap.parse_args()
    {'extract': stage_extract, 'split': stage_split, 'search': stage_search,
     'test': stage_test}[args.stage]()


if __name__ == '__main__':
    main()
