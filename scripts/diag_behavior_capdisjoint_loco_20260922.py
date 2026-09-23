# -*- coding: utf-8 -*-
"""Leave-one-capture-out (LOCO) robustness evidence for the behavior
capture-disjoint FINAL_FAIL.

The registered capture-disjoint validation holds only 4 window-bearing
captures (audio has exactly ONE), so a single bad capture dominates the
gate. To make the final ruling robust -- without ever touching the sealed
capture-disjoint TEST captures -- this script runs LOCO over the TRAIN+
VALIDATION captures only (17 captures; the 4 test captures are excluded by
the registered split and never loaded into any fit or metric here):
for each non-test capture C, train on the other non-test captures and
evaluate C's windows. Reports pooled per-class recall, per-capture accuracy,
and the share of captures that are majority-misread, for each normalization
arm (raw / capz / caprank) at the best-search observation setting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/workspace/Huawei')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import run_closure_behavior_capdisjoint_20260922 as cl  # noqa: E402
from run_scenario_behavior_agg_20260922 import LABELS  # noqa: E402

OUT = cl.OUT


def loco(cfg: str, obs: str, norm: str, k: int, model: str):
    df = cl.load_variant(cfg, obs)
    nontest = df[df._split.isin(['train', 'validation'])].reset_index(drop=True)
    fs_all = [c for c in df.columns if not c.startswith('_')]
    rows = []
    for cap in sorted(nontest['_source_file'].unique()):
        tr = nontest[nontest._source_file != cap].reset_index(drop=True)
        te = nontest[nontest._source_file == cap].reset_index(drop=True)
        if len(tr) < 20 or len(te) < 3:
            continue
        rank = cl.rank_features(tr, fs_all)
        fs = rank[:k] if k else list(fs_all)
        tr2, te2, _ = cl.apply_norm(tr, te, te.iloc[0:0], fs, norm)
        p = cl.fit_predict(tr2, te2, fs, model, cl.SEED + 10)
        rows.append({'capture': cap, 'true': te['_label'].iloc[0],
                     'n': len(te), 'acc': float((p == te['_label_id']
                                                 .astype(int).to_numpy()).mean()),
                     'pred': LABELS[int(np.bincount(p).argmax())]})
    t = pd.DataFrame(rows)
    pooled = {}
    for lab in LABELS:
        m = t['true'] == lab
        pooled[lab] = {'n_captures': int(m.sum()),
                       'mean_capture_acc': float(t[m]['acc'].mean()) if m.any() else None,
                       'captures_majority_wrong': int((t[m]['pred'] != lab).sum())}
    macro = float(np.mean([v['mean_capture_acc'] for v in pooled.values()
                           if v['mean_capture_acc'] is not None]))
    return t, pooled, macro


def main() -> None:
    lb = pd.read_csv(OUT / 'validation_leaderboard.csv')
    results = {}
    for _, top in lb.sort_values(['macro_recall', 'accuracy'],
                                 ascending=False).head(3).iterrows():
        k = 0 if int(top['k']) == 197 else int(top['k'])
        t, pooled, macro = loco(top['win_cfg'], top['obs'], top['norm'], k,
                                top['model'])
        key = f"{top['win_cfg']}/{top['obs']}/{top['norm']}/k{int(top['k'])}/{top['model']}"
        results[key] = {'per_class': pooled, 'loco_macro_capture_acc': macro,
                        'captures': t.to_dict('records')}
        print(key, 'LOCO macro capture acc', round(macro, 4), flush=True)
        print(t.to_string(index=False), flush=True)
    summary = {
        'purpose': 'robustness evidence for the FINAL_FAIL ruling; sealed '
                   'test captures excluded by the registered split',
        'nontest_captures_used': 'train+validation captures only',
        'results': results,
    }
    (OUT / 'loco_diagnostic.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print('written loco_diagnostic.json', flush=True)


if __name__ == '__main__':
    main()
