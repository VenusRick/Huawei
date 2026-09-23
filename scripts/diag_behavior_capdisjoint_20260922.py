# -*- coding: utf-8 -*-
"""Root-cause diagnostic for the behavior capture-disjoint closure.

Answers one question with numbers: is the failure window-level noise or
capture-level wholesale misclassification? For every validation capture we
report the true label, the majority prediction over its windows, and the
share of windows predicted as the majority class. If most captures are
majority-wrong, the features carry capture-specific structure that a
label-free per-capture normalization cannot remove (background composition),
which is the FINAL_FAIL evidence.
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

OUT = cl.OUT


def main() -> None:
    lb = pd.read_csv(OUT / 'validation_leaderboard.csv')
    top = lb.sort_values(['target_pass', 'macro_recall', 'accuracy'],
                         ascending=False).iloc[0]
    df = cl.load_variant(top['win_cfg'], top['obs'])
    fs_all = [c for c in df.columns if not c.startswith('_')]
    tr = df[df._split == 'train'].reset_index(drop=True)
    va = df[df._split == 'validation'].reset_index(drop=True)
    rank = cl.rank_features(tr, fs_all)
    k = 0 if int(top['k']) == 197 else int(top['k'])
    fs = rank[:k] if k else list(fs_all)
    tr2, va2, _ = cl.apply_norm(tr, va, va.iloc[0:0], fs, top['norm'])
    p = cl.fit_predict(tr2, va2, fs, top['model'], cl.SEED + 10)
    va = va.assign(_pred=[cl.LABELS[i] for i in p])
    rows = []
    for f, g in va.groupby('_source_file'):
        maj = g['_pred'].value_counts()
        rows.append({'capture': f, 'true': g['_label'].iloc[0],
                     'n_windows': len(g),
                     'majority_pred': maj.index[0],
                     'majority_share': float(maj.iloc[0] / len(g)),
                     'accuracy': float((g['_pred'] == g['_label']).mean())})
    t = pd.DataFrame(rows).sort_values(['true', 'capture'])
    t.to_csv(OUT / 'per_capture_validation_diagnostic.csv', index=False)
    caps = t['capture'].nunique()
    maj_wrong = int((t['majority_pred'] != t['true']).sum())
    summary = {
        'config': {'win_cfg': top['win_cfg'], 'obs': top['obs'],
                   'norm': top['norm'], 'k': int(top['k']),
                   'model': top['model']},
        'n_validation_captures': int(caps),
        'captures_majority_wrong': maj_wrong,
        'captures_majority_wrong_frac': maj_wrong / caps,
        'mean_majority_share': float(t['majority_share'].mean()),
        'verdict': ('capture-level wholesale misclassification dominates' if
                    maj_wrong / caps >= 0.4 else
                    'failure is window-level; capture normalization angle '
                    'remains plausible'),
    }
    (OUT / 'capture_level_diagnostic.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(t.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
