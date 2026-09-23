# -*- coding: utf-8 -*-
"""Quick dev-side exploration (train/validation ONLY, test untouched).

Tests, on the fixed fresh split:
1. seq-subset budget cost: top64 vs top96 vs top128 vs full pool.
2. ensemble variants incl. hgb and weighted voting.
3. minority-class oversampling (generic, train-side only).
4. per-class additive logit calibration fitted on train OOF predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/Huawei')
import importlib.util
spec = importlib.util.spec_from_file_location(
    'fp', '/workspace/Huawei/scripts/run_mixed_dataset_full_pass_20260921.py')
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)

from sklearn.model_selection import StratifiedKFold

OUT = fp.OUT


def main() -> None:
    df = fp.load_split()
    train = df[df['_split'] == 'train_pool'].reset_index(drop=True)
    val = df[df['_split'] == 'validation'].reset_index(drop=True)
    labels = sorted(df['_label'].unique())
    fs_all = fp.common_shape_features(df.columns)
    seq = [f for f in fs_all if f.startswith('seq_')]
    rt = [f for f in fs_all if not f.startswith('seq_')]

    # rank seq features by pooled score on train (same as search would)
    tab = fp.build_score_table(train, fs_all)
    pooled_rank = fp.rank_variant(tab, 'pooled')
    seq_rank = [f for f in pooled_rank if f.startswith('seq_')]
    combos = {
        'seq_top64': seq_rank[:64],
        'seq_top96': seq_rank[:96],
        'seq_top128': seq_rank[:128],
        'seq_full': seq,
        'seq_rt_top96': pooled_rank[:96],
        'rr16_plus_seq48': None,  # filled below
    }
    rr = []
    for d in fp.DATASETS:
        own = tab.sort_values([f'score_{d}', 'pooled_score'], ascending=False).feature.tolist()
        rr += [f for f in own[:16] if f not in rr]
    for f in seq_rank:
        if len(rr) >= 64:
            break
        if f not in rr:
            rr.append(f)
    combos['rr16_plus_seq48'] = rr[:64]

    def run(tag, fs, model, seed=20260922 + 33):
        m = fp.evaluate(train, val, fs, model, seed, labels)
        print(f'{tag:34s} {model:9s} acc={m["accuracy"]:.4f} mP={m["macro_precision"]:.4f} '
              f'mR={m["macro_recall"]:.4f} maxFPR={m["max_per_class_fpr"]:.4f}', flush=True)
        return m

    for name, fs in combos.items():
        if fs is None:
            continue
        for model in ('ens_xe', 'ens_all', 'ens_eh'):
            run(name, fs, model)

    # minority oversampling on the best pool
    fs = combos['seq_full']
    counts = train['_label'].value_counts()
    med = int(counts.median())
    parts = [train]
    for c, n in counts.items():
        if n < med:
            g = train[train['_label'] == c]
            rng = np.random.RandomState(7)
            take = rng.choice(g.index, size=med - n, replace=True)
            parts.append(train.loc[take])
    tr_os = pd.concat(parts, ignore_index=True)
    m = fp.evaluate(tr_os, val, fs, 'ens_xe', 20260922 + 33, labels)
    print(f'{"seq_full + minority-oversample":34s} {"ens_xe":9s} acc={m["accuracy"]:.4f} '
          f'mP={m["macro_precision"]:.4f} mR={m["macro_recall"]:.4f} '
          f'maxFPR={m["max_per_class_fpr"]:.4f}', flush=True)

    # per-class additive calibration fitted on train OOF probabilities
    for model in ('ens_xe',):
        skf = StratifiedKFold(5, shuffle=True, random_state=11)
        lid = {c: i for i, c in enumerate(labels)}
        y = train['_label'].map(lid).to_numpy()
        oof = np.zeros((len(train), len(labels)))
        for tr_i, te_i in skf.split(np.zeros(len(y)), y):
            tr_fold = train.iloc[tr_i]
            oof[te_i] = fp.fit_predict_proba(tr_fold, train.iloc[te_i], fs, model,
                                             20260922 + 77, labels)
        # coordinate ascent on additive offsets maximizing macro recall with a
        # precision/FPR guard, evaluated on OOF
        base = oof.copy()

        def macro_metrics(prob, yy):
            pred = prob.argmax(axis=1)
            rec = np.array([(pred[yy == i] == i).mean() if (yy == i).any() else 1.0
                            for i in range(len(labels))])
            prec = np.array([(pred == i)[yy == i].sum() / max((pred == i).sum(), 1)
                             for i in range(len(labels))])
            return rec.mean(), prec.mean()

        offsets = np.zeros(len(labels))
        best_score = -1
        for it in range(6):
            for i in range(len(labels)):
                for delta in (0.05, 0.1, 0.2, 0.35, 0.5, -0.1, -0.2):
                    trial = offsets.copy()
                    trial[i] += delta
                    mr, mp = macro_metrics(base + trial[None, :], y)
                    score = mr + 0.3 * min(mp, 0.97)
                    if score > best_score:
                        best_score = score
                        offsets = trial
        print('OOF macroR before/after calibration:',
              macro_metrics(base, y)[0], macro_metrics(base + offsets[None, :], y)[0],
              flush=True)
        prob_val = fp.fit_predict_proba(train, val, fs, model, 20260922 + 33, labels)
        pv = prob_val + offsets[None, :]
        pred = [labels[i] for i in pv.argmax(axis=1)]
        mm = fp.official_metrics(val['_label'].to_numpy(), np.array(pred), labels)
        print(f'{"seq_full + OOF-calibrated ens_xe":34s} acc={mm["accuracy"]:.4f} '
              f'mP={mm["macro_precision"]:.4f} mR={mm["macro_recall"]:.4f} '
              f'maxFPR={mm["max_per_class_fpr"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
