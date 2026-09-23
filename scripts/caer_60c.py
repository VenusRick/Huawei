# -*- coding: utf-8 -*-
"""60 类实验的 C/A/E/R 特征账本（冻结后计算，不影响选择）。

口径与 output/fast_dpi_20260918_03/caer_accounting.json 一致：
- A = 冻结配置实际输出的特征集（本实验=冻结 Top64）；
- E = A 中在 validation 上类间单因素 ANOVA p<0.05 的特征（输出有效率）；
- R = A 中最终模型 importance>0（真实被引用）的特征；
- E/A 对齐官方"特征维度输出有效率>=80%"；人工修正可用率(>=90%)仍需
  人工修正记录，无记录则如实标 BLOCKED_DATA。

用法::

    python3 scripts/caer_60c.py --run-dir output/cstnet60_20260918_exp
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default='output/cstnet60_20260918_exp')
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)

    res = json.loads((run_dir / 'experiment_60c_results.json').read_text())
    frozen = res['frozen']
    prefix = res['validation'][frozen]['prefix']
    k = res['validation'][frozen]['n_features']

    rank_csv = run_dir / ('ranking_prefix64_trainonly.csv'
                          if prefix == 64 else
                          'ranking_prefix128_trainonly.csv')
    rank = pd.read_csv(rank_csv)
    A = rank['feature'].tolist()[:k]
    imp = dict(zip(rank['feature'], rank['importance']))

    cache = run_dir / f'features_prefix{prefix}_train_validation_test.pkl'
    if not cache.exists():
        cache = run_dir / f'features_prefix{prefix}_train_validation.pkl'
    df = pd.read_pickle(cache)
    va = df[df['_split'] == 'validation']

    labels_sorted = sorted(va['_label'].unique())
    masks = {c: (va['_label'] == c).to_numpy() for c in labels_sorted}
    e_pass, e_fail = [], []
    for f in A:
        x = pd.to_numeric(va[f], errors='coerce').replace(
            [np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
        samples = [x[masks[c]] for c in labels_sorted]
        try:
            _, p = f_oneway(*samples)
        except Exception:
            p = 1.0
        (e_pass if p < 0.05 else e_fail).append(f)

    r_used = [f for f in A if imp.get(f, 0) > 0]
    out = {
        'experiment': str(run_dir),
        'frozen': frozen,
        'A_published_frozen': {
            'n': len(A), 'names': A,
            'note': f'冻结 Top{k}@prefix{prefix}，train-only 排名，未用 validation/test 筛选'},
        'E_validation_supported': {
            'n': len(e_pass), 'names': sorted(e_pass),
            'failed': sorted(e_fail),
            'note': 'validation 60 类单因素 ANOVA p<0.05（输出有效率口径）'},
        'R_model_referenced': {
            'n': len(r_used), 'names': sorted(r_used),
            'note': 'train-only 排名模型 importance>0'},
        'rates': {
            'E_over_A': round(len(e_pass) / len(A), 4),
            'R_over_A': round(len(r_used) / len(A), 4),
            'target_E_over_A': 0.80,
            'E_target_met': len(e_pass) / len(A) >= 0.80,
        },
        'manual_revision_usable_rate': {
            'status': 'BLOCKED_DATA',
            'note': '人工修正后可用率>=90% 需人工修正记录，无记录不可验证'},
    }
    (run_dir / 'caer_accounting_60c.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"A={len(A)} E={len(e_pass)} R={len(r_used)} "
          f"E/A={out['rates']['E_over_A']:.1%} "
          f"(target 80%: {'MET' if out['rates']['E_target_met'] else 'NOT MET'})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
