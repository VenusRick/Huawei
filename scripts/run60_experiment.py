# -*- coding: utf-8 -*-
"""60 类扩展实验：64 包 + 64 维主配置训练/验证/封存测试（cstnet60_20260918_exp）。

纪律：
- 特征排名只看 train（train-only 可用列过滤 + XGB importance）；
- validation 只用于配置选择/瓶颈对照（64包Top96 / 128包Top64 最小对照）；
- test 封存，冻结配置后一次性评估；未达标也如实报告，不得回头调参；
- 每类提取读包上限与其 survey tier cap 一致（observation_id 才能对齐）。

冻结规则（先声明后执行，validation 驱动）：
1. 主配置 64包+Top64 四项全过官方线（R>=.98 A>=.95 P>=.95 FPR<=.05）→ 冻结；
2. 否则在 64包Top96 / 128包Top64 里选 validation 更优者（优先 64 包/更少维）；
3. 全不过线则冻结 validation 最优配置做一次 test，如实报告缺口。

用法::

    TRAIN_THREADS=2 python3 scripts/run60_experiment.py \
        --run-dir output/cstnet60_20260918_exp
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.features.runtime import extract_feature_records_with_stats  # noqa: E402

# 与上一轮一致：识别类特征里禁用端口/指纹哈希
BLOCKED = {'src_port', 'tls_ja3_hash_prefix', 'tls_ja4_hash_prefix'}

TARGETS = {'recall': 0.98, 'accuracy': 0.95, 'precision': 0.95, 'fpr': 0.05}


def make_model(seed: int = 42) -> XGBClassifier:
    n_jobs = max(1, int(os.environ.get('TRAIN_THREADS', '2') or 2))
    return XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.06,
        min_child_weight=2, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
        n_jobs=n_jobs, eval_metric='mlogloss')


def calc_metrics(y, p, labels) -> dict:
    P, R, F1, _ = precision_recall_fscore_support(
        y, p, labels=labels, zero_division=0)
    cm = confusion_matrix(y, p, labels=labels)
    fprs = []
    for i, _ in enumerate(labels):
        fp = cm[:, i].sum() - cm[i, i]
        neg = cm.sum() - cm[i, :].sum()
        fprs.append(fp / neg if neg else 0.0)
    return {
        'accuracy': float(accuracy_score(y, p)),
        'macro_precision': float(np.mean(P)),
        'macro_recall': float(np.mean(R)),
        'macro_f1': float(np.mean(F1)),
        'max_fpr': float(max(fprs)),
        'errors': int(np.sum(np.asarray(y) != np.asarray(p))),
    }


def meets_targets(m: dict) -> bool:
    return (m['macro_recall'] >= TARGETS['recall']
            and m['accuracy'] >= TARGETS['accuracy']
            and m['macro_precision'] >= TARGETS['precision']
            and m['max_fpr'] <= TARGETS['fpr'])


def build_cache(run_dir: Path, manifest, class_cfg, prefix: int,
                splits=('train', 'validation', 'test')) -> pd.DataFrame:
    """按 manifest 逐文件提取特征；cap 与 survey tier 一致以对齐 observation_id。"""
    cache = run_dir / f'features_prefix{prefix}_{"_".join(splits)}.pkl'
    if cache.exists():
        df = pd.read_pickle(cache)
        print(f'[CACHE] {cache.name}: {len(df)} rows', flush=True)
        return df

    wanted = [m for m in manifest if m['split'] in splits]
    by_file: dict = {}
    for m in wanted:
        by_file.setdefault(m['file'], []).append(m)

    rows = []
    t0 = time.time()
    for fname, mrows in sorted(by_file.items()):
        cls = mrows[0]['label']
        cap = class_cfg[cls]['per_file_cap']
        pcap = ROOT / 'data' / 'all_data' / fname
        records, stats = extract_feature_records_with_stats(
            str(pcap), max_packets=prefix, max_read_packets=cap)
        by_idx = {r.observation.session_index: r for r in records}
        for m in mrows:
            idx = int(m['observation_id'].rsplit(':', 1)[1])
            if idx not in by_idx:
                raise RuntimeError(
                    f'prefix={prefix} 观测缺失（cap 对齐被破坏？）: '
                    f"{m['observation_id']} cap={cap}")
            row = {'_obs': m['observation_id'], '_label': m['label'],
                   '_split': m['split'], '_file': m['file'],
                   '_lid': m['_lid']}
            row.update(by_idx[idx].features)
            rows.append(row)
        print(f'  {fname:48s} wanted={len(mrows):3d} '
              f'records={len(records):4d} pkts={stats.packets_read}',
              flush=True)
    df = pd.DataFrame(rows)
    assert (df['_lid'] >= 0).all(), 'lid 未注入（manifest 需先带 _lid）'
    df.to_pickle(cache)
    print(f'prefix={prefix}: {len(df)} rows in {time.time() - t0:.0f}s '
          f'-> {cache.name}', flush=True)
    return df


def train_only_rank(tr: pd.DataFrame, usable: list) -> pd.Series:
    Xtr = (tr[usable].apply(pd.to_numeric, errors='coerce')
           .replace([np.inf, -np.inf], np.nan).fillna(0))
    rank_model = make_model()
    rank_model.fit(Xtr, tr['_lid'].astype(int).to_numpy())
    return pd.Series(rank_model.feature_importances_,
                     index=usable).sort_values(ascending=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default='output/cstnet60_20260918_exp')
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)

    manifest = [json.loads(x) for x in
                (run_dir / 'manifest_60c.jsonl').read_text().splitlines() if x]
    meta = json.loads(
        (run_dir / 'manifest_60c_class_config.json').read_text())
    class_cfg = meta['classes']
    labels = sorted(class_cfg)
    lid = {c: i for i, c in enumerate(labels)}
    for m in manifest:
        m['_lid'] = lid[m['label']]

    timing = {'stage_extraction_64': None, 'stage_extraction_128_dev': None,
              'stage_grid': None, 'stage_test': None}

    # ---- 主缓存（全部 split）与 128 包 dev 对照缓存 ----
    t0 = time.time()
    df64 = build_cache(run_dir, manifest, class_cfg, prefix=64)
    timing['stage_extraction_64'] = round(time.time() - t0, 1)

    tr = df64[df64['_split'] == 'train'].reset_index(drop=True)
    va = df64[df64['_split'] == 'validation'].reset_index(drop=True)
    print(f'train={len(tr)} validation={len(va)} '
          f'(test={int((df64["_split"] == "test").sum())} 封存)', flush=True)

    # ---- train-only 排名（与上一轮同口径）----
    candidates = [c for c in df64.columns
                  if not c.startswith('_') and c not in BLOCKED]
    usable = [c for c in candidates
              if pd.to_numeric(tr[c], errors='coerce').nunique(dropna=True) > 1]

    t0 = time.time()
    importance = train_only_rank(tr, usable)
    importance.rename_axis('feature').rename('importance').to_csv(
        run_dir / 'ranking_prefix64_trainonly.csv')
    print(f'ranking: candidate={len(candidates)} usable={len(usable)}',
          flush=True)

    def XY(d: pd.DataFrame, feats: list):
        X = (d[feats].apply(pd.to_numeric, errors='coerce')
             .replace([np.inf, -np.inf], np.nan).fillna(0))
        return X, d['_lid'].astype(int).to_numpy()

    yva = va['_lid'].astype(int).to_numpy()
    results = {}

    def eval_config(name, prefix, feats):
        model = make_model()
        Xtr, ytr = XY(tr, feats)
        model.fit(Xtr, ytr)
        Xva, _ = XY(va, feats)
        m = calc_metrics(yva, model.predict(Xva), list(range(len(labels))))
        m.update({'prefix': prefix, 'n_features': len(feats)})
        results[name] = m
        ok = meets_targets(m)
        print(f'  [VAL] {name:14s} acc={m["accuracy"]:.4f} '
              f'P={m["macro_precision"]:.4f} R={m["macro_recall"]:.4f} '
              f'F1={m["macro_f1"]:.4f} maxFPR={m["max_fpr"]:.4f} '
              f'err={m["errors"]:3d} pass={ok}', flush=True)
        return ok

    top64 = list(importance.head(64).index)
    top96 = list(importance.head(96).index)
    pass_main = eval_config('64pkt_top64', 64, top64)
    if not pass_main:
        eval_config('64pkt_top96', 64, top96)
        t1 = time.time()
        df128 = build_cache(run_dir, manifest, class_cfg, prefix=128,
                            splits=('train', 'validation'))
        timing['stage_extraction_128_dev'] = round(time.time() - t1, 1)
        tr128 = df128[df128['_split'] == 'train'].reset_index(drop=True)
        va128 = df128[df128['_split'] == 'validation'].reset_index(drop=True)
        imp128 = train_only_rank(tr128, [c for c in df128.columns
                                         if not c.startswith('_')
                                         and c not in BLOCKED
                                         and pd.to_numeric(
                                             tr128[c], errors='coerce')
                                         .nunique(dropna=True) > 1])
        imp128.rename_axis('feature').rename('importance').to_csv(
            run_dir / 'ranking_prefix128_trainonly.csv')
        Xtr, ytr = XY(tr128, list(imp128.head(64).index))
        model = make_model(); model.fit(Xtr, ytr)
        Xva, _ = XY(va128, list(imp128.head(64).index))
        m = calc_metrics(va128['_lid'].astype(int).to_numpy(),
                         model.predict(Xva), list(range(len(labels))))
        m.update({'prefix': 128, 'n_features': 64})
        results['128pkt_top64'] = m
        print(f'  [VAL] 128pkt_top64   acc={m["accuracy"]:.4f} '
              f'P={m["macro_precision"]:.4f} R={m["macro_recall"]:.4f} '
              f'F1={m["macro_f1"]:.4f} maxFPR={m["max_fpr"]:.4f} '
              f'err={m["errors"]:3d} pass={meets_targets(m)}', flush=True)
    timing['stage_grid'] = round(time.time() - t0, 1)

    # ---- 冻结（声明式规则，validation 驱动）----
    prefer = [n for n in ('64pkt_top64', '64pkt_top96', '128pkt_top64')
              if n in results]
    passing = [n for n in prefer if meets_targets(results[n])]
    if passing:
        frozen = passing[0]  # 优先主配置，其次效率（64包/更少维）
    else:
        frozen = max(prefer, key=lambda n: (
            results[n]['macro_recall'], results[n]['accuracy'],
            -results[n]['max_fpr']))
    print(f'FROZEN = {frozen}', flush=True)

    # ---- 一次性封存测试 ----
    t0 = time.time()
    prefix_frozen = results[frozen]['prefix']
    k = results[frozen]['n_features']
    if prefix_frozen == 64:
        dff = df64
        feats = list(importance.head(k).index)
    else:
        dff = build_cache(run_dir, manifest, class_cfg, prefix=128)
        feats = pd.read_csv(
            run_dir / 'ranking_prefix128_trainonly.csv')['feature'].tolist()[:k]
    trf = dff[dff['_split'] == 'train'].reset_index(drop=True)
    tef = dff[dff['_split'] == 'test'].reset_index(drop=True)
    model = make_model()
    Xtr, ytr = XY(trf, feats)
    model.fit(Xtr, ytr)
    Xte, yte = XY(tef, feats)
    pred = model.predict(Xte)
    m = calc_metrics(yte, pred, list(range(len(labels))))
    P, R, F1, sup = precision_recall_fscore_support(
        yte, pred, labels=list(range(len(labels))), zero_division=0)
    cm = confusion_matrix(yte, pred, labels=list(range(len(labels))))
    per_class = {}
    for i, c in enumerate(labels):
        fpr_i = ((cm[:, i].sum() - cm[i, i])
                 / (cm.sum() - cm[i, :].sum() or 1))
        per_class[c] = {'precision': float(P[i]), 'recall': float(R[i]),
                        'f1': float(F1[i]), 'fpr': float(fpr_i),
                        'support': int(sup[i]),
                        'split_mode': class_cfg[c]['split_mode']}
    timing['stage_test'] = round(time.time() - t0, 1)

    out = {
        'run': str(run_dir), 'n_classes': len(labels),
        'n_train': len(tr), 'n_val': len(va), 'n_test': len(tef),
        'split_mode_counts': meta['split_mode_counts'],
        'targets': TARGETS,
        'validation': results,
        'frozen': frozen,
        'test_one_shot': {**m, 'target_pass': meets_targets(m)},
        'per_class': per_class,
        'features_frozen': list(feats),
        'timing_sec': timing,
        'cache_files': [p.name for p in sorted(run_dir.glob('features_*.pkl'))],
    }
    (run_dir / 'experiment_60c_results.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: out[k] for k in
                      ('frozen', 'test_one_shot', 'timing_sec')},
                     ensure_ascii=False, indent=1))
    print('saved:', run_dir / 'experiment_60c_results.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
