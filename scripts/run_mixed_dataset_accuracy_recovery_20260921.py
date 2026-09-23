# -*- coding: utf-8 -*-
"""Recover accuracy while retaining cross-dataset feature stability.

Uses the frozen 1,280-row prefix64 cache from the 2026-09-20 diagnostic.
A new deterministic recovery split is created. Feature/model/allocation choices
use train/validation only; test is evaluated after a frozen config is written.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


ROOT = Path('/workspace/Huawei')
SRC = ROOT / 'output/mixed_dataset_feature_experiment_20260920'
OUT = ROOT / 'output/mixed_dataset_feature_recovery_20260921'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260921
DATASETS = ['CSTNET', 'CrossPlatform', 'USTC', 'VisQUIC']


IDENTIFIER_EXACT = {
    'src_port', 'dst_port', 'is_well_known_src_port', 'is_well_known_dst_port',
    'tls_ja3_hash_prefix', 'tls_ja4_hash_prefix',
}
TCP_FLAG_FEATURES = {
    'syn_flag_count', 'ack_flag_count', 'fin_flag_count', 'rst_flag_count', 'psh_flag_count',
    'urg_flag_count', 'ece_flag_count', 'cwr_flag_count',
    'fwd_syn_flag_count', 'fwd_ack_flag_count', 'fwd_fin_flag_count', 'fwd_rst_flag_count',
    'fwd_psh_flag_count', 'fwd_urg_flag_count', 'fwd_ece_flag_count', 'fwd_cwr_flag_count',
    'bwd_syn_flag_count', 'bwd_ack_flag_count', 'bwd_fin_flag_count', 'bwd_rst_flag_count',
    'bwd_psh_flag_count', 'bwd_urg_flag_count', 'bwd_ece_flag_count', 'bwd_cwr_flag_count',
}


def common_shape_features(df: pd.DataFrame) -> List[str]:
    feats = [c for c in df.columns if not c.startswith('_') and c not in IDENTIFIER_EXACT]
    out = []
    for f in feats:
        if f.startswith(('tls_', 'quic_', 'protocol_', 'tcp_window_')):
            continue
        if f in TCP_FLAG_FEATURES or f in {'syn_ack_rtt', 'num_retransmissions', 'retransmission_ratio'}:
            continue
        out.append(f)
    return sorted(out)


def clean(df: pd.DataFrame, fs: List[str]) -> pd.DataFrame:
    return df[fs].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)


def xgb(seed: int, fast: bool = False):
    return XGBClassifier(
        n_estimators=120 if fast else 320,
        max_depth=5 if fast else 6,
        learning_rate=.06 if fast else .035,
        min_child_weight=1,
        subsample=.90,
        colsample_bytree=.90,
        reg_lambda=1.8,
        random_state=seed,
        n_jobs=8,
        eval_metric='mlogloss',
        tree_method='hist',
    )


def extra(seed: int):
    return ExtraTreesClassifier(
        n_estimators=1000, max_features='sqrt', min_samples_leaf=1,
        class_weight='balanced', random_state=seed, n_jobs=8)


def rf(seed: int):
    return RandomForestClassifier(
        n_estimators=1000, max_features='sqrt', min_samples_leaf=1,
        class_weight='balanced', random_state=seed, n_jobs=8)


def split_rows(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (d, c), g in df.groupby(['_dataset', '_class'], sort=True):
        g = g.copy().reset_index(drop=True)
        idx = list(range(len(g)))
        random.Random(f'{SEED}:{d}:{c}:recovery').shuffle(idx)
        # 80/class -> 48 train pool, 16 val, 16 test.
        gp = g.iloc[idx].reset_index(drop=True)
        gp['_recovery_split'] = ['train_pool'] * 48 + ['validation'] * 16 + ['test'] * 16
        gp['_train_priority'] = list(range(48)) + [-1] * 32
        parts.append(gp)
    out = pd.concat(parts, ignore_index=True)
    if out.groupby(['_dataset', '_class', '_recovery_split']).size().min() <= 0:
        raise RuntimeError('bad recovery split')
    return out


def take_train(df: pd.DataFrame, allocation: Dict[str, int]) -> pd.DataFrame:
    parts = []
    for (d, c), g in df[df._recovery_split == 'train_pool'].groupby(['_dataset', '_class']):
        n = allocation[d]
        parts.append(g.sort_values('_train_priority').head(n))
    return pd.concat(parts, ignore_index=True)


def norm(a: np.ndarray) -> np.ndarray:
    if len(a) == 0:
        return a
    lo, hi = float(np.min(a)), float(np.max(a))
    return (a - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(a)


def fused_score(df: pd.DataFrame, ycol: str, fs: List[str], seed: int) -> Dict[str, float]:
    usable = [f for f in fs if pd.to_numeric(df[f], errors='coerce').fillna(0).std() > 1e-10]
    X = clean(df, usable)
    labels = sorted(df[ycol].unique())
    lid = {c: i for i, c in enumerate(labels)}
    y = df[ycol].map(lid).to_numpy()
    m = xgb(seed, fast=True)
    m.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    imp = np.asarray(m.feature_importances_, float)
    try:
        mi = np.asarray(mutual_info_classif(X, y, random_state=seed), float)
    except Exception:
        mi = np.zeros(len(usable), float)
    s = .6 * norm(imp) + .4 * norm(mi)
    return {f: float(v) for f, v in zip(usable, s)}


def build_score_table(train: pd.DataFrame, fs: List[str]) -> pd.DataFrame:
    per = {}
    for i, d in enumerate(DATASETS):
        dd = train[train._dataset == d]
        per[d] = fused_score(dd, '_class', fs, SEED + i)
    pooled = fused_score(train, '_label', fs, SEED + 100)
    rows = []
    for f in fs:
        vals = np.array([per[d].get(f, 0.0) for d in DATASETS], float)
        rows.append({
            'feature': f,
            **{f'score_{d}': float(per[d].get(f, 0.0)) for d in DATASETS},
            'mean_score': float(vals.mean()),
            'min_score': float(vals.min()),
            'std_score': float(vals.std()),
            'pooled_score': float(pooled.get(f, 0.0)),
        })
    tab = pd.DataFrame(rows)
    # Support uses each dataset's Top64 under the same recovery training rows.
    top_sets = {}
    for d in DATASETS:
        top_sets[d] = set(sorted(per[d], key=lambda f: (-per[d][f], f))[:64])
    tab['support64'] = [sum(f in top_sets[d] for d in DATASETS) for f in tab.feature]
    # Normalized columns across features for multi-objective formulas.
    for c in ['mean_score', 'min_score', 'std_score', 'pooled_score']:
        tab[f'n_{c}'] = norm(tab[c].to_numpy(float))
    tab['n_support'] = tab.support64 / 4.0
    return tab


def rank_variant(tab: pd.DataFrame, variant: str) -> List[str]:
    t = tab.copy()
    if variant == 'consensus_mean':
        t['objective'] = t.n_mean_score
    elif variant == 'robust_worst':
        t['objective'] = .42*t.n_mean_score + .25*t.n_min_score + .18*t.n_pooled_score + .20*t.n_support - .05*t.n_std_score
    elif variant == 'robust_stable':
        t['objective'] = .35*t.n_mean_score + .15*t.n_min_score + .20*t.n_pooled_score + .35*t.n_support - .05*t.n_std_score
    elif variant == 'pooled':
        t['objective'] = t.n_pooled_score
    else:
        raise ValueError(variant)
    return t.sort_values(['objective','support64','mean_score'], ascending=[False,False,False]).feature.tolist()


def hybrid_subset(tab: pd.DataFrame, core_k: int, total_k: int = 64) -> List[str]:
    robust = rank_variant(tab, 'robust_stable')
    # Core is explicitly cross-dataset supported where possible.
    core = [f for f in robust if int(tab.loc[tab.feature == f, 'support64'].iloc[0]) >= 2][:core_k]
    if len(core) < core_k:
        core += [f for f in robust if f not in core][:core_k-len(core)]
    pooled = rank_variant(tab, 'pooled')
    out = list(core)
    for f in pooled:
        if f not in out:
            out.append(f)
        if len(out) >= total_k:
            break
    return out[:total_k]


def encode(train: pd.DataFrame, target: str):
    labels = sorted(train[target].unique())
    return labels, {c: i for i, c in enumerate(labels)}


def evaluate_model(train: pd.DataFrame, eval_df: pd.DataFrame, fs: List[str], model_name: str, seed: int) -> Dict[str, object]:
    labels, lid = encode(train, '_label')
    Xtr, Xe = clean(train, fs), clean(eval_df, fs)
    ytr = train._label.map(lid).to_numpy()
    ye = eval_df._label.map(lid).to_numpy()
    if (ye < 0).any():
        raise RuntimeError('unknown eval label')
    probs = []
    if model_name in {'xgb','ens_xe','ens_all'}:
        m = xgb(seed)
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
        probs.append(m.predict_proba(Xe))
    if model_name in {'extra','ens_xe','ens_all'}:
        m = extra(seed+1); m.fit(Xtr, ytr); probs.append(m.predict_proba(Xe))
    if model_name in {'rf','ens_all'}:
        m = rf(seed+2); m.fit(Xtr, ytr); probs.append(m.predict_proba(Xe))
    p = np.argmax(np.mean(probs, axis=0), axis=1)
    overall = {'accuracy': float(accuracy_score(ye,p)), 'macro_f1': float(f1_score(ye,p,average='macro',zero_division=0)), 'errors': int((p!=ye).sum())}
    per = {}
    for d in DATASETS:
        idx = np.where(eval_df._dataset.to_numpy() == d)[0]
        per[d] = {'accuracy': float(accuracy_score(ye[idx], p[idx])), 'macro_f1': float(f1_score(ye[idx], p[idx], average='macro', zero_division=0)), 'n': int(len(idx))}
    overall['per_dataset'] = per
    overall['worst_dataset_accuracy'] = min(v['accuracy'] for v in per.values())
    return overall


def sharedness(tab: pd.DataFrame, fs: List[str]) -> Dict[str, object]:
    support = [int(tab.loc[tab.feature == f, 'support64'].iloc[0]) for f in fs]
    return {'support_distribution': dict(Counter(support)), 'at_least_2': int(sum(x>=2 for x in support)), 'at_least_3': int(sum(x>=3 for x in support)), 'all4': int(sum(x==4 for x in support))}


def objective(m: Dict[str, object]) -> Tuple[float,float,float]:
    return (m['accuracy'], m['macro_f1'], m['worst_dataset_accuracy'])


def main():
    raw = pd.read_csv(SRC/'all_datasets_prefix64.csv')
    df = split_rows(raw)
    split_cols = ['_dataset','_class','_label','_source_file','_observation_id','_recovery_split','_train_priority']
    df[split_cols].to_csv(OUT/'recovery_split.csv', index=False)
    split_sha = hashlib.sha256((OUT/'recovery_split.csv').read_bytes()).hexdigest()
    fs_all = common_shape_features(df)
    val = df[df._recovery_split == 'validation'].reset_index(drop=True)
    test = df[df._recovery_split == 'test'].reset_index(drop=True)

    # Equal fixed training budget: 20/class => 320 rows total.
    equal_alloc = {d:20 for d in DATASETS}
    train_equal = take_train(df, equal_alloc)
    score_tab = build_score_table(train_equal, fs_all)
    score_tab.to_csv(OUT/'recovery_feature_scores_equal320.csv', index=False)

    subset_defs = {}
    for variant in ['consensus_mean','robust_worst','robust_stable','pooled']:
        rank = rank_variant(score_tab, variant)
        for k in [48,64,80,96]:
            subset_defs[f'{variant}_k{k}'] = rank[:k]
    for core in [40,48,52,56,60]:
        subset_defs[f'hybrid_core{core}_k64'] = hybrid_subset(score_tab, core, 64)
    # 64-dimensional architecture remains primary; >64 are exploratory ceilings.

    val_rows = []
    best = None
    for name, fs in subset_defs.items():
        sh = sharedness(score_tab, fs)
        for model in ['xgb','extra','ens_xe','ens_all']:
            m = evaluate_model(train_equal, val, fs, model, SEED+10)
            row = {'subset':name,'k':len(fs),'model':model,'allocation':'equal20',
                   'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'worst_dataset_accuracy':m['worst_dataset_accuracy'],
                   'shared_at_least_2':sh['at_least_2'],'shared_at_least_3':sh['at_least_3'],'shared_all4':sh['all4']}
            val_rows.append(row)
            # Primary selection: <=64 dims and at least 80% features supported by >=2 datasets.
            feasible = len(fs) <= 64 and sh['at_least_2'] >= math.ceil(.80*len(fs))
            if feasible and (best is None or objective(m) > objective(best[0])):
                best = (m, name, fs, model, sh)
            print('VAL',row,flush=True)
    pd.DataFrame(val_rows).to_csv(OUT/'01_feature_model_validation.csv', index=False)
    if best is None:
        raise RuntimeError('no feasible shared feature configuration')
    best_m, best_name, best_fs, best_model, best_sh = best
    print('BEST_FEATURE_MODEL',best_name,best_model,best_m,best_sh,flush=True)

    # Allocation recovery under exactly 320 training rows. Per-dataset count is per class,
    # so sum of four counts must be 80 -> 4 classes * 80 = 320.
    allocs = []
    vals = list(range(8, 37, 4))
    for a,b,c,d in product(vals, repeat=4):
        if a+b+c+d != 80:
            continue
        allocs.append(dict(zip(DATASETS,[a,b,c,d])))
    fast_rows=[]
    for i, alloc in enumerate(allocs):
        tr = take_train(df, alloc)
        m = evaluate_model(tr, val, best_fs, 'xgb', SEED+100+i)
        fast_rows.append({'allocation':json.dumps(alloc,sort_keys=True),'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'worst_dataset_accuracy':m['worst_dataset_accuracy'],**{f'acc_{d}':m['per_dataset'][d]['accuracy'] for d in DATASETS}})
    fast_tab=pd.DataFrame(fast_rows).sort_values(['accuracy','macro_f1','worst_dataset_accuracy'],ascending=False)
    fast_tab.to_csv(OUT/'02_allocation_fast_validation.csv',index=False)
    # Final model comparison on top-8 allocations plus equal baseline.
    candidate_allocs=[json.loads(x) for x in fast_tab.head(8).allocation]
    if equal_alloc not in candidate_allocs:candidate_allocs.append(equal_alloc)
    alloc_rows=[];best_alloc=None
    for ai,alloc in enumerate(candidate_allocs):
        tr=take_train(df,alloc)
        for model in [best_model,'ens_xe','ens_all']:
            m=evaluate_model(tr,val,best_fs,model,SEED+500+ai)
            row={'allocation':json.dumps(alloc,sort_keys=True),'model':model,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'worst_dataset_accuracy':m['worst_dataset_accuracy'],**{f'acc_{d}':m['per_dataset'][d]['accuracy'] for d in DATASETS}}
            alloc_rows.append(row)
            if best_alloc is None or objective(m)>objective(best_alloc[0]):best_alloc=(m,alloc,model)
            print('ALLOC_VAL',row,flush=True)
    pd.DataFrame(alloc_rows).to_csv(OUT/'03_allocation_model_validation.csv',index=False)
    alloc_m,final_alloc,final_model=best_alloc

    frozen={'seed':SEED,'split_sha256':split_sha,'feature_space':'common_shape','selected_subset':best_name,'selected_features':best_fs,'selected_k':len(best_fs),'sharedness':best_sh,'training_budget_rows':320,'allocation_per_class_by_dataset':final_alloc,'model':final_model,'validation':alloc_m,'test_policy':'recovery test rows not used in feature/model/allocation selection','note':'post-exploration recovery split; not a pristine external holdout'}
    fp=OUT/'frozen_config.json';fp.write_text(json.dumps(frozen,indent=2,ensure_ascii=False));freeze_sha=hashlib.sha256(fp.read_bytes()).hexdigest()

    # First recovery-test evaluation after freeze.
    train_final=take_train(df,final_alloc)
    test320=evaluate_model(train_final,test,best_fs,final_model,SEED+900)
    # Moderate-budget result: still only half of the available 80/class, using the
    # exact same frozen features/model, with no retuning.
    train640=take_train(df,{d:40 for d in DATASETS})
    test640=evaluate_model(train640,test,best_fs,final_model,SEED+901)
    # Architecture ceiling: same 320 budget, best >64 validation subset if it helps.
    valtab=pd.DataFrame(val_rows)
    ceiling_rows=valtab[valtab.k>64].sort_values(['accuracy','macro_f1','worst_dataset_accuracy'],ascending=False)
    ceiling=None
    if len(ceiling_rows):
        rr=ceiling_rows.iloc[0];cfs=subset_defs[rr['subset']];cmodel=rr['model'];ceiling=evaluate_model(train_final,test,cfs,cmodel,SEED+902);ceiling.update({'subset':rr['subset'],'k':len(cfs),'model':cmodel,'sharedness':sharedness(score_tab,cfs)})

    baseline_fs=rank_variant(score_tab,'consensus_mean')[:64]
    baseline320=evaluate_model(take_train(df,equal_alloc),test,baseline_fs,'xgb',SEED+903)
    results={'freeze_sha256':freeze_sha,'baseline_equal320_consensus_xgb':baseline320,'recovered_fixed320':test320,'recovered_moderate640':test640,'exploratory_over64_ceiling':ceiling,'frozen':frozen}
    (OUT/'test_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))

    # Feature support details.
    support_rows=[]
    for f in best_fs:
        r=score_tab[score_tab.feature==f].iloc[0]
        support_rows.append({'feature':f,'support64':int(r.support64),'mean_score':float(r.mean_score),'min_score':float(r.min_score),'pooled_score':float(r.pooled_score)})
    pd.DataFrame(support_rows).to_csv(OUT/'selected_feature_support.csv',index=False)

    def fmt(m):
        return f"Acc {100*m['accuracy']:.2f}% / F1 {100*m['macro_f1']:.2f}% / worst-dataset {100*m['worst_dataset_accuracy']:.2f}%"
    lines=['# 多数据集共享特征精度恢复实验（2026-09-21）','',
           '## 冻结策略','',
           f"- 新recovery split：每类48 train-pool / 16 validation / 16 test；split SHA256 `{split_sha}`。",
           '- 所有特征子集、模型和320条训练预算分配仅使用Train/Validation选择；Test在frozen_config写入后首次评估。',
           f"- 最终特征：{best_name}，{len(best_fs)}维；至少2个数据集支持 {best_sh['at_least_2']}/{len(best_fs)}，至少3个支持 {best_sh['at_least_3']}/{len(best_fs)}。",
           f"- 320训练预算最终allocation：{final_alloc}；模型：{final_model}。",'',
           '## Test结果','',
           '| 配置 | Train rows | Feature dims | Accuracy | Macro-F1 | Worst dataset Acc |',
           '|---|---:|---:|---:|---:|---:|',
           f"| Baseline equal20 + consensus64 + XGB | 320 | 64 | {100*baseline320['accuracy']:.2f}% | {100*baseline320['macro_f1']:.2f}% | {100*baseline320['worst_dataset_accuracy']:.2f}% |",
           f"| Recovered shared-feature fixed budget | 320 | {len(best_fs)} | {100*test320['accuracy']:.2f}% | {100*test320['macro_f1']:.2f}% | {100*test320['worst_dataset_accuracy']:.2f}% |",
           f"| Recovered shared-feature moderate budget | 640 | {len(best_fs)} | {100*test640['accuracy']:.2f}% | {100*test640['macro_f1']:.2f}% | {100*test640['worst_dataset_accuracy']:.2f}% |"]
    if ceiling:
        lines.append(f"| Exploratory >64 ceiling | 320 | {ceiling['k']} | {100*ceiling['accuracy']:.2f}% | {100*ceiling['macro_f1']:.2f}% | {100*ceiling['worst_dataset_accuracy']:.2f}% |")
    lines += ['', '### Fixed-320 per-dataset Test Accuracy', '', '| Dataset | Baseline | Recovered | Delta |', '|---|---:|---:|---:|']
    for d in DATASETS:
        a=baseline320['per_dataset'][d]['accuracy'];b=test320['per_dataset'][d]['accuracy'];lines.append(f"| {d} | {100*a:.2f}% | {100*b:.2f}% | {(b-a)*100:+.2f} pp |")
    lines += ['', '## 裁决', '',
              '本轮不是通过重新引入端口/TLS/QUIC显式指纹来恢复精度，而是在common_shape内用worst-dataset/stability、少量pooled complement、难度自适应训练预算和轻量ensemble寻找Accuracy–Sharedness Pareto点。',
              '320条训练预算结果用于检验“在总预算不增加时能否追回精度”；640条结果表示每类只保留40/80样本时的实用上界。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status={'freeze_sha256':freeze_sha,'baseline320_accuracy':baseline320['accuracy'],'recovered320_accuracy':test320['accuracy'],'recovered320_gain_pp':(test320['accuracy']-baseline320['accuracy'])*100,'moderate640_accuracy':test640['accuracy'],'selected_subset':best_name,'selected_k':len(best_fs),'shared_at_least_2':best_sh['at_least_2'],'shared_at_least_3':best_sh['at_least_3'],'allocation':final_alloc,'model':final_model}
    (OUT/'status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False))
    print((OUT/'summary.md').read_text(),flush=True)


if __name__=='__main__':
    main()
