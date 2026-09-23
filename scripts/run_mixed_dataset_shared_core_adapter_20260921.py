# -*- coding: utf-8 -*-
"""Shared-core + small domain-adapter feature experts.

Each expert uses 64 traffic-shape features. A large core (>=48 dims) is shared
across all domains; only the residual 4-16 dims are chosen from that domain's
train-only ranking. The latent domain gate sees shared features only and never
receives dataset identity as input.
"""
from __future__ import annotations

import hashlib, json, random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import scripts.run_mixed_dataset_accuracy_recovery_20260921 as base

ROOT=Path('/workspace/Huawei')
SRC=ROOT/'output/mixed_dataset_feature_experiment_20260920'
OUT=ROOT/'output/mixed_dataset_feature_recovery_20260921/shared_core_adapter'
OUT.mkdir(parents=True,exist_ok=True)
SEED=20260955
DATASETS=base.DATASETS

def split(df):
    parts=[]
    for (d,c),g in df.groupby(['_dataset','_class'],sort=True):
        idx=list(range(len(g)));random.Random(f'{SEED}:{d}:{c}').shuffle(idx);z=g.iloc[idx].reset_index(drop=True);z['_split']=['train']*40+['validation']*20+['test']*20;parts.append(z)
    return pd.concat(parts,ignore_index=True)

def met(true,pred,domains):
    r={'accuracy':float(accuracy_score(true,pred)),'macro_f1':float(f1_score(true,pred,average='macro',zero_division=0)),'errors':int(np.sum(np.asarray(true)!=np.asarray(pred)))};per={}
    for d in DATASETS:
        ix=np.where(np.asarray(domains)==d)[0];per[d]={'accuracy':float(accuracy_score(np.asarray(true)[ix],np.asarray(pred)[ix])),'macro_f1':float(f1_score(np.asarray(true)[ix],np.asarray(pred)[ix],average='macro',zero_division=0)),'n':int(len(ix))}
    r['per_dataset']=per;r['worst_dataset_accuracy']=min(v['accuracy'] for v in per.values());return r

def xgb(seed):return XGBClassifier(n_estimators=320,max_depth=6,learning_rate=.035,min_child_weight=1,subsample=.9,colsample_bytree=.9,reg_lambda=1.8,random_state=seed,n_jobs=8,eval_metric='mlogloss',tree_method='hist')

def main():
    df=split(pd.read_csv(SRC/'all_datasets_prefix64.csv'));df[['_dataset','_class','_label','_observation_id','_split']].to_csv(OUT/'replication_split.csv',index=False);split_sha=hashlib.sha256((OUT/'replication_split.csv').read_bytes()).hexdigest();tr=df[df._split=='train'].reset_index(drop=True);va=df[df._split=='validation'].reset_index(drop=True);te=df[df._split=='test'].reset_index(drop=True);fsall=base.common_shape_features(df);tab=base.build_score_table(tr,fsall);tab.to_csv(OUT/'feature_scores.csv',index=False)
    shared_rank=base.rank_variant(tab,'robust_stable');shared_candidates=[f for f in shared_rank if int(tab.loc[tab.feature==f,'support64'].iloc[0])>=2]
    # Gate gets the frozen 64-d robust shared representation.
    gate_fs=shared_candidates[:64]
    dlid={d:i for i,d in enumerate(DATASETS)};Xg=base.clean(tr,gate_fs);yg=tr._dataset.map(dlid).to_numpy();gate=ExtraTreesClassifier(n_estimators=1400,max_features='sqrt',class_weight='balanced',random_state=SEED,n_jobs=8);gate.fit(Xg,yg)
    # Per-domain train-only rankings for residual adapters.
    dranks={}
    for i,d in enumerate(DATASETS):
        sc=base.fused_score(tr[tr._dataset==d], '_class', fsall, SEED+100+i);dranks[d]=sorted(sc,key=lambda f:(-sc[f],f))

    def run(ev,core_k,expert_kind,oracle=False):
        core=shared_candidates[:core_k];expert_fs={}
        experts={};elabels={}
        for i,d in enumerate(DATASETS):
            residual=[f for f in dranks[d] if f not in core][:64-core_k];efs=core+residual;expert_fs[d]=efs;td=tr[tr._dataset==d].reset_index(drop=True);labels=sorted(td._class.unique());lid={c:j for j,c in enumerate(labels)};X=base.clean(td,efs);y=td._class.map(lid).to_numpy()
            if expert_kind=='extra':m=ExtraTreesClassifier(n_estimators=1400,max_features='sqrt',class_weight='balanced',random_state=SEED+200+i,n_jobs=8);m.fit(X,y)
            else:m=xgb(SEED+200+i);m.fit(X,y,sample_weight=compute_sample_weight('balanced',y))
            experts[d]=m;elabels[d]=labels
        gp=gate.predict(base.clean(ev,gate_fs));pred=[]
        for i in range(len(ev)):
            d=ev.iloc[i]._dataset if oracle else DATASETS[int(gp[i])];efs=expert_fs[d];q=experts[d].predict(base.clean(ev.iloc[[i]],efs))[0];pred.append(f'{d}::{elabels[d][int(q)]}')
        gacc=float(np.mean(gp==ev._dataset.map(dlid).to_numpy()));return np.asarray(pred),gacc,expert_fs

    rows=[];best=None;truth=va._label.to_numpy()
    for core_k in [48,52,56,60]:
      for model in ['extra','xgb']:
        pred,gacc,efs=run(va,core_k,model);m=met(truth,pred,va._dataset.to_numpy());row={'core_k':core_k,'adapter_k':64-core_k,'model':model,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'worst_dataset_accuracy':m['worst_dataset_accuracy'],'domain_gate_accuracy':gacc};rows.append(row);print('VAL',row,flush=True);score=(m['accuracy'],m['macro_f1'],m['worst_dataset_accuracy']);
        if best is None or score>best[0]:best=(score,core_k,model,m,efs)
    pd.DataFrame(rows).to_csv(OUT/'validation.csv',index=False);score,core_k,model,vm,efs=best;freeze={'seed':SEED,'split_sha256':split_sha,'gate_features':gate_fs,'shared_core_k':core_k,'adapter_k':64-core_k,'expert_features':efs,'expert_model':model,'validation':vm,'test_policy':'test evaluated only after this file is written','note':'latent gate; per-domain expert uses shared core plus small train-only residual adapter'};fp=OUT/'frozen_config.json';fp.write_text(json.dumps(freeze,indent=2));fsha=hashlib.sha256(fp.read_bytes()).hexdigest()
    pred,gacc,_=run(te,core_k,model);tm=met(te._label.to_numpy(),pred,te._dataset.to_numpy());opred,_,_=run(te,core_k,model,oracle=True);ot=met(te._label.to_numpy(),opred,te._dataset.to_numpy());res={'freeze_sha256':fsha,'test':tm,'domain_gate_accuracy':gacc,'oracle_domain_test_diagnostic':ot,'frozen':freeze};(OUT/'test_results.json').write_text(json.dumps(res,indent=2));lines=['# 共享核心 + 域适配器实验（2026-09-21）','',f'- Shared core: {core_k}/64 ({100*core_k/64:.1f}%)；adapter: {64-core_k}/64。',f'- Latent gate features: 64个跨数据集shared shape；不使用dataset ID/端口/TLS/QUIC显式字段。',f'- Test Accuracy: **{100*tm["accuracy"]:.2f}%**；Macro-F1: **{100*tm["macro_f1"]:.2f}%**；Worst-domain: **{100*tm["worst_dataset_accuracy"]:.2f}%**；gate Acc: **{100*gacc:.2f}%**。','', '| Dataset | Acc | F1 |','|---|---:|---:|'];
    for d,v in tm['per_dataset'].items():lines.append(f'| {d} | {100*v["accuracy"]:.2f}% | {100*v["macro_f1"]:.2f}% |')
    lines += ['',f'Oracle-domain diagnostic: {100*ot["accuracy"]:.2f}%（仅上限）。','', '该方案的含义是共享表示负责跨数据集稳定性，少量域内adapter负责恢复数据集特异判别力；不是纯域不变特征。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n');print((OUT/'summary.md').read_text(),flush=True)

if __name__=='__main__':main()
