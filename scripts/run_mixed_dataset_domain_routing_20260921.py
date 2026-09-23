# -*- coding: utf-8 -*-
"""Second-stage accuracy recovery with a shared representation and latent domain routing.

No explicit port/TLS/QUIC/protocol feature is reintroduced. The 64-d feature
set is selected for cross-dataset stability. A latent domain gate optionally
routes to a per-domain 4-class expert; dataset identity is not supplied at
inference. A new deterministic replication split is frozen before test.
"""
from __future__ import annotations

import hashlib, json, random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import scripts.run_mixed_dataset_accuracy_recovery_20260921 as base

ROOT=Path('/workspace/Huawei')
SRC=ROOT/'output/mixed_dataset_feature_experiment_20260920'
OUT=ROOT/'output/mixed_dataset_feature_recovery_20260921/domain_routing'
OUT.mkdir(parents=True,exist_ok=True)
SEED=20260938
DATASETS=base.DATASETS

def split(df: pd.DataFrame)->pd.DataFrame:
    parts=[]
    for (d,c),g in df.groupby(['_dataset','_class'],sort=True):
        idx=list(range(len(g)));random.Random(f'{SEED}:{d}:{c}').shuffle(idx);z=g.iloc[idx].reset_index(drop=True)
        z['_split']=['train']*40+['validation']*20+['test']*20;parts.append(z)
    return pd.concat(parts,ignore_index=True)

def metrics(true, pred, datasets):
    r={'accuracy':float(accuracy_score(true,pred)),'macro_f1':float(f1_score(true,pred,average='macro',zero_division=0)),'errors':int(np.sum(np.asarray(true)!=np.asarray(pred)))}
    per={}
    for d in DATASETS:
        ix=np.where(np.asarray(datasets)==d)[0];per[d]={'accuracy':float(accuracy_score(np.asarray(true)[ix],np.asarray(pred)[ix])),'macro_f1':float(f1_score(np.asarray(true)[ix],np.asarray(pred)[ix],average='macro',zero_division=0)),'n':int(len(ix))}
    r['per_dataset']=per;r['worst_dataset_accuracy']=min(v['accuracy'] for v in per.values());return r

def shared_features(tr: pd.DataFrame)->tuple[list[str],dict]:
    fsall=base.common_shape_features(tr);tab=base.build_score_table(tr,fsall)
    # Fixed strategy learned from recovery-1, not retuned on test.
    fs=base.hybrid_subset(tab,56,64);sh=base.sharedness(tab,fs)
    tab.to_csv(OUT/'feature_scores.csv',index=False);pd.DataFrame([{'feature':f,'support64':int(tab.loc[tab.feature==f,'support64'].iloc[0])} for f in fs]).to_csv(OUT/'selected_features.csv',index=False)
    return fs,sh

def prep(train, ev, fs):return base.clean(train,fs),base.clean(ev,fs)

def fit_flat(train, ev, fs, kind):
    labels=sorted(train._label.unique());lid={c:i for i,c in enumerate(labels)};Xtr,Xe=prep(train,ev,fs);y=train._label.map(lid).to_numpy()
    if kind=='extra':m=ExtraTreesClassifier(n_estimators=1400,max_features='sqrt',class_weight='balanced',random_state=SEED,n_jobs=8);m.fit(Xtr,y);p=m.predict(Xe)
    elif kind.startswith('svm'):
        C=float(kind.split('_')[1]);m=make_pipeline(StandardScaler(),SVC(C=C,gamma='scale',class_weight='balanced',probability=True,random_state=SEED));m.fit(Xtr,y);p=m.predict(Xe)
    elif kind=='xgb':
        m=XGBClassifier(n_estimators=360,max_depth=6,learning_rate=.03,min_child_weight=1,subsample=.9,colsample_bytree=.9,reg_lambda=1.8,random_state=SEED,n_jobs=8,eval_metric='mlogloss',tree_method='hist');m.fit(Xtr,y,sample_weight=compute_sample_weight('balanced',y));p=m.predict(Xe)
    else:raise ValueError(kind)
    return np.array([labels[int(i)] for i in p])

def domain_moe(train, ev, fs, gate_kind='extra', expert_kind='extra', soft=True, oracle=False):
    Xtr,Xe=prep(train,ev,fs);dlid={d:i for i,d in enumerate(DATASETS)};yd=train._dataset.map(dlid).to_numpy()
    if gate_kind=='extra':gate=ExtraTreesClassifier(n_estimators=1200,max_features='sqrt',class_weight='balanced',random_state=SEED+1,n_jobs=8);gate.fit(Xtr,yd)
    else:gate=make_pipeline(StandardScaler(),SVC(C=float(gate_kind.split('_')[1]),gamma='scale',class_weight='balanced',probability=True,random_state=SEED+1));gate.fit(Xtr,yd)
    gate_prob=gate.predict_proba(Xe);gate_pred=np.argmax(gate_prob,axis=1)
    experts={};expert_labels={}
    for di,d in enumerate(DATASETS):
        td=train[train._dataset==d].reset_index(drop=True);labels=sorted(td._class.unique());lid={c:i for i,c in enumerate(labels)};Xd=base.clean(td,fs);y=td._class.map(lid).to_numpy()
        if expert_kind=='extra':m=ExtraTreesClassifier(n_estimators=1200,max_features='sqrt',class_weight='balanced',random_state=SEED+10+di,n_jobs=8);m.fit(Xd,y)
        elif expert_kind.startswith('svm'):
            C=float(expert_kind.split('_')[1]);m=make_pipeline(StandardScaler(),SVC(C=C,gamma='scale',class_weight='balanced',probability=True,random_state=SEED+10+di));m.fit(Xd,y)
        else:raise ValueError(expert_kind)
        experts[d]=m;expert_labels[d]=labels
    pred=[]
    for i in range(len(ev)):
        if oracle:chosen=ev.iloc[i]._dataset
        elif not soft:chosen=DATASETS[int(gate_pred[i])]
        else:chosen=None
        if chosen is not None:
            pr=experts[chosen].predict(Xe.iloc[[i]])[0];cls=expert_labels[chosen][int(pr)];pred.append(f'{chosen}::{cls}')
        else:
            best=(-1,None)
            for di,d in enumerate(DATASETS):
                ep=experts[d].predict_proba(Xe.iloc[[i]])[0]
                for ci,p in enumerate(ep):
                    score=float(gate_prob[i,di])*float(p)
                    if score>best[0]:best=(score,f'{d}::{expert_labels[d][ci]}')
            pred.append(best[1])
    domain_true=ev._dataset.map(dlid).to_numpy();domain_acc=float(np.mean(gate_pred==domain_true))
    return np.asarray(pred),domain_acc

def main():
    raw=pd.read_csv(SRC/'all_datasets_prefix64.csv');df=split(raw);df[['_dataset','_class','_label','_observation_id','_split']].to_csv(OUT/'replication_split.csv',index=False);split_sha=hashlib.sha256((OUT/'replication_split.csv').read_bytes()).hexdigest();tr=df[df._split=='train'].reset_index(drop=True);va=df[df._split=='validation'].reset_index(drop=True);te=df[df._split=='test'].reset_index(drop=True);fs,sh=shared_features(tr)
    valrows=[];best=None
    configs=[]
    for k in ['extra','xgb','svm_1','svm_10','svm_100']:configs.append(('flat',k,None,None))
    for gate in ['extra','svm_10']:
      for expert in ['extra','svm_10','svm_100']:
        configs += [('moe_soft',gate,expert,True),('moe_hard',gate,expert,False)]
    truth=va._label.to_numpy()
    for ci,cfg in enumerate(configs):
        typ,a,b,soft=cfg
        if typ=='flat':pred=fit_flat(tr,va,fs,a);gacc=None
        else:pred,gacc=domain_moe(tr,va,fs,a,b,soft=soft)
        m=metrics(truth,pred,va._dataset.to_numpy());row={'type':typ,'a':a,'b':b or '', 'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'worst_dataset_accuracy':m['worst_dataset_accuracy'],'domain_gate_accuracy':gacc if gacc is not None else ''};valrows.append(row);print('VAL',row,flush=True)
        score=(m['accuracy'],m['macro_f1'],m['worst_dataset_accuracy'])
        if best is None or score>best[0]:best=(score,cfg,m)
    # Oracle domain routing is diagnostic only, never eligible for selection.
    opred,_=domain_moe(tr,va,fs,'extra','extra',soft=False,oracle=True);oracle=metrics(truth,opred,va._dataset.to_numpy());print('ORACLE_VAL',oracle,flush=True)
    pd.DataFrame(valrows).to_csv(OUT/'model_validation.csv',index=False);score,cfg,vm=best
    freeze={'seed':SEED,'split_sha256':split_sha,'features':fs,'sharedness':sh,'selected_model':cfg,'validation':vm,'oracle_domain_validation_diagnostic':oracle,'test_policy':'test rows not evaluated until this file was written','note':'replication split; shared common_shape only, no explicit dataset/protocol identifiers'};fp=OUT/'frozen_config.json';fp.write_text(json.dumps(freeze,indent=2));fsha=hashlib.sha256(fp.read_bytes()).hexdigest()
    typ,a,b,soft=cfg
    if typ=='flat':pred=fit_flat(tr,te,fs,a);gacc=None
    else:pred,gacc=domain_moe(tr,te,fs,a,b,soft=soft)
    tm=metrics(te._label.to_numpy(),pred,te._dataset.to_numpy());opred,_=domain_moe(tr,te,fs,'extra','extra',soft=False,oracle=True);ot=metrics(te._label.to_numpy(),opred,te._dataset.to_numpy())
    result={'freeze_sha256':fsha,'test':tm,'domain_gate_accuracy':gacc,'oracle_domain_test_diagnostic':ot,'frozen':freeze};(OUT/'test_results.json').write_text(json.dumps(result,indent=2))
    lines=['# 多数据集共享特征 + 潜在域路由恢复实验（2026-09-21）','',f'- Split SHA256: `{split_sha}`；每类40 train / 20 val / 20 test。',f"- 共享特征：64维，至少2域支持 {sh['at_least_2']}/64，至少3域支持 {sh['at_least_3']}/64。",f'- Validation选中的模型：{cfg}；Test在冻结后首次评估。','', '| Metric | Result |','|---|---:|',f"| Test Accuracy | {100*tm['accuracy']:.2f}% |",f"| Test Macro-F1 | {100*tm['macro_f1']:.2f}% |",f"| Worst-dataset Accuracy | {100*tm['worst_dataset_accuracy']:.2f}% |"]
    if gacc is not None:lines.append(f"| Latent domain gate Accuracy | {100*gacc:.2f}% |")
    lines += ['', 'Per-dataset:', '', '| Dataset | Accuracy | F1 |','|---|---:|---:|']
    for d,v in tm['per_dataset'].items():lines.append(f"| {d} | {100*v['accuracy']:.2f}% | {100*v['macro_f1']:.2f}% |")
    lines += ['',f"Oracle-domain diagnostic（仅上限，不是可部署成绩）：{100*ot['accuracy']:.2f}% Accuracy。",'', '该实验仍只使用common_shape共享特征；latent gate从流量形状中推断域，不读取dataset ID。它用于判断模型分解是否能追回混合16类的精度，不应被解释为域不变学习。']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n');print((OUT/'summary.md').read_text(),flush=True)

if __name__=='__main__':main()
