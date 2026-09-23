# -*- coding: utf-8 -*-
"""Repeated training-budget curve with one fixed shared-feature method."""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
import scripts.run_mixed_dataset_accuracy_recovery_20260921 as base

ROOT=Path('/workspace/Huawei');SRC=ROOT/'output/mixed_dataset_feature_experiment_20260920/all_datasets_prefix64.csv';OUT=ROOT/'output/mixed_dataset_feature_recovery_20260921/budget_curve';OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[20260938,20260955,20260977,20260999,20261021];BUDGETS=[20,30,40,50]

def split(raw,seed):
    parts=[]
    for (d,c),g in raw.groupby(['_dataset','_class'],sort=True):
        idx=list(range(len(g)));random.Random(f'{seed}:{d}:{c}:budget').shuffle(idx);z=g.iloc[idx].reset_index(drop=True);z['_part']=['train_pool']*50+['validation_unused']*15+['test']*15;z['_prio']=list(range(50))+[-1]*30;parts.append(z)
    return pd.concat(parts,ignore_index=True)

def main():
    raw=pd.read_csv(SRC);rows=[]
    for seed in SEEDS:
        df=split(raw,seed);te=df[df._part=='test'].reset_index(drop=True)
        for budget in BUDGETS:
            tr=pd.concat([g.sort_values('_prio').head(budget) for _,g in df[df._part=='train_pool'].groupby(['_dataset','_class'])],ignore_index=True);fsall=base.common_shape_features(df);tab=base.build_score_table(tr,fsall);fs=base.hybrid_subset(tab,56,64);sh=base.sharedness(tab,fs);labels=sorted(tr._label.unique());lid={c:i for i,c in enumerate(labels)};m=ExtraTreesClassifier(n_estimators=1400,max_features='sqrt',class_weight='balanced',random_state=seed+budget,n_jobs=8);m.fit(base.clean(tr,fs),tr._label.map(lid).to_numpy());p=m.predict(base.clean(te,fs));pred=np.array([labels[int(i)] for i in p]);truth=te._label.to_numpy();r={'seed':seed,'train_per_class':budget,'train_rows':16*budget,'accuracy':float(accuracy_score(truth,pred)),'macro_f1':float(f1_score(truth,pred,average='macro',zero_division=0)),'shared_at_least_2':sh['at_least_2'],'shared_at_least_3':sh['at_least_3']}
            for d in base.DATASETS:
                ix=te._dataset.to_numpy()==d;r[f'acc_{d}']=float(accuracy_score(truth[ix],pred[ix]))
            rows.append(r);print(r,flush=True)
    tab=pd.DataFrame(rows);tab.to_csv(OUT/'budget_curve.csv',index=False);summ=[]
    for b,g in tab.groupby('train_per_class'):
        summ.append({'train_per_class':int(b),'train_rows':int(g.train_rows.iloc[0]),'accuracy_mean':float(g.accuracy.mean()),'accuracy_std':float(g.accuracy.std(ddof=1)),'accuracy_min':float(g.accuracy.min()),'accuracy_max':float(g.accuracy.max()),'macro_f1_mean':float(g.macro_f1.mean()),**{f'acc_{d}_mean':float(g[f'acc_{d}'].mean()) for d in base.DATASETS},'shared_at_least_2_mean':float(g.shared_at_least_2.mean())})
    (OUT/'summary.json').write_text(json.dumps(summ,indent=2));print(json.dumps(summ,indent=2),flush=True)

if __name__=='__main__':main()
