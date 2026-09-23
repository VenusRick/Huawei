# -*- coding: utf-8 -*-
"""Repeated fixed evaluation for the final simple shared-feature model."""
from __future__ import annotations

import json, random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

import scripts.run_mixed_dataset_accuracy_recovery_20260921 as base

ROOT=Path('/workspace/Huawei')
SRC=ROOT/'output/mixed_dataset_feature_experiment_20260920/all_datasets_prefix64.csv'
OUT=ROOT/'output/mixed_dataset_feature_recovery_20260921/repeated_fixed'
OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[20260938,20260955,20260977,20260999,20261021]

def split(raw,seed):
    parts=[]
    for (d,c),g in raw.groupby(['_dataset','_class'],sort=True):
        idx=list(range(len(g)));random.Random(f'{seed}:{d}:{c}:fixed').shuffle(idx);z=g.iloc[idx].reset_index(drop=True);z['_split']=['train']*40+['validation_unused']*20+['test']*20;parts.append(z)
    return pd.concat(parts,ignore_index=True)

def main():
    raw=pd.read_csv(SRC);rows=[];feature_sets=[]
    for si,seed in enumerate(SEEDS):
        df=split(raw,seed);tr=df[df._split=='train'].reset_index(drop=True);te=df[df._split=='test'].reset_index(drop=True);fsall=base.common_shape_features(df);tab=base.build_score_table(tr,fsall);fs=base.hybrid_subset(tab,56,64);sh=base.sharedness(tab,fs);feature_sets.append(set(fs))
        labels=sorted(tr._label.unique());lid={c:i for i,c in enumerate(labels)};m=ExtraTreesClassifier(n_estimators=1400,max_features='sqrt',class_weight='balanced',random_state=seed,n_jobs=8);m.fit(base.clean(tr,fs),tr._label.map(lid).to_numpy());p=m.predict(base.clean(te,fs));pred=np.array([labels[int(i)] for i in p]);true=te._label.to_numpy();r={'seed':seed,'accuracy':float(accuracy_score(true,pred)),'macro_f1':float(f1_score(true,pred,average='macro',zero_division=0)),'shared_at_least_2':sh['at_least_2'],'shared_at_least_3':sh['at_least_3']}
        for d in base.DATASETS:
            ix=te._dataset.to_numpy()==d;r[f'acc_{d}']=float(accuracy_score(true[ix],pred[ix]))
        rows.append(r);print(r,flush=True)
    tab=pd.DataFrame(rows);tab.to_csv(OUT/'repeated_results.csv',index=False);acc=tab.accuracy.to_numpy();f1=tab.macro_f1.to_numpy();summary={'n_seeds':len(SEEDS),'accuracy_mean':float(acc.mean()),'accuracy_std':float(acc.std(ddof=1)),'accuracy_min':float(acc.min()),'accuracy_max':float(acc.max()),'macro_f1_mean':float(f1.mean()),'per_dataset_mean':{d:float(tab[f'acc_{d}'].mean()) for d in base.DATASETS},'shared_at_least_2_mean':float(tab.shared_at_least_2.mean()),'shared_at_least_3_mean':float(tab.shared_at_least_3.mean()),'feature_set_pairwise_jaccard_mean':float(np.mean([len(a&b)/len(a|b) for i,a in enumerate(feature_sets) for b in feature_sets[i+1:]]))};(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
