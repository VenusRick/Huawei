# -*- coding: utf-8 -*-
"""Official ISCXVPN 15s feature-vector RuleBundle benchmark.

This benchmark follows the dataset-provided 15-second ARFF representation.
It is intentionally labelled a feature-vector functional benchmark rather than
raw-PCAP cross-capture generalization evidence.
"""
from __future__ import annotations

import hashlib, json, os, resource, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.engine.matcher.optimized_engine import OptimizedDPIEngine
from src.engine.model_io import save_rule_bundle
from src.engine.rule_compiler.optimized_generator import OptimizedRuleGenerator

ROOT=Path('/workspace/Huawei')
OUT=ROOT/'output/protocol_and_system_acceptance_20260920/behavior_official_arff'
ARFF=Path('/workspace/datasets/ISCXVPN2016/CSVs/Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-VPN.arff')
LABEL_MAP={'VPN-CHAT':'chat','VPN-VOIP':'audio','VPN-STREAMING':'video'}
LABELS=['chat','audio','video'];LID={x:i for i,x in enumerate(LABELS)};SEED=20260920

def load_arff():
    lines=ARFF.read_text(errors='replace').splitlines();attrs=[];start=None
    for i,line in enumerate(lines):
        s=line.strip()
        if s.lower().startswith('@attribute'):
            attrs.append(s[len('@attribute'):].strip().split()[0].strip("'\""))
        if s.lower().startswith('@data'):
            start=i+1;break
    if start is None:raise RuntimeError('ARFF @data missing')
    rows=[]
    for line in lines[start:]:
        s=line.strip()
        if not s or s.startswith('%'):continue
        z=[x.strip() for x in s.split(',')]
        if len(z)>=len(attrs):rows.append(z[:len(attrs)])
    df=pd.DataFrame(rows,columns=attrs);label_col=attrs[-1];mask=df[label_col].isin(LABEL_MAP);df=df[mask].reset_index(drop=True);df['_label_name']=df[label_col].map(LABEL_MAP);df['_label_id']=df._label_name.map(LID);features=attrs[:-1]
    for c in features:df[c]=pd.to_numeric(df[c],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(-1)
    return df,features

def met(y,p):
    P,R,F1,s=precision_recall_fscore_support(y,p,labels=range(3),zero_division=0);cm=confusion_matrix(y,p,labels=range(3));f=[]
    for i in range(3):
        fp=cm[:,i].sum()-cm[i,i];neg=cm.sum()-cm[i,:].sum();f.append(fp/neg if neg else 0)
    r={'accuracy':float(accuracy_score(y,p)),'macro_precision':float(P.mean()),'macro_recall':float(R.mean()),'macro_f1':float(F1.mean()),'max_class_fpr':float(max(f)),'errors':int((p!=y).sum()),'per_class':{LABELS[i]:{'P':float(P[i]),'R':float(R[i]),'F1':float(F1[i]),'FPR':float(f[i]),'N':int(s[i])} for i in range(3)},'cm':cm.tolist()};r['target_pass']=r['accuracy']>=.95 and r['macro_precision']>=.95 and r['macro_recall']>=.98 and r['max_class_fpr']<=.05;return r
def key(r):return (1 if r['target_pass'] else 0,r['macro_recall'],r['accuracy'],r['macro_precision'],-r['max_class_fpr'])

def main():
    OUT.mkdir(parents=True,exist_ok=True);df,features=load_arff();idx=np.arange(len(df));y=df._label_id.to_numpy()
    train_idx,tmp_idx=train_test_split(idx,test_size=.40,random_state=SEED,stratify=y);val_idx,test_idx=train_test_split(tmp_idx,test_size=.50,random_state=SEED,stratify=y[tmp_idx])
    split=np.full(len(df),'',dtype=object);split[train_idx]='train';split[val_idx]='validation';split[test_idx]='test';df['_split']=split
    split_csv=OUT/'split_rows.csv';pd.DataFrame({'row_id':idx,'label':df._label_name,'split':split}).to_csv(split_csv,index=False);split_sha=hashlib.sha256(split_csv.read_bytes()).hexdigest();json.dump({'source':str(ARFF),'seed':SEED,'split_unit':'official 15s feature vector','counts':pd.crosstab(df._label_name,df._split).to_dict(),'split_sha256':split_sha,'warning':'random stratified official feature-vector benchmark; not raw-PCAP cross-capture generalization'},open(OUT/'split_manifest.json','w'),indent=2)
    tr=df[df._split=='train'].reset_index(drop=True);va=df[df._split=='validation'].reset_index(drop=True);te=df[df._split=='test'].reset_index(drop=True);ytr=tr._label_id.to_numpy();yv=va._label_id.to_numpy()
    ranker=XGBClassifier(n_estimators=220,max_depth=6,learning_rate=.04,min_child_weight=1,subsample=.95,colsample_bytree=.95,reg_lambda=1.2,random_state=SEED,n_jobs=8,eval_metric='mlogloss',tree_method='hist');ranker.fit(tr[features],ytr,sample_weight=compute_sample_weight('balanced',ytr));order=np.argsort(ranker.feature_importances_)[::-1];rank=[features[i] for i in order];pd.DataFrame({'feature':rank,'importance':[float(ranker.feature_importances_[i]) for i in order]}).to_csv(OUT/'train_feature_ranking.csv',index=False)
    upper=[];cand=[];best=None
    for k in [12,16,20,len(features)]:
        fs=rank[:min(k,len(rank))]
        up=XGBClassifier(n_estimators=350,max_depth=7,learning_rate=.03,min_child_weight=1,subsample=.95,colsample_bytree=.95,reg_lambda=1.2,random_state=42,n_jobs=8,eval_metric='mlogloss',tree_method='hist');up.fit(tr[fs],ytr,sample_weight=compute_sample_weight('balanced',ytr));um=met(yv,up.predict(va[fs]));upper.append({'k':len(fs),**{z:v for z,v in um.items() if z not in ('per_class','cm')}})
        for depth in [8,12,16,20,24]:
            for leaf in [1,2]:
                gen=OptimizedRuleGenerator(confidence_threshold=0,min_confidence=0,tree_max_depth=depth,tree_min_samples_leaf=leaf,tree_class_weight='balanced');rules=gen.fit_and_generate(tr[fs],tr._label_id,fs,{0:'chat',1:'audio',2:'video'},validation=(va[fs],va._label_id));deploy=[r for r in rules if r.get('type')!='ensemble_classifier']
                for th in [0,.5,.7,.8,.9,.95]:
                    eng=OptimizedDPIEngine();eng.rules=deploy;eng.selected_features=fs;eng.label_names={0:'chat',1:'audio',2:'video'};eng.confidence_threshold=th;eng.classifier=None;pred=[]
                    for _,row in va.iterrows():
                        m=eng.match({f:row[f] for f in fs});pred.append(LID.get(m[0].result,-1) if m else -1)
                    mm=met(yv,np.asarray(pred));rr={'k':len(fs),'depth':depth,'leaf':leaf,'threshold':th,'n_rules':len(deploy),**{z:v for z,v in mm.items() if z not in ('per_class','cm')}};cand.append(rr)
                    if best is None or key(mm)>key(best[0]):best=(mm,fs,depth,leaf,th,rules,deploy)
    pd.DataFrame(upper).to_csv(OUT/'offline_upper_sweep.csv',index=False);pd.DataFrame(cand).to_csv(OUT/'validation_rule_sweep.csv',index=False);vm,fs,depth,leaf,th,rules,deploy=best
    bundle=OUT/'bundle';import shutil;shutil.rmtree(bundle,ignore_errors=True);save_rule_bundle(rules,fs,{0:'chat',1:'audio',2:'video'},th,str(bundle),bundle_meta={'task':'behavior_feature_vector','source':'ISCXVPN official 15s VPN ARFF','split_unit':'official-15s-vector','raw_pcap_runtime':False});freeze={'split_sha256':split_sha,'selected_features':fs,'depth':depth,'leaf':leaf,'threshold':th,'validation':vm,'test_policy':'test rows first evaluated after this config freeze'};fp=OUT/'frozen_config.json';fp.write_text(json.dumps(freeze,indent=2));freeze_sha=hashlib.sha256(fp.read_bytes()).hexdigest()
    features_json=OUT/'test_features.json';features_json.write_text(json.dumps([{f:float(row[f]) for f in fs} for _,row in te.iterrows()]));truth=te._label_id.to_numpy()
    # independent pure-rule feature-vector inference
    guard=OUT/'import_guard';guard.mkdir(exist_ok=True);(guard/'sitecustomize.py').write_text("import builtins\n_o=builtins.__import__\ndef g(n,*a,**k):\n    if n.split('.')[0] in {'sklearn','xgboost','scipy'}: raise ImportError('blocked '+n)\n    return _o(n,*a,**k)\nbuiltins.__import__=g\n");outjson=OUT/'independent_test.json';env=os.environ.copy();env['PYTHONPATH']=str(guard)+os.pathsep+str(ROOT);t=time.perf_counter();proc=subprocess.run([sys.executable,'-m','src.engine.dpi_infer','--rules',str(bundle),'--features',str(features_json),'-o',str(outjson)],capture_output=True,text=True,env=env);wall=time.perf_counter()-t;rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss;(OUT/'independent_cli.log').write_text(proc.stdout+'\nSTDERR\n'+proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stderr)
    rows=json.loads(outjson.read_text())['results'];pred=np.asarray([LID.get(r['predicted_label'],-1) for r in rows]);tm=met(truth,pred);json.dump(tm,open(OUT/'metrics.json','w'),indent=2);perf={'wall_sec':wall,'peak_rss_mib':rss/1024.0,'vectors':len(rows),'vectors_per_sec':len(rows)/max(wall,1e-9),'feature_count':len(fs),'rule_count':len(deploy)};json.dump(perf,open(OUT/'performance.json','w'),indent=2);acc={'scope':'official 15s ARFF random-stratified feature-vector RuleBundle benchmark','warning':'functional feature-vector benchmark; not raw-PCAP/cross-capture generalization','validation':vm,'test':tm,'formal_test_pass':tm['target_pass'],'freeze_sha256':freeze_sha};json.dump(acc,open(OUT/'acceptance.json','w'),indent=2);print('VALIDATION',vm);print('TEST',tm);print('PERF',perf)

if __name__=='__main__':main()
