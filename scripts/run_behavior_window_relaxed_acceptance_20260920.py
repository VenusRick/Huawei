# -*- coding: utf-8 -*-
"""Behavior15 optimistic competition benchmark with window-stratified split.

Important: different 15-second windows from the same reconstructed session may
appear in different splits. This is deliberately weaker than session/file
disjoint evaluation and MUST NOT be presented as cross-session generalization.
"""
from __future__ import annotations

import hashlib, json, os, resource, shutil, subprocess, sys, time, random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from scripts.run_behavior_relaxed_acceptance_20260920 import metric, key, PAT, LABELS, LID
from src.engine.matcher.optimized_engine import OptimizedDPIEngine
from src.engine.model_io import save_rule_bundle
from src.engine.rule_compiler.optimized_generator import OptimizedRuleGenerator
from src.profile_specs import get_profile_spec

ROOT=Path('/workspace/Huawei')
SRC=ROOT/'output/final_profile_runtime_20260919/behavior'
OUT=ROOT/'output/protocol_and_system_acceptance_20260920/behavior_window_relaxed'
SEED=20260920

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(SRC/'mine_flow/raw_features.csv')
    # unique observation guard
    if df['_observation_id'].duplicated().any():
        raise RuntimeError('duplicate behavior observations')
    split={}
    for lab in LABELS:
        idx=list(df.index[df._label_name==lab])
        random.Random(f'{SEED}:{lab}:window').shuffle(idx)
        n=len(idx);nt=max(1,round(n*.20));nv=max(1,round(n*.20))
        for i in idx[:nt]:split[int(i)]='test'
        for i in idx[nt:nt+nv]:split[int(i)]='validation'
        for i in idx[nt+nv:]:split[int(i)]='train'
    df['_relaxed_split']=[split[int(i)] for i in df.index]
    split_rows=df[['_observation_id','_source_file','_label_name','_relaxed_split']].copy()
    split_rows.to_csv(OUT/'window_split_rows.csv',index=False)
    split_sha=hashlib.sha256((OUT/'window_split_rows.csv').read_bytes()).hexdigest()
    counts=df.groupby(['_label_name','_relaxed_split']).size().unstack(fill_value=0).to_dict('index')
    json.dump({'seed':SEED,'split_unit':'15s-window','same_session_may_cross_split':True,
               'warning':'competition-oriented optimistic split; not cross-session/cross-capture generalization',
               'counts':counts,'split_sha256':split_sha},open(OUT/'split_manifest.json','w'),indent=2)
    tr=df[df._relaxed_split=='train'].reset_index(drop=True);va=df[df._relaxed_split=='validation'].reset_index(drop=True);te=df[df._relaxed_split=='test'].reset_index(drop=True)
    features=[c for c in df.columns if not c.startswith('_')]
    X=tr[features].replace([np.inf,-np.inf],np.nan).fillna(0);ytr=tr._label_id.astype(int).to_numpy()
    ranker=XGBClassifier(n_estimators=180,max_depth=5,learning_rate=.05,min_child_weight=1,subsample=.9,colsample_bytree=.9,reg_lambda=1.5,random_state=SEED,n_jobs=8,eval_metric='mlogloss',tree_method='hist')
    ranker.fit(X,ytr,sample_weight=compute_sample_weight('balanced',ytr));order=np.argsort(ranker.feature_importances_)[::-1];rank=[features[i] for i in order]
    pd.DataFrame({'feature':rank,'importance':[float(ranker.feature_importances_[i]) for i in order]}).to_csv(OUT/'train_feature_ranking.csv',index=False)
    yv=va._label_id.astype(int).to_numpy();best=None;cand=[];upper=[]
    for k in [16,26,32,48,64]:
        fs=rank[:min(k,len(rank))]
        up=XGBClassifier(n_estimators=220,max_depth=6,learning_rate=.04,min_child_weight=1,subsample=.9,colsample_bytree=.9,reg_lambda=1.5,random_state=42,n_jobs=8,eval_metric='mlogloss',tree_method='hist')
        up.fit(tr[fs].fillna(0),ytr,sample_weight=compute_sample_weight('balanced',ytr));um=metric(yv,up.predict(va[fs].fillna(0)));upper.append({'k':len(fs),**{z:v for z,v in um.items() if z not in ('per_class','cm')}})
        for depth in [6,10,14,18]:
            for leaf in [1,2]:
                gen=OptimizedRuleGenerator(confidence_threshold=0,min_confidence=0,tree_max_depth=depth,tree_min_samples_leaf=leaf,tree_class_weight='balanced')
                rules=gen.fit_and_generate(tr[fs],tr._label_id,fs,{0:'chat',1:'audio',2:'video'},validation=(va[fs],va._label_id))
                deploy=[r for r in rules if r.get('type')!='ensemble_classifier']
                for th in [0,.5,.7,.8,.9,.95]:
                    eng=OptimizedDPIEngine();eng.rules=deploy;eng.selected_features=fs;eng.label_names={0:'chat',1:'audio',2:'video'};eng.confidence_threshold=th;eng.classifier=None
                    pp=[]
                    for _,row in va.iterrows():
                        m=eng.match({f:row[f] for f in fs});pp.append(LID.get(m[0].result,-1) if m else -1)
                    mm=metric(yv,np.asarray(pp));rr={'k':len(fs),'depth':depth,'leaf':leaf,'threshold':th,'n_rules':len(deploy),**{z:v for z,v in mm.items() if z not in ('per_class','cm')}};cand.append(rr)
                    if best is None or key(mm)>key(best[0]):best=(mm,fs,depth,leaf,th,rules,deploy)
    pd.DataFrame(upper).to_csv(OUT/'offline_upper_sweep.csv',index=False);pd.DataFrame(cand).to_csv(OUT/'validation_rule_sweep.csv',index=False)
    vm,fs,depth,leaf,th,rules,deploy=best
    bundle=OUT/'bundle';shutil.rmtree(bundle,ignore_errors=True);spec=get_profile_spec('behavior15','behavior');spec['required_features']=fs
    save_rule_bundle(rules,fs,{0:'chat',1:'audio',2:'video'},th,str(bundle),bundle_meta={'task':'behavior','feature_profile':'behavior15','profile_spec':spec,'context':spec['context'],'split_unit':'15s-window','same_session_may_cross_split':True,'max_read_packets_per_file':50000})
    freeze={'seed':SEED,'split_sha256':split_sha,'selected_features':fs,'tree_max_depth':depth,'tree_min_samples_leaf':leaf,'confidence_threshold':th,'validation':vm,'test_policy':'window test rows untouched until config freeze','warning':'same session may cross splits'}
    fp=OUT/'frozen_config.json';fp.write_text(json.dumps(freeze,indent=2));freeze_sha=hashlib.sha256(fp.read_bytes()).hexdigest()
    # Direct test after freeze.
    eng=OptimizedDPIEngine();eng.load_rule_bundle(str(bundle));yt=te._label_id.astype(int).to_numpy();pred=[];times=[];direct={}
    for _,row in te.iterrows():
        t=time.perf_counter();m=eng.match({f:row[f] for f in fs});times.append((time.perf_counter()-t)*1000);lab=m[0].result if m else 'unknown';pred.append(LID.get(lab,-1));direct[str(row._observation_id)]=lab
    dm=metric(yt,np.asarray(pred))
    # Independent all-PCAP replay and exact frozen observation-id filtering.
    pcapdir=OUT/'pcaps';shutil.rmtree(pcapdir,ignore_errors=True);pcapdir.mkdir();files=sorted(set(Path(x).resolve() for x in df._source_file.astype(str)))
    for p in files:(pcapdir/p.name).symlink_to(p)
    guard=OUT/'import_guard';guard.mkdir(exist_ok=True);(guard/'sitecustomize.py').write_text("import builtins\n_o=builtins.__import__\ndef g(n,*a,**k):\n    if n.split('.')[0] in {'sklearn','xgboost','scipy'}: raise ImportError('blocked '+n)\n    return _o(n,*a,**k)\nbuiltins.__import__=g\n")
    outjson=OUT/'independent_all_windows.json';env=os.environ.copy();env['PYTHONPATH']=str(guard)+os.pathsep+str(ROOT);t=time.perf_counter();proc=subprocess.run([sys.executable,'-m','src.engine.dpi_infer','--rules',str(bundle),'--pcap-dir',str(pcapdir),'-o',str(outjson)],capture_output=True,text=True,env=env);wall=time.perf_counter()-t;rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    (OUT/'independent_cli.log').write_text(proc.stdout+'\nSTDERR\n'+proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stderr)
    test_ids=set(te._observation_id.astype(str));truth=dict(zip(te._observation_id.astype(str),te._label_name));rows=[r for r in json.loads(outjson.read_text())['results'] if str(r['observation_id']) in test_ids]
    if len(rows)!=len(te):raise RuntimeError(f'independent rows {len(rows)} != test {len(te)}')
    yi=np.asarray([LID[truth[str(r['observation_id'])]] for r in rows]);pi=np.asarray([LID.get(r['predicted_label'],-1) for r in rows]);im=metric(yi,pi);mis=[r['observation_id'] for r in rows if direct.get(str(r['observation_id']))!=r['predicted_label']]
    json.dump(im,open(OUT/'metrics.json','w'),indent=2);json.dump({'consistent':not mis,'compared':len(rows),'mismatches':mis[:50]},open(OUT/'parity.json','w'),indent=2)
    perf={'independent_wall_sec':wall,'peak_rss_mib':rss/1024.0,'test_windows':len(rows),'windows_per_sec':len(rows)/max(wall,1e-9),'input_bytes':sum(p.stat().st_size for p in files),'mib_per_sec':(sum(p.stat().st_size for p in files)/(1024*1024))/max(wall,1e-9),'matcher_ms_mean':float(np.mean(times)),'matcher_ms_p95':float(np.quantile(times,.95)),'feature_count':len(fs),'rule_count':len(deploy)};json.dump(perf,open(OUT/'performance.json','w'),indent=2)
    strict=json.load(open(SRC/'acceptance.json'))['independent_rule_validation'];session=json.load(open(ROOT/'output/protocol_and_system_acceptance_20260920/behavior_relaxed/acceptance.json')) if (ROOT/'output/protocol_and_system_acceptance_20260920/behavior_relaxed/acceptance.json').exists() else None
    acc={'scope':'window-stratified competition-oriented split; same session/capture may cross splits','warning':'optimistic functional benchmark, not cross-session/cross-capture generalization','strict_file_disjoint_reference':strict,'session_disjoint_reference':session,'validation':vm,'test_direct':dm,'test_independent':im,'parity_consistent':not mis,'formal_test_pass':im['target_pass'],'freeze_sha256':freeze_sha};json.dump(acc,open(OUT/'acceptance.json','w'),indent=2)
    print('VALIDATION',vm);print('TEST',im);print('PARITY',not mis);print('PERF',perf)

if __name__=='__main__':main()
