# -*- coding: utf-8 -*-
import json, time, resource, subprocess, sys, os, shutil
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from xgboost import XGBClassifier

from src.engine.rule_compiler.optimized_generator import OptimizedRuleGenerator
from src.engine.matcher.optimized_engine import OptimizedDPIEngine
from src.engine.model_io import save_rule_bundle, load_rule_bundle
from src.features.runtime import extract_feature_records_with_stats
from src.profile_specs import get_profile_spec, filter_forbidden_features

ROOT=Path('/workspace/Huawei')
OUT=ROOT/'output/final_profile_runtime_20260919/application'
OLD=Path('/workspace/taskquay-data/worktrees/Huawei-610224a7/output/highmetric_search')
OUT.mkdir(parents=True,exist_ok=True)
LABELS=['FTP','Gmail','MySQL','WorldOfWarcraft']; lid={x:i for i,x in enumerate(LABELS)}
spec=get_profile_spec('application64','app')

df=pd.read_pickle(OLD/'ustc4_prefix64_dev.pkl')
ranking=pd.read_csv(OLD/'ustc4_ranking.csv')['feature'].tolist()
ranked=filter_forbidden_features(ranking,spec)
selected=ranked[:64]
(OUT/'selected_features.txt').write_text('\n'.join(selected)+'\n')
print('selected',len(selected),'blocked from frozen ranking',
      [x for x in ranking[:80] if x not in selected][:20],flush=True)

tr=df[df._split=='train'].reset_index(drop=True)
va=df[df._split=='validation'].reset_index(drop=True)
Xtr=tr[selected].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0)
Xv=va[selected].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(0)
ytr=tr._lid.astype(int).to_numpy(); yv=va._lid.astype(int).to_numpy()

def metrics(y,p):
    P,R,F1,s=precision_recall_fscore_support(y,p,labels=range(4),zero_division=0)
    cm=confusion_matrix(y,p,labels=range(4)); fprs=[]
    for i in range(4):
        fp=cm[:,i].sum()-cm[i,i]; neg=cm.sum()-cm[i,:].sum()
        fprs.append(fp/neg if neg else 0)
    r={'accuracy':float(accuracy_score(y,p)),'macro_precision':float(P.mean()),
       'macro_recall':float(R.mean()),'macro_f1':float(F1.mean()),
       'max_class_fpr':float(max(fprs)),'errors':int((p!=y).sum()),
       'per_class':{LABELS[i]:{'P':float(P[i]),'R':float(R[i]),'F1':float(F1[i]),
                               'FPR':float(fprs[i]),'N':int(s[i])} for i in range(4)},
       'cm':cm.tolist()}
    r['target_pass']=r['accuracy']>=.95 and r['macro_precision']>=.95 and r['macro_recall']>=.98 and r['max_class_fpr']<=.05
    return r
def key(r):
    return (1 if r['target_pass'] else 0,r['macro_recall'],r['accuracy'],r['macro_precision'],-r['max_class_fpr'])

# Offline upper bound on exactly deployable identifier-free features.
upper=XGBClassifier(n_estimators=180,max_depth=5,learning_rate=.05,min_child_weight=2,
                    subsample=.9,colsample_bytree=.9,reg_lambda=1.5,random_state=42,
                    n_jobs=8,eval_metric='mlogloss',tree_method='hist')
upper.fit(Xtr,ytr); upper_val=metrics(yv,upper.predict(Xv))
print('UPPER VAL',upper_val,flush=True)

# Validation-only deployment search. Test is not loaded here.
candidates=[]
best_obj=None
for depth in [4,5,6,8,10]:
  for leaf in [1,2,4]:
    gen=OptimizedRuleGenerator(confidence_threshold=0.0,min_confidence=0.0,
                               tree_max_depth=depth,tree_min_samples_leaf=leaf,
                               tree_class_weight='balanced')
    rules=gen.fit_and_generate(tr[selected],tr._lid,selected,
                               {i:x for i,x in enumerate(LABELS)},
                               validation=(va[selected],va._lid),
                               groups=tr._file)
    deploy=[r for r in rules if r.get('type')!='ensemble_classifier']
    for th in [0.0,.5,.7,.8,.9,.95]:
      eng=OptimizedDPIEngine();eng.rules=deploy;eng.selected_features=selected
      eng.label_names={i:x for i,x in enumerate(LABELS)};eng.confidence_threshold=th;eng.classifier=None
      pred=[]
      for row in Xv.to_dict('records'):
        m=eng.match(row);pred.append(lid[m[0].result] if m and m[0].result in lid else -1)
      mm=metrics(yv,np.asarray(pred))
      rec={'depth':depth,'leaf':leaf,'threshold':th,'n_rules':len(deploy),**{k:v for k,v in mm.items() if k not in ('per_class','cm')}}
      candidates.append(rec)
      if best_obj is None or key(mm)>key(best_obj[0]):
        best_obj=(mm,depth,leaf,th,rules,deploy)
pd.DataFrame(candidates).to_csv(OUT/'validation_rule_sweep.csv',index=False)
best,depth,leaf,th,rules,deploy=best_obj
print('BEST RULE VAL',depth,leaf,th,best,flush=True)

bundle=OUT/'bundle'
if bundle.exists():shutil.rmtree(bundle)
pspec=dict(spec);pspec['required_features']=selected
save_rule_bundle(rules,selected,{i:x for i,x in enumerate(LABELS)},th,str(bundle),
                 bundle_meta={'task':'app','source':'frozen-ustc-runtime-acceptance',
                              'feature_profile':'application64','profile_spec':pspec,
                              'temporal_vote':1,'max_packets_per_session':64,
                              'validation_only_selection':True})
json.dump({'offline_upper_validation':upper_val,'rule_validation':best,
           'tree_max_depth':depth,'tree_min_samples_leaf':leaf,
           'confidence_threshold':th,'selected_features':selected,
           'fidelity_macro_f1_gap':upper_val['macro_f1']-best['macro_f1']},
          open(OUT/'frozen_config.json','w'),indent=2)

# Re-extract raw PCAPs with the deployed runtime. This occurs only after config freeze.
manifest=[json.loads(x) for x in (OLD/'ustc4_manifest.jsonl').read_text().splitlines() if x.strip()]
test=[x for x in manifest if x['split']=='test']
wanted={(Path(x['file']).name,int(x['session_index'])):x for x in test}
raw_features={}; extract_perf=[]
parity_mismatches=[]; parity_checked=0
for pstr in sorted(set(x['file'] for x in manifest)):
    t=time.perf_counter()
    recs,st=extract_feature_records_with_stats(pstr,max_packets=64,max_read_packets=100000)
    sec=time.perf_counter()-t
    extract_perf.append({'file':pstr,'seconds':sec,'records':len(recs),'packets_read':st.packets_read,
                         'ms_per_record':1000*sec/max(len(recs),1)})
    byidx={r.observation.session_index:r for r in recs}
    # parity against frozen train/val pkl for up to 100 rows/file
    sub=df[df._file==pstr].head(100)
    for _,row in sub.iterrows():
        idx=int(str(row['_obs']).rsplit(':',1)[1]); rr=byidx.get(idx)
        if rr is None:
            parity_mismatches.append(f'{Path(pstr).name}:{idx}:missing');continue
        parity_checked+=1
        for f in selected:
            a=float(row[f]) if pd.notna(row[f]) else 0.0
            b=float(rr.features.get(f,0.0) or 0.0)
            if not np.isclose(a,b,rtol=0,atol=1e-9):
                parity_mismatches.append(f'{Path(pstr).name}:{idx}:{f}:{a}!={b}')
                if len(parity_mismatches)>=50:break
        if len(parity_mismatches)>=50:break
    for idx,rr in byidx.items():
        k=(Path(pstr).name,idx)
        if k in wanted: raw_features[k]=rr.features

assert len(raw_features)==len(test),(len(raw_features),len(test))
json.dump({'checked_rows':parity_checked,'mismatches':parity_mismatches[:50],
           'consistent':not parity_mismatches},open(OUT/'parity.json','w'),indent=2)
pd.DataFrame(extract_perf).to_csv(OUT/'extraction_performance.csv',index=False)

# Matcher-only exact test subset (no test tuning).
eng=OptimizedDPIEngine();eng.load_rule_bundle(str(bundle))
pred=[]; match_times=[]
for x in test:
    f=raw_features[(Path(x['file']).name,int(x['session_index']))]
    t=time.perf_counter();m=eng.match(f);match_times.append((time.perf_counter()-t)*1000)
    pred.append(lid[m[0].result] if m and m[0].result in lid else -1)
ytest=np.array([lid[x['label']] for x in test])
test_metrics=metrics(ytest,np.asarray(pred))
json.dump(test_metrics,open(OUT/'rule_test_metrics.json','w'),indent=2)
pd.DataFrame([{'true':x['label'],'pred':LABELS[p] if p>=0 else 'unknown',
               'file':x['file'],'session_index':x['session_index']}
              for x,p in zip(test,pred)]).to_csv(OUT/'rule_test_predictions.csv',index=False)

# Freeze test feature vectors for independent rules-only subprocess replay.
feature_json=OUT/'test_features.json'
feature_json.write_text(json.dumps([raw_features[(Path(x['file']).name,int(x['session_index']))] for x in test]))
guard=OUT/'import_guard';guard.mkdir(exist_ok=True)
(guard/'sitecustomize.py').write_text(
"import builtins\n_o=builtins.__import__\ndef g(n,*a,**k):\n"
"\n    if n.split('.')[0] in {'sklearn','xgboost','scipy'}: raise ImportError('training lib blocked: '+n)\n"
"    return _o(n,*a,**k)\nbuiltins.__import__=g\n")
outjson=OUT/'independent_predictions.json'
env=os.environ.copy();env['PYTHONPATH']=str(guard)+os.pathsep+str(ROOT)
t=time.perf_counter()
proc=subprocess.run([sys.executable,'-m','src.engine.dpi_infer','--rules',str(bundle),
                     '--features',str(feature_json),'-o',str(outjson)],
                    text=True,capture_output=True,env=env)
cli_sec=time.perf_counter()-t
(OUT/'independent_cli.log').write_text(proc.stdout+'\nSTDERR\n'+proc.stderr)
if proc.returncode!=0: raise RuntimeError(proc.stderr)
ip=json.loads(outjson.read_text())['results']
assert len(ip)==len(test)
ind_pred=np.array([lid.get(r['predicted_label'],-1) for r in ip])
assert np.array_equal(ind_pred,np.asarray(pred))
bundle_bytes=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
performance={'extraction':extract_perf,
             'extraction_ms_per_record_mean':float(np.mean([x['ms_per_record'] for x in extract_perf])),
             'matcher_ms_mean':float(np.mean(match_times)),'matcher_ms_p95':float(np.quantile(match_times,.95)),
             'independent_feature_replay_wall_sec':cli_sec,'records_per_sec':len(test)/cli_sec,
             'bundle_bytes':bundle_bytes,'rule_count':len(deploy)}
json.dump(performance,open(OUT/'performance.json','w'),indent=2)
accept={'profile':'application64','offline_upper_validation':upper_val,'rule_validation':best,
        'rule_test':test_metrics,'parity_consistent':not parity_mismatches,
        'independent_rules_only_match':True,
        'formal_test_pass':test_metrics['target_pass']}
json.dump(accept,open(OUT/'acceptance.json','w'),indent=2)
print('RULE TEST',test_metrics,flush=True)
print('PARITY',not parity_mismatches,'checked',parity_checked,'mismatch',parity_mismatches[:3],flush=True)
print('PERFORMANCE',performance,flush=True)
