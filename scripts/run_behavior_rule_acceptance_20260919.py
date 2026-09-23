# -*- coding: utf-8 -*-
import json, os, resource, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.engine.rule_compiler.optimized_generator import OptimizedRuleGenerator
from src.engine.matcher.optimized_engine import OptimizedDPIEngine
from src.engine.model_io import save_rule_bundle
from src.profile_specs import get_profile_spec

ROOT=Path('/workspace/Huawei')
OUT=ROOT/'output/final_profile_runtime_20260919/behavior'
MINE=OUT/'mine_flow'
df=pd.read_csv(MINE/'raw_features.csv')
LABELS=['chat','audio','video']; lid={x:i for i,x in enumerate(LABELS)}
tr=df[df._split=='train'].reset_index(drop=True);va=df[df._split=='validation'].reset_index(drop=True)
allf=[c for c in df.columns if not c.startswith('_')]
Xall=tr[allf].replace([np.inf,-np.inf],np.nan).fillna(0);ytr=tr._label_id.astype(int).to_numpy()
# train-only XGB ranking
ranker=XGBClassifier(n_estimators=180,max_depth=5,learning_rate=.05,min_child_weight=1,
 subsample=.9,colsample_bytree=.9,reg_lambda=1.5,random_state=20260919,n_jobs=8,
 eval_metric='mlogloss',tree_method='hist')
ranker.fit(Xall,ytr,sample_weight=compute_sample_weight('balanced',ytr))
rank=[x for _,x in sorted(zip(ranker.feature_importances_,allf),reverse=True)]
pd.DataFrame({'feature':rank,'importance':sorted(ranker.feature_importances_,reverse=True)}).to_csv(OUT/'train_feature_ranking.csv',index=False)

def met(y,p):
 P,R,F1,s=precision_recall_fscore_support(y,p,labels=range(3),zero_division=0)
 cm=confusion_matrix(y,p,labels=range(3));f=[]
 for i in range(3):
  fp=cm[:,i].sum()-cm[i,i];neg=cm.sum()-cm[i,:].sum();f.append(fp/neg if neg else 0)
 r={'accuracy':float(accuracy_score(y,p)),'macro_precision':float(P.mean()),
    'macro_recall':float(R.mean()),'macro_f1':float(F1.mean()),
    'max_class_fpr':float(max(f)),'errors':int((p!=y).sum()),
    'per_class':{LABELS[i]:{'P':float(P[i]),'R':float(R[i]),'F1':float(F1[i]),
                            'FPR':float(f[i]),'N':int(s[i])} for i in range(3)},
    'cm':cm.tolist()}
 r['target_pass']=r['accuracy']>=.95 and r['macro_precision']>=.95 and r['macro_recall']>=.98 and r['max_class_fpr']<=.05
 return r
def key(r):return (1 if r['target_pass'] else 0,r['macro_recall'],r['accuracy'],r['macro_precision'],-r['max_class_fpr'])

# Validation only: rank dimensionality + deployable tree-rule sweep.
yv=va._label_id.astype(int).to_numpy();upper_rows=[];cand=[];best=None
for k in [16,26,32,48,64]:
 fs=rank[:min(k,len(rank))]
 Xtr=tr[fs].fillna(0);Xv=va[fs].fillna(0)
 up=XGBClassifier(n_estimators=220,max_depth=6,learning_rate=.04,min_child_weight=1,
   subsample=.9,colsample_bytree=.9,reg_lambda=1.5,random_state=42,n_jobs=8,
   eval_metric='mlogloss',tree_method='hist')
 up.fit(Xtr,ytr,sample_weight=compute_sample_weight('balanced',ytr))
 um=met(yv,up.predict(Xv));upper_rows.append({'k':len(fs),**{x:v for x,v in um.items() if x not in ('per_class','cm')}})
 for depth in [6,10,14]:
  for leaf in [1,2]:
   gen=OptimizedRuleGenerator(confidence_threshold=0.0,min_confidence=0.0,
     tree_max_depth=depth,tree_min_samples_leaf=leaf,tree_class_weight='balanced')
   rules=gen.fit_and_generate(tr[fs],tr._label_id,fs,{i:x for i,x in enumerate(LABELS)},
                              validation=(va[fs],va._label_id),groups=tr._capture_id)
   deploy=[r for r in rules if r.get('type')!='ensemble_classifier']
   for th in [0,.5,.7,.8,.9,.95]:
    eng=OptimizedDPIEngine();eng.rules=deploy;eng.selected_features=fs
    eng.label_names={i:x for i,x in enumerate(LABELS)};eng.confidence_threshold=th;eng.classifier=None
    pp=[]
    for row in va[fs].fillna(0).to_dict('records'):
      m=eng.match(row);pp.append(lid[m[0].result] if m and m[0].result in lid else -1)
    mm=met(yv,np.array(pp))
    rr={'k':len(fs),'depth':depth,'leaf':leaf,'threshold':th,'n_rules':len(deploy),**{x:v for x,v in mm.items() if x not in ('per_class','cm')}}
    cand.append(rr)
    if best is None or key(mm)>key(best[0]):best=(mm,fs,depth,leaf,th,rules,deploy)
pd.DataFrame(upper_rows).to_csv(OUT/'offline_upper_sweep.csv',index=False)
pd.DataFrame(cand).to_csv(OUT/'validation_rule_sweep.csv',index=False)
bm,fs,depth,leaf,th,rules,deploy=best
best_upper=max(upper_rows,key=lambda r:(r['macro_recall'],r['accuracy']))
print('BEST UPPER',best_upper,flush=True);print('BEST RULE',depth,leaf,th,bm,flush=True)

bundle=OUT/'bundle'
if bundle.exists():shutil.rmtree(bundle)
spec=get_profile_spec('behavior15','behavior');spec['required_features']=fs
save_rule_bundle(rules,fs,{i:x for i,x in enumerate(LABELS)},th,str(bundle),
 bundle_meta={'task':'behavior','source':'real-pcap-runtime-validation','feature_profile':'behavior15',
              'profile_spec':spec,'context':spec['context'],'temporal_vote':1,
              'max_read_packets_per_file':50000,'validation_only_selection':True})
json.dump({'selected_features':fs,'tree_max_depth':depth,'tree_min_samples_leaf':leaf,
           'confidence_threshold':th,'offline_upper_best':best_upper,
           'rule_validation':bm,'fidelity_macro_f1_gap':best_upper['macro_f1']-bm['macro_f1']},
          open(OUT/'frozen_config.json','w'),indent=2)

# Direct validation predictions by observation ID.
eng=OptimizedDPIEngine();eng.load_rule_bundle(str(bundle)); direct={}
times=[]
for _,row in va.iterrows():
 t=time.perf_counter();m=eng.match({f:row[f] for f in fs});times.append((time.perf_counter()-t)*1000)
 direct[row['_observation_id']]=m[0].result if m else 'unknown'
json.dump({'n':len(direct),'predictions':direct},open(OUT/'direct_validation_predictions.json','w'))

# Independent rules-only PCAP replay on validation files.
manifest=[json.loads(x) for x in (OUT/'manifest.jsonl').read_text().splitlines() if x.strip()]
val=[x for x in manifest if x['split']=='validation']
valdir=OUT/'validation_pcaps';shutil.rmtree(valdir,ignore_errors=True);valdir.mkdir()
truth={}
for x in val:
 p=Path(x['pcap']);dst=valdir/p.name
 try:dst.symlink_to(p)
 except FileExistsError:pass
 truth[p.name]=x['behavior']
guard=OUT/'import_guard';guard.mkdir(exist_ok=True)
(guard/'sitecustomize.py').write_text(
"import builtins\n_o=builtins.__import__\ndef g(n,*a,**k):\n"
"    if n.split('.')[0] in {'sklearn','xgboost','scipy'}: raise ImportError('training lib blocked: '+n)\n"
"    return _o(n,*a,**k)\nbuiltins.__import__=g\n")
outjson=OUT/'independent_validation.json';env=os.environ.copy();env['PYTHONPATH']=str(guard)+os.pathsep+str(ROOT)
t=time.perf_counter();before=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
p=subprocess.run([sys.executable,'-m','src.engine.dpi_infer','--rules',str(bundle),
                  '--pcap-dir',str(valdir),'-o',str(outjson)],capture_output=True,text=True,env=env)
wall=time.perf_counter()-t;rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
(OUT/'independent_cli.log').write_text(p.stdout+'\nSTDERR\n'+p.stderr)
if p.returncode:raise RuntimeError(p.stderr)
data=json.loads(outjson.read_text());rows=data['results']
yt=[];yp=[];parity=[]
for r in rows:
 lab=truth[r['source_file']];yt.append(lid[lab]);yp.append(lid.get(r['predicted_label'],-1))
 if r['observation_id'] in direct and direct[r['observation_id']]!=r['predicted_label']:
  parity.append((r['observation_id'],direct[r['observation_id']],r['predicted_label']))
im=met(np.array(yt),np.array(yp))
json.dump(im,open(OUT/'independent_validation_metrics.json','w'),indent=2)
json.dump({'consistent':not parity,'mismatches':parity[:50],'compared':sum(r['observation_id'] in direct for r in rows)},
          open(OUT/'parity.json','w'),indent=2)

bundle_bytes=sum(x.stat().st_size for x in bundle.rglob('*') if x.is_file())
perf={'mine':json.load(open(OUT/'mine_flow_performance.json')),
      'independent_validation_wall_sec':wall,'peak_rss_kb':rss,
      'windows':len(rows),'windows_per_sec':len(rows)/max(wall,1e-9),
      'matcher_ms_mean':float(np.mean(times)),'matcher_ms_p95':float(np.quantile(times,.95)),
      'bundle_bytes':bundle_bytes,'rule_count':len(deploy)}
json.dump(perf,open(OUT/'performance.json','w'),indent=2)
accept={'profile':'behavior15','scope':'file-disjoint real-PCAP validation (not sealed test)',
        'offline_upper_validation':best_upper,'rule_validation_direct':bm,
        'independent_rule_validation':im,'parity_consistent':not parity,
        'formal_validation_pass':im['target_pass'],
        'official_ARFF_reference_not_runtime':'VPN 15s ExtraTrees Acc99.39/P99.17/R99.15/FPR0.73'}
json.dump(accept,open(OUT/'acceptance.json','w'),indent=2)
print('INDEPENDENT',im,flush=True);print('PARITY',not parity,'compared',sum(r['observation_id'] in direct for r in rows),flush=True);print('PERF',perf,flush=True)
