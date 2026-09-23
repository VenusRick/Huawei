# -*- coding: utf-8 -*-
import json, os, re, resource, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.engine.rule_compiler.optimized_generator import OptimizedRuleGenerator
from src.engine.matcher.optimized_engine import OptimizedDPIEngine
from src.engine.model_io import save_rule_bundle
from src.profile_specs import get_profile_spec, temporal_vote_rows

ROOT=Path('/workspace/Huawei')
OUT=ROOT/'output/final_profile_runtime_20260919/tunnel'
MINE=OUT/'mine'
df=pd.read_csv(MINE/'raw_features.csv')
LABELS=['NonTor','Tor'];lid={x:i for i,x in enumerate(LABELS)}
tr=df[df._split=='train'].reset_index(drop=True);va=df[df._split=='validation'].reset_index(drop=True)
allf=[c for c in df.columns if not c.startswith('_')]
Xall=tr[allf].replace([np.inf,-np.inf],np.nan).fillna(0);ytr=tr._label_id.astype(int).to_numpy()

ranker=XGBClassifier(n_estimators=180,max_depth=5,learning_rate=.05,min_child_weight=1,
 subsample=.9,colsample_bytree=.9,reg_lambda=1.5,random_state=20260919,n_jobs=8,
 eval_metric='logloss',tree_method='hist')
ranker.fit(Xall,ytr,sample_weight=compute_sample_weight('balanced',ytr))
order=np.argsort(ranker.feature_importances_)[::-1]
rank=[allf[i] for i in order]
pd.DataFrame({'feature':rank,'importance':[float(ranker.feature_importances_[i]) for i in order]}).to_csv(OUT/'train_feature_ranking.csv',index=False)

pat=re.compile(r':tw(\d+):(\d+)$')
def row_meta(row,label,conf,key='predicted_label'):
 m=pat.search(str(row['_observation_id']))
 if not m: raise ValueError(row['_observation_id'])
 return {'observation_id':row['_observation_id'],'source_file':Path(row['_source_file']).name,
         'session_index':int(m.group(1)),'window_start':int(m.group(2))*15.0,
         'window_end':(int(m.group(2))+1)*15.0,key:label,'confidence':float(conf)}

def met(y,p):
 P,R,F1,s=precision_recall_fscore_support(y,p,labels=[0,1],zero_division=0)
 cm=confusion_matrix(y,p,labels=[0,1]);f=[]
 for i in [0,1]:
  fp=cm[:,i].sum()-cm[i,i];neg=cm.sum()-cm[i,:].sum();f.append(fp/neg if neg else 0)
 r={'accuracy':float(accuracy_score(y,p)),'macro_precision':float(P.mean()),
    'macro_recall':float(R.mean()),'macro_f1':float(F1.mean()),
    'max_class_fpr':float(max(f)),'errors':int((p!=y).sum()),
    'per_class':{LABELS[i]:{'P':float(P[i]),'R':float(R[i]),'F1':float(F1[i]),
                            'FPR':float(f[i]),'N':int(s[i])} for i in [0,1]},
    'cm':cm.tolist()}
 r['target_pass']=r['accuracy']>=.95 and r['macro_precision']>=.95 and r['macro_recall']>=.98 and r['max_class_fpr']<=.05
 return r
def keyfun(r):return (1 if r['target_pass'] else 0,r['macro_recall'],r['accuracy'],r['macro_precision'],-r['max_class_fpr'])

def vote_eval(rows, truths):
 out=[];yt=[];byfile={}
 for rr,t in zip(rows,truths):
  byfile.setdefault(rr['source_file'],[]).append((rr,t))
 for fn,pairs in byfile.items():
  seq=[x[0] for x in pairs]
  truth=pairs[0][1]
  voted=temporal_vote_rows(seq,3,label_key='predicted_label')
  for v in voted:
   out.append(v);yt.append(lid[truth] if isinstance(truth,str) else int(truth))
 yp=np.array([lid.get(v['predicted_label'],-1) for v in out])
 return met(np.array(yt),yp),out,np.array(yt),yp

yv=va._label_id.astype(int).to_numpy()
upper_rows=[];cand=[];best=None
for k in [16,28,32,48,64]:
 fs=rank[:min(k,len(rank))]
 Xtr=tr[fs].fillna(0);Xv=va[fs].fillna(0)
 up=XGBClassifier(n_estimators=220,max_depth=6,learning_rate=.04,min_child_weight=1,
  subsample=.9,colsample_bytree=.9,reg_lambda=1.5,random_state=42,n_jobs=8,
  eval_metric='logloss',tree_method='hist')
 up.fit(Xtr,ytr,sample_weight=compute_sample_weight('balanced',ytr))
 pro=up.predict_proba(Xv);pred=np.argmax(pro,axis=1)
 rawm=met(yv,pred)
 urows=[row_meta(row,LABELS[int(p)],float(c)) for (_,row),p,c in zip(va.iterrows(),pred,pro.max(1))]
 votem,_,_,_=vote_eval(urows,[LABELS[int(x)] for x in yv])
 upper_rows.append({'k':len(fs),'raw_accuracy':rawm['accuracy'],'raw_macro_recall':rawm['macro_recall'],
                    'raw_macro_f1':rawm['macro_f1'],'vote_accuracy':votem['accuracy'],
                    'vote_macro_precision':votem['macro_precision'],'vote_macro_recall':votem['macro_recall'],
                    'vote_macro_f1':votem['macro_f1'],'vote_max_fpr':votem['max_class_fpr'],
                    'vote_target_pass':votem['target_pass']})
 for depth in [5,8,10,14]:
  for leaf in [1,2,4]:
   gen=OptimizedRuleGenerator(confidence_threshold=0.0,min_confidence=0.0,
    tree_max_depth=depth,tree_min_samples_leaf=leaf,tree_class_weight='balanced')
   rules=gen.fit_and_generate(tr[fs],tr._label_id,fs,{0:'NonTor',1:'Tor'},
      validation=(va[fs],va._label_id),groups=tr._capture_id)
   deploy=[r for r in rules if r.get('type')!='ensemble_classifier']
   for th in [0,.5,.7,.8,.9,.95]:
    eng=OptimizedDPIEngine();eng.rules=deploy;eng.selected_features=fs
    eng.label_names={0:'NonTor',1:'Tor'};eng.confidence_threshold=th;eng.classifier=None
    raw=[];raw_pred=[]
    for _,row in va.iterrows():
     m=eng.match({f:row[f] for f in fs})
     lab=m[0].result if m else 'unknown';conf=m[0].confidence if m else 0.0
     raw.append(row_meta(row,lab,conf));raw_pred.append(lid.get(lab,-1))
    rawm=met(yv,np.array(raw_pred))
    votem,voted,vy,vp=vote_eval(raw,[LABELS[int(x)] for x in yv])
    rr={'k':len(fs),'depth':depth,'leaf':leaf,'threshold':th,'n_rules':len(deploy),
        'raw_accuracy':rawm['accuracy'],'raw_macro_recall':rawm['macro_recall'],'raw_macro_f1':rawm['macro_f1'],
        'vote_accuracy':votem['accuracy'],'vote_macro_precision':votem['macro_precision'],
        'vote_macro_recall':votem['macro_recall'],'vote_macro_f1':votem['macro_f1'],
        'vote_max_fpr':votem['max_class_fpr'],'vote_errors':votem['errors'],
        'vote_target_pass':votem['target_pass'],'n_voted':len(vy)}
    cand.append(rr)
    if best is None or keyfun(votem)>keyfun(best[0]):
     best=(votem,rawm,fs,depth,leaf,th,rules,deploy,raw,voted)
pd.DataFrame(upper_rows).to_csv(OUT/'offline_upper_sweep.csv',index=False)
pd.DataFrame(cand).to_csv(OUT/'validation_rule_sweep.csv',index=False)
vm,rm,fs,depth,leaf,th,rules,deploy,direct_raw,direct_voted=best
best_upper=max(upper_rows,key=lambda r:(1 if r['vote_target_pass'] else 0,r['vote_macro_recall'],r['vote_accuracy']))
print('BEST UPPER',best_upper,flush=True);print('BEST RULE VOTE',depth,leaf,th,vm,'RAW',rm,flush=True)

bundle=OUT/'bundle';shutil.rmtree(bundle,ignore_errors=True)
spec=get_profile_spec('tunnel15','tool');spec['required_features']=fs
save_rule_bundle(rules,fs,{0:'NonTor',1:'Tor'},th,str(bundle),
 bundle_meta={'task':'tool','source':'real-pcap-runtime-validation','feature_profile':'tunnel15',
              'profile_spec':spec,'context':spec['context'],'temporal_vote':3,
              'max_read_packets_per_file':30000,'validation_only_selection':True})
json.dump({'selected_features':fs,'tree_max_depth':depth,'tree_min_samples_leaf':leaf,
           'confidence_threshold':th,'offline_upper_best':best_upper,
           'rule_raw_validation':rm,'rule_vote_validation':vm},
          open(OUT/'frozen_config.json','w'),indent=2)

# independent rules-only validation replay
manifest=[json.loads(x) for x in (OUT/'manifest.jsonl').read_text().splitlines() if x.strip()]
val=[x for x in manifest if x['split']=='validation'];truth={Path(x['pcap']).name:x['app'] for x in val}
valdir=OUT/'validation_pcaps';shutil.rmtree(valdir,ignore_errors=True);valdir.mkdir()
for x in val:
 p=Path(x['pcap']);(valdir/p.name).symlink_to(p)
guard=OUT/'import_guard';guard.mkdir(exist_ok=True)
(guard/'sitecustomize.py').write_text(
"import builtins\n_o=builtins.__import__\ndef g(n,*a,**k):\n"
"    if n.split('.')[0] in {'sklearn','xgboost','scipy'}: raise ImportError('training lib blocked: '+n)\n"
"    return _o(n,*a,**k)\nbuiltins.__import__=g\n")
outjson=OUT/'independent_validation.json';env=os.environ.copy();env['PYTHONPATH']=str(guard)+os.pathsep+str(ROOT)
t=time.perf_counter()
p=subprocess.run([sys.executable,'-m','src.engine.dpi_infer','--rules',str(bundle),
                  '--pcap-dir',str(valdir),'-o',str(outjson)],capture_output=True,text=True,env=env)
wall=time.perf_counter()-t;rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
(OUT/'independent_cli.log').write_text(p.stdout+'\nSTDERR\n'+p.stderr)
if p.returncode:raise RuntimeError(p.stderr)
rows=json.loads(outjson.read_text())['results']
yt=np.array([lid[truth[r['source_file']]] for r in rows]);yp=np.array([lid.get(r['predicted_label'],-1) for r in rows])
im=met(yt,yp);json.dump(im,open(OUT/'independent_validation_metrics.json','w'),indent=2)

# direct voted parity by observation id + file (winner observation is stable)
dmap={(r['source_file'],r['observation_id']):r['predicted_label'] for r in direct_voted}
mis=[]
for r in rows:
 k=(r['source_file'],r['observation_id'])
 if k in dmap and dmap[k]!=r['predicted_label']:mis.append((k,dmap[k],r['predicted_label']))
json.dump({'consistent':not mis,'compared':sum((r['source_file'],r['observation_id']) in dmap for r in rows),
           'mismatches':mis[:50]},open(OUT/'parity.json','w'),indent=2)

# matcher-only raw-window cost
eng=OptimizedDPIEngine();eng.load_rule_bundle(str(bundle));mt=[]
for _,row in va.iterrows():
 t=time.perf_counter();eng.match({f:row[f] for f in fs});mt.append((time.perf_counter()-t)*1000)
bundle_bytes=sum(x.stat().st_size for x in bundle.rglob('*') if x.is_file())
perf={'mine':json.load(open(OUT/'mine_performance.json')),'independent_validation_wall_sec':wall,
      'peak_rss_kb':rss,'voted_observations':len(rows),'voted_per_sec':len(rows)/max(wall,1e-9),
      'matcher_ms_mean_raw_window':float(np.mean(mt)),'matcher_ms_p95_raw_window':float(np.quantile(mt,.95)),
      'bundle_bytes':bundle_bytes,'rule_count':len(deploy)}
json.dump(perf,open(OUT/'performance.json','w'),indent=2)
accept={'profile':'tunnel15','scope':'file-disjoint real-PCAP validation / 3-window per-flow vote; not sealed test',
        'offline_upper_vote_validation':best_upper,'rule_vote_validation_direct':vm,
        'independent_rule_vote_validation':im,'parity_consistent':not mis,
        'formal_validation_pass':im['target_pass'],
        'prior_grouped_offline_reference':'Hybrid Top64+3-window mean Acc98.29/P98.27/R98.18/FPR3.05'}
json.dump(accept,open(OUT/'acceptance.json','w'),indent=2)
print('INDEPENDENT',im,flush=True);print('PARITY',not mis,'compared',sum((r['source_file'],r['observation_id']) in dmap for r in rows),flush=True);print('PERF',perf,flush=True)
