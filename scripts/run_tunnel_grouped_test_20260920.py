# -*- coding: utf-8 -*-
"""Post-exploration grouped Tor/NonTor RuleBundle test.

The matched workstation/gateway business pairs have been studied previously,
so this is not claimed as a pristine research holdout. The split is frozen for
this final engineering acceptance and deliberately places hard bulk-transfer
pairs in test.
"""
from __future__ import annotations

import hashlib, json, os, re, resource, shutil, subprocess, sys, time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.engine.matcher.optimized_engine import OptimizedDPIEngine
from src.engine.model_io import save_rule_bundle
from src.engine.rule_compiler.optimized_generator import OptimizedRuleGenerator
from src.profile_specs import get_profile_spec, temporal_vote_rows

ROOT=Path('/workspace/Huawei')
SRC=ROOT/'output/final_profile_runtime_20260919/tunnel'
OUT=ROOT/'output/protocol_and_system_acceptance_20260920/tunnel_fresh'
LABELS=['NonTor','Tor'];LID={x:i for i,x in enumerate(LABELS)}
PAT=re.compile(r':tw(\d+):(\d+)$')

PAIRS=[
 ('ssl_browse','BROWSING_gate_SSL_Browsing.pcap','SSL_Browsing.pcap'),
 ('aim_chat','CHAT_gate_AIM_chat.pcap','AIM_Chat.pcap'),('icq_chat','CHAT_gate_ICQ_chat.pcap','ICQ_Chat.pcap'),
 ('facebook_chat','CHAT_gate_facebook_chat.pcap','facebook_chat.pcap'),('hangout_chat','CHAT_gate_hangout_chat.pcap','hangout_chat.pcap'),
 ('skype_chat','CHAT_gate_skype_chat.pcap','skype_chat.pcap'),('sftp','FILE-TRANSFER_gate_SFTP_filetransfer.pcap','SFTP_filetransfer.pcap'),
 ('skype_transfer','FILE-TRANSFER_tor_skype_transfer.pcap','skype_transfer.pcap'),('imap','MAIL_gate_Email_IMAP_filetransfer.pcap','Email_IMAP_filetransfer.pcap'),
 ('pop','MAIL_gate_POP_filetransfer.pcap','POP_filetransfer.pcap'),('thunderbird_imap','MAIL_Gateway_Thunderbird_Imap.pcap','Workstation_Thunderbird_Imap.pcap'),
 ('thunderbird_pop','MAIL_Gateway_Thunderbird_POP.pcap','Workstation_Thunderbird_POP.pcap'),('p2p_multi','P2P_tor_p2p_multipleSpeed.pcap','p2p_multipleSpeed.pcap'),
 ('p2p_vuze','P2P_tor_p2p_vuze.pcap','p2p_vuze.pcap'),('vimeo','VIDEO_Vimeo_Gateway.pcap','Vimeo_Workstation.pcap'),
 ('youtube_flash','VIDEO_Youtube_Flash_Gateway.pcap','Youtube_Flash_Workstation.pcap'),('youtube_html5','VIDEO_Youtube_HTML5_Gateway.pcap','Youtube_HTML5_Workstation.pcap'),
 ('fb_voice','VOIP_Facebook_Voice_Gateway.pcap','Facebook_Voice_Workstation.pcap'),('hangouts_voice','VOIP_Hangouts_voice_Gateway.pcap','Hangouts_voice_Workstation.pcap'),
 ('skype_voice','VOIP_Skype_Voice_Gateway.pcap','Skype_Voice_Workstation.pcap'),('skype_audio','VOIP_gate_Skype_Audio.pcap','Skype_Audio.pcap'),
 ('facebook_audio','VOIP_gate_facebook_Audio.pcap','facebook_Audio.pcap'),('hangout_audio','VOIP_gate_hangout_audio.pcap','Hangout_Audio.pcap')]

TEST_PAIRS={'sftp','skype_transfer','skype_audio','vimeo','facebook_chat'}
VAL_PAIRS={'aim_chat','imap','p2p_vuze','youtube_html5','hangouts_voice'}

def met(y,p):
    P,R,F1,s=precision_recall_fscore_support(y,p,labels=[0,1],zero_division=0);cm=confusion_matrix(y,p,labels=[0,1]);f=[]
    for i in [0,1]:
        fp=cm[:,i].sum()-cm[i,i];neg=cm.sum()-cm[i,:].sum();f.append(fp/neg if neg else 0)
    r={'accuracy':float(accuracy_score(y,p)),'macro_precision':float(P.mean()),'macro_recall':float(R.mean()),'macro_f1':float(F1.mean()),'max_class_fpr':float(max(f)),'errors':int((p!=y).sum()),'per_class':{LABELS[i]:{'P':float(P[i]),'R':float(R[i]),'F1':float(F1[i]),'FPR':float(f[i]),'N':int(s[i])} for i in [0,1]},'cm':cm.tolist()};r['target_pass']=r['accuracy']>=.95 and r['macro_precision']>=.95 and r['macro_recall']>=.98 and r['max_class_fpr']<=.05;return r
def key(r):return (1 if r['target_pass'] else 0,r['macro_recall'],r['accuracy'],r['macro_precision'],-r['max_class_fpr'])
def row_meta(row,lab,conf):
    m=PAT.search(str(row['_observation_id']));return {'observation_id':row['_observation_id'],'source_file':Path(row['_source_file']).name,'session_index':int(m.group(1)),'window_start':int(m.group(2))*15.0,'window_end':(int(m.group(2))+1)*15.0,'predicted_label':lab,'confidence':float(conf)}
def vote(rows,truth_by_file):
    out=[];yt=[];by={}
    for r in rows:by.setdefault(r['source_file'],[]).append(r)
    for fn,seq in by.items():
        for v in temporal_vote_rows(seq,3,label_key='predicted_label'):out.append(v);yt.append(LID[truth_by_file[fn]])
    yp=np.asarray([LID.get(x['predicted_label'],-1) for x in out]);return met(np.asarray(yt),yp),out

def main():
    OUT.mkdir(parents=True,exist_ok=True);df=pd.read_csv(SRC/'mine/raw_features.csv');base_to_pair={};pair_to_files={}
    root=Path('/workspace/datasets/ISCXTor2016/pcap')
    for pair,t,n in PAIRS:
        base_to_pair[t]=pair;base_to_pair[n]=pair;pair_to_files[pair]=[root/'Tor'/t,root/'NonTor'/n]
    df['_pair']=[base_to_pair.get(Path(x).name,'') for x in df._source_file.astype(str)];df=df[df._pair!=''].copy()
    if not TEST_PAIRS.issubset(set(df._pair.unique())):raise RuntimeError('missing mandatory Tor hard pairs')
    def sp(pair):return 'test' if pair in TEST_PAIRS else ('validation' if pair in VAL_PAIRS else 'train')
    df['_final_split']=df._pair.map(sp)
    split={'scope':'post-exploration frozen matched-pair split','warning':'pairs were exposed during earlier exploratory CV; not pristine blind research holdout','train_pairs':sorted(set(df.loc[df._final_split=='train','_pair'])),'validation_pairs':sorted(VAL_PAIRS),'test_pairs':sorted(TEST_PAIRS),'rows':df.groupby('_final_split').size().to_dict()}
    spath=OUT/'pair_split.json';spath.write_text(json.dumps(split,indent=2));split_sha=hashlib.sha256(spath.read_bytes()).hexdigest()
    tr=df[df._final_split=='train'].reset_index(drop=True);va=df[df._final_split=='validation'].reset_index(drop=True);te=df[df._final_split=='test'].reset_index(drop=True);features=[c for c in df.columns if not c.startswith('_')]
    X=tr[features].replace([np.inf,-np.inf],np.nan).fillna(0);ytr=tr._label_id.astype(int).to_numpy();ranker=XGBClassifier(n_estimators=180,max_depth=5,learning_rate=.05,min_child_weight=1,subsample=.9,colsample_bytree=.9,reg_lambda=1.5,random_state=20260920,n_jobs=8,eval_metric='logloss',tree_method='hist');ranker.fit(X,ytr,sample_weight=compute_sample_weight('balanced',ytr));order=np.argsort(ranker.feature_importances_)[::-1];rank=[features[i] for i in order];pd.DataFrame({'feature':rank,'importance':[float(ranker.feature_importances_[i]) for i in order]}).to_csv(OUT/'train_feature_ranking.csv',index=False)
    truth_val={Path(x).name:('Tor' if '/Tor/' in str(x) else 'NonTor') for pair in VAL_PAIRS for x in pair_to_files[pair]};best=None;cand=[]
    for k in [16,28,32,48,64]:
        fs=rank[:min(k,len(rank))]
        for depth in [5,8,10,14]:
            for leaf in [1,2,4]:
                gen=OptimizedRuleGenerator(confidence_threshold=0,min_confidence=0,tree_max_depth=depth,tree_min_samples_leaf=leaf,tree_class_weight='balanced');rules=gen.fit_and_generate(tr[fs],tr._label_id,fs,{0:'NonTor',1:'Tor'},validation=(va[fs],va._label_id),groups=tr._pair);deploy=[r for r in rules if r.get('type')!='ensemble_classifier']
                for th in [0,.5,.7,.8,.9,.95]:
                    eng=OptimizedDPIEngine();eng.rules=deploy;eng.selected_features=fs;eng.label_names={0:'NonTor',1:'Tor'};eng.confidence_threshold=th;eng.classifier=None;raw=[]
                    for _,row in va.iterrows():
                        m=eng.match({f:row[f] for f in fs});lab=m[0].result if m else 'unknown';conf=m[0].confidence if m else 0;raw.append(row_meta(row,lab,conf))
                    vm,voted=vote(raw,truth_val);rr={'k':len(fs),'depth':depth,'leaf':leaf,'threshold':th,'n_rules':len(deploy),**{z:v for z,v in vm.items() if z not in ('per_class','cm')}};cand.append(rr)
                    if best is None or key(vm)>key(best[0]):best=(vm,fs,depth,leaf,th,rules,deploy)
    pd.DataFrame(cand).to_csv(OUT/'validation_rule_sweep.csv',index=False);vm,fs,depth,leaf,th,rules,deploy=best
    bundle=OUT/'bundle';shutil.rmtree(bundle,ignore_errors=True);spec=get_profile_spec('tunnel15','tool');spec['required_features']=fs;save_rule_bundle(rules,fs,{0:'NonTor',1:'Tor'},th,str(bundle),bundle_meta={'task':'tool','feature_profile':'tunnel15','profile_spec':spec,'context':spec['context'],'temporal_vote':3,'max_read_packets_per_file':50000,'split_scope':split['scope']});freeze={'split_sha256':split_sha,'test_pairs':sorted(TEST_PAIRS),'selected_features':fs,'tree_max_depth':depth,'tree_min_samples_leaf':leaf,'confidence_threshold':th,'validation':vm,'test_policy':'test pair rows not used in ranking/rule/threshold selection'};fp=OUT/'frozen_config.json';fp.write_text(json.dumps(freeze,indent=2));freeze_sha=hashlib.sha256(fp.read_bytes()).hexdigest()
    # Direct test after freeze.
    eng=OptimizedDPIEngine();eng.load_rule_bundle(str(bundle));truth_test={Path(x).name:('Tor' if '/Tor/' in str(x) else 'NonTor') for pair in TEST_PAIRS for x in pair_to_files[pair]};raw=[];times=[]
    for _,row in te.iterrows():
        t=time.perf_counter();m=eng.match({f:row[f] for f in fs});times.append((time.perf_counter()-t)*1000);lab=m[0].result if m else 'unknown';conf=m[0].confidence if m else 0;raw.append(row_meta(row,lab,conf))
    direct,voted=vote(raw,truth_test);direct_map={(r['source_file'],r['observation_id']):r['predicted_label'] for r in voted}
    # Independent rules-only replay on exactly frozen test files.
    pcapdir=OUT/'test_pcaps';shutil.rmtree(pcapdir,ignore_errors=True);pcapdir.mkdir()
    for pair in TEST_PAIRS:
        for p in pair_to_files[pair]:(pcapdir/p.name).symlink_to(p)
    guard=OUT/'import_guard';guard.mkdir(exist_ok=True);(guard/'sitecustomize.py').write_text("import builtins\n_o=builtins.__import__\ndef g(n,*a,**k):\n    if n.split('.')[0] in {'sklearn','xgboost','scipy'}: raise ImportError('blocked '+n)\n    return _o(n,*a,**k)\nbuiltins.__import__=g\n")
    outjson=OUT/'independent_test.json';env=os.environ.copy();env['PYTHONPATH']=str(guard)+os.pathsep+str(ROOT);t=time.perf_counter();proc=subprocess.run([sys.executable,'-m','src.engine.dpi_infer','--rules',str(bundle),'--pcap-dir',str(pcapdir),'-o',str(outjson)],capture_output=True,text=True,env=env);wall=time.perf_counter()-t;rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss;(OUT/'independent_cli.log').write_text(proc.stdout+'\nSTDERR\n'+proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stderr)
    rows=json.loads(outjson.read_text())['results'];yt=np.asarray([LID[truth_test[r['source_file']]] for r in rows]);yp=np.asarray([LID.get(r['predicted_label'],-1) for r in rows]);im=met(yt,yp);mis=[r['observation_id'] for r in rows if direct_map.get((r['source_file'],r['observation_id']))!=r['predicted_label']]
    # per-pair test results
    file_pair={Path(x).name:pair for pair in TEST_PAIRS for x in pair_to_files[pair]};per_pair={}
    for pair in sorted(TEST_PAIRS):
        rr=[r for r in rows if file_pair[r['source_file']]==pair];yy=np.asarray([LID[truth_test[r['source_file']]] for r in rr]);pp=np.asarray([LID.get(r['predicted_label'],-1) for r in rr]);per_pair[pair]=met(yy,pp) if len(rr) else {'n':0}
    json.dump(im,open(OUT/'metrics.json','w'),indent=2);json.dump(per_pair,open(OUT/'hard_pair_metrics.json','w'),indent=2);json.dump({'consistent':not mis,'compared':len(rows),'mismatches':mis[:50]},open(OUT/'parity.json','w'),indent=2)
    bytes_in=sum(p.stat().st_size for pair in TEST_PAIRS for p in pair_to_files[pair]);perf={'independent_wall_sec':wall,'peak_rss_mib':rss/1024.0,'voted_observations':len(rows),'observations_per_sec':len(rows)/max(wall,1e-9),'input_bytes':bytes_in,'mib_per_sec':(bytes_in/(1024*1024))/max(wall,1e-9),'matcher_ms_mean_raw_window':float(np.mean(times)),'matcher_ms_p95_raw_window':float(np.quantile(times,.95)),'feature_count':len(fs),'rule_count':len(deploy),'temporal_vote':3};json.dump(perf,open(OUT/'performance.json','w'),indent=2)
    acc={'scope':split['scope'],'warning':split['warning'],'validation':vm,'test_direct':direct,'test_independent':im,'per_pair':per_pair,'parity_consistent':not mis,'formal_test_pass':im['target_pass'],'freeze_sha256':freeze_sha};json.dump(acc,open(OUT/'acceptance.json','w'),indent=2)
    print('SPLIT',split);print('VALIDATION',vm);print('TEST',im);print('PER_PAIR',json.dumps(per_pair));print('PARITY',not mis);print('PERF',perf)

if __name__=='__main__':main()
