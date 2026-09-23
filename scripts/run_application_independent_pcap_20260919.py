import json,os,resource,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix

ROOT=Path('/workspace/Huawei');OUT=ROOT/'output/final_profile_runtime_20260919/application'
OLD=Path('/workspace/taskquay-data/worktrees/Huawei-610224a7/output/highmetric_search')
bundle=OUT/'bundle'
# Exact raw-read cap used by frozen USTC extraction.
cfg=json.loads((bundle/'bundle_config.json').read_text());cfg['max_read_packets_per_file']=100000
(bundle/'bundle_config.json').write_text(json.dumps(cfg,indent=2))

manifest=[json.loads(x) for x in (OLD/'ustc4_manifest.jsonl').read_text().splitlines() if x.strip()]
test=[x for x in manifest if x['split']=='test'];labels=['FTP','Gmail','MySQL','WorldOfWarcraft'];lid={x:i for i,x in enumerate(labels)}
pcdir=OUT/'pcap_replay';shutil.rmtree(pcdir,ignore_errors=True);pcdir.mkdir()
for p in sorted(set(x['file'] for x in manifest)):
    (pcdir/Path(p).name).symlink_to(Path(p))
guard=OUT/'import_guard';env=os.environ.copy();env['PYTHONPATH']=str(guard)+os.pathsep+str(ROOT)
out=OUT/'independent_pcap_all_predictions.json'
cmd=[sys.executable,'-m','src.engine.dpi_infer','--rules',str(bundle),'--pcap-dir',str(pcdir),'-o',str(out)]
t=time.perf_counter();p=subprocess.run(cmd,capture_output=True,text=True,env=env);wall=time.perf_counter()-t
rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
(OUT/'independent_pcap_cli.log').write_text(p.stdout+'\nSTDERR\n'+p.stderr)
if p.returncode:raise RuntimeError(p.stderr)
rows=json.loads(out.read_text())['results'];predmap={(r['source_file'],int(str(r['observation_id']).rsplit(':',1)[1])):r['predicted_label'] for r in rows}
yt=[];yp=[];missing=[]
for x in test:
    k=(Path(x['file']).name,int(x['session_index']));lab=predmap.get(k)
    if lab is None:missing.append(k);lab='unknown'
    yt.append(lid[x['label']]);yp.append(lid.get(lab,-1))
yt=np.array(yt);yp=np.array(yp)
P,R,F1,s=precision_recall_fscore_support(yt,yp,labels=range(4),zero_division=0);cm=confusion_matrix(yt,yp,labels=range(4));f=[]
for i in range(4):
    fp=cm[:,i].sum()-cm[i,i];neg=cm.sum()-cm[i,:].sum();f.append(fp/neg if neg else 0)
m={'accuracy':float(accuracy_score(yt,yp)),'macro_precision':float(P.mean()),'macro_recall':float(R.mean()),'macro_f1':float(F1.mean()),'max_class_fpr':float(max(f)),'errors':int((yt!=yp).sum()),'missing':len(missing),'cm':cm.tolist()}
m['target_pass']=m['accuracy']>=.95 and m['macro_precision']>=.95 and m['macro_recall']>=.98 and m['max_class_fpr']<=.05
json.dump(m,open(OUT/'independent_pcap_test_metrics.json','w'),indent=2)
perf={'wall_sec':wall,'peak_rss_kb':rss,'all_runtime_records':len(rows),'records_per_sec':len(rows)/wall}
json.dump(perf,open(OUT/'independent_pcap_performance.json','w'),indent=2)
print('METRICS',m);print('PERF',perf);print('MISSING',missing[:10])
