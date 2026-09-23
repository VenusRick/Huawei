import json,os,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path('/workspace/Huawei');OUT=ROOT/'output/final_profile_runtime_20260919/application'
bundle=OUT/'bundle';pcdir=OUT/'pcap_replay';guard=OUT/'import_guard'
env=os.environ.copy();env['PYTHONPATH']=str(guard)+os.pathsep+str(ROOT)
out=OUT/'independent_pcap_on_demand.json'
t=time.perf_counter()
p=subprocess.run([sys.executable,'-m','src.engine.dpi_infer','--rules',str(bundle),
                  '--pcap-dir',str(pcdir),'--on-demand','-o',str(out)],
                 capture_output=True,text=True,env=env)
wall=time.perf_counter()-t;rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
(OUT/'independent_pcap_on_demand.log').write_text(p.stdout+'\nSTDERR\n'+p.stderr)
if p.returncode:raise RuntimeError(p.stderr)
a=json.loads((OUT/'independent_pcap_all_predictions.json').read_text())
b=json.loads(out.read_text())
ma={(r['source_file'],r['observation_id']):r['predicted_label'] for r in a['results']}
mb={(r['source_file'],r['observation_id']):r['predicted_label'] for r in b['results']}
keys=set(ma)|set(mb);mis=[(k,ma.get(k),mb.get(k)) for k in keys if ma.get(k)!=mb.get(k)]
perf={'wall_sec':wall,'peak_rss_kb':rss,'records':len(b['results']),
      'records_per_sec':len(b['results'])/wall,'operator_calls':b.get('operator_calls',{}),
      'prediction_mismatches_vs_full':len(mis)}
json.dump(perf,open(OUT/'on_demand_performance.json','w'),indent=2)
print('PERF',perf);print('MISMATCH_SAMPLE',mis[:5])
