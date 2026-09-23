# -*- coding: utf-8 -*-
from pathlib import Path
import json, shutil, hashlib, os

ROOT=Path('/workspace/Huawei')
SRC=ROOT/'output/final_profile_runtime_20260919'
OUT=ROOT/'output/three_profile_acceptance_20260919'

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

profiles=['application','behavior','tunnel']

def readj(p):
    return json.loads(Path(p).read_text())

def copy_file(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)

def copy_tree(src,dst):
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src,dst)

summary={'date':'2026-09-19','formal_thresholds':{'accuracy':0.95,'macro_precision':0.95,'macro_recall':0.98,'max_class_fpr':0.05},
         'profiles':{},'boundaries':{}}

# Application
s=SRC/'application'; d=OUT/'application'; d.mkdir()
for fn in ['frozen_config.json','rule_test_metrics.json','rule_test_predictions.csv','parity.json','performance.json',
           'acceptance.json','independent_pcap_test_metrics.json','independent_pcap_performance.json','on_demand_performance.json',
           'selected_features.txt','validation_rule_sweep.csv']:
    if (s/fn).exists(): copy_file(s/fn,d/fn)
copy_tree(s/'bundle',d/'bundle')
copy_file(ROOT/'output/experiment_archive_20260919/ustc4/ustc4_manifest.jsonl',d/'manifest.jsonl')
app=readj(s/'acceptance.json')
appm=readj(s/'rule_test_metrics.json')
appfc=readj(s/'frozen_config.json')
json.dump({
    'offline_upper_validation':app['offline_upper_validation'],
    'rule_validation':app['rule_validation'],
    'rule_test':app['rule_test'],
    'macro_f1_gap_validation':app['offline_upper_validation']['macro_f1']-app['rule_validation']['macro_f1'],
    'parity_consistent':app['parity_consistent'],
    'independent_rules_only_match':app['independent_rules_only_match']
},open(d/'fidelity.json','w'),indent=2)
summary['profiles']['application']={
    'profile':'application64','status':'PASS','scope':'USTC4 real-PCAP sealed test n=800',
    'metrics':appm,'selected_features':len(appfc['selected_features']),
    'rule_count':readj(s/'performance.json')['rule_count'],
    'parity_consistent':True,
    'deployment_note':'CESNET-QUIC22 remains the modern QUIC offline main evidence; current PCAP RuleBundle full-chain acceptance uses USTC4 because CESNET source is structured QUIC CSV/PPI rather than raw PCAP in this pipeline.'
}
summary['boundaries']['application']='CESNET W47 is not re-used for RuleBundle fitting/tuning; USTC4 verifies raw-PCAP RuleBundle/DPI deployment fidelity.'

# Behavior
s=SRC/'behavior'; d=OUT/'behavior'; d.mkdir()
for fn in ['frozen_config.json','acceptance.json','performance.json','parity.json','manifest.jsonl',
           'independent_validation_metrics.json','direct_validation_predictions.json','validation_rule_sweep.csv',
           'offline_upper_sweep.csv','train_feature_ranking.csv']:
    if (s/fn).exists(): copy_file(s/fn,d/fn)
copy_tree(s/'bundle',d/'bundle')
ba=readj(s/'acceptance.json')
bf=readj(s/'frozen_config.json')
copy_file(s/'independent_validation_metrics.json',d/'metrics.json')
json.dump({
    'official_ARFF_reference':ba.get('official_ARFF_reference_not_runtime'),
    'raw_pcap_offline_upper_validation':ba['offline_upper_validation'],
    'rule_validation_direct':ba['rule_validation_direct'],
    'independent_rule_validation':ba['independent_rule_validation'],
    'parity_consistent':ba['parity_consistent'],
    'macro_f1_rule_vs_raw_xgb_gap':ba['rule_validation_direct']['macro_f1']-ba['offline_upper_validation']['macro_f1'],
    'interpretation':'Official 15s ARFF random-window result is not equivalent to the stricter real-PCAP file-disjoint runtime acceptance. Runtime RuleBundle remains below formal thresholds.'
},open(d/'fidelity.json','w'),indent=2)
summary['profiles']['behavior']={
    'profile':'behavior15','status':'FAIL',
    'scope':ba['scope'],'metrics':ba['independent_rule_validation'],
    'selected_features':len(bf['selected_features']),
    'rule_count':readj(s/'performance.json')['rule_count'],
    'parity_consistent':ba['parity_consistent'],
    'offline_reference':'ISCXVPN VPN 15s official ARFF ExtraTrees ~99.39% Acc / 99.15% Macro-R / 0.73% maxFPR',
    'deployment_gap':'real-PCAP file-disjoint RuleBundle Macro-R 87.43%, maxFPR 12.10%'
}
summary['boundaries']['behavior']='No sealed test is claimed. Acceptance is file-disjoint real-PCAP validation. The official ARFF result remains offline/reference evidence and is not represented as RuleBundle performance.'

# Tunnel
s=SRC/'tunnel'; d=OUT/'tunnel'; d.mkdir()
for fn in ['frozen_config.json','acceptance.json','performance.json','parity.json','manifest.jsonl',
           'independent_validation_metrics.json','validation_rule_sweep.csv','offline_upper_sweep.csv',
           'train_feature_ranking.csv']:
    if (s/fn).exists(): copy_file(s/fn,d/fn)
copy_tree(s/'bundle',d/'bundle')
ta=readj(s/'acceptance.json'); tf=readj(s/'frozen_config.json')
copy_file(s/'independent_validation_metrics.json',d/'metrics.json')
json.dump({
    'offline_upper_vote_validation':ta['offline_upper_vote_validation'],
    'rule_vote_validation_direct':ta['rule_vote_validation_direct'],
    'independent_rule_vote_validation':ta['independent_rule_vote_validation'],
    'parity_consistent':ta['parity_consistent'],
    'prior_grouped_offline_reference':ta['prior_grouped_offline_reference'],
    'macro_f1_gap_rule_vs_upper':ta['offline_upper_vote_validation']['vote_macro_f1']-ta['rule_vote_validation_direct']['macro_f1']
},open(d/'fidelity.json','w'),indent=2)
summary['profiles']['tunnel']={
    'profile':'tunnel15','status':'PASS','scope':ta['scope'],
    'metrics':ta['independent_rule_vote_validation'],
    'selected_features':len(tf['selected_features']),
    'rule_count':readj(s/'performance.json')['rule_count'],
    'parity_consistent':ta['parity_consistent'],
    'temporal_vote':3,
    'deployment_note':'15s flow windows, non-overlapping 3-window causal vote; identifier-free 64-d candidate family, selected subset persisted in bundle.'
}
summary['boundaries']['tunnel']='PASS is file-disjoint validation, not a newly created sealed holdout. A fresh grouped holdout is still required for final competition-level anonymous-tool acceptance.'

json.dump(summary,open(OUT/'summary.json','w'),indent=2,ensure_ascii=False)

# status.json
json.dump({
    'overall':'PARTIAL_PASS',
    'application':'PASS',
    'behavior':'FAIL',
    'tunnel':'PASS',
    'all_parity_consistent':all(v.get('parity_consistent',False) for v in summary['profiles'].values()),
    'formal_interpretation':'Two of three deployed RuleBundle profiles meet formal recognition thresholds under their stated scopes. Behavior15 remains the only deployment-level blocker.'
},open(OUT/'status.json','w'),indent=2)

# summary.md
def pct(v): return f"{100*float(v):.2f}%"
rows=[]
for k,v in summary['profiles'].items():
    m=v['metrics']
    rows.append((k,v['status'],v['scope'],pct(m['accuracy']),pct(m['macro_precision']),pct(m['macro_recall']),pct(m['macro_f1']),pct(m['max_class_fpr']),v['selected_features'],v['rule_count']))
md=[
'# 三Profile RuleBundle / DPI 最终验收（2026-09-19）','',
'| Profile | Status | Scope | Acc | P | R | F1 | MaxFPR | Features | Rules |',
'|---|---|---|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    md.append('| '+' | '.join(map(str,r))+' |')
md += [
'',
'## 结论',
'',
'- Application64：真实PCAP→RuleBundle→rules-only独立进程→Test全链通过；Test Macro-R 99.75%，MaxFPR 0.17%。',
'- Behavior15：提取/规则/独立进程 parity 一致，但严格 file-disjoint 原始PCAP Validation Macro-R 87.43%、MaxFPR 12.10%，正式失败。官方ARFF 99%+只能作为离线参考，不能替代部署成绩。',
'- Tunnel15：15s窗口+3窗投票，独立RuleBundle Validation Macro-R 98.51%、MaxFPR 2.70%，正式通过；但仍是file-disjoint validation而非fresh sealed holdout。',
'',
'## 工程边界',
'',
'- CESNET-QUIC22主实验是结构化QUIC CSV/PPI数据，当前PCAP RuleBundle链不能假装直接重放CESNET W47；Application部署验收因此使用USTC真实PCAP。',
'- Behavior是当前唯一需要继续修的Profile；不允许通过反复使用sealed test调阈值。',
'- Tunnel还需补一个fresh grouped holdout，尤其覆盖Skype file-transfer/bulk encrypted traffic。',
]
(OUT/'summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8')

# reproduce.sh
repro='''#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Huawei
export PYTHONPATH=/workspace/Huawei
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/usr/bin/python3

$PY -m pytest -q --basetemp=output/pytest_profile_runtime_20260919 tests/regression/test_profile_runtime_20260919.py
$PY tests/test_all.py

$PY scripts/run_application_rule_acceptance_20260919.py

$PY scripts/build_profile_manifests_20260919.py
$PY -m src.pipeline --mode mine --manifest output/final_profile_runtime_20260919/behavior/manifest.jsonl --task behavior --profile behavior15 --max-read-packets 50000 --output output/final_profile_runtime_20260919/behavior/mine_flow
$PY scripts/run_behavior_rule_acceptance_20260919.py

$PY -m src.pipeline --mode mine --manifest output/final_profile_runtime_20260919/tunnel/manifest.jsonl --task tool --profile tunnel15 --max-read-packets 50000 --output output/final_profile_runtime_20260919/tunnel/mine
$PY scripts/run_tunnel_rule_acceptance_20260919.py
'''
(OUT/'reproduce.sh').write_text(repro,encoding='utf-8')
os.chmod(OUT/'reproduce.sh',0o755)

# hash all files except manifest itself
files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
lines=[]
for p in files:
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    lines.append(f'{h}  {p.relative_to(OUT)}')
(OUT/'SHA256SUMS').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2,ensure_ascii=False))
print('files',len(files),'archive',OUT)
