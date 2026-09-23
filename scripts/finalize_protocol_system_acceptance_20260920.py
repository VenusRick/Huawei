# -*- coding: utf-8 -*-
from pathlib import Path
import json, hashlib, os

ROOT=Path('/workspace/Huawei')
OUT=ROOT/'output/protocol_and_system_acceptance_20260920'
def J(p):return json.loads(Path(p).read_text())

quic=J(OUT/'quic/acceptance.json')
tcp=J(OUT/'tcp_session/acceptance.json')
ooo=J(OUT/'tcp_session/real_ooo_scan.json')
beh_strict=J(ROOT/'output/three_profile_acceptance_20260919/behavior/metrics.json')
beh_session=J(OUT/'behavior_relaxed/acceptance.json')
beh_window=J(OUT/'behavior_window_relaxed/acceptance.json')
beh_arff=J(OUT/'behavior_official_arff/acceptance.json')
tun_strict=J(ROOT/'output/three_profile_acceptance_20260919/tunnel/metrics.json')
tun_hard=J(OUT/'tunnel_fresh/acceptance.json')
perf=J(OUT/'performance/engineering_table.json')

summary={
 'date':'2026-09-20',
 'formal_thresholds':{'accuracy':.95,'precision':.95,'recall':.98,'max_fpr':.05},
 'quic':{
   'status':'PASS','protocol_acceptance':quic['acceptance_pass'],
   'real_visquic':{k:v for k,v in quic['real'].items() if k!='files'},
   'fixture':quic['fixture'],
   'classification_reference':'CESNET-QUIC22 W47 offline application classification remains 98.375% Acc/Recall; separate from raw-PCAP parser acceptance.'
 },
 'tcp_session':{
   'status':'PASS','acceptance':tcp['acceptance_pass'],'semantic_fixture':tcp['semantic_fixture'],
   'real_summary':{k:v for k,v in tcp['real'].items() if k!='files'},
   'real_ooo_scan':ooo,
 },
 'behavior':{
   'strict_raw_pcap':{'status':'FAIL','scope':'file-disjoint raw-PCAP RuleBundle','metrics':beh_strict},
   'session_disjoint_raw_pcap':{'status':'FAIL','scope':beh_session['scope'],'validation':beh_session['validation'],'test':beh_session['test_independent']},
   'window_stratified_raw_pcap':{'status':'FAIL','scope':beh_window['scope'],'warning':beh_window['warning'],'validation':beh_window['validation'],'test':beh_window['test_independent']},
   'official_arff_rulebundle':{'status':'PASS','scope':beh_arff['scope'],'warning':beh_arff['warning'],'validation':beh_arff['validation'],'test':beh_arff['test']},
   'competition_use':'Use official 15s feature-vector RuleBundle as functional benchmark only. Raw-PCAP generalization remains below formal recall target.'
 },
 'tunnel':{
   'original_file_disjoint':{'status':'PASS','scope':'file-disjoint validation + 3-window vote','metrics':tun_strict},
   'hard_pair_grouped_test':{'status':'FAIL','scope':tun_hard['scope'],'warning':tun_hard['warning'],'validation':tun_hard['validation'],'test':tun_hard['test_independent'],'per_pair':tun_hard['per_pair']},
   'competition_use':'Core Tor detection passes ordinary file-disjoint validation, but hard-pair stress test exposes Skype-transfer/Vimeo weakness.'
 },
 'performance_table':'performance/engineering_table.csv',
 'tests':{'targeted_protocol':'2 passed','regression':'128 passed, 2 warnings','test_all':'6/6 passed'},
 'overall':'FUNCTIONALLY_COMPLETE_WITH_GENERALIZATION_GAPS',
 'remaining_gaps':[
   'Behavior raw-PCAP cross-session/cross-capture generalization remains below Recall 98%.',
   'Tor hard-pair grouped stress test fails because Skype file-transfer and Vimeo Tor recall are low.',
   'No release package is produced in this phase by user request.'
 ]
}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))

def pct(x):return f'{100*float(x):.2f}%'
md=['# 协议处理与系统验收补充（2026-09-20）','',
'## 总结','',
'| 项目 | 结果 | 结论 |',
'|---|---|---|',
f"| QUIC真实PCAP处理 | 30 VisQUIC PCAP；12,312 UDP/QUIC-like包；68 Initial；12,047 short-header；parse error=0 | PASS |",
f"| TCP会话重建 | 325,000包；111,632会话；语义fixture 9/9；真实重传/乱序均有证据 | PASS |",
f"| Behavior官方15s RuleBundle | Test Acc {pct(beh_arff['test']['accuracy'])} / P {pct(beh_arff['test']['macro_precision'])} / R {pct(beh_arff['test']['macro_recall'])} / FPR {pct(beh_arff['test']['max_class_fpr'])} | PASS；仅feature-vector功能基准 |",
f"| Behavior raw-PCAP strict | R {pct(beh_strict['macro_recall'])} / FPR {pct(beh_strict['max_class_fpr'])} | FAIL |",
f"| Behavior raw-PCAP window-relaxed | Test R {pct(beh_window['test_independent']['macro_recall'])} / FPR {pct(beh_window['test_independent']['max_class_fpr'])} | FAIL |",
f"| Tunnel原file-disjoint | R {pct(tun_strict['macro_recall'])} / FPR {pct(tun_strict['max_class_fpr'])} | PASS |",
f"| Tunnel hard-pair grouped | Test R {pct(tun_hard['test_independent']['macro_recall'])} / FPR {pct(tun_hard['test_independent']['max_class_fpr'])} | FAIL；Skype-transfer主导 |",
'',
'## 工程性能表','',
(OUT/'performance/engineering_table.md').read_text(encoding='utf-8'),
'',
'## 口径边界','',
'- Behavior官方15s ARFF RuleBundle用于证明官方窗口特征可以被纯规则引擎高精度执行；不等同于raw-PCAP跨capture泛化。',
'- Behavior raw-PCAP strict/session-disjoint/window-stratified结果均保留，不能用宽松口径覆盖严格失败。',
'- Tor hard-pair测试故意包含SFTP、Skype-transfer、Skype-audio、Vimeo、Facebook-chat；业务对在前期探索中已暴露，因此是post-exploration engineering stress test，不称为 pristine blind holdout。',
'- QUIC协议能力由真实VisQUIC PCAP证明；QUIC应用分类能力仍由CESNET-QUIC22跨周结果证明，两者分别核账。',
]
(OUT/'summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
json.dump({'overall':summary['overall'],'quic':'PASS','tcp_session':'PASS',
           'behavior_official_functional':'PASS','behavior_raw_runtime':'FAIL',
           'tunnel_core':'PASS','tunnel_hard_pair':'FAIL','release_package':'NOT_REQUESTED'},
          open(OUT/'status.json','w'),indent=2)

repro='''#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Huawei
export PYTHONPATH=/workspace/Huawei
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/usr/bin/python3
$PY scripts/run_protocol_acceptance_20260920.py
$PY scripts/run_behavior_relaxed_acceptance_20260920.py
$PY scripts/run_behavior_window_relaxed_acceptance_20260920.py
$PY scripts/run_behavior_official_arff_rule_acceptance_20260920.py
$PY scripts/run_tunnel_grouped_test_20260920.py
$PY scripts/build_engineering_performance_20260920.py
BT=output/pytest_protocol_system_repro_20260920
rm -rf "$BT"; mkdir -p "$BT"
$PY -m pytest -q --basetemp="$BT" tests/regression/test_protocol_system_acceptance_20260920.py
BT2=output/pytest_regression_protocol_system_repro_20260920
rm -rf "$BT2"; mkdir -p "$BT2"
$PY -m pytest -q --basetemp="$BT2" tests/regression
$PY tests/test_all.py
'''
(OUT/'reproduce.sh').write_text(repro,encoding='utf-8');os.chmod(OUT/'reproduce.sh',0o755)

files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
lines=[]
for p in files:lines.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}')
(OUT/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
print((OUT/'summary.md').read_text())
print('files',len(files))
