# -*- coding: utf-8 -*-
"""Build a normalized engineering performance table from frozen evidence."""
from __future__ import annotations

import csv, json, hashlib
from pathlib import Path

ROOT=Path('/workspace/Huawei')
BASE=ROOT/'output/protocol_and_system_acceptance_20260920'
OUT=BASE/'performance'

def J(p): return json.loads(Path(p).read_text())
def mib_from_kb(v): return float(v)/1024.0

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    app=J(ROOT/'output/three_profile_acceptance_20260919/application/performance.json')
    app_pcap=J(ROOT/'output/three_profile_acceptance_20260919/application/independent_pcap_performance.json')
    beh=J(ROOT/'output/three_profile_acceptance_20260919/behavior/performance.json')
    tun=J(ROOT/'output/three_profile_acceptance_20260919/tunnel/performance.json')
    quic=J(BASE/'quic/acceptance.json')['real']
    tcp=J(BASE/'tcp_session/acceptance.json')['real']
    barff=J(BASE/'behavior_official_arff/performance.json')
    bw=J(BASE/'behavior_window_relaxed/performance.json')
    tg=J(BASE/'tunnel_fresh/performance.json')

    rows=[
      {
       'name':'QUIC protocol processing','scope':'30 real VisQUIC PCAPs; parse+UDP session+QUIC state detection',
       'context':'raw UDP/QUIC PCAP','features':'N/A','rules':'N/A','observations':quic['pcapreader_sessions'],
       'packets':quic['packets_scanned_direct'],'wall_sec':quic['wall_sec'],'end_to_end_throughput':quic['packets_per_sec'],
       'throughput_unit':'packets/s','mib_per_sec':quic['mib_per_sec'],'matcher_ms_mean':'','matcher_ms_p95':'',
       'peak_rss_mib':quic['peak_rss_mib'],'measurement_note':'single-process direct state scan plus PCAPReader replay; input fully below 50k cap'
      },
      {
       'name':'TCP/session reconstruction','scope':'4 real USTC/CSTNET PCAPs; parse+session reconstruction',
       'context':'TCP 5-tuple/state machine','features':'N/A','rules':'N/A','observations':tcp['sessions'],
       'packets':tcp['packets'],'wall_sec':tcp['wall_sec'],'end_to_end_throughput':tcp['packets_per_sec'],
       'throughput_unit':'packets/s','mib_per_sec':tcp['mib_per_sec'],'matcher_ms_mean':'','matcher_ms_p95':'',
       'peak_rss_mib':tcp['peak_rss_mib'],'measurement_note':'100k packet/file cap; MiB/s uses source-file footprint, so packet/s is the authoritative capped-read throughput'
      },
      {
       'name':'Application64 RuleBundle','scope':'USTC real-PCAP independent rules-only replay',
       'context':'64 packets/flow','features':64,'rules':app['rule_count'],'observations':app_pcap['all_runtime_records'],
       'packets':'','wall_sec':app_pcap['wall_sec'],'end_to_end_throughput':app_pcap['records_per_sec'],
       'throughput_unit':'records/s','mib_per_sec':'','matcher_ms_mean':app['matcher_ms_mean'],'matcher_ms_p95':app['matcher_ms_p95'],
       'peak_rss_mib':mib_from_kb(app_pcap['peak_rss_kb']),'measurement_note':'end-to-end raw PCAP extraction+RuleBundle; matcher latency measured over frozen feature rows'
      },
      {
       'name':'Behavior15 strict RuleBundle','scope':'ISCXVPN file-disjoint PCAP validation',
       'context':'15s flow windows','features':48,'rules':beh['rule_count'],'observations':beh['windows'],
       'packets':'','wall_sec':beh['independent_validation_wall_sec'],'end_to_end_throughput':beh['windows_per_sec'],
       'throughput_unit':'windows/s','mib_per_sec':'','matcher_ms_mean':beh['matcher_ms_mean'],'matcher_ms_p95':beh['matcher_ms_p95'],
       'peak_rss_mib':mib_from_kb(beh['peak_rss_kb']),'measurement_note':'independent PCAP rules-only validation; strict generalization result is FAIL'
      },
      {
       'name':'Behavior15 official ARFF RuleBundle','scope':'official 15s feature-vector test; rules-only subprocess',
       'context':'15s official feature vectors','features':barff['feature_count'],'rules':barff['rule_count'],'observations':barff['vectors'],
       'packets':'','wall_sec':barff['wall_sec'],'end_to_end_throughput':barff['vectors_per_sec'],
       'throughput_unit':'vectors/s','mib_per_sec':'','matcher_ms_mean':'','matcher_ms_p95':'',
       'peak_rss_mib':barff['peak_rss_mib'],'measurement_note':'feature-vector functional benchmark; excludes raw-PCAP parsing and is not cross-capture evidence'
      },
      {
       'name':'Behavior15 window-relaxed PCAP','scope':'window-stratified competition-oriented PCAP test',
       'context':'15s windows; same session may cross split','features':bw['feature_count'],'rules':bw['rule_count'],'observations':bw['test_windows'],
       'packets':'','wall_sec':bw['independent_wall_sec'],'end_to_end_throughput':bw['windows_per_sec'],
       'throughput_unit':'selected-test-windows/s','mib_per_sec':'','matcher_ms_mean':bw['matcher_ms_mean'],'matcher_ms_p95':bw['matcher_ms_p95'],
       'peak_rss_mib':bw['peak_rss_mib'],'measurement_note':'independent replay parses all source captures then filters frozen test window IDs; throughput is conservative for selected test outputs'
      },
      {
       'name':'Tunnel15 RuleBundle','scope':'original file-disjoint PCAP validation + 3-window vote',
       'context':'15s x 3 causal vote','features':16,'rules':tun['rule_count'],'observations':tun['voted_observations'],
       'packets':'','wall_sec':tun['independent_validation_wall_sec'],'end_to_end_throughput':tun['voted_per_sec'],
       'throughput_unit':'voted obs/s','mib_per_sec':'','matcher_ms_mean':tun['matcher_ms_mean_raw_window'],'matcher_ms_p95':tun['matcher_ms_p95_raw_window'],
       'peak_rss_mib':mib_from_kb(tun['peak_rss_kb']),'measurement_note':'independent PCAP rules-only validation; original random file split'
      },
      {
       'name':'Tunnel15 hard-pair grouped test','scope':'post-exploration matched-pair frozen test + 3-window vote',
       'context':'15s x 3 causal vote','features':tg['feature_count'],'rules':tg['rule_count'],'observations':tg['voted_observations'],
       'packets':'','wall_sec':tg['independent_wall_sec'],'end_to_end_throughput':tg['observations_per_sec'],
       'throughput_unit':'voted obs/s','mib_per_sec':'','matcher_ms_mean':tg['matcher_ms_mean_raw_window'],'matcher_ms_p95':tg['matcher_ms_p95_raw_window'],
       'peak_rss_mib':tg['peak_rss_mib'],'measurement_note':'hard pairs deliberately included; input disk footprint is capped-read and not used for authoritative MiB/s'
      },
    ]
    fields=list(rows[0].keys())
    with open(OUT/'engineering_table.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (OUT/'engineering_table.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    md=['# 统一工程性能表（2026-09-20）','',
        '口径：end-to-end wall time 为独立运行/协议runner从输入处理到预测或会话产出的wall time；matcher latency单列；RSS统一MiB。不同任务的Observation Unit不同，因此吞吐单位必须保留，不横向伪比较。','',
        '| Chain | Context | Feat | Rules | Obs | E2E throughput | Matcher mean/p95 ms | Peak RSS MiB | Note |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        feat=r['features'];rules=r['rules'];obs=r['observations'];thr=f"{float(r['end_to_end_throughput']):.2f} {r['throughput_unit']}"
        mm='—' if r['matcher_ms_mean']=='' else f"{float(r['matcher_ms_mean']):.4f}/{float(r['matcher_ms_p95']):.4f}"
        md.append(f"| {r['name']} | {r['context']} | {feat} | {rules} | {obs} | {thr} | {mm} | {float(r['peak_rss_mib']):.1f} | {r['measurement_note']} |")
    (OUT/'engineering_table.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))

if __name__=='__main__':main()
