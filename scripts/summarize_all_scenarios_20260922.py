# -*- coding: utf-8 -*-
"""Aggregate the 2026-09-22 all-scenarios metric push into one summary table.

Reads each scenario's outcome artifacts under
output/all_scenarios_metric_push_20260922/ and writes summary_table.md + .csv.
Only reads; never modifies scenario artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path('/workspace/Huawei')
BASE = ROOT / 'output/all_scenarios_metric_push_20260922'


def pct(x):
    return None if x is None else round(100 * float(x), 2)


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def rows() -> list[dict]:
    out = []

    def add(scenario, granularity, m, passed, blindness, notes, unit='window'):
        out.append({
            'Scenario': scenario,
            'Eval granularity': granularity,
            'Unit': unit,
            'Acc%': pct(m.get('accuracy')),
            'MacroP%': pct(m.get('macro_precision')),
            'MacroR%': pct(m.get('macro_recall')),
            'MaxFPR%': pct(m.get('max_per_class_fpr', m.get('max_class_fpr'))),
            'PASS/FAIL': 'PASS' if passed else 'FAIL',
            'Blindness': blindness,
            'Notes': notes,
        })

    # A. mixed16 rich (handshake-visible, competition-oriented)
    d = BASE / 'mixed16_rich'
    t = load_json(d / 'test_metrics.json')
    s = load_json(d / 'search_outcome.json')
    f = load_json(d / 'frozen_config.json')
    if t:
        add('A mixed16 rich (handshake-visible, competition-oriented)',
            'fresh sealed test (held files)', t['metrics'], t['all_pass'],
            'fresh split v2, test opened once',
            'ports/TLS/QUIC/protocol/TCP visible fields + seq197',
            unit='session')
    else:
        best = (s or {}).get('validation_best_single', {})
        add('A mixed16 rich (handshake-visible, competition-oriented)',
            'validation (gate not passed; test sealed)', best,
            False, 'test never opened',
            'best validation config; '
            + ((s or {}).get('reason') or ''), unit='session')
    # A2. mixed16 ipfeed (identifier-assisted)
    d = BASE / 'mixed16_ipfeed'
    t = load_json(d / 'test_metrics.json')
    s = load_json(d / 'search_outcome.json')
    if t:
        add('A2 mixed16 IP-feed (IDENTIFIER-ASSISTED)', 'fresh sealed test',
            t['metrics'], t['all_pass'],
            'fresh split v2, test opened once',
            'remote server IP (u32, /24) + hs pool; IP->class learned on train '
            'files only', unit='session')
    else:
        best = (s or {}).get('validation_best_single', {})
        add('A2 mixed16 IP-feed (IDENTIFIER-ASSISTED)',
            'validation (gate not passed; test sealed)', best, False,
            'test never opened', (s or {}).get('reason', ''), unit='session')

    # B. behavior raw-PCAP aggregate windows
    for g, label in (('win_stratified', 'window-stratified'),
                     ('session_disjoint', 'session-disjoint'),
                     ('capture_disjoint', 'capture-disjoint')):
        d = BASE / 'behavior_rawpcap'
        t = load_json(d / f'test_metrics_{g}.json')
        s = load_json(d / f'search_outcome_{g}.json')
        if t:
            add(f'B behavior raw-PCAP multi-flow ({label})', 'fresh test',
                t['metrics'], t['all_pass'],
                'fresh pre-registered split (seed 20260922) over previously '
                'explored captures',
                f"win={t['win_cfg']} (W={int(t.get('window_seconds', 0))}s "
                f"multi-flow aggregate)", unit='aggregate window')
        else:
            best = (s or {}).get('validation_best', {})
            add(f'B behavior raw-PCAP multi-flow ({label})',
                'validation (gate not passed)', best, False,
                'test never opened',
                (s or {}).get('reason', ''), unit='aggregate window')

    # C. Tor hard-pair engineering stress
    d = BASE / 'tor_hardpair'
    r = load_json(d / 'engineering_stress_results.json')
    if r:
        for e in r['results']:
            m = e['test']
            add(f"C Tor hard-pair ({e['unit']}, {e['variant']})",
                'post-exploration frozen hard split', m, m['target_pass'],
                'post-exploration engineering stress (all pairs previously '
                'exposed)',
                'baseline 20260920: Acc 91.39/P 94.83/R 83.02/FPR 33.96',
                unit=e['unit'])

    # D. MIRAGE activity fresh v2
    d = BASE / 'mirage_activity'
    t = load_json(d / 'test_metrics.json')
    s = load_json(d / 'search_outcome.json')
    h = load_json(d / 'hybrid_search_outcome.json')
    if h and not t:
        best = h.get('validation_best', {})
        add('D2 MIRAGE activity (hybrid representation)',
            'validation (gate not passed; test sealed)', best, False,
            'test never opened',
            'prefix/fullflow aggregates recomputed for 5 apps + merged-stream; '
            'same pre-registered split as D', unit='capture')
    if t:
        add('D MIRAGE activity (merged-stream + cross-flow)',
            'fresh file-disjoint test', t['metrics'], t['all_pass'],
            'fresh pre-registered split (seed 20260922)', 'capture-level '
            'observation, 5 apps x 3 activities', unit='capture')
    else:
        best = (s or {}).get('validation_best', {})
        add('D MIRAGE activity (merged-stream + cross-flow)',
            'validation (gate not passed; test sealed)', best, False,
            'test never opened', (s or {}).get('reason', ''), unit='capture')

    # E. CrossPlatform Android<->iOS
    d = BASE / 'crossplatform'
    r = load_json(d / 'crossplatform_results.json')
    if r:
        for e in r['evaluations']:
            setting = (e.get('setting')
                       or f"{e['train_platform']}->{e['test_platform']}")
            m = e['metrics']
            add(f"E CrossPlatform china ({e['pool']}, {setting})",
                'platform transfer (capture-disjoint)' if e['capture_disjoint']
                else 'pooled control (captures cross splits)',
                m, m['four_pass'],
                'first formal mapping round; captures never used for tuning '
                'before this scenario',
                '9 common apps; earlier sealed MIRAGE-style claim not reused',
                unit='session')
    return out


def main() -> None:
    rs = rows()
    df = pd.DataFrame(rs)
    df.to_csv(BASE / 'summary_table.csv', index=False)
    md = ['# 全场景指标冲刺总表（2026-09-22）', '',
          '正式阈值: Acc>=95% / Macro-P>=95% / Macro-R>=98% / Max per-class '
          'FPR<=5%。PASS/FAIL 按四项同时满足判定。', '',
          df.to_string(index=False), '']
    (BASE / 'summary_table.md').write_text('\n'.join(md), encoding='utf-8')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
