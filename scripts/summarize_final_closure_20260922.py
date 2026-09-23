# -*- coding: utf-8 -*-
"""Assemble the final-experiment-closure summary artifacts.

Reads each scenario closure directory under
output/final_experiment_closure_20260922/ and produces summary.md,
summary_table.md, summary_table.csv, status.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path('/workspace/Huawei')
OUT = ROOT / 'output/final_experiment_closure_20260922'


def pct(x):
    return None if x is None else round(100.0 * float(x), 2)


def collect() -> list:
    rows = []

    # 1. behavior capture-disjoint
    d = OUT / 'behavior_capture_disjoint'
    so = json.loads((d / 'search_outcome.json').read_text())
    loco = json.loads((d / 'loco_diagnostic.json').read_text())
    raw_loco = next((v['loco_macro_capture_acc'] for k, v in
                     loco['results'].items() if '/raw/' in k), None)
    rows.append({
        'scenario': 'B behavior raw-PCAP capture-disjoint',
        'track': 'shared-shape (multi-flow aggregate windows)',
        'status': 'FINAL_FAIL',
        'test_opened': False,
        'acc': pct(so['validation_best']['accuracy']),
        'macro_p': pct(so['validation_best']['macro_precision']),
        'macro_r': pct(so['validation_best']['macro_recall']),
        'max_fpr': pct(so['validation_best']['max_class_fpr']),
        'evidence': ('gate failed over 216 configs (obs topK x capz/caprank x '
                     'w30-60); capture-relative normalization WORSE than raw; '
                     f'LOCO over 17 non-test captures raw={pct(raw_loco)}%; '
                     'single audio validation capture facebook_audio2 is '
                     'chat-shaped (2.96 pkt/s vs 91-135 for other audio '
                     'captures) -> label-composition defect; see root_cause.md'),
    })

    # 2. mixed16 endpoint aggregation
    d = OUT / 'mixed16_endpoint_agg'
    if (d / 'frozen_config.json').exists():
        fz = json.loads((d / 'frozen_config.json').read_text())
        row = {
            'scenario': 'A2+ mixed16 fresh split v2 (identifier-assisted + '
                        'capture-scoped endpoint aggregation)',
            'track': 'identifier-assisted (IP feed + label-free within-file '
                     '//24 endpoint pooling)',
            'status': 'FROZEN->TEST_OPENED' if (d / 'test_metrics.json').exists()
                      else 'FROZEN (test pending)',
            'test_opened': (d / 'test_metrics.json').exists(),
            'acc': pct(fz['validation_best_single']['accuracy']),
            'macro_p': pct(fz['validation_best_single']['macro_precision']),
            'macro_r': pct(fz['validation_best_single']['macro_recall']),
            'max_fpr': pct(fz['validation_best_single']['max_per_class_fpr']),
            'evidence': 'validation gate passed (single + 3-seed stability)',
        }
        if (d / 'test_metrics.json').exists():
            tm = json.loads((d / 'test_metrics.json').read_text())
            row['test_acc'] = pct(tm['metrics']['accuracy'])
            row['test_macro_p'] = pct(tm['metrics']['macro_precision'])
            row['test_macro_r'] = pct(tm['metrics']['macro_recall'])
            row['test_max_fpr'] = pct(tm['metrics']['max_per_class_fpr'])
            row['test_pass'] = tm['all_pass']
            row['status'] = 'PASS' if tm['all_pass'] else 'FAIL (test opened once)'
        rows.append(row)
    else:
        so = json.loads((d / 'search_outcome.json').read_text())
        rows.append({
            'scenario': 'A2+ mixed16 endpoint aggregation',
            'track': 'identifier-assisted', 'status': 'FINAL_FAIL',
            'test_opened': False,
            'acc': pct(so['validation_best_single']['accuracy']),
            'macro_p': pct(so['validation_best_single']['macro_precision']),
            'macro_r': pct(so['validation_best_single']['macro_recall']),
            'max_fpr': pct(so['validation_best_single']['max_per_class_fpr']),
            'evidence': (f"single-fit passed all four ({pct(so['validation_best_single']['macro_recall'])}% R) "
                         f"but 3-seed stability Macro-R mean {pct(so['validation_stability']['macro_recall_mean'])}% "
                         f"(min {pct(so['validation_stability']['macro_recall_min'])}%) < 98; "
                         'sealed test stays closed permanently'),
        })

    # 3. mirage group-split
    d = OUT / 'mirage_groupsplit'
    src = ROOT / 'output/all_scenarios_metric_push_20260922/mirage_activity'
    if (d / 'frozen_config.json').exists():
        fz = json.loads((d / 'frozen_config.json').read_text())
        row = {
            'scenario': 'D3 MIRAGE fresh grouped (flow-length-grouped hybrid)',
            'track': 'shared-shape (activity-capture observation)',
            'status': 'FROZEN->TEST_OPENED' if (d / 'test_metrics.json').exists()
                      else 'FROZEN (test pending)',
            'test_opened': (d / 'test_metrics.json').exists(),
            'acc': pct(fz['validation_best']['accuracy']),
            'macro_p': pct(fz['validation_best']['macro_precision']),
            'macro_r': pct(fz['validation_best']['macro_recall']),
            'max_fpr': pct(fz['validation_best']['max_class_fpr']),
            'evidence': 'validation gate passed',
        }
        if (d / 'test_metrics.json').exists():
            tm = json.loads((d / 'test_metrics.json').read_text())
            row['test_acc'] = pct(tm['metrics']['accuracy'])
            row['test_macro_p'] = pct(tm['metrics']['macro_precision'])
            row['test_macro_r'] = pct(tm['metrics']['macro_recall'])
            row['test_max_fpr'] = pct(tm['metrics']['max_class_fpr'])
            row['test_pass'] = tm['all_pass']
            row['status'] = 'PASS' if tm['all_pass'] else 'FAIL (test opened once)'
        rows.append(row)
    else:
        so = json.loads((d / 'search_outcome.json').read_text())
        rows.append({
            'scenario': 'D3 MIRAGE fresh grouped (flow-length-grouped hybrid)',
            'track': 'shared-shape', 'status': 'FINAL_FAIL',
            'test_opened': False,
            'acc': pct(so['validation_best']['accuracy']),
            'macro_p': pct(so['validation_best']['macro_precision']),
            'macro_r': pct(so['validation_best']['macro_recall']),
            'max_fpr': pct(so['validation_best']['max_class_fpr']),
            'evidence': so.get('reason', ''),
        })

    # 4. crossplatform alignment
    d = OUT / 'crossplatform_align'
    fr = json.loads((d / 'final_results.json').read_text())
    full = [a for a in fr['arms'] if a.get('metrics')
            and a['arm'] in ('session_capnorm', 'coral_on_capz')]
    best = max(full, key=lambda a: a['metrics']['accuracy'])
    met = best['metrics']
    ipf = {f"{a['train_platform']}->{a['test_platform']}": a
           for a in fr['arms'] if a['arm'] == 'ipfeed_transfer'}
    cap_agg = max((a for a in fr['arms']
                   if a['arm'] == 'capture_level_aggregation'
                   and a.get('metrics')), key=lambda a: a['metrics']['accuracy'])
    rows.append({
        'scenario': 'E CrossPlatform Android<->iOS (final alignment round)',
        'track': 'shape / hs_visible / capnorm / CORAL / capture-agg / IP feed',
        'status': 'FINAL_FAIL (structural: 1 capture per (app,platform))',
        'test_opened': False,
        'acc': pct(met['accuracy']),
        'macro_p': pct(met['macro_precision']),
        'macro_r': pct(met['macro_recall']),
        'max_fpr': pct(met['max_class_fpr']),
        'evidence': (f"best session-level arm {best['arm']}/{best.get('pool','')}"
                     f" {best.get('train_platform')}->{best.get('test_platform')}"
                     f" {best.get('model')} acc={pct(met['accuracy'])}%; "
                     f"capture-agg best {pct(cap_agg['metrics']['accuracy'])}% "
                     f"(n=9 captures); IP-feed coverage "
                     f"{pct(ipf['android->ios']['feed_coverage_slash24'])}%//24 "
                     'with only '
                     f"{pct(ipf['android->ios']['accuracy_on_covered_sessions'])}"
                     '% precision on covered sessions (shared multi-app '
                     'endpoints); capz/caprank HURT transfer'),
    })

    # 5. manual review readiness
    d = OUT / 'manual_review_readiness'
    st = json.loads((d / 'status.json').read_text())
    rows.append({
        'scenario': 'F 人工修正后可用率 >= 90%',
        'track': 'external human review',
        'status': 'EXTERNAL_REVIEW (machine side complete)',
        'test_opened': None,
        'acc': None, 'macro_p': None, 'macro_r': None, 'max_fpr': None,
        'evidence': (f"{st['n_review_rows']} review rows prepared over 3 "
                     'machine-validated packages; no human records exist; '
                     'compute_usable_rate.py refuses until verdicts exist'),
    })
    return rows


def main() -> None:
    rows = collect()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'summary_table.csv', index=False)

    cols = ['scenario', 'status', 'test_opened', 'acc', 'macro_p', 'macro_r',
            'max_fpr', 'evidence']
    header = ('| 场景 | 状态 | Test是否打开 | Acc% | Macro-P% | Macro-R% | '
              'MaxFPR% | 证据 |\n|---|---|---|---:|---:|---:|---:|---|\n')
    lines = []
    for r in rows:
        def f(k):
            v = r.get(k)
            return '-' if v is None else v
        lines.append(
            f"| {r['scenario']} | {r['status']} | {r['test_opened']} | "
            f"{f('acc')} | {f('macro_p')} | {f('macro_r')} | {f('max_fpr')} | "
            f"{r['evidence'][:160]} |")
        for tk in ('test_acc', 'test_macro_p', 'test_macro_r', 'test_max_fpr'):
            if r.get(tk) is not None:
                lines.append(
                    f"| ↳ 一次性 Test | {'PASS' if r.get('test_pass') else 'FAIL'}"
                    f" | opened once | {r.get('test_acc')} | "
                    f"{r.get('test_macro_p')} | {r.get('test_macro_r')} | "
                    f"{r.get('test_max_fpr')} | one-shot, frozen config |")
    (OUT / 'summary_table.md').write_text(header + '\n'.join(lines) + '\n')

    md = ('# 最终实验收尾（final_experiment_closure_20260922）\n\n'
          '目标：把全部未闭合实验收账为 已完成 / FINAL_FAIL / 外部依赖，'
          '不保留"继续试试"式待办。评价口径与第 30/31 节一致'
          '（Acc≥95 / Macro-P≥95 / Macro-R≥98 / MaxFPR≤5%）。\n\n'
          '## 总表\n\n' + header + '\n'.join(lines) + '\n\n'
          '## 各场景目录\n\n'
          '- behavior_capture_disjoint/ — FINAL_FAIL + root_cause.md + LOCO 证据\n'
          '- mixed16_endpoint_agg/ — 冻结/一次性 Test 产物\n'
          '- mirage_groupsplit/ — 冻结或 FINAL_FAIL 产物\n'
          '- crossplatform_align/ — 最终对齐轮全部臂 + 结构性裁决\n'
          '- manual_review_readiness/ — 人工审阅就绪包（EXTERNAL_REVIEW）\n\n'
          '## 纪律声明\n\n'
          '- 所有 sealed Test 开启均发生在对应 validation 四指标门（单次 + '
          '3种子稳定性）通过、frozen_config 写盘之后；未过门的 Test 保持密封。\n'
          '- 不删困难样本、不挑 seed、无 label/路径/文件名泄漏；'
          'identifier-assisted 轨道与泛化口径分开标注。\n')
    (OUT / 'summary.md').write_text(md)

    status = {
        'date': '2026-09-22',
        'scenarios': [{k: v for k, v in r.items() if k != 'evidence'}
                      for r in rows],
        'closure_standard': {'PASS': '四指标按协议闭合（含一次性 Test）',
                             'FINAL_FAIL': '至少一轮针对根因的最终实验且'
                                           '证据表明同类搜索价值低',
                             'EXTERNAL_REVIEW': '机器侧到边界，仅剩人工签字'},
    }
    (OUT / 'status.json').write_text(
        json.dumps(status, indent=2, ensure_ascii=False))
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
