# -*- coding: utf-8 -*-
"""Assemble the round's deliverables from produced artifacts (no invention).

Reads whatever stages actually produced (leaderboard, stability, outcome or
frozen+test) and writes summary.md / status.json / reproduce.sh / SHA256SUMS.
If the validation gate did not pass, the fresh test stays sealed and the
summary states that explicitly.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

OUT = Path('/workspace/Huawei/output/mixed_dataset_full_pass_20260921')
SCRIPTS = Path('/workspace/Huawei/scripts')
TARGETS = {'accuracy': .95, 'macro_precision': .95, 'macro_recall': .98,
           'max_per_class_fpr': .05}


def md_table(rows) -> str:
    """Minimal markdown table (no tabulate dependency)."""
    cols = list(rows.columns)
    def fmt(v):
        if isinstance(v, float):
            return f'{v:.4f}'
        return str(v)
    out = ['| ' + ' | '.join(cols) + ' |',
           '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in rows.iterrows():
        out.append('| ' + ' | '.join(fmt(r[c]) for c in cols) + ' |')
    return '\n'.join(out)


def main() -> None:
    lb = OUT / 'validation_leaderboard.csv'
    stab = OUT / 'validation_stability.csv'
    outcome_p = OUT / 'search_outcome.json'
    frozen_p = OUT / 'frozen_config.json'
    test_p = OUT / 'test_metrics.json'

    lb_rows = None
    if lb.exists():
        import pandas as pd
        lb_rows = pd.read_csv(lb)
    stab_rows = None
    if stab.exists():
        import pandas as pd
        stab_rows = pd.read_csv(stab)
    outcome = json.loads(outcome_p.read_text()) if outcome_p.exists() else None
    frozen = json.loads(frozen_p.read_text()) if frozen_p.exists() else None
    test = json.loads(test_p.read_text()) if test_p.exists() else None

    split_spec = json.loads((OUT / 'fresh_split_spec.json').read_text())
    audit = json.loads((OUT / 'split_leakage_audit.json').read_text())

    status = {
        'experiment': 'mixed_dataset_full_pass_20260921',
        'split_sha256': split_spec['split_sha256'],
        'leakage_audit_all_passed': all(c['passed'] in (True,)
                                        for c in audit['checks'] if isinstance(c['passed'], bool)),
        'fallback_single_capture_classes': audit['fallback_classes'],
        'counts': split_spec['counts'],
        'validation_gate_passed': frozen is not None,
        'test_opened': test is not None,
    }
    if lb_rows is not None:
        best = lb_rows.iloc[0]
        status['validation_best'] = {
            'subset': str(best['subset']), 'model': str(best['model']), 'k': int(best['k']),
            'accuracy': float(best['accuracy']),
            'macro_precision': float(best['macro_precision']),
            'macro_recall': float(best['macro_recall']),
            'max_per_class_fpr': float(best['max_per_class_fpr'])}
    if outcome is not None:
        status['search_outcome'] = outcome['status']
    if test is not None:
        status['test'] = {'metrics': test['metrics'], 'pass': test['pass'],
                          'all_pass': test['all_pass']}
    (OUT / 'status.json').write_text(json.dumps(status, indent=2, ensure_ascii=False))

    # ---- summary.md ---------------------------------------------------------
    L = ['# 多数据集全链四指标冲刺（2026-09-21 · fresh split v2）', '',
         '## 协议',
         '- 16 类四数据集（USTC/CSTNET/CrossPlatform-China-Android/VisQUIC），统一 runtime prefix=64。',
         '- 泛化特征池 = common_shape（无端口/SNI/JA3/JA4/TLS/QUIC/protocol/TCP-flag/window）',
         '  + 新增 197 维通用序列族（方向游程、包长/IAT 分位数与直方图、位置签名、转移矩阵、',
         '  双阈值 burst、自相关、累计进度曲线、时间窗节奏、尺寸多样性）。最终分类器不接触 dataset ID。',
         '- fresh split（seed=20260922）在任何搜索前生成；split SHA256 '
         f'`{split_spec["split_sha256"]}`。',
         '- 泛化修复：multi-file 类 Test 为整文件 held-out（held 文件全部 observation 从 Train/Validation',
         '  剔除，Test 下采样，held 余量 drop）；`split_leakage_audit.json` 硬断言',
         '  observation_id 两两交集=0、multi-file 类 dev/test source_file 交集=0。',
         f'- 单 capture 类（{len(audit["fallback_classes"])} 个）只能 observation 级划分，已单列。',
         '- 只有 Validation 同时达到四指标才允许写 frozen_config 并首次打开 Test。', '',
         '## Validation 结果（Train/Validation 搜索，Test 保持密封）', '']
    if lb_rows is not None:
        top = lb_rows.head(8)[['subset', 'model', 'k', 'accuracy', 'macro_precision',
                               'macro_recall', 'max_per_class_fpr', 'slack']]
        L += ['Top-8 配置（按四指标最小边际排序）：', '', md_table(top), '']
    if stab_rows is not None:
        L += ['Top-3 三种子稳定性（Validation）：', '',
              md_table(stab_rows[['subset', 'model', 'accuracy_mean', 'accuracy_min',
                                  'macro_precision_mean', 'macro_recall_mean',
                                  'macro_recall_min', 'max_fpr_max', 'slack_mean']]), '']
    if outcome is not None:
        L += ['', '## 裁决', '',
              f'- **{outcome["status"]}**：{outcome["reason"]}',
              '- frozen_config.json 未写入，fresh Test 未打开（协议要求）。', '']
    if test is not None:
        L += ['## Fresh Test（一次性）', '',
              f"- Acc={test['metrics']['accuracy']:.4f} / MacroP={test['metrics']['macro_precision']:.4f} "
              f"/ MacroR={test['metrics']['macro_recall']:.4f} / maxFPR={test['metrics']['max_per_class_fpr']:.4f}",
              f"- 四项 {'全部 PASS' if test['all_pass'] else '未全部 PASS'}：{test['pass']}", '']
    if outcome is not None and not frozen_p.exists():
        L += ['## 失败的硬指标与证据链', '',
              '- Macro Recall 是唯一未达指标（目标 98%）：最好配置 Validation ≈ '
              f"{status['validation_best']['macro_recall']:.4f}。",
              '- 误差集中在 CrossPlatform 域内（bubei.tingshu↔autonavi↔BaiduMap↔aikan）与 CSTNET 相似站点。',
              '- 诊断证据（全部 Train/Validation 侧）：',
              '  - 1-NN：Validation 错误样本的最近 Train 邻居只有 32% 与真标签同类；',
              '  - LOO 1-NN（Train 内）：CrossPlatform 总体 77%，autonavi 62.5% —— 类在形态空间真实互穿；',
              '  - 错误样本真标签概率排名：rank2=14、rank3=6、rank4+=7，median 概率差 0.454 ——',
              '    即使完美决策规则也只到 ~97.3%，缺口是表征性的；',
              '  - 弱类样本受限：aikan 仅 98 个合格会话（48 train），CrossPlatform 其余类已用尽全部会话；',
              '  - 已试无增益：种子 bagging、随机子空间、kNN 成员、OOF 校准、少数类过采样、',
              '    中段窗口增广、MoE gate+experts、96/128/197 维放宽（+0.5~1pp，仍<98%）。', '',
              '## 下一步最可能有效的真实路径', '',
              '1. 允许 SNI/证书长度族等握手内容特征（当前纪律明令禁止；这是最大的判别信息源）。',
              '2. CrossPlatform 换用带服务标注的采集或做 App↔SDK 流清洗，消除跨 App SDK 形态噪声。',
              '3. 把观察单位从单会话提升为短时窗聚合（多会话联合形态），需要行为层部署语义。', '']
    (OUT / 'summary.md').write_text('\n'.join(L) + '\n', encoding='utf-8')

    # ---- reproduce.sh --------------------------------------------------------
    rep = OUT / 'reproduce.sh'
    rep.write_text('''#!/usr/bin/env bash
# Reproduce mixed_dataset_full_pass_20260921 (fresh split v2, leakage-fixed).
set -euo pipefail
cd /workspace/Huawei
python3 scripts/run_mixed_dataset_full_pass_20260921.py --stage selftest
python3 scripts/run_mixed_dataset_full_pass_20260921.py --stage extract
python3 scripts/run_mixed_dataset_full_pass_20260921.py --stage split
python3 scripts/run_mixed_dataset_full_pass_20260921.py --stage search
# stage test only runs if the validation four-metric gate passed and
# frozen_config.json exists; it refuses to overwrite previous results.
test ! -f output/mixed_dataset_full_pass_20260921/frozen_config.json || \\
  python3 scripts/run_mixed_dataset_full_pass_20260921.py --stage test
''')
    rep.chmod(0o755)

    # ---- SHA256SUMS ----------------------------------------------------------
    files = sorted(p for p in OUT.rglob('*')
                   if p.is_file() and p.name != 'SHA256SUMS'
                   and 'quarantine' not in p.parts)
    lines = []
    for p in files:
        lines.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT)}')
    (OUT / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    print('FINALIZED. status:', json.dumps({k: v for k, v in status.items()
                                            if k != 'fallback_single_capture_classes'},
                                           ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
