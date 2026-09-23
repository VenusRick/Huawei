# -*- coding: utf-8 -*-
"""Assemble the machine side of the "人工修正后可用率 >= 90%" closure
(final_experiment_closure_20260922).

Fact base: NO human correction record exists anywhere in this project (all
prior rounds recorded manual_revision_usable_rate = BLOCKED_DATA). This
builder therefore does NOT invent a usable rate. It assembles everything a
human reviewer needs so that the only remaining step is an actual sign-off:

1. candidate feature packages (machine-selected, machine-validated):
   - CSTNET60 frozen 128pkt_top64 (E=63/64=98.4% output-effectiveness),
   - CESNET-QUIC22 corrected56 (machine-corrected 56-dim, four-metric PASS),
   - Application64 production RuleBundle selected features (deployed, sealed
     test 99.75/99.75/99.75/0.17);
2. a before/after review table template with the machine status pre-filled
   and the human verdict columns empty;
3. the usable-rate computation script (refuses to produce a number until
   real human verdicts exist; computes keep+modify_accepted / reviewed and
   compares against the 90% threshold);
4. status.json declaring EXTERNAL_REVIEW with the exact missing evidence.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path('/workspace/Huawei')
OUT = ROOT / 'output/final_experiment_closure_20260922/manual_review_readiness'
CAND = OUT / 'candidates'

CESNET = ROOT / 'output/experiment_archive_20260919/cesnet_quic22'
CST = ROOT / 'output/cstnet60_20260918_exp'
APP = ROOT / 'output/three_profile_acceptance_20260919/application'


def main() -> None:
    CAND.mkdir(parents=True, exist_ok=True)

    # ---- 1. candidate packages --------------------------------------------
    packages = {}

    ces_val = json.loads((CESNET / 'corrected56_validation.json').read_text())
    ces_feats = (CESNET / 'corrected56_features.txt').read_text().split()
    (CAND / 'cesnet_corrected56_features.txt').write_text(
        '\n'.join(ces_feats) + '\n')
    packages['cesnet_quic22_corrected56'] = {
        'source': 'output/experiment_archive_20260919/cesnet_quic22',
        'n_features': len(ces_feats),
        'machine_evidence': {
            'validation': {k: ces_val[k] for k in (
                'accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
                'max_class_fpr', 'errors', 'target_pass')},
            'correction_kind': 'machine selection/correction to 56 dims',
        },
        'features_file': 'candidates/cesnet_corrected56_features.txt',
    }

    acc = json.loads((CST / 'caer_accounting_60c.json').read_text())
    cst_names = acc['A_published_frozen']['names']
    (CAND / 'cstnet60_top64_features.txt').write_text(
        '\n'.join(cst_names) + '\n')
    failed = acc['E_validation_supported']['failed']
    packages['cstnet60_frozen_top64'] = {
        'source': 'output/cstnet60_20260918_exp/caer_accounting_60c.json',
        'n_features': len(cst_names),
        'machine_evidence': {
            'output_effectiveness_E_over_A': acc['rates']['E_over_A'],
            'model_referenced_R_over_A': acc['rates']['R_over_A'],
            'anova_failed_features': failed,
        },
        'features_file': 'candidates/cstnet60_top64_features.txt',
    }

    app_feats = [l.strip() for l in
                 (APP / 'selected_features.txt').read_text().splitlines()
                 if l.strip()]
    (CAND / 'application64_bundle_features.txt').write_text(
        '\n'.join(app_feats) + '\n')
    app_test = json.loads(
        (APP / 'independent_pcap_test_metrics.json').read_text())
    packages['application64_rule_bundle'] = {
        'source': 'output/three_profile_acceptance_20260919/application',
        'n_features': len(app_feats),
        'machine_evidence': {
            'sealed_test': {k: app_test.get(k) for k in (
                'accuracy', 'macro_precision', 'macro_recall',
                'max_per_class_fpr')},
        },
        'features_file': 'candidates/application64_bundle_features.txt',
    }
    (CAND / 'packages.json').write_text(
        json.dumps(packages, indent=2, ensure_ascii=False))

    # ---- 2. before/after review template -----------------------------------
    machine_status = {}
    machine_status.update({f: 'selected(machine)' for f in ces_feats})
    eff = set(acc['E_validation_supported']['names'])
    for f in cst_names:
        machine_status[f] = ('effective(ANOVA p<0.05)' if f in eff
                             else 'ineffective(ANOVA)')
    app_val = json.loads((APP / 'fidelity.json').read_text()) \
        if (APP / 'fidelity.json').exists() else {}
    for f in app_feats:
        machine_status[f] = 'bundle_selected(deployed)'
    rows = []
    for pkg, feats in (('cesnet_quic22_corrected56', ces_feats),
                       ('cstnet60_frozen_top64', cst_names),
                       ('application64_rule_bundle', app_feats)):
        for f in feats:
            rows.append({
                'package': pkg, 'feature': f,
                'machine_status': machine_status.get(f, ''),
                'human_verdict': '',       # keep | modify_accepted | drop
                'human_note': '',          # required when verdict != keep
                'reviewer': '', 'review_date': '',
            })
    tpl = pd.DataFrame(rows)
    tpl.to_csv(OUT / 'review_before_after_template.csv', index=False)

    # ---- 3. usable-rate computation script ---------------------------------
    (OUT / 'compute_usable_rate.py').write_text('''\
# -*- coding: utf-8 -*-
"""Compute the manual-revision usable rate from FILLED review records.

usable = human_verdict in {keep, modify_accepted}
rate   = usable / reviewed   (reviewed = rows with a non-empty human_verdict)

Refuses to emit a rate until at least one row carries a human verdict AND
every reviewed row names a reviewer. Threshold: >= 0.90.

Usage: python3 compute_usable_rate.py <filled_review.csv>
"""
import sys

import pandas as pd

USABLE = {'keep', 'modify_accepted'}
THRESHOLD = 0.90


def main() -> None:
    df = pd.read_csv(sys.argv[1])
    df['human_verdict'] = df['human_verdict'].fillna('').str.strip()
    df['reviewer'] = df['reviewer'].fillna('').str.strip()
    reviewed = df[df['human_verdict'] != '']
    if reviewed.empty:
        print('STATUS: not_computable - no human verdicts recorded yet '
              '(external review pending)')
        sys.exit(2)
    bad = reviewed[~reviewed['human_verdict'].isin(USABLE | {'drop'})]
    if len(bad):
        print(f'STATUS: invalid_verdicts - {len(bad)} rows'); sys.exit(3)
    no_name = reviewed[reviewed['reviewer'] == '']
    if len(no_name):
        print(f'STATUS: missing_reviewer - {len(no_name)} rows'); sys.exit(4)
    usable = reviewed['human_verdict'].isin(USABLE)
    rate = float(usable.sum()) / len(reviewed)
    print(f'reviewed={len(reviewed)} usable={int(usable.sum())} '
          f'rate={rate:.4f} threshold={THRESHOLD} '
          f'pass={rate >= THRESHOLD}')
    for pkg, g in reviewed.groupby('package'):
        r = float(g['human_verdict'].isin(USABLE).sum()) / len(g)
        print(f'  {pkg}: {r:.4f} (n={len(g)})')
    sys.exit(0 if rate >= THRESHOLD else 1)


if __name__ == '__main__':
    main()
''')

    # demo: the template itself must be refused
    demo = subprocess.run(
        [sys.executable, str(OUT / 'compute_usable_rate.py'),
         str(OUT / 'review_before_after_template.csv')],
        capture_output=True, text=True)
    demo_line = (demo.stdout + demo.stderr).strip().splitlines()

    # ---- 4. checklist + status ---------------------------------------------
    (OUT / 'review_checklist.md').write_text(f'''\
# 人工修正后可用率 ≥ 90% —— 审阅清单（机器侧已就绪）

状态：**EXTERNAL_REVIEW**。本项目至今无任何真实人工修正记录（历轮
manual_revision_usable_rate 均为 BLOCKED_DATA），因此不给出、也不预测可用率数值。

## 审阅对象（三个候选特征包，机器证据齐备）

| 包 | 维数 | 机器证据 |
|---|---:|---|
| cesnet_quic22_corrected56 | {len(ces_feats)} | Validation Acc {ces_val['accuracy']:.4f} / Macro-R {ces_val['macro_recall']:.4f} / MaxFPR {ces_val['max_class_fpr']:.4f}（四项过） |
| cstnet60_frozen_top64 | {len(cst_names)} | 输出有效率 E/A={acc['rates']['E_over_A']:.4f}，模型引用 R/A={acc['rates']['R_over_A']:.4f}；ANOVA 未过：{failed} |
| application64_rule_bundle | {len(app_feats)} | 独立推理 sealed Test {app_test.get('accuracy')} / {app_test.get('macro_recall')}（已部署口径） |

## 人工步骤（预计工作量：184 行逐行判定）

1. 打开 `review_before_after_template.csv`（machine_status 已预填）。
2. 对每个特征填 `human_verdict` ∈ {{keep, modify_accepted, drop}}；
   非 keep 必须写 `human_note`（改了什么/为什么删）。
3. 每行签署 `reviewer` 与 `review_date`。
4. 运行 `python3 compute_usable_rate.py <填好的表>.csv`：
   rate = (keep + modify_accepted) / 已审阅行数，阈值 ≥ 0.90。
5. 修正后的特征清单回流 `src/pipeline.py --mode revise-rules`
   （校验→验证集回放 diff→新版本，不覆盖原 bundle），留档 revision_diff。

## 机器侧当前自检（演示）

`{demo_line[0] if demo_line else ''}`

退出码 2 = 无人工判定（当前真实状态）；0/1 = 有记录后的过/不过。
''')

    status = {
        'experiment': 'final_experiment_closure_20260922/manual_review_readiness',
        'official_requirement': '人工修正后可用率 >= 90%',
        'status': 'EXTERNAL_REVIEW',
        'machine_side_complete': True,
        'missing_evidence': '真实人工逐特征判定记录（reviewer+日期+verdict）',
        'n_review_rows': int(len(tpl)),
        'packages': {k: v['n_features'] for k, v in packages.items()},
        'no_human_records_claim_base': [
            'output/chain_improve_20260918_02/reports/competition_accounting.json '
            '(manual_revision_usable_rate: BLOCKED_DATA)',
            'output/cstnet60_20260918_exp/caer_accounting_60c.json '
            "(manual_revision_usable_rate: BLOCKED_DATA)"],
        'not_an_experiment_todo': ('机器侧无可再优化目标；仅剩人工签字，'
                                  '不再列为实验待办'),
    }
    (OUT / 'status.json').write_text(
        json.dumps(status, indent=2, ensure_ascii=False))
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
