# -*- coding: utf-8 -*-
from pathlib import Path
import json, hashlib
import pandas as pd

ROOT=Path('/workspace/Huawei')
OUT=ROOT/'output/mixed_dataset_feature_experiment_20260920'
def J(p): return json.loads(Path(p).read_text())
def pct(x): return f'{100*float(x):.2f}%'
def pp(x): return f'{100*float(x):+.2f} pp'

common=J(OUT/'results_common_shape.json')
broad=J(OUT/'results_broad_no_identifier.json')
cs=common['individual_common']
pool_c=pd.read_csv(OUT/'pooled_mix_summary_common_shape.csv')

specific_mean=sum(v['specific_metrics']['accuracy'] for v in cs['per_dataset'].values())/4
cons_mean=sum(v['consensus_metrics']['accuracy'] for v in cs['per_dataset'].values())/4
pooled_mean=sum(v['pooled_top64_metrics']['accuracy'] for v in cs['per_dataset'].values())/4

def row(scenario,k,df):
    r=df[(df.scenario==scenario)&(df.n_datasets==k)].iloc[0]
    return float(r.accuracy_mean),float(r.accuracy_std),float(r.macro_f1_mean)

fixed_c={k:row('fixed_total_about_320',k,pool_c) for k in [1,2,3,4]}
fixed_pc={k:row('fixed_per_class_40',k,pool_c) for k in [1,2,3,4]}

lines=[
'# 多数据集混合特征选择诊断（2026-09-20）','',
'## 实验设计','',
'- 数据集：USTC / CSTNET / CrossPlatform-China-Android / VisQUIC。',
'- 每个数据集4类，每类统一保留80个flow/session样本，总计1,280条。',
'- 全部由当前共享runtime重新从原始PCAP提取，统一前64包。',
'- common_shape特征空间剔除端口、TLS/QUIC/protocol显式字段、TCP flag/window/retransmission等明显协议或数据集标识，主要保留包长、IAT、方向、burst、payload/header、rate和quantile统计。',
'- 选择分数沿用项目思路：0.6×XGBoost importance + 0.4×mutual information。','',
'## 1. 各数据集关键特征确实不同','',
'单数据集Top64两两Jaccard仅约0.24–0.42（交集25–38维/64）。四个数据集Top64同时共有的只有9维：','',
(', '.join(cs['features_in_all_4_specific_top64'])),'',
f'全特征空间中，至少进入3/4个数据集各自Top64的特征共有 {len(cs["features_in_3plus_specific_top64"])} 维。','',
'## 2. 强制使用“跨数据集共同Top64”会有一定精度损失','',
'| Dataset | Dataset-specific Top64 | Consensus Top64 | Delta | Pooled-16class Top64 | Delta |',
'|---|---:|---:|---:|---:|---:|'
]
for d,v in sorted(cs['per_dataset'].items()):
    a=v['specific_metrics']['accuracy'];c=v['consensus_metrics']['accuracy'];p=v['pooled_top64_metrics']['accuracy']
    lines.append(f'| {d} | {pct(a)} | {pct(c)} | {pp(c-a)} | {pct(p)} | {pp(p-a)} |')
lines += [
'',
f'- 四数据集specific Top64平均Accuracy：**{pct(specific_mean)}**。',
f'- Consensus Top64平均Accuracy：**{pct(cons_mean)}**，平均下降 **{abs((cons_mean-specific_mean)*100):.2f} pp**。',
f'- 直接在16类混合训练集上选出的Pooled Top64平均Accuracy：**{pct(pooled_mean)}**，平均下降 **{abs((pooled_mean-specific_mean)*100):.2f} pp**。',
'- 损失主要集中在CSTNET和CrossPlatform；USTC与VisQUIC在该固定split上基本不受影响。','',
'Consensus Top64的跨数据集支持度：','',
f'- 4/4数据集都进入各自Top64：{cs["consensus_top64_support_distribution"].get("4",0)}维；',
f'- 3/4：{cs["consensus_top64_support_distribution"].get("3",0)}维；',
f'- 2/4：{cs["consensus_top64_support_distribution"].get("2",0)}维；',
f'- 仅1/4：{cs["consensus_top64_support_distribution"].get("1",0)}维。','',
'即Consensus Top64中61/64至少在两个数据集的单独Top64里出现，确实明显向“共享形状特征”集中。','',
'反过来，直接Pooled 16-class Top64并不等于交集：其中有8维甚至不在任何单数据集Top64中，说明混合训练还会选出用于区分数据集/类别边界的交互特征。','',
'## 3. 多混数据集是否必然降低准确率？','',
'### 固定每类40条样本','',
'| Dataset count | Mean Accuracy | Std | Macro-F1 |',
'|---:|---:|---:|---:|'
]
for k in [1,2,3,4]:
    a,s,f=fixed_pc[k];lines.append(f'| {k} | {pct(a)} | {pct(s)} | {pct(f)} |')
lines += [
'',
'在每类样本量不变时，1→4个数据集没有出现单调下降，Accuracy大致维持在87%–89%。因此“数据集混合本身”并不会自动导致大幅掉点。','',
'### 固定总样本预算约320条','',
'| Dataset count | Classes | Mean Accuracy | Std | Macro-F1 |',
'|---:|---:|---:|---:|---:|'
]
for k in [1,2,3,4]:
    a,s,f=fixed_c[k];lines.append(f'| {k} | {4*k} | {pct(a)} | {pct(s)} | {pct(f)} |')
lines += [
'',
f'固定总预算下，1→4个数据集Accuracy从 **{pct(fixed_c[1][0])}** 降到 **{pct(fixed_c[4][0])}**，下降 **{(fixed_c[1][0]-fixed_c[4][0])*100:.2f} pp**。',
'这条曲线与你的直觉一致，但原因不是单一的“共享特征变弱”：同时发生了类别数4→16增加，以及每类训练样本显著减少。','',
'## 4. 结论','',
'实验对原假设是“部分支持”：','',
'1. **支持：不同数据集的关键指纹确实不同。** Top64重合度只有中等水平，四集真正共有Top64仅9维。',
'2. **支持：如果主动把选择器推向跨数据集共享特征，会有可测的准确率代价。** common_shape下Consensus Top64平均约损失1.95个百分点，CSTNET/CrossPlatform更明显。',
'3. **不完全支持：直接把更多数据集混在一起，并不会在每类样本充足时必然掉精度。** Pooled selector仍会保留一部分数据集特异/交互特征。',
'4. **明显支持：在固定总样本预算下，多数据集混合会明显掉点。** 但这里的主因是“共享特征约束 + 类别数上升 + 单类样本稀释”的共同作用。','',
'因此，如果后续真要利用“多数据集混合训练”逼出更泛化的特征，建议不要简单pool后做普通Top-K，而应显式优化跨数据集稳定性，例如按dataset分别计算importance/stability，再做consensus或worst-dataset约束；否则普通pooled selector仍可能抓住数据集特异边界。','',
'## 5. 关键产物','',
'- all_datasets_prefix64.csv：统一runtime提取的1,280条样本。',
'- results_common_shape.json：保守共享形状特征空间完整结果。',
'- results_broad_no_identifier.json：较宽特征空间对照。',
'- pooled_mix_common_shape.csv / pooled_mix_summary_common_shape.csv：混合闭集明细与汇总。',
'- summary.md：本摘要。'
]
(OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

status={
 'hypothesis':'partially_supported',
 'dataset_fingerprint_heterogeneity':'supported',
 'consensus_feature_penalty_mean_pp':round((cons_mean-specific_mean)*100,4),
 'fixed_total_1_to_4_dataset_accuracy_drop_pp':round((fixed_c[4][0]-fixed_c[1][0])*100,4),
 'fixed_per_class_monotonic_drop':False,
 'all4_top64_intersection':len(cs['features_in_all_4_specific_top64']),
 'consensus_top64_at_least_2_dataset_support':sum(v for k,v in cs['consensus_top64_support_distribution'].items() if int(k)>=2),
 'note':'exploratory small-sample diagnostic; not formal competition accuracy'
}
(OUT/'status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False))
files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='SHA256SUMS')
(OUT/'SHA256SUMS').write_text('\n'.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}' for p in files)+'\n')
print((OUT/'summary.md').read_text())
print(json.dumps(status,indent=2,ensure_ascii=False))
