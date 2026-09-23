"""
端到端流水线主入口
整合所有模块：PCAP解析 → 特征提取 → 特征选择 → 规则生成 → DPI匹配

使用方式：
    python -m src.pipeline --mode mine --input data/pcaps/ --output output/rules/
    python -m src.pipeline --mode detect --rules output/rules/rules.yaml --input data/pcaps/
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import AppConfig, DEFAULT_CONFIG
from src.parser.pcap_reader import PCAPReader
from src.parser.session.session_manager import SessionManager
from src.features.basic.feature_extractor import BasicFeatureExtractor
from src.features.advanced.advanced_extractor import AdvancedFeatureExtractor
from src.features.selection.feature_selector import FeatureSelector
from src.engine.rule_compiler.optimized_generator import OptimizedRuleGenerator as RuleGenerator
from src.engine.matcher.optimized_engine import OptimizedDPIEngine as DPIRuleEngine


LEGAL_OPS = ("<=", ">", ">=", "<", "==", "!=", "exists", "not_exists")


def _revise_rules(bundle_dir: str, edits_path: str, manifest: str,
                  output: str, task=None) -> None:
    """人工修正规则：结构化编辑 + 校验 + 验证集回放 + 新版本发布。

    红线：未知函数与任意Python表达式禁止入规则（只接受结构化edit）；
    原bundle不被覆盖；不读取封存测试数据。
    """
    import json as _json
    from src.data_manifest import load_manifest
    from src.engine.model_io import load_rule_bundle

    rules, selected, label_names, cfg = load_rule_bundle(bundle_dir)
    btask = task or cfg.get('task', 'app')
    edits = _json.loads(Path(edits_path).read_text(encoding='utf-8'))
    by_id = {r.get('id'): r for r in rules}
    problems = []
    n_applied = 0
    for e in edits.get('rules', []):
        rid, action = e.get('rule_id'), e.get('action')
        r = by_id.get(rid)
        if r is None:
            problems.append(f"未知规则ID: {rid}")
            continue
        if action == 'disable':
            r['_disabled'] = True
            n_applied += 1
        elif action == 'enable':
            r['_disabled'] = False
            n_applied += 1
        elif action == 'edit':
            for c in e.get('conditions', []):
                if c.get('op') not in LEGAL_OPS:
                    problems.append(f"{rid}: 非法运算符 {c.get('op')}")
                if c.get('feature') not in selected:
                    problems.append(
                        f"{rid}: 不支持特征 {c.get('feature')}"
                        "（未部署，需先重挖）")
                if c.get('op') not in ('exists', 'not_exists') and not isinstance(
                        c.get('value'), (int, float)):
                    problems.append(f"{rid}: value 必须为数值")
            if not any(p.startswith(rid) for p in problems):
                r['conditions'] = e['conditions']
                n_applied += 1
        else:
            problems.append(f"{rid}: 未知动作 {action}")
    if problems:
        print("编辑校验失败，拒绝发布：")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    new_rules = [r for r in rules if not r.get('_disabled')]
    for r in new_rules:
        r.pop('_disabled', None)

    out = Path(output)
    (out / 'bundle').mkdir(parents=True, exist_ok=True)
    from src.engine.model_io import save_rule_bundle
    save_rule_bundle(new_rules, selected, label_names,
                     cfg.get('confidence_threshold', 0.7),
                     str(out / 'bundle'),
                     bundle_meta={'task': btask, 'source': 'revise-rules',
                                  'context': cfg.get('context'),
                                  'feature_profile': cfg.get('feature_profile', 'full'),
                                  'profile_spec': cfg.get('profile_spec'),
                                  'temporal_vote': cfg.get('temporal_vote', 1),
                                  'max_packets_per_session': cfg.get(
                                      'max_packets_per_session'),
                                  'parent_bundle': str(bundle_dir),
                                  'edits_applied': n_applied})
    # 验证集回放 diff（新 vs 旧）
    records = [r for r in load_manifest(manifest)
               if r.split in (None, 'validation', 'val')]
    base = Path(manifest).parent
    from src.engine.matcher.optimized_engine import OptimizedDPIEngine
    old_eng = OptimizedDPIEngine()
    old_eng.load_rule_bundle(bundle_dir)
    new_eng = OptimizedDPIEngine()
    new_eng.load_rule_bundle(str(out / 'bundle'))
    diff = {'n_windows': 0, 'changed': 0, 'old_matched': 0, 'new_matched': 0}
    from src.features.runtime import (extract_feature_records,
                                      extract_behavior_feature_records)
    ctx = cfg.get('context') or {}
    for r in records:
        p = Path(r.pcap)
        p = p if p.is_absolute() else base / p
        if btask == 'behavior':
            wrecs = extract_behavior_feature_records(
                str(p), ctx, terminal=getattr(r, 'terminal_ip', '') or None)
            vecs = [{k: (v if v is not None else -1.0)
                     for k, v in w.features.items()} for w in wrecs]
        else:
            vecs = [rec.features
                    for rec in extract_feature_records(str(p))]
        for v in vecs:
            o = old_eng.match(v)
            n_ = new_eng.match(v)
            ol = o[0].result if o else 'unknown'
            nl = n_[0].result if n_ else 'unknown'
            diff['n_windows'] += 1
            diff['old_matched'] += bool(o)
            diff['new_matched'] += bool(n_)
            diff['changed'] += (ol != nl)
    (out / 'revision_diff.json').write_text(
        _json.dumps(diff, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"修订完成：应用 {n_applied} 条编辑，禁用 "
          f"{len(rules) - len(new_rules)} 条规则")
    print(f"验证集回放：{diff['n_windows']} 样本，"
          f"{diff['changed']} 条预测变化（old匹配 {diff['old_matched']}"
          f" -> new匹配 {diff['new_matched']}）")
    print(f"新 bundle: {out / 'bundle'}（原 bundle 未改动）")


def _r_pcap_of(rec, base: Path) -> Path:
    """manifest 记录 -> pcap 绝对路径（相对路径基于 manifest 目录）。"""
    p = Path(rec.pcap)
    return p if p.is_absolute() else base / p


class FeatureMiningPipeline:
    """智能化加密流量特征挖掘流水线"""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or DEFAULT_CONFIG

        # 初始化各模块
        self.session_manager = SessionManager(
            tcp_timeout=self.config.parser.tcp_timeout,
            udp_timeout=self.config.parser.udp_timeout,
            max_sessions=self.config.parser.max_sessions,
        )
        self.pcap_reader = PCAPReader(self.session_manager)
        self.basic_extractor = BasicFeatureExtractor()
        self.advanced_extractor = AdvancedFeatureExtractor(
            ngram_size=self.config.feature.ngram_size,
            bow_vocab_size=self.config.feature.bow_vocabulary_size,
        )
        self.feature_selector = FeatureSelector(
            correlation_threshold=self.config.selection.correlation_threshold,
            max_features=self.config.selection.max_features,
            min_features=self.config.selection.min_features,
        )
        self.rule_generator = RuleGenerator(
            confidence_threshold=self.config.rule.confidence_threshold,
            min_support=self.config.rule.min_support,
            min_confidence=self.config.rule.min_confidence,
        )
        self.dpi_engine = DPIRuleEngine()

    # ==================== 模式1：特征挖掘 ====================

    @staticmethod
    def _parity_probe(pcap_path: str, records, max_packets: int,
                      active_groups=None) -> Dict:
        """训练侧 vs 独立推理入口的实测 parity 探针（同文件二次提取比对）。

        返回 {"parity_features": set, "mismatches": [...], "probed_file": ...}；
        consistent 判定只用实测证据，不再默认 True。
        active_groups：fast profile 时两侧同组按需提取（探针与训练同路径，
        不拿319维全集冒充16维profile的parity）。
        """
        from src.features.runtime import feature_names
        from src.engine.dpi_infer import extract_features_from_pcap
        again = extract_features_from_pcap(pcap_path, max_packets=max_packets,
                                           active_groups=active_groups)
        parity_features = feature_names(records)
        mismatches = []
        if len(again) != len(records):
            mismatches.append(f"记录数不一致: mine={len(records)} "
                              f"infer={len(again)}")
        for r, o in zip(records, again):
            for k in set(r.features) | set(o):
                a, b = r.features.get(k), o.get(k)
                if (a is None) != (b is None) or (
                        a is not None and b is not None
                        and float(a) != float(b)):
                    mismatches.append(f"{r.observation.observation_id}:{k}")
                    if len(mismatches) > 50:
                        break
        return {"parity_features": set(parity_features),
                "mismatches": mismatches[:50],
                "probed_file": str(pcap_path),
                "n_records": len(records)}

    def _replay_validation(self, rules, selected, label_names,
                           val_rows: List[Dict]) -> Dict:
        """验证集回放：纯规则引擎在验证行上的成绩（字符串标签口径）。"""
        return self._replay_validation_raw(rules, selected, label_names,
                                           val_rows)

    def mine(self, pcap_dir: str, output_dir: str,
            allow_demo: bool = False,
            labels: Optional[Dict[str, int]] = None,
            manifest: Optional[str] = None,
            task: str = 'app',
            max_read_packets: Optional[int] = None,
            max_packets: Optional[int] = None,
            behavior_include_background: bool = False,
            profile: str = 'full') -> Dict:
        """
        特征挖掘模式：从PCAP文件挖掘特征并生成规则

        Args:
            pcap_dir: PCAP文件目录
            output_dir: 输出目录
            labels: 标签字典 {文件名(basename精确匹配): 标签ID}
            manifest: 试次清单（正式入口；标签=记录字段，不猜文件名）
            task: app | behavior | tool
            max_read_packets: 每文件真正的读包上限（读满即停）
            max_packets: 每会话前缀截断上限（默认 runtime 500；
                profile 为 fast16/fast32 且未显式指定时默认 32）
            behavior_include_background: behavior 任务是否把背景窗
                作为显式类别入训练（默认 False 维持首版口径）
            profile: 特征profile——full=全量319维（兼容默认）；
                fast16/fast32=低成本少量特征（2026-09-18 第三轮，
                bundle 携带 profile 与前缀包数，独立CLI同参）

        红线（审计 2026-09-18）：
        - 标签只来自清单记录字段或 labels 的 basename 精确匹配；
          匹配不到显式报错，绝不回落类0、绝不做子串猜测；
        - FeatureSelector 与规则生成只在 train split 上拟合；
          validation 只用于阈值/模型选择与有效性判定证据，
          按 capture_id 分组隔离，绝不混入拟合；
        - 无 validation split 时明确记录"无该证据"，不随机按会话混拆。
        """
        print("=" * 60)
        print("  智能化加密流量特征挖掘工具")
        print("  模式：特征挖掘与规则生成")
        print("=" * 60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        from src.features.runtime import (DEFAULT_MAX_PACKETS, ExtractStats,
                                          extract_feature_records_with_stats)
        from src.features.fast_profile import (FAST_PROFILES,
                                               default_prefix_packets)
        from src.profile_specs import (get_profile_spec,
                                       filter_forbidden_features)
        profile_spec = get_profile_spec(profile, task=task)
        if max_packets is None and profile_spec.get('max_packets'):
            max_packets = int(profile_spec['max_packets'])
        elif max_packets is None and profile in FAST_PROFILES:
            max_packets = default_prefix_packets(profile)
        _active = {profile} if profile in FAST_PROFILES else None
        if profile in FAST_PROFILES:
            print(f"  特征profile: {profile}"
                  f"（{len(FAST_PROFILES[profile])}特征，"
                  f"会话前缀 {max_packets} 包）")
        _mp = int(max_packets or DEFAULT_MAX_PACKETS)
        self._behavior_vocab: Dict[str, int] = {}
        self._last_profile = profile
        self._last_profile_spec = profile_spec

        # Step 1/2: 载入清单（或目录）并按共享 runtime 提取特征
        print("\n[Step 1/5] 解析PCAP文件 + 提取特征（共享runtime入口）...")
        stats_total = ExtractStats()
        rows: List[Dict] = []           # 每行 = 特征 + _label_id/_split/...
        parity_probe_report: Optional[Dict] = None

        if manifest is not None:
            from src.data_manifest import (load_manifest, validate_manifest,
                                           build_label_vocab)
            records = [r for r in load_manifest(manifest)
                       if r.split in (None, 'train', 'validation')]
            base = Path(manifest).parent
            validate_manifest(records, manifest_dir=str(base), task=task)
            vocab = build_label_vocab(records, task)
            label_map = vocab['label_map']
            n_train_rec = sum(1 for r in records
                              if r.split in (None, 'train'))
            n_val_rec = sum(1 for r in records if r.split == 'validation')
            print(f"  清单: {len(records)} 试次（train {n_train_rec} / "
                  f"validation {n_val_rec}；test 不入挖掘）")
            if n_val_rec == 0:
                print("  注意：清单无 validation split——验证证据将标记为"
                      " 不可用，不随机按会话混拆造验证集")
            for r in records:
                pcap = _r_pcap_of(r, base)
                split = 'validation' if r.split == 'validation' else 'train'
                if (task == 'behavior'
                        and profile_spec.get('observation_unit') == 'behavior_window'):
                    wrecs = self._behavior_window_rows(
                        r, pcap, max_read_packets,
                        include_background=behavior_include_background,
                        context_override=profile_spec.get('context'))
                    for row in wrecs:
                        row['_split'] = split
                        row['_capture_id'] = r.capture_id
                        row['_trial_id'] = r.trial_id
                    rows.extend(wrecs)
                    stats_total.files += 1
                    continue
                if (task == 'behavior'
                        and profile_spec.get('observation_unit') == 'flow_window'):
                    wrecs = self._behavior_flow_window_rows(
                        r, pcap, max_read_packets,
                        include_background=behavior_include_background,
                        context_override=profile_spec.get('context'))
                    for row in wrecs:
                        row['_split'] = split
                        row['_capture_id'] = r.capture_id
                        row['_trial_id'] = r.trial_id
                    rows.extend(wrecs)
                    stats_total.files += 1
                    continue
                if (task == 'tool'
                        and profile_spec.get('observation_unit') == 'tunnel_window'):
                    wrecs = self._tunnel_window_rows(
                        r, pcap, max_read_packets,
                        label_id=label_map[r.label(task)],
                        context_override=profile_spec.get('context'))
                    for row in wrecs:
                        row['_split'] = split
                        row['_capture_id'] = r.capture_id
                        row['_trial_id'] = r.trial_id
                    rows.extend(wrecs)
                    stats_total.files += 1
                    continue
                if (task == 'tool'
                        and profile_spec.get('observation_unit') == 'tunnel_flow_window'):
                    wrecs = self._tunnel_flow_window_rows(
                        r, pcap, max_read_packets,
                        label_id=label_map[r.label(task)],
                        context_override=profile_spec.get('context'))
                    for row in wrecs:
                        row['_split'] = split
                        row['_capture_id'] = r.capture_id
                        row['_trial_id'] = r.trial_id
                    rows.extend(wrecs)
                    stats_total.files += 1
                    continue
                recs, st = extract_feature_records_with_stats(
                    str(pcap), max_packets=_mp, capture_id=r.capture_id,
                    max_read_packets=max_read_packets,
                    active_groups=_active)
                for k in vars(st):
                    if k == 'error_details':
                        stats_total.error_details.extend(st.error_details)
                    else:
                        setattr(stats_total, k,
                                getattr(stats_total, k) + getattr(st, k))
                exact_label = label_map[r.label(task)]   # 记录->精确标签
                for rec in recs:
                    row = dict(rec.features)
                    row['_observation_id'] = rec.observation.observation_id
                    row['_source_file'] = str(pcap)
                    row['_capture_id'] = r.capture_id
                    row['_trial_id'] = r.trial_id
                    row['_split'] = split
                    row['_label_id'] = exact_label
                    rows.append(row)
                if parity_probe_report is None and recs:
                    parity_probe_report = self._parity_probe(
                        str(pcap), recs, _mp, active_groups=_active)
        else:
            pcap_files = (list(Path(pcap_dir).rglob("*.pcap"))
                          + list(Path(pcap_dir).rglob("*.pcapng")))
            if not pcap_files:
                # R25: 正式挖掘空输入必须失败；演示仅显式 allow_demo 触发
                if allow_demo:
                    print(f"  警告：在 {pcap_dir} 中未找到PCAP文件，使用演示数据")
                    return self._generate_demo_rules(output_path)
                raise FileNotFoundError(
                    f"挖掘输入目录中未找到PCAP文件: {pcap_dir}")
            # R25: 正式挖掘必须提供标签（无标签静默全0是伪造数据）
            if not labels and not allow_demo:
                raise ValueError("正式挖掘需要标签映射（文件名→类别）；"
                                 "演示请显式使用 allow_demo")
            by_basename = {Path(k).name: v for k, v in (labels or {}).items()}
            unmatched = []
            for pcap in pcap_files:
                recs, st = extract_feature_records_with_stats(
                    str(pcap), max_packets=_mp,
                    max_read_packets=max_read_packets,
                    active_groups=_active)
                for k in vars(st):
                    if k == 'error_details':
                        stats_total.error_details.extend(st.error_details)
                    else:
                        setattr(stats_total, k,
                                getattr(stats_total, k) + getattr(st, k))
                lab = by_basename.get(pcap.name)
                if lab is None:
                    unmatched.append(pcap.name)
                    continue
                for rec in recs:
                    row = dict(rec.features)
                    row['_observation_id'] = rec.observation.observation_id
                    row['_source_file'] = str(pcap)
                    row['_capture_id'] = pcap.stem
                    row['_trial_id'] = pcap.stem
                    row['_split'] = 'train'   # 目录模式无 split 证据
                    row['_label_id'] = int(lab)
                    rows.append(row)
                if parity_probe_report is None and recs:
                    parity_probe_report = self._parity_probe(
                        str(pcap), recs, _mp, active_groups=_active)
            if unmatched:
                raise ValueError(
                    "以下PCAP无精确标签映射（basename 精确匹配失败，"
                    "不做子串猜测、不回落类0）: "
                    f"{sorted(set(unmatched))}")

        if not rows:
            # R25: 无有效会话视为输入错误，不再静默降级演示
            if allow_demo:
                return self._generate_demo_rules(output_path)
            raise ValueError("PCAP解析后无有效会话，无法进行正式挖掘")

        df = pd.DataFrame(rows)
        feat_cols_all = [c for c in df.columns if not str(c).startswith('_')]
        feat_cols = filter_forbidden_features(feat_cols_all, profile_spec)
        blocked = sorted(set(feat_cols_all) - set(feat_cols))
        if blocked:
            print(f"  identifier policy: 排除 {len(blocked)} 特征: {blocked}")
        if not feat_cols:
            raise ValueError(f"profile={profile} 过滤后无可用特征")
        y_all = df['_label_id']
        train_mask = df['_split'] == 'train'
        val_mask = df['_split'] == 'validation'
        n_train, n_val = int(train_mask.sum()), int(val_mask.sum())
        if n_train == 0:
            raise ValueError("train split 为空：清单需至少含一条 train/未分记录")
        if task == 'behavior':
            df[feat_cols] = df[feat_cols].fillna(-1.0)  # 训练侧哨兵填充
        print(f"  总特征数: {len(feat_cols)}；会话/窗口: {len(df)}"
              f"（train {n_train} / validation {n_val}）")
        if profile_spec.get('observation_unit') in (
                'behavior_window', 'flow_window', 'tunnel_window',
                'tunnel_flow_window'):
            print("  解析计数: 窗口profile使用独立runtime；包/会话/RSS/时延"
                  "见 performance 与独立CLI证据，不用flow ExtractStats的0值")
        else:
            print(f"  解析计数: 读包 {stats_total.packets_read}"
                  f"（上限截断文件 {stats_total.read_capped_files}），"
                  f"会话 {stats_total.sessions_total}，"
                  f"过滤(<3包) {stats_total.sessions_filtered_min_packets}，"
                  f"过滤(<100B) {stats_total.sessions_filtered_min_bytes}，"
                  f"截断 {stats_total.sessions_truncated}，"
                  f"提取失败 {stats_total.extract_errors}")
        df.to_csv(str(output_path / 'raw_features.csv'), index=False)

        # Step 3: 特征选择（只在 train split 拟合）
        print("\n[Step 3/5] 智能特征选择（仅 train split 拟合）...")
        df_tr = df[train_mask].reset_index(drop=True)
        y_tr = df_tr['_label_id']
        self.feature_selector.fit(df_tr[feat_cols], y_tr, feat_cols)
        selected = list(self.feature_selector.selected_features_)
        feature_budget = profile_spec.get('feature_budget')
        if feature_budget is not None and len(selected) > int(feature_budget):
            selected = selected[:int(feature_budget)]
            self.feature_selector.selected_features_ = list(selected)
        report = self.feature_selector.get_selection_report()
        print(f"  选中特征数: {len(selected)}")
        print(f"  特征保留率(选中/全部): {len(selected)}/{len(feat_cols)} "
              f"({len(selected)/len(feat_cols)*100:.1f}%)"
              f"（保留率≠特征有效率，有效率见 analysis/）")
        print(f"  Top-5重要特征:")
        for feat, imp in report['importance_ranking'][:5]:
            print(f"    {feat}: {imp:.4f}")

        # Step 4: 规则生成（train 拟合；validation 只作一次性离线对照）
        print("\n[Step 4/5] 生成DPI规则（train拟合，validation仅对照）...")
        if task == 'behavior' and manifest is not None:
            label_names = {v: k for k, v in self._behavior_vocab.items()}
        elif manifest is not None:
            label_names = {v: k for k, v in label_map.items()}
        else:
            label_names = {int(v): k for k, v in (labels or {}).items()
                           if v in set(y_all)}
        if not label_names:
            label_names = {0: 'benign', 1: 'unknown_app'}

        validation = None
        if n_val > 0:
            df_va = df[val_mask].reset_index(drop=True)
            validation = (df_va[feat_cols], df_va['_label_id'])
        groups = df_tr['_capture_id']
        rules = self.rule_generator.fit_and_generate(  # R28: 真实API
            df_tr[feat_cols], y_tr, selected, label_names,
            validation=validation, groups=groups)
        print(f"  生成规则数: {len(rules)}")

        # 导出规则文件
        rule_path = str(output_path / 'rules.yaml')
        self.rule_generator.export_yaml(rule_path, label_names)
        print(f"  规则文件已保存: {rule_path}")

        # Step 5: 验证证据 + 报告 + bundle
        print("\n[Step 5/5] 保存报告与证据...")
        # 验证集回放（规则引擎纯规则路径；无验证集则明确记录）
        val_report = {'available': bool(n_val), 'n_validation': n_val}
        if n_val:
            df_va = df[val_mask].reset_index(drop=True)
            val_rows = df_va.to_dict('records')
            replay = self._replay_validation_raw(rules, selected,
                                                 label_names, val_rows)
            val_report.update(replay)
        else:
            val_report['reason'] = ('清单无 validation split；'
                                    '不随机按会话混拆造验证集')
        (output_path / 'validation_report.json').write_text(
            json.dumps(val_report, ensure_ascii=False, indent=2),
            encoding='utf-8')
        print(f"  validation_report: available={val_report['available']}"
              + (f", acc={val_report.get('accuracy')}" if n_val else ""))

        # M3: AnalysisPackage mine 层（判别证据用验证集；无则如实标注训练集）
        try:
            from src.reporting import build_mine_layer
            if n_val:
                ev_df, ev_y = df[val_mask], df[val_mask]['_label_id']
                evidence_split = 'validation'
            else:
                ev_df, ev_y = df[train_mask], df[train_mask]['_label_id']
                evidence_split = ('train（无validation split，'
                                  '判别证据来自训练集，非验证证据）')
            parity_feats = None
            if parity_probe_report is not None:
                parity_feats = (set(parity_probe_report['parity_features'])
                                if not parity_probe_report['mismatches']
                                else set())
            build_mine_layer(ev_df, ev_y, selected, rules, label_names,
                             str(output_path / 'analysis'),
                             confidence_threshold=getattr(
                                 self.rule_generator, 'confidence_threshold',
                                 0.7),
                             parity_features=parity_feats,
                             engine_supported=set(selected),
                             evidence_split=evidence_split)
            print(f"  AnalysisPackage(mine层): {output_path / 'analysis'}")
        except Exception as _e:  # noqa: BLE001
            print(f"  警告：mine层分析包生成失败（不阻塞挖掘）: {_e}")
        if parity_probe_report is not None:
            (output_path / 'parity_report.json').write_text(
                json.dumps({k: (sorted(v) if isinstance(v, set) else v)
                            for k, v in parity_probe_report.items()},
                           ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"  parity_report: 失配 "
                  f"{len(parity_probe_report['mismatches'])} 项")
        (output_path / 'extraction_stats.json').write_text(
            json.dumps(stats_total.to_dict(), ensure_ascii=False, indent=2),
            encoding='utf-8')

        # P0: 正式 bundle 三件套（独立推理的加载单位）
        from src.engine.model_io import save_rule_bundle
        bundle_dir = output_path / 'bundle'
        # Profile contract is persisted with the exact selected feature/operator
        # dependency list, so independent inference does not require CLI repeats.
        pmeta = dict(profile_spec)
        pmeta['required_features'] = list(selected)
        try:
            if pmeta.get('observation_unit') in ('flow', 'flow_prefix'):
                from src.features.operators import resolve_required_families
                pmeta['required_operators'] = sorted(
                    resolve_required_families(selected))
            else:
                pmeta['required_operators'] = []
        except Exception:
            pmeta['required_operators'] = []
        save_rule_bundle(rules, selected, label_names,
                         getattr(self.rule_generator, 'confidence_threshold', 0.7),
                         str(bundle_dir),
                         bundle_meta={'task': task or 'app', 'source': 'mine',
                                      'context': getattr(self, '_last_context', None),
                                      'n_train_rows': n_train,
                                      'n_validation_rows': n_val,
                                      'max_packets_per_session': _mp,
                                      'max_read_packets_per_file': max_read_packets,
                                      'feature_profile': profile,
                                      'profile_spec': pmeta,
                                      'temporal_vote': int(
                                          profile_spec.get('temporal_vote', 1))})
        print(f"  bundle 已保存: {bundle_dir}")
        self._save_reports(df_tr, selected, report, rules, output_path)

        return {
            'total_sessions': len(df),
            'n_train': n_train,
            'n_validation': n_val,
            'total_features': len(feat_cols),
            'selected_features': len(selected),
            'total_rules': len(rules),
            'rule_file': rule_path,
        }

    def _behavior_window_rows(self, record, pcap, max_read_packets,
                              include_background: bool = False,
                              context_override: Optional[Dict] = None) -> List[Dict]:
        """M2 行为窗口 -> 特征行（标签=区间重叠判定，含可选背景类）。"""
        from src.features.runtime import extract_behavior_feature_records
        from src.data_manifest import label_window
        ctx_cfg = dict(context_override or {})
        if not ctx_cfg:
            ctx_cfg = getattr(getattr(self.config, 'features', None),
                              'context_window', None) or {}
            ctx_cfg = dict(ctx_cfg) if isinstance(ctx_cfg, dict) else {}
        if not ctx_cfg:
            from src.parser.context import DEFAULT_CONTEXT_CONFIG
            ctx_cfg = dict(DEFAULT_CONTEXT_CONFIG)
        self._last_context = ctx_cfg   # 写入 bundle：推理端同配置重算窗口
        wrecs = extract_behavior_feature_records(
            str(pcap), ctx_cfg,
            terminal=getattr(record, 'terminal_ip', '') or None,
            capture_id=record.capture_id,
            max_read_packets=max_read_packets)
        rows = []
        vocab = self._behavior_vocab
        for w in wrecs:
            lab = label_window(record, w.observation.window_start,
                               w.observation.window_end)
            if lab == 'background' and not include_background:
                continue          # 首版口径：背景窗不入训练（计数见评价）
            if lab not in vocab:
                vocab[lab] = len(vocab)
            row = {k: (v if v is not None else -1.0)
                   for k, v in w.features.items()}
            row['_observation_id'] = w.observation.observation_id
            row['_source_file'] = str(pcap)
            row['_label_id'] = vocab[lab]
            row['_label_name'] = lab
            rows.append(row)
        return rows

    def _tunnel_window_rows(self, record, pcap, max_read_packets,
                            label_id: int,
                            context_override: Optional[Dict] = None) -> List[Dict]:
        """Tunnel profile window rows; label is the manifest's official tool label."""
        from src.features.runtime import extract_tunnel_feature_records
        ctx_cfg = dict(context_override or {})
        self._last_context = ctx_cfg
        wrecs = extract_tunnel_feature_records(
            str(pcap), ctx_cfg,
            terminal=getattr(record, 'terminal_ip', '') or None,
            capture_id=record.capture_id,
            max_read_packets=max_read_packets)
        cap = int(getattr(self, '_last_profile_spec', {}).get(
            'max_windows_per_file', 0) or 0)
        if cap and len(wrecs) > cap:
            # Deterministic coverage of the whole capture; avoids a few very
            # long PCAPs dominating rule fitting while not peeking at labels.
            idx = np.linspace(0, len(wrecs) - 1, cap, dtype=int)
            wrecs = [wrecs[int(i)] for i in idx]
        rows = []
        for w in wrecs:
            row = {k: (v if v is not None else -1.0)
                   for k, v in w.features.items()}
            row['_observation_id'] = w.observation.observation_id
            row['_source_file'] = str(pcap)
            row['_label_id'] = int(label_id)
            row['_label_name'] = record.app
            rows.append(row)
        return rows

    def _behavior_flow_window_rows(self, record, pcap, max_read_packets,
                                   include_background: bool = False,
                                   context_override: Optional[Dict] = None) -> List[Dict]:
        """Long-flow 15s slices used by the frozen ISCXVPN Behavior profile."""
        from src.features.runtime import extract_flow_window_feature_records
        from src.data_manifest import label_window
        ctx_cfg = dict(context_override or {})
        self._last_context = ctx_cfg
        wrecs = extract_flow_window_feature_records(
            str(pcap), ctx_cfg, feature_kind='behavior',
            capture_id=record.capture_id, max_read_packets=max_read_packets)
        rows = []
        vocab = self._behavior_vocab
        for w in wrecs:
            lab = label_window(record, w.observation.window_start,
                               w.observation.window_end)
            if lab == 'background' and not include_background:
                continue
            if lab not in vocab:
                vocab[lab] = len(vocab)
            row = {k: (v if v is not None else -1.0)
                   for k, v in w.features.items()}
            row['_observation_id'] = w.observation.observation_id
            row['_source_file'] = str(pcap)
            row['_label_id'] = vocab[lab]
            row['_label_name'] = lab
            rows.append(row)
        return rows

    def _tunnel_flow_window_rows(self, record, pcap, max_read_packets,
                                 label_id: int,
                                 context_override: Optional[Dict] = None) -> List[Dict]:
        """Long-flow 15s slices used by the frozen ISCXTor Tunnel profile."""
        from src.features.runtime import extract_flow_window_feature_records
        ctx_cfg = dict(context_override or {})
        self._last_context = ctx_cfg
        wrecs = extract_flow_window_feature_records(
            str(pcap), ctx_cfg, feature_kind='tunnel',
            capture_id=record.capture_id, max_read_packets=max_read_packets)
        cap = int(getattr(self, '_last_profile_spec', {}).get(
            'max_windows_per_file', 0) or 0)
        if cap and len(wrecs) > cap:
            idx = np.linspace(0, len(wrecs) - 1, cap, dtype=int)
            wrecs = [wrecs[int(i)] for i in idx]
        rows = []
        for w in wrecs:
            row = {k: (v if v is not None else -1.0)
                   for k, v in w.features.items()}
            row['_observation_id'] = w.observation.observation_id
            row['_source_file'] = str(pcap)
            row['_label_id'] = int(label_id)
            row['_label_name'] = record.app
            rows.append(row)
        return rows

    def _replay_validation_raw(self, rules, selected, label_names,
                               val_rows: List[Dict]) -> Dict:
        """验证集回放：纯规则引擎在验证行上的成绩（字符串标签口径）。"""
        from src.engine.matcher.optimized_engine import OptimizedDPIEngine
        eng = OptimizedDPIEngine()
        eng.rules = [r for r in rules
                     if r.get('type') != 'ensemble_classifier']
        eng.label_names = dict(label_names)
        eng.selected_features = list(selected)
        eng.classifier = None
        # 与独立推理同阈值：回放阈值=生成器/bundle阈值（否则回放与 dpi_infer
        # 口径不一致，2026-09-18 第二轮）
        eng.confidence_threshold = getattr(
            self.rule_generator, 'confidence_threshold',
            eng.confidence_threshold)
        y_true_s, y_pred_s = [], []
        for row in val_rows:
            vec = {k: v for k, v in row.items() if not str(k).startswith('_')}
            matches = eng.match(vec)
            y_true_s.append(row.get('_label_name')
                            or label_names.get(row['_label_id'], '?'))
            y_pred_s.append(matches[0].result if matches else 'unknown')
        from src.evaluation import compute_metrics
        names = [label_names[i] for i in sorted(label_names)]
        ids = sorted(label_names)
        id_of = {n: i for i, n in zip(ids, names)}
        yt = [id_of.get(t, -1) for t in y_true_s]
        yp = [id_of.get(p, -1) for p in y_pred_s]
        m = compute_metrics(yt, yp, names, label_ids=ids)
        m.pop('per_class', None)
        m['confusion_true_to_pred'] = {
            t: {} for t in set(y_true_s) | set(y_pred_s)}
        for t, p in zip(y_true_s, y_pred_s):
            m['confusion_true_to_pred'].setdefault(t, {})
            m['confusion_true_to_pred'][t][p] = (
                m['confusion_true_to_pred'][t].get(p, 0) + 1)
        return m

    # ==================== 模式2：规则检测 ====================

    def detect(self, pcap_path: str, rule_file: str,
               max_packets: Optional[int] = None,
               max_read_packets: Optional[int] = None) -> List[Dict]:
        """
        规则检测模式：使用已有规则识别流量

        Args:
            pcap_path: PCAP文件路径（单文件或目录）
            rule_file: 规则包目录（bundle 三件套）或旧版 rules.yaml
            max_packets: 每会话前缀截断上限（默认 runtime 500）
            max_read_packets: 每文件真正的读包上限

        与独立推理同口径：bundle 加载（不载 sklearn/xgboost 模型）、
        共享 runtime 提取（<3包/<100B 过滤 + 前缀截断一致），
        不再走另一套未截断/未bundle入口（审计 2026-09-18）。
        """
        print("=" * 60)
        print("  高效DPI规则识别引擎")
        print("  模式：流量识别")
        print("=" * 60)

        # 加载规则：目录(bundle)优先，旧版 yaml 兼容
        print(f"\n加载规则: {rule_file}")
        rp = Path(rule_file)
        bundle_cfg = {}
        if (rp / 'rules.json').exists() or rp.is_dir():
            self.dpi_engine.load_rule_bundle(str(rp))
            from src.engine.model_io import load_rule_bundle as _load_bundle
            _, _, _, bundle_cfg = _load_bundle(str(rp))
        else:
            self.dpi_engine.load_rules(rule_file)
        print(f"  规则数: {len(self.dpi_engine.rules)}")

        from src.features.runtime import (DEFAULT_MAX_PACKETS,
                                          extract_feature_records_with_stats)
        from src.profile_specs import get_profile_spec, temporal_vote_rows
        btask = bundle_cfg.get('task', 'app')
        bprof = bundle_cfg.get('feature_profile', 'full')
        try:
            pspec = bundle_cfg.get('profile_spec') or get_profile_spec(bprof, btask)
        except Exception:
            pspec = {'observation_unit': 'flow', 'temporal_vote': 1}
        if max_packets is None:
            max_packets = (bundle_cfg.get('max_packets_per_session')
                           or pspec.get('max_packets'))
        if max_read_packets is None:
            max_read_packets = bundle_cfg.get('max_read_packets_per_file')
        _mp = int(max_packets or DEFAULT_MAX_PACKETS)
        ctx = pspec.get('context') or bundle_cfg.get('context') or {}
        obs_unit = pspec.get('observation_unit', 'flow')
        vote_n = int(pspec.get('temporal_vote',
                               bundle_cfg.get('temporal_vote', 1)) or 1)
        active_groups = {bprof} if bprof in ('fast16', 'fast32') else None

        # 解析PCAP（共享 runtime：过滤/截断与训练、独立推理一致）
        print(f"\n解析PCAP: {pcap_path}")
        results = []
        pp = Path(pcap_path)
        files = (sorted(pp.rglob('*.pcap')) + sorted(pp.rglob('*.pcapng'))
                 if pp.is_dir() else [pp])
        for f in files:
            print(f"  解析: {f.name}")
            if btask == 'behavior' and obs_unit == 'behavior_window':
                from src.features.runtime import extract_behavior_feature_records
                records = extract_behavior_feature_records(str(f), ctx)
                print(f"    behavior窗口: {len(records)}")
            elif btask == 'behavior' and obs_unit == 'flow_window':
                from src.features.runtime import extract_flow_window_feature_records
                records = extract_flow_window_feature_records(
                    str(f), ctx, feature_kind='behavior',
                    max_read_packets=max_read_packets)
                print(f"    behavior flow窗口: {len(records)}")
            elif btask == 'tool' and obs_unit == 'tunnel_window':
                from src.features.runtime import extract_tunnel_feature_records
                records = extract_tunnel_feature_records(str(f), ctx)
                print(f"    tunnel窗口: {len(records)}")
            elif btask == 'tool' and obs_unit == 'tunnel_flow_window':
                from src.features.runtime import extract_flow_window_feature_records
                records = extract_flow_window_feature_records(
                    str(f), ctx, feature_kind='tunnel',
                    max_read_packets=max_read_packets)
                cap = int(pspec.get('max_windows_per_file', 0) or 0)
                if cap and len(records) > cap:
                    idx = np.linspace(0, len(records) - 1, cap, dtype=int)
                    records = [records[int(i)] for i in idx]
                print(f"    tunnel flow窗口: {len(records)}")
            else:
                records, st = extract_feature_records_with_stats(
                    str(f), max_packets=_mp, max_read_packets=max_read_packets,
                    active_groups=active_groups)
                print(f"    会话(过滤后): {len(records)}（读包 {st.packets_read}，"
                      f"截断 {st.sessions_truncated}，失败 {st.extract_errors}）")
            file_rows = []
            for rec in records:
                feats = {k: (v if v is not None else -1.0)
                         for k, v in rec.features.items()}
                matches = self.dpi_engine.match(feats)
                if matches:
                    best = matches[0]
                    row = {
                        'observation_id': rec.observation.observation_id,
                        'source_file': f.name,
                        'app': best.app, 'behavior': best.behavior,
                        'result': best.result, 'confidence': best.confidence,
                        'rule_id': best.rule_id, 'rule_name': best.rule_name,
                    }
                else:
                    row = {
                        'observation_id': rec.observation.observation_id,
                        'source_file': f.name, 'app': 'unknown',
                        'behavior': 'unknown', 'result': 'unknown',
                        'confidence': 0.0, 'rule_id': '', 'rule_name': '',
                    }
                if rec.observation.window_start is not None:
                    row['window_start'] = rec.observation.window_start
                    row['window_end'] = rec.observation.window_end
                    row['session_index'] = rec.observation.session_index
                file_rows.append(row)
            if (btask == 'tool' and obs_unit in ('tunnel_window', 'tunnel_flow_window')
                    and vote_n > 1):
                file_rows = temporal_vote_rows(file_rows, vote_n,
                                               label_key='result')
            results.extend(file_rows)

        # 统计
        engine_stats = self.dpi_engine.get_statistics()
        print(f"\n识别完成:")
        print(f"  总会话数: {len(results)}")
        print(f"  已识别: {sum(1 for r in results if r['app'] != 'unknown')}")
        print(f"  未知: {sum(1 for r in results if r['app'] == 'unknown')}")
        print(f"  平均匹配耗时: {engine_stats['avg_match_time_ms']:.4f} ms/次")

        return results

    # ==================== 演示数据生成 ====================

    def _generate_demo_rules(self, output_path: Path) -> Dict:
        """生成演示规则（用于没有PCAP数据时的测试）"""
        demo_rules = [
            {
                'id': 'DEMO_001',
                'name': '微信视频通话识别',
                'app': 'WeChat',
                'behavior': 'video_call',
                'priority': 100,
                'confidence': 0.95,
                'conditions': [
                    {'feature': 'total_bytes', 'op': 'range', 'min': 50000, 'max': 10000000},
                    {'feature': 'fwd_bwd_byte_ratio', 'op': 'range', 'min': 0.05, 'max': 0.3},
                    {'feature': 'duration', 'op': 'range', 'min': 30.0, 'max': 3600.0},
                    {'feature': 'bwd_pkt_size_std', 'op': 'range', 'min': 200, 'max': 1500},
                ],
                'action': {'result': 'WeChat_VideoCall', 'confidence': 0.95, 'source': 'demo'}
            },
            {
                'id': 'DEMO_002',
                'name': '微信语音通话识别',
                'app': 'WeChat',
                'behavior': 'voice_call',
                'priority': 100,
                'confidence': 0.92,
                'conditions': [
                    {'feature': 'total_bytes', 'op': 'range', 'min': 10000, 'max': 2000000},
                    {'feature': 'fwd_bwd_byte_ratio', 'op': 'range', 'min': 0.3, 'max': 1.5},
                    {'feature': 'duration', 'op': 'range', 'min': 10.0, 'max': 3600.0},
                    {'feature': 'bwd_pkt_size_std', 'op': 'range', 'min': 50, 'max': 400},
                ],
                'action': {'result': 'WeChat_VoiceCall', 'confidence': 0.92, 'source': 'demo'}
            },
            {
                'id': 'DEMO_003',
                'name': 'Tor流量检测',
                'app': 'Tor',
                'behavior': 'anonymous_browsing',
                'priority': 200,
                'confidence': 0.90,
                'conditions': [
                    {'feature': 'tls_has_sni', 'op': '<=', 'value': 0},
                    {'feature': 'payload_byte_entropy', 'op': 'range', 'min': 4.0, 'max': 8.0},
                    {'feature': 'pkt_size_entropy', 'op': 'range', 'min': 3.0, 'max': 8.0},
                ],
                'action': {'result': 'Tor_Anonymous', 'confidence': 0.90, 'source': 'demo'}
            },
            {
                'id': 'DEMO_004',
                'name': 'WhatsApp视频通话',
                'app': 'WhatsApp',
                'behavior': 'video_call',
                'priority': 100,
                'confidence': 0.93,
                'conditions': [
                    {'feature': 'total_bytes', 'op': 'range', 'min': 100000, 'max': 15000000},
                    {'feature': 'fwd_bwd_byte_ratio', 'op': 'range', 'min': 0.03, 'max': 0.25},
                    {'feature': 'duration', 'op': 'range', 'min': 30.0, 'max': 7200.0},
                    {'feature': 'packet_rate', 'op': 'range', 'min': 50, 'max': 500},
                ],
                'action': {'result': 'WhatsApp_VideoCall', 'confidence': 0.93, 'source': 'demo'}
            },
            {
                'id': 'DEMO_005',
                'name': 'Instagram发帖',
                'app': 'Instagram',
                'behavior': 'post',
                'priority': 150,
                'confidence': 0.88,
                'conditions': [
                    {'feature': 'tls_has_sni', 'op': '>', 'value': 0},
                    {'feature': 'duration', 'op': 'range', 'min': 0.5, 'max': 30.0},
                    {'feature': 'total_fwd_bytes', 'op': 'range', 'min': 1000, 'max': 50000},
                ],
                'action': {'result': 'Instagram_Post', 'confidence': 0.88, 'source': 'demo'}
            },
        ]

        label_names = {
            0: 'benign',
            1: 'WeChat_VideoCall',
            2: 'WeChat_VoiceCall',
            3: 'Tor_Anonymous',
            4: 'WhatsApp_VideoCall',
            5: 'Instagram_Post',
        }

        self.rule_generator.rules = demo_rules
        self.rule_generator.export_yaml(
            str(output_path / 'rules.yaml'), label_names
        )

        return {
            'total_sessions': 0,
            'total_features': 0,
            'selected_features': 0,
            'total_rules': len(demo_rules),
            'rule_file': str(output_path / 'rules.yaml'),
            'note': '使用演示规则（未找到PCAP数据）'
        }

    # ==================== 报告保存 ====================

    def _save_reports(self, df: pd.DataFrame, selected: List[str],
                     report: Dict, rules: List[Dict], output_path: Path):
        """保存各类报告"""
        # 特征选择报告
        with open(str(output_path / 'feature_selection_report.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"  特征选择报告: {output_path / 'feature_selection_report.json'}")

        # 选中特征的统计
        if selected:
            selected_df = df[selected]
            stats_df = selected_df.describe()
            stats_df.to_csv(str(output_path / 'selected_feature_stats.csv'))
            print(f"  特征统计: {output_path / 'selected_feature_stats.csv'}")

        # 规则摘要
        with open(str(output_path / 'rules_summary.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False, default=str)
        print(f"  规则摘要: {output_path / 'rules_summary.json'}")

    # ==================== 工具方法 ====================

    def get_feature_names(self) -> List[str]:
        """获取所有特征名"""
        basic_names = list(self.basic_extractor._extract_session_info(
            type('', (), {'duration': 0, 'src_port': 0, 'dst_port': 0,
                         'protocol': type('', (), {'value': 6})()})()).keys())
        return basic_names


def main():
    parser = argparse.ArgumentParser(description='智能化加密流量特征挖掘工具')
    parser.add_argument('--mode', choices=['mine', 'detect', 'validate-data', 'evaluate', 'revise-rules'], default='mine',
                       help='运行模式：mine=特征挖掘, detect=规则检测')
    parser.add_argument('--input', help='输入PCAP文件或目录（mine目录模式/detect必填）')
    parser.add_argument('--output', default='output', help='输出目录')
    parser.add_argument('--rules', help='规则文件路径（detect模式）')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--manifest', help='试次清单jsonl（validate-data/evaluate/manifest驱动的mine）')
    parser.add_argument('--task', default='app', help='任务类型：app | behavior | tool')
    parser.add_argument('--truth', help='真值来源（evaluate模式：试次清单jsonl）')
    parser.add_argument('--observations',
                        help='预先固定的观测清单jsonl（evaluate模式：'
                             'observation_id+label，会话级批次分母，'
                             '缺预测行计错——2026-09-18第三轮）')
    parser.add_argument('--predictions', help='预测文件（evaluate模式）')
    parser.add_argument('--edits', help='规则编辑JSON（revise-rules模式）')
    parser.add_argument('--max-read-packets', type=int, default=None,
                        help='每文件真正的读包上限（读满即停，审计小批量用）')
    parser.add_argument('--max-packets', type=int, default=None,
                        help='每会话前缀截断上限（默认500；fast profile默认32）')
    parser.add_argument('--include-background', action='store_true',
                        help='behavior任务：背景窗作为显式类别入训练')
    parser.add_argument('--profile', default='full',
                        choices=['full', 'fast16', 'fast32',
                                 'application64', 'behavior15', 'tunnel15'],
                        help='特征profile：full=全量319维（兼容默认）；'
                             'fast16/fast32=低成本少量特征；'
                             'application64=单flow前64包且禁identifier；'
                             'behavior15=15秒行为窗；tunnel15=15秒隧道窗+3窗投票。'
                             'bundle携带profile契约，独立CLI自动继承。')

    args = parser.parse_args()

    # 加载配置
    config = AppConfig.from_yaml(args.config) if args.config else DEFAULT_CONFIG

    # 创建流水线
    pipeline = FeatureMiningPipeline(config)

    if args.mode == 'validate-data':
        if not args.manifest:
            print("错误：validate-data模式需要 --manifest")
            sys.exit(1)
        # 数据体检：清单校验 + pcap存在性 + 采集组统计（失败非零退出）
        from src.data_manifest import load_manifest, validate_manifest, build_label_vocab
        records = load_manifest(args.manifest)
        task = args.task or 'app'
        stats = validate_manifest(
            records, manifest_dir=str(Path(args.manifest).parent), task=task)
        vocab = build_label_vocab(records, task)
        print(json.dumps({**stats, 'labels': vocab['names']},
                         ensure_ascii=False, indent=2))
        return
    if args.mode == 'evaluate':
        # 评价只读真值清单与预测文件，不接触训练对象
        from src.data_manifest import load_manifest, build_label_vocab
        from src.evaluation import load_truth_manifest, evaluate_files
        if args.observations:
            # 观测级评价（2026-09-18 第三轮）：预先固定的观测清单=唯一分母
            from src.evaluation import (evaluate_observations,
                                        load_observation_list)
            if not args.predictions:
                print("错误：evaluate模式需要 --predictions（预测文件）")
                sys.exit(1)
            observations = load_observation_list(args.observations)
            metrics = evaluate_observations(args.predictions, observations,
                                             args.output)
            print(json.dumps({k: metrics.get(k) for k in
                              ('n_observations', 'n_missing_predictions',
                               'n_rejected', 'accuracy', 'macro_f1',
                               'macro_fpr')},
                             ensure_ascii=False, indent=2))
            return
        if not (args.truth and args.predictions):
            print("错误：evaluate模式需要 --truth（清单jsonl）与 --predictions（预测文件）")
            sys.exit(1)
        task = args.task or 'app'
        records = load_manifest(args.truth)
        truth = load_truth_manifest(records, task)
        label_map = build_label_vocab(records, task)['label_map']
        metrics = evaluate_files(args.predictions, truth, label_map,
                                 args.output, task=task)
        # M3: AnalysisPackage evaluate 层（混淆/误判/规则解释/方案/HTML）
        try:
            from src.reporting import build_evaluate_layer
            _cand = [args.rules] if args.rules else [
                str(Path(args.predictions).parent / 'bundle'),
                str(Path(args.predictions).parents[1] / 'mine' / 'bundle'),
                str(Path(args.predictions).parents[1] / 'bundle')]
            _rules_dir = next((c for c in _cand
                               if c and (Path(c) / 'rules.json').exists()), None)
            if _rules_dir is None:
                raise FileNotFoundError(
                    "未找到规则包（可用 --rules 指定 bundle 目录）")
            build_evaluate_layer(_rules_dir, args.predictions, truth,
                                 metrics, str(Path(args.output) / 'analysis'))
            print("  AnalysisPackage(evaluate层) 已生成")
        except Exception as _e:  # noqa: BLE001
            print(f"  警告：evaluate层分析包生成失败: {_e}")
        print(json.dumps({k: metrics.get(k) for k in
                          ('n_samples', 'n_rejected', 'accuracy',
                           'macro_f1', 'macro_fpr',
                           'n_unpaired_predictions')},
                         ensure_ascii=False, indent=2))
        return
    if args.mode == 'revise-rules':
        # M3: 人工修正（校验->回放->新版本发布，不覆盖原bundle）
        if not (args.rules and args.edits and args.manifest):
            print("错误：revise-rules模式需要 --rules --edits --manifest")
            sys.exit(1)
        _revise_rules(args.rules, args.edits, args.manifest, args.output,
                      args.task or None)
        return
    if args.mode == 'mine':
        if not (args.manifest or args.input):
            print("错误：mine模式需要 --manifest 或 --input")
            sys.exit(1)
        result = pipeline.mine(
            args.input, args.output,
            manifest=args.manifest, task=(args.task or 'app'),
            max_read_packets=args.max_read_packets,
            max_packets=args.max_packets,
            behavior_include_background=args.include_background,
            profile=args.profile)
        print(f"\n{'='*60}")
        print(f"  挖掘完成！")
        print(f"  规则文件: {result['rule_file']}")
        print(f"  生成规则数: {result['total_rules']}")
        print(f"  train/validation: {result['n_train']}/{result['n_validation']}")
        print(f"{'='*60}")

    elif args.mode == 'detect':
        if not (args.input and args.rules):
            print("错误：detect模式需要 --input 与 --rules")
            sys.exit(1)
        results = pipeline.detect(args.input, args.rules,
                                  max_packets=args.max_packets,
                                  max_read_packets=args.max_read_packets)

        # 输出结果
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(str(output_path / 'detection_results.csv'), index=False)
        print(f"\n结果已保存: {output_path / 'detection_results.csv'}")


if __name__ == '__main__':
    main()
