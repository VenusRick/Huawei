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

    def mine(self, pcap_dir: str, output_dir: str,
            allow_demo: bool = False,
            labels: Optional[Dict[str, int]] = None,
            manifest: Optional[str] = None,
            task: str = 'app') -> Dict:
        """
        特征挖掘模式：从PCAP文件挖掘特征并生成规则

        Args:
            pcap_dir: PCAP文件目录
            output_dir: 输出目录
            labels: 标签字典 {文件名: 标签ID}

        Returns:
            挖掘结果摘要
        """
        print("=" * 60)
        print("  智能化加密流量特征挖掘工具")
        print("  模式：特征挖掘与规则生成")
        print("=" * 60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: 解析PCAP文件
        print("\n[Step 1/5] 解析PCAP文件...")
        all_sessions = []
        if manifest is not None:
            # P0: manifest 驱动的正式挖掘（标签来自全局词表）
            from src.data_manifest import (load_manifest, validate_manifest,
                                           build_label_vocab)
            records = [r for r in load_manifest(manifest)
                       if r.split in (None, 'train', 'validation')]
            base = Path(manifest).parent
            validate_manifest(records, manifest_dir=str(base), task=task)
            vocab = build_label_vocab(records, task)
            labels = vocab['label_map']
            pcap_files = [Path(r.pcap) if Path(r.pcap).is_absolute()
                          else base / r.pcap for r in records]
        else:
            pcap_files = (list(Path(pcap_dir).rglob("*.pcap"))
                          + list(Path(pcap_dir).rglob("*.pcapng")))

        if not pcap_files:
            # R25: 正式挖掘空输入必须失败；演示数据仅经显式 allow_demo 触发
            if allow_demo:
                print(f"  警告：在 {pcap_dir} 中未找到PCAP文件，使用演示数据")
                return self._generate_demo_rules(output_path)
            raise FileNotFoundError(f"挖掘输入目录中未找到PCAP文件: {pcap_dir}")

        for pcap_file in pcap_files:
            print(f"  解析: {pcap_file.name}")
            sessions = self.pcap_reader.read_pcap(str(pcap_file))
            for s in sessions:
                s._source_file = str(pcap_file)  # A01: 标签按源文件回溯
            all_sessions.extend(sessions)

        stats = self.pcap_reader.get_statistics()
        print(f"  完成: {stats['total_packets']} 个包, "
              f"{len(all_sessions)} 个会话")

        if not all_sessions:
            # R25: 无有效会话视为输入错误，不再静默降级演示
            if allow_demo:
                return self._generate_demo_rules(output_path)
            raise ValueError("PCAP解析后无有效会话，无法进行正式挖掘")

        # R25: 正式挖掘必须提供标签（无标签静默全0是伪造数据）
        if not labels and not allow_demo:
            raise ValueError("正式挖掘需要标签映射（文件名→类别）；演示请显式使用 allow_demo")
        # Step 2: 特征提取
        print("\n[Step 2/5] 提取特征...")
        if task == 'behavior' and manifest is not None:
            # M2: 行为窗口样本（终端窗口/流段/关系特征 + 区间标签）
            from src.features.runtime import extract_behavior_feature_records
            from src.data_manifest import label_window
            ctx_cfg = getattr(getattr(self.config, 'features', None),
                              'context_window', None) or {}
            ctx_cfg = dict(ctx_cfg) if not isinstance(ctx_cfg, dict) else ctx_cfg
            if not ctx_cfg:
                from src.parser.context import DEFAULT_CONTEXT_CONFIG
                ctx_cfg = dict(DEFAULT_CONTEXT_CONFIG)
            self._last_context = ctx_cfg   # 写入 bundle：推理端同配置重算窗口
            feature_records = []
            label_series = []
            _vocab = {}
            for _r in records:
                wrecs = extract_behavior_feature_records(
                    str(_r_pcap_of(_r, base)), ctx_cfg,
                    terminal=getattr(_r, 'terminal_ip', '') or None,
                    capture_id=_r.capture_id)
                for w in wrecs:
                    lab = label_window(_r, w.observation.window_start,
                                       w.observation.window_end)
                    if lab == 'background':
                        continue          # 首版只学行为类，背景窗不入训练
                    if lab not in _vocab:
                        _vocab[lab] = len(_vocab)
                    feature_records.append(dict(w.features))
                    label_series.append(_vocab[lab])
            labels = {k: v for k, v in _vocab.items()}   # {行为名: ID}
            labels_inv = {v: k for k, v in _vocab.items()}
            df = pd.DataFrame(feature_records)
            y = pd.Series(label_series)
            feat_cols = [c for c in df.columns if not c.startswith('_')]
            df = df.fillna(-1.0)     # 训练侧哨兵填充；推理端同规则转换
            print(f"  行为窗口样本: {len(df)}, 特征: {len(feat_cols)},"
                  f" 类别: {sorted(_vocab)}")
        else:
            feature_records = []
            label_series = []
            _mp = int(getattr(getattr(self.config, 'parser', None),
                              'max_packets', 500) or 500)
            for i, session in enumerate(all_sessions):
                # 与 runtime/dpi_infer 一致的大会话截断（训练推理特征一致性）
                if len(session.packets) > _mp:
                    session.packets = session.packets[:_mp]
                    session.total_fwd_packets = sum(
                        1 for p in session.packets if p.direction == 1)
                    session.total_bwd_packets = sum(
                        1 for p in session.packets if p.direction == -1)
                    session.total_fwd_bytes = sum(
                        p.length for p in session.packets if p.direction == 1)
                    session.total_bwd_bytes = sum(
                        p.length for p in session.packets if p.direction == -1)
                # 基础特征
                basic_feats = self.basic_extractor.extract_all(session)
                # 高级特征
                advanced_feats = self.advanced_extractor.extract_all(session)
                # 合并
                all_feats = {**basic_feats, **advanced_feats}
                # 添加元信息
                all_feats['_session_idx'] = i
                all_feats['_src_ip'] = session.src_ip
                all_feats['_dst_ip'] = session.dst_ip
                all_feats['_protocol'] = 'TCP' if session.protocol.value == 6 else 'UDP'
                feature_records.append(all_feats)
                # 标签
                if labels:
                    pcap_name = None
                    for p_name in labels:
                        if p_name in str(getattr(session, '_source_file', '')):
                            pcap_name = p_name
                            break
                    label_series.append(labels.get(pcap_name, 0))
                else:
                    label_series.append(0)
            df = pd.DataFrame(feature_records)
            y = pd.Series(label_series)
            # 统计特征数量
            feat_cols = [c for c in df.columns if not c.startswith('_')]
            print(f"  总特征数: {len(feat_cols)}")
            print(f"  会话数: {len(df)}")
            # 保存原始特征
            df.to_csv(str(output_path / 'raw_features.csv'), index=False)
            print(f"  原始特征已保存: {output_path / 'raw_features.csv'}")
        # Step 3: 特征选择
        print("\n[Step 3/5] 智能特征选择...")
        self.feature_selector.fit(df[feat_cols], y, feat_cols)
        selected = self.feature_selector.selected_features_
        report = self.feature_selector.get_selection_report()
        print(f"  选中特征数: {len(selected)}")
        print(f"  特征有效率: {len(selected)}/{len(feat_cols)} "
              f"({len(selected)/len(feat_cols)*100:.1f}%)")
        print(f"  Top-5重要特征:")
        for feat, imp in report['importance_ranking'][:5]:
            print(f"    {feat}: {imp:.4f}")

        # Step 4: 规则生成
        print("\n[Step 4/5] 生成DPI规则...")
        if task == 'behavior' and manifest is not None:
            label_names = labels_inv
        else:
            label_names = {v: k for k, v in (labels or {}).items()}
        if not label_names:
            label_names = {0: 'benign', 1: 'unknown_app'}

        rules = self.rule_generator.fit_and_generate(  # R28: 调用真实存在的API
            df[feat_cols], y, selected, label_names
        )
        print(f"  生成规则数: {len(rules)}")

        # 导出规则文件
        rule_path = str(output_path / 'rules.yaml')
        self.rule_generator.export_yaml(rule_path, label_names)
        print(f"  规则文件已保存: {rule_path}")

        # Step 5: 保存特征定义和选择报告
        print("\n[Step 5/5] 保存报告...")
        # M3: AnalysisPackage mine 层（catalog/effectiveness/profiles）
        try:
            from src.reporting import build_mine_layer
            build_mine_layer(df, y, selected, rules, label_names,
                             str(output_path / 'analysis'),
                             confidence_threshold=getattr(
                                 self.rule_generator, 'confidence_threshold',
                                 0.7))
            print(f"  AnalysisPackage(mine层): {output_path / 'analysis'}")
        except Exception as _e:  # noqa: BLE001
            print(f"  警告：mine层分析包生成失败（不阻塞挖掘）: {_e}")
        # P0: 正式 bundle 三件套（独立推理的加载单位）
        from src.engine.model_io import save_rule_bundle
        bundle_dir = output_path / 'bundle'
        save_rule_bundle(rules, selected, label_names,
                         getattr(self.rule_generator, 'confidence_threshold', 0.7),
                         str(bundle_dir),
                         bundle_meta={'task': task or 'app', 'source': 'mine',
                                      'context': getattr(self, '_last_context', None)})
        print(f"  bundle 已保存: {bundle_dir}")
        self._save_reports(df, selected, report, rules, output_path)

        return {
            'total_sessions': len(all_sessions),
            'total_features': len(feat_cols),
            'selected_features': len(selected),
            'total_rules': len(rules),
            'rule_file': rule_path,
        }

    # ==================== 模式2：规则检测 ====================

    def detect(self, pcap_path: str, rule_file: str) -> List[Dict]:
        """
        规则检测模式：使用已有规则识别流量

        Args:
            pcap_path: PCAP文件路径
            rule_file: 规则文件路径

        Returns:
            识别结果列表
        """
        print("=" * 60)
        print("  高效DPI规则识别引擎")
        print("  模式：流量识别")
        print("=" * 60)

        # 加载规则
        print(f"\n加载规则: {rule_file}")
        self.dpi_engine.load_rules(rule_file)
        print(f"  规则数: {len(self.dpi_engine.rules)}")

        # 解析PCAP
        print(f"\n解析PCAP: {pcap_path}")
        sessions = self.pcap_reader.read_pcap(pcap_path)
        print(f"  会话数: {len(sessions)}")

        # 识别
        print("\n执行规则匹配...")
        results = []
        for session in sessions:
            # 提取特征
            basic_feats = self.basic_extractor.extract_all(session)
            advanced_feats = self.advanced_extractor.extract_all(session)
            all_feats = {**basic_feats, **advanced_feats}

            # 匹配
            matches = self.dpi_engine.match(all_feats)

            if matches:
                best_match = matches[0]  # 取最高优先级
                results.append({
                    'src_ip': session.src_ip,
                    'dst_ip': session.dst_ip,
                    'src_port': session.src_port,
                    'dst_port': session.dst_port,
                    'protocol': 'TCP' if session.protocol.value == 6 else 'UDP',
                    'duration': session.duration,
                    'total_bytes': session.total_bytes,
                    'app': best_match.app,
                    'behavior': best_match.behavior,
                    'result': best_match.result,
                    'confidence': best_match.confidence,
                    'rule_id': best_match.rule_id,
                    'rule_name': best_match.rule_name,
                })
            else:
                results.append({
                    'src_ip': session.src_ip,
                    'dst_ip': session.dst_ip,
                    'src_port': session.src_port,
                    'dst_port': session.dst_port,
                    'protocol': 'TCP' if session.protocol.value == 6 else 'UDP',
                    'duration': session.duration,
                    'total_bytes': session.total_bytes,
                    'app': 'unknown',
                    'behavior': 'unknown',
                    'result': 'unknown',
                    'confidence': 0.0,
                    'rule_id': '',
                    'rule_name': '',
                })

        # 统计
        engine_stats = self.dpi_engine.get_statistics()
        print(f"\n识别完成:")
        print(f"  总会话数: {len(sessions)}")
        print(f"  已识别: {sum(1 for r in results if r['app'] != 'unknown')}")
        print(f"  未知: {sum(1 for r in results if r['app'] == 'unknown')}")
        print(f"  平均匹配耗时: {engine_stats['avg_match_time_ms']:.4f} ms/包")

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
    parser.add_argument('--predictions', help='预测文件（evaluate模式）')
    parser.add_argument('--edits', help='规则编辑JSON（revise-rules模式）')

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
        if not (args.truth and args.predictions):
            print("错误：evaluate模式需要 --truth（清单jsonl）与 --predictions（预测文件）")
            sys.exit(1)
        task = args.task or 'app'
        records = load_manifest(args.truth)
        truth = load_truth_manifest(records, task)
        label_map = build_label_vocab(records, task)['label_map']
        metrics = evaluate_files(args.predictions, truth, label_map, args.output)
        # M3: AnalysisPackage evaluate 层（混淆/误判/规则解释/方案/HTML）
        try:
            from src.reporting import build_evaluate_layer
            _rules_dir = args.rules or (
                str(Path(args.predictions).parents[1] / 'bundle'))
            build_evaluate_layer(_rules_dir, args.predictions, truth,
                                 metrics, str(Path(args.output) / 'analysis'))
            print("  AnalysisPackage(evaluate层) 已生成")
        except Exception as _e:  # noqa: BLE001
            print(f"  警告：evaluate层分析包生成失败: {_e}")
        print(json.dumps({k: metrics[k] for k in
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
            manifest=args.manifest, task=(args.task or 'app'))
        print(f"\n{'='*60}")
        print(f"  挖掘完成！")
        print(f"  规则文件: {result['rule_file']}")
        print(f"  生成规则数: {result['total_rules']}")
        print(f"{'='*60}")

    elif args.mode == 'detect':
        if not (args.input and args.rules):
            print("错误：detect模式需要 --input 与 --rules")
            sys.exit(1)
        if not args.rules:
            print("错误：detect模式需要指定 --rules 参数")
            sys.exit(1)
        results = pipeline.detect(args.input, args.rules)

        # 输出结果
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(str(output_path / 'detection_results.csv'), index=False)
        print(f"\n结果已保存: {output_path / 'detection_results.csv'}")


if __name__ == '__main__':
    main()
