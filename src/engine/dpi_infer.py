"""
DPI独立推理工具
脱离训练流水线，直接加载模型包进行流量识别。

用法:
  # 从PCAP文件推理（完整流程：解析→特征提取→分类）
  python -m src.engine.dpi_infer --model output/results/ --pcap test.pcap

  # 从特征JSON推理（已有特征向量）
  python -m src.engine.dpi_infer --model output/results/ --features features.json

  # 批量推理（目录下所有PCAP）
  python -m src.engine.dpi_infer --model output/results/ --pcap-dir data/test/

  # 输出为JSON文件
  python -m src.engine.dpi_infer --model output/results/ --pcap test.pcap --output results.json
"""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from pathlib import Path


def extract_features_from_pcap(pcap_path: str, max_packets: int = 500,
                              active_groups=None, max_read_packets=None):
    """从PCAP文件提取特征向量（解析→会话→特征）。

    薄包装 src.features.runtime 共享入口（过滤/截断/提取单一实现；
    训练与推理 parity 由此保证，勿在此复制提取逻辑）。
    """
    from src.features.runtime import extract_feature_records_with_stats
    records, _stats = extract_feature_records_with_stats(
        pcap_path, max_packets=max_packets, active_groups=active_groups,
        max_read_packets=max_read_packets)
    return [r.features for r in records]


def extract_records_from_pcap(pcap_path: str, max_packets: int = 500,
                              active_groups=None, max_read_packets=None):
    """从PCAP文件提取特征记录（含 ObservationSpec；观察可追溯）。

    与 extract_features_from_pcap 同一共享入口；app 模式推理用本入口
    保留 observation_id（"{文件名}:{会话序号}"），评价端才能按观测去重
    与对账（2026-09-18 第二轮）。
    """
    from src.features.runtime import extract_feature_records_with_stats
    records, _stats = extract_feature_records_with_stats(
        pcap_path, max_packets=max_packets, active_groups=active_groups,
        max_read_packets=max_read_packets)
    return records


def infer_features(engine, features_list, output_path=None):
    """对特征向量列表执行推理"""
    all_results = []
    for i, feats in enumerate(features_list):
        matches = engine.match(feats)
        if matches:
            m = matches[0]
            result = {
                'session_id': i,
                'predicted_label': m.result,
                'predicted_app': m.app,
                'predicted_behavior': m.behavior,
                'confidence': round(m.confidence, 6),
                'source': m.source,
                'match_score': round(m.match_score, 4),
                'match_time_ms': round(m.match_time_ms, 4),
            }
        else:
            result = {
                'session_id': i,
                'predicted_label': 'unknown',
                'confidence': 0.0,
                'source': 'no_match',
            }
        all_results.append(result)

    # 统计
    total = len(all_results)
    matched = sum(1 for r in all_results if r['predicted_label'] != 'unknown')
    labels = {}
    for r in all_results:
        lbl = r['predicted_label']
        labels[lbl] = labels.get(lbl, 0) + 1

    print(f"\n  推理结果: {total} 个会话, {matched} 个已识别 ({matched/max(total,1)*100:.1f}%)")
    for lbl, cnt in sorted(labels.items(), key=lambda x: -x[1]):
        print(f"    {lbl}: {cnt}")

    # 输出
    if output_path:
        output = {
            'version': '1.0',
            'engine': 'OptimizedDPIEngine',
            'total_sessions': total,
            'matched': matched,
            'results': all_results,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  输出文件: {output_path}")

    return all_results


def _write_results(output_path, all_results, extra=None):
    """统一结果输出（results + 可选的算子调用计数等元信息）。"""
    if not output_path:
        return
    output = {
        'version': '1.0',
        'engine': 'OptimizedDPIEngine',
        'total_sessions': len(all_results),
        'results': all_results,
    }
    if extra:
        output.update(extra)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  输出文件: {output_path}")
    if extra and 'operator_calls' in extra:
        print(f"  算子调用计数: {extra['operator_calls']}")


def _operator_calls_extra(active_groups):
    """按需模式的算子调用计数（真实独立CLI的算子成本证据）。"""
    if active_groups is None:
        return None
    from src.features.operators import OPERATOR_CALLS
    return {'operator_calls': dict(OPERATOR_CALLS)}


def _infer_window_records(engine, records, source_file: str):
    """Shared behavior/tunnel window inference with traceable window metadata."""
    out = []
    for w in records:
        feats = {k: (v if v is not None else -1.0)
                 for k, v in w.features.items()}
        matches = engine.match(feats)
        if matches:
            m = matches[0]
            row = {
                'observation_id': w.observation.observation_id,
                'source_file': source_file,
                'window_start': w.observation.window_start,
                'window_end': w.observation.window_end,
                'session_index': w.observation.session_index,
                'terminal': w.observation.terminal,
                'predicted_label': m.result,
                'confidence': round(m.confidence, 6),
                'source': m.source,
            }
        else:
            row = {
                'observation_id': w.observation.observation_id,
                'source_file': source_file,
                'window_start': w.observation.window_start,
                'window_end': w.observation.window_end,
                'session_index': w.observation.session_index,
                'terminal': w.observation.terminal,
                'predicted_label': 'unknown', 'confidence': 0.0,
                'source': 'no_match',
            }
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(
        description='DPI独立推理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从PCAP文件推理
  python -m src.engine.dpi_infer --model output/results/ --pcap test.pcap

  # 从特征JSON推理
  python -m src.engine.dpi_infer --model output/results/ --features features.json

  # 批量推理目录下所有PCAP
  python -m src.engine.dpi_infer --model output/results/ --pcap-dir data/test/

  # 输出结果到文件
  python -m src.engine.dpi_infer --model output/results/ --pcap test.pcap -o result.json
        """
    )

    parser.add_argument('--model',
                        help='[对照模式] 旧模型目录（XGBoost模型包）')
    parser.add_argument('--rules',
                        help='[正式模式] 规则包目录（rules.json+selected_features.json+bundle_config.json）')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--pcap', help='单个PCAP/PCAPNG文件')
    input_group.add_argument('--pcap-dir', help='PCAP文件目录（批量推理）')
    input_group.add_argument('--features', help='特征向量JSON文件')

    parser.add_argument('-o', '--output', help='输出结果JSON文件路径')
    parser.add_argument('--terminal', help='可见终端IP（behavior任务窗口方向基准）')
    parser.add_argument('--on-demand', action='store_true',
                        help='按需计算：按bundle所需算子组提取（调用计数入输出）')
    parser.add_argument('--max-packets', type=int, default=None,
                        help='每会话最大包数（默认取 bundle 的 '
                             'max_packets_per_session，无 bundle 配置时 500）')
    parser.add_argument('--max-read-packets', type=int, default=None,
                        help='每文件真正的读包上限（读满即停；与mine同语义）')
    parser.add_argument('--confidence', type=float, default=None,
                        help='覆盖置信度阈值')

    args = parser.parse_args()

    # 加载引擎
    from src.engine.matcher.optimized_engine import OptimizedDPIEngine
    engine = OptimizedDPIEngine()
    bundle_task = 'app'          # 默认 app；--rules 时按 bundle 配置覆盖
    bundle_ctx = {}
    profile_spec = {'observation_unit': 'flow', 'temporal_vote': 1}
    temporal_vote = 1
    active_groups = None
    if args.rules:
        # 正式模式：独立规则推理（不加载预测器/训练库）
        engine.load_rule_bundle(args.rules)
        from src.engine.model_io import load_rule_bundle as _lrb
        _rules, _selected, _, _bcfg = _lrb(args.rules)
        bundle_task = _bcfg.get('task', 'app')
        from src.profile_specs import get_profile_spec
        _prof_default = _bcfg.get('feature_profile', 'full')
        try:
            profile_spec = (_bcfg.get('profile_spec')
                            or get_profile_spec(_prof_default, bundle_task))
        except Exception:
            profile_spec = {'observation_unit': 'flow', 'temporal_vote': 1}
        bundle_ctx = (profile_spec.get('context')
                      or _bcfg.get('context') or {})
        temporal_vote = int(profile_spec.get(
            'temporal_vote', _bcfg.get('temporal_vote', 1)) or 1)
        # 前缀包数默认读 bundle（训练32包时CLI不得悄悄用500；
        # 显式 --max-packets 优先）——2026-09-18 第三轮
        if args.max_packets is None:
            _mp = _bcfg.get('max_packets_per_session')
            if _mp:
                args.max_packets = int(_mp)
                print(f"  会话前缀包数(bundle默认): {args.max_packets}")
        if args.max_read_packets is None:
            _mr = _bcfg.get('max_read_packets_per_file')
            if _mr:
                args.max_read_packets = int(_mr)
                print(f"  文件读包上限(bundle默认): {args.max_read_packets}")
        # 特征profile默认读 bundle：fast bundle 的CLI默认只算profile特征
        # （非--on-demand时也如此；训练端同profile同前缀同缺失语义）
        if _prof_default in ('fast16', 'fast32'):
            active_groups = {_prof_default}
            print(f"  特征profile(bundle默认): {_prof_default}")
        if getattr(args, 'on_demand', False):
            from src.features.operators import (required_features_of_rules,
                                                resolve_required_families,
                                                reset_calls)
            from src.features.fast_profile import (profile_of_features,
                                                    FAST_PROFILES)
            reset_calls()
            _req = required_features_of_rules(_rules, _selected)
            # fast profile 优先：bundle 声明 profile 且覆盖依赖，或依赖
            # 整组可落入某 fast profile——只算 ≤32 个低成本特征；
            # 否则按族映射（同名特征同一批族方法，逐值一致）
            _prof_cfg = _bcfg.get('feature_profile')
            _prof_cov = profile_of_features(_req)
            if _prof_cfg in ('fast16', 'fast32') and set(_req) <= set(
                    FAST_PROFILES.get(_prof_cfg, ())):
                active_groups = {_prof_cfg}
            elif _prof_cov is not None:
                active_groups = {_prof_cov}
            else:
                active_groups = resolve_required_families(_req)
            print(f"  按需计算: 需要算子族 {sorted(active_groups)}"
                  f"（{len(_req)}特征）")
    elif args.model:
        # 对照模式：旧XGBoost模型包
        print("  [对照模式] 使用旧模型包（XGBoost预测，非正式DPI链路）")
        engine.load_model_dir(args.model)
    else:
        print("错误: 必须指定 --rules（规则包）或 --model（对照模式）")
        sys.exit(2)

    if args.max_packets is None:
        args.max_packets = 500      # 无 bundle 前缀配置时的旧默认

    if args.confidence is not None:
        engine.confidence_threshold = args.confidence

    t0 = time.time()

    if args.pcap:
        # 单文件PCAP推理
        print(f"\n  PCAP: {args.pcap}")
        obs_unit = profile_spec.get('observation_unit', 'flow')
        if bundle_task == 'behavior' and obs_unit == 'behavior_window':
            # M2: behavior bundle 必须走窗口模式（与训练端同参重算），
            # 否则单文件路径静默退化为 app 会话模式，口径不一致
            from src.features.runtime import extract_behavior_feature_records
            wrecs = extract_behavior_feature_records(
                args.pcap, bundle_ctx, terminal=args.terminal)
            print(f"  提取: {len(wrecs)} 个窗口")
            all_results = _infer_window_records(
                engine, wrecs, Path(args.pcap).name)
            _write_results(args.output, all_results,
                           {'profile_spec': profile_spec})
            print(f"\n  窗口推理: {len(all_results)} 个窗口, "
                  f"{sum(1 for r in all_results if r['predicted_label'] != 'unknown')} 个已识别")
        elif bundle_task == 'behavior' and obs_unit == 'flow_window':
            from src.features.runtime import extract_flow_window_feature_records
            wrecs = extract_flow_window_feature_records(
                args.pcap, bundle_ctx, feature_kind='behavior',
                max_read_packets=args.max_read_packets)
            all_results = _infer_window_records(
                engine, wrecs, Path(args.pcap).name)
            print(f"  提取: {len(wrecs)} 个flow窗口")
            _write_results(args.output, all_results,
                           {'profile_spec': profile_spec})
        elif bundle_task == 'tool' and obs_unit == 'tunnel_window':
            from src.features.runtime import extract_tunnel_feature_records
            from src.profile_specs import temporal_vote_rows
            wrecs = extract_tunnel_feature_records(
                args.pcap, bundle_ctx, terminal=args.terminal,
                max_read_packets=args.max_read_packets)
            raw_results = _infer_window_records(
                engine, wrecs, Path(args.pcap).name)
            all_results = temporal_vote_rows(
                raw_results, temporal_vote, label_key='predicted_label')
            print(f"  提取: {len(wrecs)} 个15s窗口 -> "
                  f"{len(all_results)} 个投票观测")
            _write_results(args.output, all_results, {
                'profile_spec': profile_spec,
                'raw_window_count': len(raw_results),
            })
        elif bundle_task == 'tool' and obs_unit == 'tunnel_flow_window':
            from src.features.runtime import extract_flow_window_feature_records
            from src.profile_specs import temporal_vote_rows
            wrecs = extract_flow_window_feature_records(
                args.pcap, bundle_ctx, feature_kind='tunnel',
                max_read_packets=args.max_read_packets)
            cap = int(profile_spec.get('max_windows_per_file', 0) or 0)
            if cap and len(wrecs) > cap:
                idx = np.linspace(0, len(wrecs) - 1, cap, dtype=int)
                wrecs = [wrecs[int(i)] for i in idx]
            raw_results = _infer_window_records(
                engine, wrecs, Path(args.pcap).name)
            all_results = temporal_vote_rows(
                raw_results, temporal_vote, label_key='predicted_label')
            print(f"  提取: {len(wrecs)} 个15s flow窗口 -> "
                  f"{len(all_results)} 个投票观测")
            _write_results(args.output, all_results, {
                'profile_spec': profile_spec,
                'raw_window_count': len(raw_results),
            })
        else:
            recs = extract_records_from_pcap(
                args.pcap, args.max_packets, active_groups=active_groups,
                max_read_packets=args.max_read_packets)
            print(f"  提取: {len(recs)} 个会话")
            features_list = [r.features for r in recs]
            all_results = infer_features(engine, features_list)
            for r, rec in zip(all_results, recs):
                r['observation_id'] = rec.observation.observation_id
                r['source_file'] = Path(args.pcap).name
            _write_results(args.output, all_results,
                           _operator_calls_extra(active_groups))

    elif args.pcap_dir:
        # 批量PCAP推理
        pcap_dir = Path(args.pcap_dir)
        pcap_files = sorted(pcap_dir.rglob('*.pcap')) + sorted(pcap_dir.rglob('*.pcapng'))  # 递归支持类别子目录
        print(f"\n  目录: {pcap_dir}")
        print(f"  文件: {len(pcap_files)} 个")

        all_results = []
        obs_unit = profile_spec.get('observation_unit', 'flow')
        if bundle_task == 'behavior' and obs_unit == 'behavior_window':
            # M2: 窗口模式（bundle context 配置驱动，与训练端同参重算）
            from src.features.runtime import extract_behavior_feature_records
            for f in pcap_files:
                print(f"\n  --- {f.name} (behavior windows) ---")
                wrecs = extract_behavior_feature_records(
                    str(f), bundle_ctx, terminal=args.terminal)
                print(f"  提取: {len(wrecs)} 个窗口")
                all_results.extend(_infer_window_records(engine, wrecs, f.name))
        elif bundle_task == 'behavior' and obs_unit == 'flow_window':
            from src.features.runtime import extract_flow_window_feature_records
            for f in pcap_files:
                print(f"\n  --- {f.name} (behavior flow windows) ---")
                wrecs = extract_flow_window_feature_records(
                    str(f), bundle_ctx, feature_kind='behavior',
                    max_read_packets=args.max_read_packets)
                all_results.extend(_infer_window_records(engine, wrecs, f.name))
        elif bundle_task == 'tool' and obs_unit == 'tunnel_window':
            from src.features.runtime import extract_tunnel_feature_records
            from src.profile_specs import temporal_vote_rows
            for f in pcap_files:
                print(f"\n  --- {f.name} (tunnel windows) ---")
                wrecs = extract_tunnel_feature_records(
                    str(f), bundle_ctx, terminal=args.terminal,
                    max_read_packets=args.max_read_packets)
                raw = _infer_window_records(engine, wrecs, f.name)
                voted = temporal_vote_rows(
                    raw, temporal_vote, label_key='predicted_label')
                print(f"  提取: {len(wrecs)} -> 投票观测 {len(voted)}")
                all_results.extend(voted)
        elif bundle_task == 'tool' and obs_unit == 'tunnel_flow_window':
            from src.features.runtime import extract_flow_window_feature_records
            from src.profile_specs import temporal_vote_rows
            for f in pcap_files:
                print(f"\n  --- {f.name} (tunnel flow windows) ---")
                wrecs = extract_flow_window_feature_records(
                    str(f), bundle_ctx, feature_kind='tunnel',
                    max_read_packets=args.max_read_packets)
                cap = int(profile_spec.get('max_windows_per_file', 0) or 0)
                if cap and len(wrecs) > cap:
                    idx = np.linspace(0, len(wrecs) - 1, cap, dtype=int)
                    wrecs = [wrecs[int(i)] for i in idx]
                raw = _infer_window_records(engine, wrecs, f.name)
                voted = temporal_vote_rows(
                    raw, temporal_vote, label_key='predicted_label')
                print(f"  提取: {len(wrecs)} -> 投票观测 {len(voted)}")
                all_results.extend(voted)
        else:
            for f in pcap_files:
                print(f"\n  --- {f.name} ---")
                recs = extract_records_from_pcap(
                    str(f), args.max_packets, active_groups=active_groups,
                    max_read_packets=args.max_read_packets)
                print(f"  提取: {len(recs)} 个会话")
                results = infer_features(engine, [r.features for r in recs])
                for r, rec in zip(results, recs):
                    r['observation_id'] = rec.observation.observation_id
                    r['source_file'] = f.name
                all_results.extend(results)

        extra = _operator_calls_extra(active_groups) or {}
        extra['profile_spec'] = profile_spec
        _write_results(args.output, all_results, extra)

    elif args.features:
        # 特征JSON推理
        print(f"\n  特征文件: {args.features}")
        with open(args.features, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            features_list = data
        elif isinstance(data, dict):
            features_list = [data]
        else:
            print("错误: 特征文件格式不正确"); return
        print(f"  加载: {len(features_list)} 个特征向量")
        infer_features(engine, features_list, args.output)

    elapsed = time.time() - t0
    stats = engine.get_statistics()
    print(f"\n  耗时: {elapsed:.2f}s, "
          f"平均匹配: {stats['avg_match_time_ms']:.2f}ms, "
          f"匹配率: {stats['match_rate']:.1%}")


if __name__ == '__main__':
    main()
