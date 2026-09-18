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
                              active_groups=None):
    """从PCAP文件提取特征向量（解析→会话→特征）"""
    from src.parser.pcap_reader import PCAPReader
    from src.parser.session.session_manager import SessionManager
    from src.features.basic.feature_extractor import BasicFeatureExtractor
    from src.features.advanced.advanced_extractor import AdvancedFeatureExtractor

    sm = SessionManager(tcp_timeout=300, udp_timeout=60)
    reader = PCAPReader(sm)
    sessions = reader.read_pcap(pcap_path)

    basic_ext = BasicFeatureExtractor()
    advanced_ext = AdvancedFeatureExtractor()

    results = []
    for session in sessions:
        if session.total_packets < 3:
            continue
        # 跳过无有效数据的会话（纯控制流）
        if session.total_bytes < 100:
            continue
        # 大会话截断（与训练流水线一致）
        if len(session.packets) > max_packets:
            session.packets = session.packets[:max_packets]
            session.total_fwd_packets = sum(1 for p in session.packets if p.direction == 1)
            session.total_bwd_packets = sum(1 for p in session.packets if p.direction == -1)
            session.total_fwd_bytes = sum(p.length for p in session.packets if p.direction == 1)
            session.total_bwd_bytes = sum(p.length for p in session.packets if p.direction == -1)
        try:
            if active_groups is None:
                feats = {**basic_ext.extract_all(session),
                         **advanced_ext.extract_all(session)}
            else:
                from src.features.operators import extract_on_demand
                feats = extract_on_demand(session, active_groups,
                                          basic_ext, advanced_ext)
            results.append(feats)
        except Exception:
            continue

    return results


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
    parser.add_argument('--max-packets', type=int, default=500,
                        help='每会话最大包数（默认500）')
    parser.add_argument('--confidence', type=float, default=None,
                        help='覆盖置信度阈值')

    args = parser.parse_args()

    # 加载引擎
    from src.engine.matcher.optimized_engine import OptimizedDPIEngine
    engine = OptimizedDPIEngine()
    if args.rules:
        # 正式模式：独立规则推理（不加载预测器/训练库）
        engine.load_rule_bundle(args.rules)
        from src.engine.model_io import load_rule_bundle as _lrb
        _rules, _selected, _, _bcfg = _lrb(args.rules)
        bundle_task = _bcfg.get('task', 'app')
        bundle_ctx = _bcfg.get('context') or {}
        active_groups = None
        if getattr(args, 'on_demand', False):
            from src.features.operators import (required_features_of_rules,
                                                resolve_required_operators,
                                                reset_calls, OPERATOR_CALLS)
            reset_calls()
            _req = required_features_of_rules(_rules, _selected)
            active_groups = resolve_required_operators(_req)
            print(f"  按需计算: 需要算子组 {sorted(active_groups)}")
    elif args.model:
        # 对照模式：旧XGBoost模型包
        print("  [对照模式] 使用旧模型包（XGBoost预测，非正式DPI链路）")
        engine.load_model_dir(args.model)
    else:
        print("错误: 必须指定 --rules（规则包）或 --model（对照模式）")
        sys.exit(2)

    if args.confidence is not None:
        engine.confidence_threshold = args.confidence

    t0 = time.time()

    if args.pcap:
        # 单文件PCAP推理
        print(f"\n  PCAP: {args.pcap}")
        features_list = extract_features_from_pcap(args.pcap, args.max_packets)
        print(f"  提取: {len(features_list)} 个会话")
        infer_features(engine, features_list, args.output)

    elif args.pcap_dir:
        # 批量PCAP推理
        pcap_dir = Path(args.pcap_dir)
        pcap_files = sorted(pcap_dir.rglob('*.pcap')) + sorted(pcap_dir.rglob('*.pcapng'))  # 递归支持类别子目录
        print(f"\n  目录: {pcap_dir}")
        print(f"  文件: {len(pcap_files)} 个")

        all_results = []
        if bundle_task == 'behavior':
            # M2: 窗口模式（bundle context 配置驱动，与训练端同参重算）
            from src.features.runtime import extract_behavior_feature_records
            for f in pcap_files:
                print(f"\n  --- {f.name} (behavior windows) ---")
                wrecs = extract_behavior_feature_records(
                    str(f), bundle_ctx, terminal=args.terminal)
                print(f"  提取: {len(wrecs)} 个窗口")
                for w in wrecs:
                    feats = {k: (v if v is not None else -1.0)
                             for k, v in w.features.items()}
                    matches = engine.match(feats)
                    if matches:
                        m = matches[0]
                        all_results.append({
                            'observation_id': w.observation.observation_id,
                            'source_file': f.name,
                            'window_start': w.observation.window_start,
                            'window_end': w.observation.window_end,
                            'predicted_label': m.result,
                            'confidence': round(m.confidence, 6),
                            'source': m.source,
                        })
                    else:
                        all_results.append({
                            'observation_id': w.observation.observation_id,
                            'source_file': f.name,
                            'window_start': w.observation.window_start,
                            'window_end': w.observation.window_end,
                            'predicted_label': 'unknown',
                            'confidence': 0.0, 'source': 'no_match'})
        else:
            for f in pcap_files:
                print(f"\n  --- {f.name} ---")
                features_list = extract_features_from_pcap(str(f), args.max_packets)
                print(f"  提取: {len(features_list)} 个会话")
                results = infer_features(engine, features_list)
                for r in results:
                    r['source_file'] = f.name
                all_results.extend(results)

        if args.output:
            output = {
                'version': '1.0',
                'engine': 'OptimizedDPIEngine',
                'total_sessions': len(all_results),
                'results': all_results,
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"\n  输出文件: {args.output}")

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
