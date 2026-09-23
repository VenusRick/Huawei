"""
测试脚本 - 验证各模块基本功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser.tls.tls_parser import TLSParser
from src.parser.quic.quic_parser import QUICParser, is_quic_packet
from src.parser.session.session_manager import (
    SessionManager, PacketInfo, Protocol, FlowSession
)
from src.features.basic.feature_extractor import BasicFeatureExtractor
from src.features.advanced.advanced_extractor import AdvancedFeatureExtractor
from src.engine.matcher.optimized_engine import OptimizedDPIEngine as DPIRuleEngine
from src.engine.rule_compiler.optimized_generator import OptimizedRuleGenerator as RuleGenerator
from src.features.selection.feature_selector import FeatureSelector
from src.pipeline import FeatureMiningPipeline

import numpy as np
import pandas as pd
import time


def test_tls_parser():
    """测试TLS解析器"""
    print("=" * 50)
    print("测试 TLS 解析器")
    print("=" * 50)

    parser = TLSParser()

    # 模拟一个ClientHello消息
    # 构建最小ClientHello
    client_hello = bytearray()

    # TLS版本: TLS 1.0 (0x0301)
    client_hello.extend(b'\x03\x03')

    # Random (32字节)
    client_hello.extend(b'\x00' * 32)

    # Session ID Length = 0
    client_hello.append(0)

    # Cipher Suites: 2个套件
    client_hello.extend(b'\x00\x04')  # 长度=4
    client_hello.extend(b'\x13\x01')  # TLS_AES_128_GCM_SHA256
    client_hello.extend(b'\x13\x02')  # TLS_AES_256_GCM_SHA384

    # Compression Methods: 无压缩
    client_hello.append(1)
    client_hello.append(0)

    # Extensions
    ext_data = bytearray()
    # SNI扩展
    sni = b'www.google.com'
    ext_data.extend(b'\x00\x00')  # 类型=server_name
    ext_data.extend((5 + len(sni)).to_bytes(2, 'big'))  # 长度
    ext_data.extend((3 + len(sni)).to_bytes(2, 'big'))  # SNI列表长度
    ext_data.append(0)  # 主机名类型
    ext_data.extend(len(sni).to_bytes(2, 'big'))
    ext_data.extend(sni)

    # Supported Versions扩展
    ext_data.extend(b'\x00\x2b')  # 类型=43
    ext_data.extend(b'\x00\x03')  # 长度=3
    ext_data.append(2)            # 版本列表长度=2
    ext_data.extend(b'\x03\x04')  # TLS 1.3

    client_hello.extend(len(ext_data).to_bytes(2, 'big'))
    client_hello.extend(ext_data)

    # 构建完整TLS记录
    tls_record = bytearray()
    tls_record.append(0x16)  # Handshake
    tls_record.extend(b'\x03\x01')  # TLS 1.0
    tls_record.extend((len(client_hello) + 4).to_bytes(2, 'big'))
    tls_record.append(1)  # ClientHello
    tls_record.extend((len(client_hello)).to_bytes(3, 'big'))
    tls_record.extend(client_hello)

    record = parser.parse_record(bytes(tls_record))

    assert record is not None and record.client_hello, "TLS ClientHello 解析失败"
    ch = record.client_hello
    print(f"  ✓ TLS版本: 0x{ch.tls_version:04x}")
    print(f"  ✓ SNI: {ch.sni}")
    print(f"  ✓ 密码套件数: {len(ch.cipher_suites)}")
    print(f"  ✓ 扩展数: {len(ch.extensions)}")
    print(f"  ✓ 支持版本: {[hex(v) for v in ch.supported_versions]}")
    print(f"  ✓ JA3哈希: {ch.ja3_hash}")
    print(f"  ✓ JA4哈希: {ch.ja4_hash[:16]}...")
    features = parser.extract_features(record)
    print(f"  ✓ 提取特征数: {len(features)}")
    assert ch.sni == 'www.google.com', f"SNI 不符: {ch.sni}"
    assert len(ch.cipher_suites) == 2
    assert len(features) > 0, "TLS 特征提取为空"

    print()


def test_session_manager():
    """测试会话管理器"""
    print("=" * 50)
    print("测试 会话管理器")
    print("=" * 50)

    sm = SessionManager(tcp_timeout=300, udp_timeout=60)

    # 模拟TCP三次握手
    ts = 1000.0

    # SYN
    syn = PacketInfo(
        timestamp=ts, src_ip='192.168.1.1', dst_ip='10.0.0.1',
        src_port=12345, dst_port=443, protocol=Protocol.TCP,
        length=66, tcp_flags=0x02, tcp_seq=1000
    )
    sm.process_packet(syn)

    # SYN-ACK
    syn_ack = PacketInfo(
        timestamp=ts + 0.05, src_ip='10.0.0.1', dst_ip='192.168.1.1',
        src_port=443, dst_port=12345, protocol=Protocol.TCP,
        length=66, tcp_flags=0x12, tcp_seq=5000, tcp_ack=1001
    )
    sm.process_packet(syn_ack)

    # ACK
    ack = PacketInfo(
        timestamp=ts + 0.1, src_ip='192.168.1.1', dst_ip='10.0.0.1',
        src_port=12345, dst_port=443, protocol=Protocol.TCP,
        length=54, tcp_flags=0x10, tcp_seq=1001, tcp_ack=5001
    )
    sm.process_packet(ack)

    # 数据包
    for i in range(10):
        data_pkt = PacketInfo(
            timestamp=ts + 0.2 + i * 0.1,
            src_ip='192.168.1.1' if i % 2 == 0 else '10.0.0.1',
            dst_ip='10.0.0.1' if i % 2 == 0 else '192.168.1.1',
            src_port=12345 if i % 2 == 0 else 443,
            dst_port=443 if i % 2 == 0 else 12345,
            protocol=Protocol.TCP,
            length=1000 + i * 100,
            payload_length=940 + i * 100,
            tcp_flags=0x18,  # PSH+ACK
            tcp_seq=1001 + i * 940,
        )
        sm.process_packet(data_pkt)

    print(f"  ✓ 活跃会话数: {sm.get_active_session_count()}")
    print(f"  ✓ 已处理包数: {len(sm.active_sessions) + sm.get_closed_session_count()}")

    sessions = sm.flush_all()
    print(f"  ✓ 关闭后会话数: {len(sessions)}")

    assert len(sessions) == 1, f"三次握手+数据应归并为1个会话, 实得 {len(sessions)}"
    s = sessions[0]
    print(f"  ✓ 首个会话: {s.src_ip}:{s.src_port} → {s.dst_ip}:{s.dst_port}")
    print(f"    持续时间: {s.duration:.3f}s")
    print(f"    总包数: {s.total_packets}")
    print(f"    前向/后向: {s.total_fwd_packets}/{s.total_bwd_packets}")
    assert s.src_ip == '192.168.1.1' and s.dst_ip == '10.0.0.1'
    # 合成数据的反向seq不连续，个别数据包按状态机计为重传被丢弃；
    # 不变量：计入会话的包数 = 输入13 - 重传数，且方向计数守恒。
    assert s.total_packets == 13 - s.num_retransmissions, \
        f"total_packets {s.total_packets} != 13 - retrans {s.num_retransmissions}"
    assert s.total_fwd_packets + s.total_bwd_packets == s.total_packets
    assert abs(s.duration - 1.1) < 1e-6, f"duration 应为1.1s, 实得 {s.duration}"
    assert len(s.packets) == s.total_packets

    print()


def test_feature_extraction():
    """测试特征提取"""
    print("=" * 50)
    print("测试 特征提取")
    print("=" * 50)

    # 创建测试会话
    session = FlowSession(
        src_ip='192.168.1.1', dst_ip='10.0.0.1',
        src_port=12345, dst_port=443, protocol=Protocol.TCP,
        start_time=1000.0, end_time=1010.0
    )

    # 模拟包
    for i in range(50):
        direction = 1 if i % 3 != 0 else -1
        pkt = PacketInfo(
            timestamp=1000.0 + i * 0.2,
            src_ip='192.168.1.1' if direction == 1 else '10.0.0.1',
            dst_ip='10.0.0.1' if direction == 1 else '192.168.1.1',
            src_port=12345 if direction == 1 else 443,
            dst_port=443 if direction == 1 else 12345,
            protocol=Protocol.TCP,
            length=200 + np.random.randint(0, 1300),
            payload_length=140 + np.random.randint(0, 1240),
            tcp_flags=0x18,
            direction=direction,
        )
        session.packets.append(pkt)
        if direction == 1:
            session.total_fwd_packets += 1
            session.total_fwd_bytes += pkt.length
        else:
            session.total_bwd_packets += 1
            session.total_bwd_bytes += pkt.length

    # 基础特征
    basic_extractor = BasicFeatureExtractor()
    basic_feats = basic_extractor.extract_all(session)
    print(f"  ✓ 基础特征数: {len(basic_feats)}")

    # 显示部分特征
    for key in ['duration', 'total_packets', 'total_bytes', 'fwd_bwd_byte_ratio',
                'iat_mean', 'pkt_size_mean', 'pkt_size_entropy', 'payload_byte_entropy']:
        if key in basic_feats:
            print(f"    {key}: {basic_feats[key]:.4f}")

    # 高级特征
    advanced_extractor = AdvancedFeatureExtractor()
    advanced_feats = advanced_extractor.extract_all(session)
    print(f"  ✓ 高级特征数: {len(advanced_feats)}")

    all_feats = {**basic_feats, **advanced_feats}
    print(f"  ✓ 总特征数: {len(all_feats)}")

    assert len(basic_feats) > 0 and len(advanced_feats) > 0, "特征提取为空"
    assert len(all_feats) == len(basic_feats) + len(advanced_feats), \
        "基础/高级特征键冲突"
    for key in ['duration', 'total_packets', 'total_bytes',
                'fwd_bwd_byte_ratio', 'iat_mean', 'pkt_size_mean']:
        assert key in basic_feats, f"缺少基础特征 {key}"
        assert np.isfinite(basic_feats[key]), f"{key} 非有限值"

    print()


def test_feature_selection():
    """测试特征选择"""
    print("=" * 50)
    print("测试 特征选择")
    print("=" * 50)

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 200
    n_features = 30

    feature_names = [f'feature_{i}' for i in range(n_features)]
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    # 让前5个特征与标签有强关联
    y = (X['feature_0'] * 2 + X['feature_1'] * 1.5 +
         X['feature_2'] * 0.5 + np.random.randn(n_samples) * 0.3 > 0).astype(int)

    selector = FeatureSelector(max_features=10)
    selector.fit(X, y, feature_names)

    report = selector.get_selection_report()
    print(f"  ✓ 选中特征数: {report['num_selected']}")
    print(f"  ✓ 选中特征: {report['selected_features'][:5]}...")
    print(f"  ✓ Top-3重要性:")
    for feat, imp in report['importance_ranking'][:3]:
        print(f"    {feat}: {imp:.4f}")

    assert len(selector.selected_features_) > 0, "特征选择结果为空"
    assert 0 < report['num_selected'] <= 10
    assert set(report['selected_features']) == set(selector.selected_features_)
    # 与标签强关联的 feature_0/1/2 应至少有一个入选
    assert {'feature_0', 'feature_1', 'feature_2'} & set(
        selector.selected_features_), "强关联特征未入选"

    print()


def test_dpi_engine():
    """测试DPI引擎"""
    print("=" * 50)
    print("测试 DPI引擎")
    print("=" * 50)

    engine = DPIRuleEngine()

    # 创建临时规则文件
    import tempfile, yaml
    rule_data = {
        'version': '2.0',
        'classifier': {
            'type': 'xgboost_ensemble',
            'confidence_threshold': 0.65,
            'num_classes': 2,
            'label_names': {0: 'TestApp', 1: 'AnonApp'},
        },
        'rules': [
            {
                'id': 'TEST_001', 'name': '测试规则1',
                'type': 'decision_tree',
                'app': 'TestApp', 'behavior': 'test_behavior',
                'priority': 100, 'confidence': 0.95,
                'conditions': [
                    {'feature': 'total_bytes', 'op': 'range', 'min': 1000, 'max': 50000},
                    {'feature': 'duration', 'op': 'range', 'min': 1.0, 'max': 60.0},
                ],
                'action': {'result': 'TestApp', 'confidence': 0.95, 'source': 'decision_tree'}
            },
        ],
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(rule_data, f, allow_unicode=True)
        tmp_path = f.name

    engine.load_rules(tmp_path)
    os.unlink(tmp_path)
    print(f"  ✓ 加载规则数: {len(engine.rules)}")
    assert len(engine.rules) == 1

    # 测试匹配
    test_features = {
        'total_bytes': 25000,
        'duration': 10.0,
        'tls_has_sni': 1,
    }
    matches = engine.match(test_features)
    print(f"  ✓ 匹配结果: {len(matches)} 条")
    assert matches, "满足全部条件的会话未命中规则"
    print(f"    最佳: {matches[0].result} (置信度: {matches[0].confidence})")
    assert matches[0].result == 'TestApp'
    assert matches[0].confidence >= 0.65

    # 反例：越界不命中（严格AND）
    bad = engine.match({'total_bytes': 999999, 'duration': 10.0})
    assert not bad, "越界特征不应命中"

    stats = engine.get_statistics()
    print(f"  ✓ 引擎统计: 平均匹配时间 {stats['avg_match_time_ms']:.4f} ms")

    print()


def test_end_to_end():
    """端到端测试"""
    print("=" * 50)
    print("测试 端到端流水线")
    print("=" * 50)

    pipeline = FeatureMiningPipeline()

    # 测试挖掘模式（无PCAP，使用演示数据）。
    # 输出隔离：不再写 output/results（那是历史正式产物目录），
    # 固定写审计测试目录，可用环境变量 TEST_ALL_OUTPUT 覆盖。
    out_dir = os.environ.get(
        'TEST_ALL_OUTPUT',
        'output/chain_audit_20260918/test_all_out')
    result = pipeline.mine(
        pcap_dir='data/pcaps',
        output_dir=out_dir,
        allow_demo=True  # 显式演示模式（R25后demo不再隐式触发）
    )
    print(f"  ✓ 挖掘结果:")
    print(f"    规则数: {result['total_rules']}")
    print(f"    规则文件: {result['rule_file']}")
    if 'note' in result:
        print(f"    备注: {result['note']}")
    assert result['total_rules'] > 0, "演示规则生成失败"
    assert os.path.exists(result['rule_file']), "规则文件未落盘"

    # 测试检测模式
    if result['total_rules'] > 0:
        stats = pipeline.dpi_engine.get_statistics()
        print(f"  ✓ 引擎已就绪")

    print()


def main():
    print("\n" + "★" * 50)
    print("  智能化加密流量特征挖掘工具 - 测试套件")
    print("★" * 50 + "\n")

    tests = [
        ("TLS解析器", test_tls_parser),
        ("会话管理器", test_session_manager),
        ("特征提取", test_feature_extraction),
        ("特征选择", test_feature_selection),
        ("DPI引擎", test_dpi_engine),
        ("端到端流水线", test_end_to_end),
    ]

    results = []
    for name, test_func in tests:
        # 通过标准 = 断言全部成立正常返回（函数不再返回布尔值，
        # 断言失败抛异常计为失败——pytest 与脚本入口同一口径）
        try:
            start = time.time()
            test_func()
            elapsed = time.time() - start
            results.append((name, True, elapsed))
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, 0))

    print("=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    for name, success, elapsed in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {status} | {name} ({elapsed:.2f}s)")

    total = len(results)
    passed = sum(1 for _, s, _ in results if s)
    print(f"\n  通过: {passed}/{total}")

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
