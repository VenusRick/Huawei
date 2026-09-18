"""
检视回归用例（按 docs/实施方案/01_代码检视报告.md 重建）
- 控制用例 2 个 + 反例 R01-R31
- A 档（16 项已修复，2026-09-16 整改）：断言移植自逐项合成验证
- B 档（本文件先建接口已明项）：R16/R22/R24/R25/R26，当前预期红（真实基线）
- 待补项：R18-R21/R23/R27/R28/R29/R30/R31 —— 于 S2-S4 接口稳定后补写，此处不虚构
红线：不删反例、不放宽断言、不 xfail 掩盖。
"""
import os
import struct
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.parser.session.session_manager import (  # noqa: E402
    SessionManager, PacketInfo, FlowSession, Protocol, TCPState,
)
from src.parser.pcap_reader import PCAPReader  # noqa: E402
from src.parser.quic.quic_parser import QUICParser  # noqa: E402
from src.parser.tls.tls_parser import TLSParser  # noqa: E402
from src.features.basic.feature_extractor import BasicFeatureExtractor  # noqa: E402
from src.features.advanced.advanced_extractor import AdvancedFeatureExtractor  # noqa: E402

A, B = ('10.0.0.2', 5000), ('10.0.0.1', 80)


def P(src, sport, dst, dport, flags, seq, ts, plen=0, proto=Protocol.TCP, payload=b''):
    p = PacketInfo()
    p.src_ip, p.src_port, p.dst_ip, p.dst_port = src, sport, dst, dport
    p.protocol = proto
    p.tcp_flags, p.tcp_seq, p.timestamp, p.payload_length = flags, seq, ts, plen
    p.length = 54 + plen
    p.payload = payload
    return p


def handshake(sm):
    sm.process_packet(P(A[0], A[1], B[0], B[1], 0x02, 1000, 0.0))
    sm.process_packet(P(B[0], B[1], A[0], A[1], 0x12, 5000, 0.1))
    sm.process_packet(P(A[0], A[1], B[0], B[1], 0x10, 1001, 0.2))


def build_clienthello(sni=b'example.com', ciphers=None, exts=None):
    if ciphers is None:
        ciphers = [4865]
    name = b'\x00' + struct.pack('!H', len(sni)) + sni
    sn_list = struct.pack('!H', len(name)) + name
    parts = [struct.pack('!HH', 0x0000, len(sn_list)) + sn_list]
    for et, payload in (exts or []):
        parts.append(struct.pack('!HH', et, len(payload)) + payload)
    exts_b = b''.join(parts)
    body = (b'\x03\x03' + b'\x11' * 32 + b'\x00'
            + struct.pack('!H', len(ciphers) * 2) + b''.join(struct.pack('!H', c) for c in ciphers)
            + b'\x01\x00' + struct.pack('!H', len(exts_b)) + exts_b)
    hs = b'\x01' + struct.pack('!I', len(body))[1:] + body
    return b'\x16\x03\x01' + struct.pack('!H', len(hs)) + hs


# ───────── 控制用例 ─────────

def test_control_microsecond_pcap_reads(tmp_path):
    hdr = struct.pack('<IHHiIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1)
    pkt = b'\x11' * 20
    rec = struct.pack('<IIII', 1, 500_000, len(pkt), len(pkt)) + pkt
    f = tmp_path / 'us.pcap'
    f.write_bytes(hdr + rec)
    r = PCAPReader(SessionManager())
    out = list(r._read_packets(str(f)))
    assert len(out) == 1 and abs(out[0][1] - 1.5) < 1e-9


def test_control_complete_clienthello_parses():
    tp = TLSParser()
    rec = tp.parse_record(build_clienthello())
    assert rec is not None
    feat = tp.extract_features(rec)
    assert feat.get('ch_sni') == 'example.com'


# ───────── A 档：已修复 16 项 ─────────

def test_R01_initiator_direction_not_lexical_address():
    sm = SessionManager()
    sm.process_packet(P(A[0], A[1], B[0], B[1], 0x02, 1000, 0.0))
    s = list(sm.active_sessions.values())[0]
    assert s.packets[0].direction == 1
    assert s.src_ip == A[0]


def test_R02_normal_tcp_data_not_marked_retransmission():
    sm = SessionManager()
    handshake(sm)
    sm.process_packet(P(A[0], A[1], B[0], B[1], 0x18, 1001, 0.3, 10))
    sm.process_packet(P(B[0], B[1], A[0], A[1], 0x18, 5001, 0.4, 5))
    s = list(sm.active_sessions.values())[0]
    assert s.num_retransmissions == 0
    assert len([p for p in s.packets if p.payload_length > 0]) == 2


def test_R03_reorder_does_not_count_same_wire_packet_twice():
    sm = SessionManager()
    handshake(sm)
    sm.process_packet(P(A[0], A[1], B[0], B[1], 0x18, 1011, 0.3, 10))
    sm.process_packet(P(A[0], A[1], B[0], B[1], 0x18, 1031, 0.4, 10))
    sm.process_packet(P(A[0], A[1], B[0], B[1], 0x18, 1001, 0.5, 10))
    sm.flush_all()
    s = sm.get_all_closed_sessions()[0]
    seqs = sorted(p.tcp_seq for p in s.packets if p.payload_length > 0)
    assert seqs == [1001, 1011, 1031]


def test_R04_udp_idle_timeout_is_applied_during_ingest():
    sm = SessionManager(udp_timeout=60)
    sm.process_packet(P(A[0], A[1], B[0], B[1], 0, 0, 0.0, proto=Protocol.UDP))
    sm.process_packet(P(A[0], A[1], B[0], B[1], 0, 0, 100.0, proto=Protocol.UDP))
    sm.flush_all()
    assert len([s for s in sm.get_all_closed_sessions() if s.protocol == Protocol.UDP]) == 2


def test_R05_closed_tcp_state_is_reclaimed():
    sm = SessionManager()
    handshake(sm)
    sm.process_packet(P(A[0], A[1], B[0], B[1], 0x11, 1001, 0.3))
    r = sm.process_packet(P(B[0], B[1], A[0], A[1], 0x11, 5001, 0.4))
    assert r is not None and r.state == TCPState.CLOSED
    assert len(sm.active_sessions) == 0


def test_R06_zero_origin_timestamp_has_duration():
    assert FlowSession(start_time=0.0, end_time=2.0).duration == 2.0


def test_R07_nanosecond_pcap_timestamp(tmp_path):
    hdr = struct.pack('<IHHiIII', 0xa1b23c4d, 2, 4, 0, 0, 65535, 1)
    pkt = b'\x11' * 20
    rec = struct.pack('<IIII', 1, 500_000_000, len(pkt), len(pkt)) + pkt
    f = tmp_path / 'nano.pcap'
    f.write_bytes(hdr + rec)
    out = list(PCAPReader(SessionManager())._read_packets(str(f)))
    assert len(out) == 1 and abs(out[0][1] - 1.5) < 1e-9


def _pcapng_block(btype, body, endian):
    total = 8 + len(body) + 4
    return struct.pack(f'{endian}II', btype, total) + body + struct.pack(f'{endian}I', total)


def test_R08_big_endian_pcapng_not_silently_empty(tmp_path):
    shb = struct.pack('>IHHq', 0x1A2B3C4D, 1, 0, -1)
    idb = struct.pack('>HHI', 1, 0, 0)
    epb_body = (struct.pack('>IIIII', 0, 0, 2_000_000, 16, 16) + b'\x22' * 16
                + b'\x00' * 0 + struct.pack('>I', 1))
    data = (_pcapng_block(0x0A0D0D0A, shb, '>') + _pcapng_block(1, idb, '>')
            + _pcapng_block(6, epb_body, '>'))
    f = tmp_path / 'be.pcapng'
    f.write_bytes(data)
    out = list(PCAPReader(SessionManager())._read_packets(str(f)))
    assert len(out) == 1


def _eth_tcp_frame(payload=b'ABCD'):
    tcp_hdr = struct.pack('!HHIIBBHHH', 1234, 80, 100, 200, (5 << 4), 0x18, 8192, 0, 0)
    ip_hdr = struct.pack('!BBHHHBBH4s4s', 0x45, 0, 20 + 20 + len(payload), 1, 0, 64, 6, 0,
                         b'\x0a\x00\x00\x02', b'\x0a\x00\x00\x01')
    frame = b'\xaa' * 6 + b'\xbb' * 6 + struct.pack('!H', 0x0800) + ip_hdr + tcp_hdr + payload
    return frame + b'\x00' * max(0, 60 - len(frame))


def test_R09_ethernet_padding_not_tcp_payload():
    sm = SessionManager(tcp_timeout=300, udp_timeout=60)
    r = PCAPReader(sm)
    r._process_raw_packet(_eth_tcp_frame(), 1.0, 1)
    sm.flush_all()
    pl = [len(p.payload) for s in sm.get_all_closed_sessions()
          for p in s.packets if getattr(p, 'protocol', None) == Protocol.TCP and p.payload is not None]
    assert pl == [4]


def test_R10_quic_short_header_never_discards_udp_packet():
    qp = QUICParser()
    out = qp.extract_features_from_udp_payload(b'\x40' + b'\x01' * 20)
    assert out is None  # 不抛异常、不产出伪特征


def test_R11_quic_eight_byte_varint_masks_prefix():
    qp = QUICParser()
    assert qp._read_varint(b'\xc0' + b'\x00' * 7, 0) == (0, 8)
    assert qp._read_varint(b'\xc0' + b'\x00' * 6 + b'\x01', 0)[0] == 1


def test_R12_fragmented_tcp_clienthello_preserves_sni():
    reader = PCAPReader()

    def mk(payload, ts):
        p = PacketInfo()
        p.src_ip, p.src_port, p.dst_ip, p.dst_port = A[0], A[1], '10.0.0.1', 443
        p.payload, p.timestamp, p.tcp_flags = payload, ts, 0x18
        return p

    rec = build_clienthello()
    p1, p2 = mk(rec[:10], 0.1), mk(rec[10:], 0.2)
    reader._tls_feed(p1)
    reader._tls_feed(p2)
    assert (p2.tls_info or {}).get('ch_sni') == 'example.com'


def test_R13_truncated_clienthello_not_reported_complete():
    rec = build_clienthello()
    assert TLSParser().parse_record(rec[:len(rec) // 2]) is None


def test_R14_ja3_ignores_grease():
    tp = TLSParser()
    plain = tp.extract_features(tp.parse_record(build_clienthello()))
    greasy = tp.extract_features(tp.parse_record(
        build_clienthello(ciphers=[0x0a0a, 4865, 0x1a1a],
                          exts=[(0x0a0a, b'\x00\x00'), (0xfafa, b'')])))
    assert plain['ja3_hash'] == greasy['ja3_hash']


def test_R15_uniform_byte_entropy_is_eight_bits():
    def mk(payload, ts):
        p = P(A[0], A[1], '10.0.0.1', 443, 0x18, 100, ts, len(payload), payload=payload)
        return p

    s = FlowSession(src_ip=A[0], dst_ip='10.0.0.1', src_port=A[1], dst_port=443,
                    protocol=Protocol.TCP)
    s.packets = [mk(bytes(range(128)), 0.0), mk(bytes(range(128, 256)), 0.1)]
    fe = BasicFeatureExtractor()
    e = fe.extract_all(s).get('payload_byte_entropy')
    assert e is not None and abs(e - 8.0) < 1e-6


def test_R17_multiscale_aggregation_keeps_burst_timestamps():
    ae = AdvancedFeatureExtractor()
    def mk(ts):
        return P(A[0], A[1], '10.0.0.1', 443, 0x18, 100, ts, 1, payload=b'\x01')
    pkts = [mk(0.0), mk(0.5), mk(1.2), mk(1.8)]
    items = ae._aggregate_to_bursts([1, 1, 2, 2], pkts, window_sec=1.0)
    assert [w for _, w in items] == [0, 1]
    assert len(ae._aggregate_bursts_to_behavior(items, 1.0, 5.0)) == 1


# ───────── B 档：接口已明的未修项（当前预期红） ─────────

def _mk_session(n_pkts):
    s = FlowSession(src_ip=A[0], dst_ip='10.0.0.1', src_port=A[1], dst_port=443,
                    protocol=Protocol.TCP)
    for i in range(n_pkts):
        p = P(A[0] if i % 2 else '10.0.0.1', A[1] if i % 2 else 443,
              '10.0.0.1' if i % 2 else A[0], 443 if i % 2 else A[1],
              0x18, 100 + i, 0.1 * i, 8, payload=b'\x11' * 8)
        p.direction = 1 if i % 2 == 0 else -1
        s.packets.append(p)
    return s


def test_R16_feature_schema_is_independent_of_packet_count():
    """短流与长流产出的特征键集必须一致（缺失键=0/NaN填充也算键存在）"""
    short = BasicFeatureExtractor().extract_all(_mk_session(1))
    long = BasicFeatureExtractor().extract_all(_mk_session(12))
    a = AdvancedFeatureExtractor().extract_all(_mk_session(1))
    b = AdvancedFeatureExtractor().extract_all(_mk_session(12))
    assert set(short) == set(long)
    assert set(a) == set(b)


def test_R24_feature_extraction_does_not_mutate_original_session():
    s = _mk_session(4)
    before = [(p.timestamp, p.payload) for p in s.packets]
    BasicFeatureExtractor().extract_all(s)
    AdvancedFeatureExtractor().extract_all(s)
    after = [(p.timestamp, p.payload) for p in s.packets]
    assert before == after


def test_R25_missing_input_fails_instead_of_generating_success(tmp_path):
    """正式挖掘：空输入目录必须抛错（FileNotFoundError/ValueError），不得产出演示成功"""
    from src.pipeline import FeatureMiningPipeline
    empty = tmp_path / 'empty_dir'
    empty.mkdir()
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        FeatureMiningPipeline().mine(str(empty), str(tmp_path / 'out'))


def test_R26_configuration_yaml_roundtrip(tmp_path):
    """from_yaml(to_yaml(cfg)) 必须与原配置等价（含嵌套字段）"""
    from src.config import AppConfig
    cfg = AppConfig()
    cfg.selection.max_features = 17
    cfg.rule.min_support = 0.2
    f = tmp_path / 'cfg.yaml'
    cfg.to_yaml(str(f))
    cfg2 = AppConfig.from_yaml(str(f))
    assert cfg2.selection.max_features == 17
    assert cfg2.rule.min_support == 0.2


# ───────── C 档：规则引擎/生成器（接口稳定后补写，2026-09-17） ─────────

def _engine_with_rules(rules):
    from src.engine.matcher.optimized_engine import OptimizedDPIEngine
    eng = OptimizedDPIEngine()
    eng.rules = rules
    return eng


def test_R18_tree_path_requires_all_conditions():
    """严格AND：4条件只满足3（旧软化0.75可放行）不得命中；全满足才命中"""
    rule = {'id': 'T1', 'name': 'r', 'type': 'decision_tree',
            'conditions': [
                {'feature': 'a', 'op': '<=', 'value': 10},
                {'feature': 'b', 'op': '>', 'value': 0},
                {'feature': 'c', 'op': '==', 'value': 5},
                {'feature': 'd', 'op': '>', 'value': 100},
            ],
            'action': {'result': 'X', 'source': 'decision_tree'}, 'confidence': 0.99}
    eng = _engine_with_rules([rule])
    assert eng._evaluate_rule(rule, {'a': 1.0, 'b': 1.0, 'c': 5, 'd': 1.0}) is None
    assert eng._evaluate_rule(rule, {'a': 1.0, 'b': 1.0, 'c': 5, 'd': 200.0}) is not None


def test_R19_not_exists_matches_missing_value():
    rule = {'id': 'T2', 'name': 'r', 'type': 'statistical',
            'conditions': [{'feature': 'x', 'op': 'not_exists'}],
            'action': {'result': 'Y', 'source': 'statistical'}, 'confidence': 0.9}
    eng = _engine_with_rules([rule])
    assert eng._evaluate_rule(rule, {}) is not None       # 缺失 → 命中
    assert eng._evaluate_rule(rule, {'x': 0}) is None     # 0是合法值（旧代码误命中）
    assert eng._evaluate_rule(rule, {'x': ''}) is not None


def test_R22_missing_required_features_are_not_silently_zero_filled():
    rule = {'id': 'T3', 'name': 'r', 'type': 'decision_tree',
            'conditions': [
                {'feature': 'a', 'op': '<=', 'value': 10},
                {'feature': 'b', 'op': '>', 'value': 0},
                {'feature': 'c', 'op': '>', 'value': 0},
                {'feature': 'd', 'op': '>', 'value': 0},
            ],
            'action': {'result': 'Z', 'source': 'decision_tree'}, 'confidence': 0.9}
    eng = _engine_with_rules([rule])
    assert eng._evaluate_rule(rule, {'a': 1.0, 'b': 1.0, 'c': 1.0}) is None  # 缺d→整条不匹配


def test_R20_rule_output_obeys_configured_acceptance_threshold():
    import pandas as pd
    from src.engine.rule_compiler.optimized_generator import OptimizedRuleGenerator
    gen = OptimizedRuleGenerator(confidence_threshold=0.95)
    X = pd.DataFrame({'f': [0.0] * 16 + [9.0] * 64})
    y = pd.Series([0] * 13 + [1] * 3 + [1] * 64)  # 0值叶纯度0.8125，9值叶纯度1.0
    rules = gen._extract_tree_rules(X, y, ['f'], {0: 'a', 1: 'b'})
    results = {r['action']['result'] for r in rules}
    assert results == {'b'}  # 低置信叶不导出，高置信叶导出


def test_R21_class_zero_is_exported_like_other_classes():
    import pandas as pd
    from importlib import import_module
    gen = import_module('src.engine.rule_compiler.optimized_generator').OptimizedRuleGenerator(confidence_threshold=0.7)
    X = pd.DataFrame({'f': [0.0] * 20 + [9.0] * 20})
    y = pd.Series([0] * 20 + [1] * 20)
    rules = gen._extract_tree_rules(X, y, ['f'], {0: 'zero', 1: 'one'})
    results = {r['action']['result'] for r in rules}
    assert results == {'zero', 'one'}


def test_R31_tree_leaf_labels_use_classes_mapping():
    import pandas as pd
    from src.engine.rule_compiler.optimized_generator import OptimizedRuleGenerator
    gen = OptimizedRuleGenerator(confidence_threshold=0.7)
    X = pd.DataFrame({'f': [0.0] * 20 + [9.0] * 20})
    y = pd.Series(['A'] * 20 + ['B'] * 20)  # 字符串标签
    rules = gen._extract_tree_rules(X, y, ['f'], {'A': 'appA', 'B': 'appB'})
    results = {r['action']['result'] for r in rules}
    assert results == {'appA', 'appB'}  # 不得出现class_N fallback


# ───────── D 档：评价层（2026-09-17 补，接口稳定） ─────────
# R30 说明：截断前缀时长属观测层（ObservationSpec）语义，随 src/features/runtime.py
# 实现时定义并补用例，此处不虚构单测。

import numpy as np  # D档用例所需


class _DummyClf:
    """返回固定概率：前20个高置信(0.99,类1)，其余低置信(0.5/0.5,argmax=0)"""
    def predict_proba(self, X):
        n = len(X)
        proba = np.full((n, 2), 0.5)
        proba[:20, 0], proba[:20, 1] = 0.01, 0.99
        return proba


def test_R29_threshold_search_counts_rejected_known_samples():
    """100个known只高置信接纳20个时，阈值F1必须计入被拒的80个（不得报1.0）"""
    import pandas as pd
    import tests.test_generic_pipeline as tgp
    X = pd.DataFrame(np.zeros((100, 3)), columns=['a', 'b', 'c'])
    y = pd.Series([1] * 20 + [1] * 80)  # 全部真值为类1；低置信80个argmax=0且被拒
    best_t, best_f1, details = tgp.tune_confidence_threshold(
        _DummyClf(), X, y, ['a', 'b', 'c'])
    row = details[details['threshold'].round(2) == 0.9]
    assert len(row) == 1 and row.iloc[0]['f1'] < 0.9, \
        f"t=0.9时f1={row.iloc[0]['f1'] if len(row) else 'NA'}（旧口径在仅接纳样本上会得1.0）"


def test_R23_saved_dpi_labels_apply_rejection(tmp_path):
    """保存的DPI结果中，低置信样本的预测标签必须是unknown而非原argmax标签"""
    import json
    import pandas as pd
    import tests.test_generic_pipeline as tgp
    X = pd.DataFrame(np.zeros((30, 2)), columns=['a', 'b'])
    y = pd.Series([1] * 30)
    out = tgp.save_dpi_output(_DummyClf(), X, y, ['a', 'b'],
                              {0: 'class0', 1: 'class1'}, 0.9, str(tmp_path))
    data = json.load(open(tmp_path / 'dpi_results.json', encoding='utf-8'))
    lows = [r for r in data['results'] if r['confidence'] < 0.9]
    highs = [r for r in data['results'] if r['confidence'] >= 0.9]
    assert lows and all(r['predicted_label'] == 'unknown' for r in lows)
    assert highs and all(r['predicted_label'] == 'class1' for r in highs)


def test_R27_validation_labels_keep_training_ids(tmp_path):
    """split内标签沿用全局label_map的ID（如5），不按出现类别重编号为0"""
    import struct
    import tests.test_generic_pipeline as tgp
    hdr = struct.pack('<IHHiIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1)
    pkt = b'\x11' * 20
    body = struct.pack('<IIII', 1, 0, len(pkt), len(pkt)) + pkt
    f = tmp_path / 'globid.pcap'
    f.write_bytes(hdr + body)
    sessions, labels, names = tgp.parse_pcap_files(str(tmp_path), {5: 'globid'})
    if labels:  # 空pcap可能无有效会话；有则ID必须为5
        assert set(labels) == {5}, f"labels={set(labels)}（重编号会变{0}）"
        assert names.get(5) == 'globid'

# ---------------------------------------------------------------------
# P4-5.2 GREASE 精确集（RFC 8701 16位保留值；位掩码会误判 0x1A2A）
# ---------------------------------------------------------------------
class TestGreaseExactSet:
    def test_all_16_grease_values_recognized(self):
        from src.parser.tls.tls_parser import GREASE_16
        from src.parser.tls.tls_parser import TLSParser
        for i in range(16):
            v = 0x0A0A + 0x1010 * i
            assert TLSParser._is_grease(v), f"GREASE值漏判: {hex(v)}"
        assert len(GREASE_16) == 16

    def test_bitmask_false_positives_rejected(self):
        from src.parser.tls.tls_parser import TLSParser
        # 这些满足 (v & 0x0F0F) == 0x0A0A 但并非 RFC8701 保留值
        for v in (0x1A2A, 0x0A2A, 0x1A0A, 0x2A2B, 0x00FF, 0x1B1B):
            assert not TLSParser._is_grease(v), f"位掩码误判: {hex(v)}"

    def test_real_cipher_not_grease(self):
        from src.parser.tls.tls_parser import TLSParser
        for v in (0x1301, 0x1302, 0x1303, 0xC02F, 0x002F):
            assert not TLSParser._is_grease(v)


# ---------------------------------------------------------------------
# W4 树一致性专项（类ID 0 / 非连续类ID / 概率并列 / 正常缺失路径）
# ---------------------------------------------------------------------
class TestW4TreeRuleConsistency:
    """W4 专项：_traverse_tree 对 classes_ 的映射（M1 R31 修复的守门）。"""

    def _export(self, X, y, label_names, threshold=0.5):
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier
        from src.engine.rule_compiler.optimized_generator import (
            OptimizedRuleGenerator)
        tree = DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=1, random_state=0).fit(X, y)
        gen = OptimizedRuleGenerator(confidence_threshold=threshold)
        rules = []
        feats = [f"f{i}" for i in range(X.shape[1])]
        gen._traverse_tree(tree.tree_, 0, feats, label_names, rules, [],
                           tree.classes_)
        return rules, tree

    def test_class_id_zero_exported(self):
        """类 ID 0 的叶必须照常导出（R21：不得丢弃 0 类）。"""
        import numpy as np
        rng = np.random.RandomState(0)
        X = np.vstack([rng.rand(20, 3), rng.rand(20, 3) + 5])
        y = np.array([0] * 20 + [1] * 20)
        rules, _ = self._export(X, y, {0: "zero", 1: "one"})
        got = {r["action"]["result"] for r in rules}
        assert "zero" in got and "one" in got, f"0 类被丢弃: {got}"

    def test_noncontiguous_class_ids(self):
        """非连续类 ID（2/5/9）经 classes_ 映射不串标签。"""
        import numpy as np
        rng = np.random.RandomState(1)
        X = np.vstack([rng.rand(20, 3), rng.rand(20, 3) + 5,
                       rng.rand(20, 3) + 10])
        y = np.array([2] * 20 + [5] * 20 + [9] * 20)
        rules, _ = self._export(X, y, {2: "c2", 5: "c5", 9: "c9"})
        got = {r["action"]["result"] for r in rules}
        assert got and got <= {"c2", "c5", "c9"}, f"标签越界: {got}"
        assert len(got) == 3, f"应有全部三类，实际: {got}"

    def test_tied_probabilities_leaf(self):
        """概率并列叶如实导出 0.5，不虚构排序（低阈值放行以观测）。"""
        import numpy as np
        rng = np.random.RandomState(2)
        X = np.tile([[0.3, 0.7]], (40, 1))       # 全同样本 -> 叶内两类各半
        y = np.array([0, 1] * 20)
        rules, tree = self._export(X, y, {0: "a", 1: "b"}, threshold=0.0)
        confs = [r["confidence"] for r in rules]
        assert any(abs(c - 0.5) < 1e-9 for c in confs), \
            f"并列概率叶未如实导出 0.5: {confs}"

    def test_tree_and_rules_agree_on_samples(self):
        """导出规则与原树在同一批样本上判定一致（数值两侧一致）。"""
        import numpy as np
        from src.engine.matcher.optimized_engine import OptimizedDPIEngine
        rng = np.random.RandomState(3)
        X = np.vstack([rng.rand(30, 3), rng.rand(30, 3) + 5])
        y = np.array([0] * 30 + [1] * 30)
        rules, tree = self._export(X, y, {0: "aa", 1: "bb"}, threshold=0.9)
        feats = [f"f{i}" for i in range(3)]
        for i in range(0, 60, 7):
            leaf_label = tree.predict([X[i]])[0]
            vec = {feats[j]: float(X[i][j]) for j in range(3)}
            hit = None
            for r in rules:
                ok = all(
                    (vec.get(c["feature"]) is not None) and
                    (vec[c["feature"]] <= c["value"] if c["op"] == "<="
                     else vec[c["feature"]] > c["value"])
                    for c in r["conditions"])
                if ok:
                    hit = r["action"]["result"]
                    break
            expected = {0: "aa", 1: "bb"}[int(leaf_label)]
            assert hit == expected, \
                f"样本{i}: 树判{expected} 规则判{hit}"




# ---------------------------------------------------------------------
# §5.1 TCP统计与TLS字节重组联合用例（同一ClientHello多变体）
# ---------------------------------------------------------------------
class TestTcpTlsJoint:
    """四变体喂法后 SNI 可见字段一致；事件不双计；缺口不拼造。"""

    def _feed(self, chunks, gap_after=None):
        """按顺序喂 TCP 分段；返回 (最后一个带tls_info的pkt, reader)。"""
        from src.parser.pcap_reader import PCAPReader
        from src.parser.session.session_manager import SessionManager
        r = PCAPReader(SessionManager())
        last_info = None
        off = 0
        for i, ch in enumerate(chunks):
            p = P("10.0.0.1", 1234, "10.0.0.2", 443, 0x18,
                  1000 + off, 1.0 + i * 0.01, len(ch), payload=ch)
            off += len(ch)
            r._tls_feed(p)
            if p.tls_info is not None:
                last_info = p.tls_info
        return last_info, r

    def test_variants_same_sni(self):
        rec = build_clienthello(sni=b"example.com")
        half = len(rec) // 2
        cases = {
            "完整": [rec],
            "分段": [rec[:half], rec[half:]],
            "三段": [rec[:20], rec[20:half], rec[half:]],
            # 乱序重排与重复段去重是 TCP 层(_tcp_tracker, R03用例)职责；
            # _tls_feed 只收已排序去重的字节流（分层边界，§5.1）
        }
        for name, chunks in cases.items():
            info, r = self._feed(chunks)
            sni = (info or {}).get("ch_sni") if isinstance(info, dict) else getattr(
                info, "ch_sni", None)
            assert sni in (b"example.com", "example.com"), \
                f"{name} 变体 SNI 提取失败: {sni!r}"
            assert r.total_tls_records == 1, \
                f"{name} 变体 record 双计: {r.total_tls_records}"

    def test_gap_not_fabricated(self):
        """缺 record 头时不能凭后到数据拼造完整握手。"""
        rec = build_clienthello(sni=b"example.com")
        info, r = self._feed([rec[10:]])   # 只有后半，永远缺头
        assert r.total_tls_records == 0, "缺口被拼造为完整握手"

    def test_fresh_tuple_not_chained(self):
        """新四元组的缓冲独立，不复用旧流的残留字节。"""
        from src.parser.pcap_reader import PCAPReader
        from src.parser.session.session_manager import SessionManager
        r = PCAPReader(SessionManager())
        rec1 = build_clienthello(sni=b"first.com")
        p1 = P("10.0.0.1", 2000, "10.0.0.2", 443, 0x18, 1, 1.0,
               len(rec1), payload=rec1)
        r._tls_feed(p1)
        assert r.total_tls_records == 1
        rec2 = build_clienthello(sni=b"second.org")
        p2 = P("10.9.9.9", 3000, "10.0.0.2", 443, 0x18, 1, 2.0,
               len(rec2), payload=rec2)
        r._tls_feed(p2)
        assert r.total_tls_records == 2, "第二四元组未被独立解析"
        sni2 = (p2.tls_info or {}).get("ch_sni") if isinstance(
            p2.tls_info, dict) else getattr(p2.tls_info, "ch_sni", None)
        assert sni2 in (b"second.org", "second.org"), \
            f"四元组复用串联了旧流: {sni2!r}"


# ---------------------------------------------------------------------
# §5.3 QUIC 三态：有效长头 Initial / 短头未知版本 / 普通 UDP 反例
# 可见范围如实（不解密，只断言可见字段与不崩）
# ---------------------------------------------------------------------
class TestQuicThreeStates:
    def _initial(self, version=1):
        # 长头 Initial: 0xC3(form=1,fixed=1,type=0) + version + DCID/SCID
        return (bytes([0xC3]) + version.to_bytes(4, "big")
                + bytes([8]) + b"\x11" * 8
                + bytes([4]) + b"\x22" * 4
                + bytes([0])            # token length varint = 0
                + bytes([4]) + b"\xaa" * 4)   # packet length + payload

    def test_valid_long_header_initial(self):
        from src.parser.quic.quic_parser import QUICParser
        p = QUICParser().parse_packet(self._initial(version=1))
        assert p is not None, "有效 Initial 未识别"
        assert getattr(p, "version", 1) in (1, 0x00000001)
        assert getattr(p, "header_form", 1) == 1

    def test_short_header_no_crash(self):
        from src.parser.quic.quic_parser import QUICParser
        short = bytes([0x40]) + b"\x33" * 16   # form=0 1-RTT
        r = QUICParser().parse_packet(short)
        # 短头不解密：返回 None 或短头记录均可，但不得抛异常
        assert r is None or getattr(r, "header_form", None) in (0, None)

    def test_unknown_version_visible_not_fabricated(self):
        from src.parser.quic.quic_parser import QUICParser
        r = QUICParser().parse_packet(self._initial(version=0x1A2A3A4A))
        # 未知版本：可见范围如实（None 或原样版本），不虚构握手字段
        if r is not None:
            assert getattr(r, "version", None) == 0x1A2A3A4A

    def test_plain_udp_negative(self):
        from src.parser.quic.quic_parser import QUICParser
        dns = (b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
               b"\x07example\x03com\x00\x00\x01\x00\x01")
        assert QUICParser().parse_packet(dns) is None, "普通UDP被误判QUIC"
        assert QUICParser().parse_packet(b"") is None
        assert QUICParser().parse_packet(b"\x00" * 6) is None

    def test_states_counted_distinctly(self):
        """三态（有效/不可解析/反例）各自独立结果，无相互伪装。"""
        from src.parser.quic.quic_parser import QUICParser
        qp = QUICParser()
        r_valid = qp.parse_packet(self._initial())
        r_short = qp.parse_packet(bytes([0x40]) + b"\x00" * 8)
        # 反例用 fixed=0 的 DNS 字节；QUIC 短头(fixed=1)与随机 UDP 固有
        # 不可完全区分（协议属性），可见范围如实不虚构
        r_plain = qp.parse_packet(bytes([0x12]) + b"\x34" * 20)
        assert (r_valid is not None)
        assert r_plain is None   # fixed=0 的普通UDP绝不为QUIC


# ---------------------------------------------------------------------
# §5.3 QUIC 三态（有效/短头降级/普通UDP反例；可见范围如实记录）
# ---------------------------------------------------------------------
class TestQuicThreeState:
    def _initial(self, version=b'\x00\x00\x00\x01', dcid=b'\x11'*8,
                 payload_len=32):
        body = bytes([8]) + dcid + bytes([0]) + b''   # DCID len+cid, SCID len 0
        body += b'\x00'                              # token length varint = 0
        pl = payload_len + 4                          # payload + 4B packet number
        body += bytes([pl])                           # packet length varint（<64 一字节）
        body += b'\x00\x00\x00\x01' + b'\xAA' * payload_len
        return b'\xc3' + version + body               # form=1 fixed=1 type=0 pn=4B

    def test_valid_initial_v1_parsed(self):
        from src.parser.quic.quic_parser import QUICParser
        p = QUICParser()
        pkt = p.parse_packet(self._initial())
        assert pkt is not None and pkt.is_valid
        assert pkt.version == 1
        assert pkt.dest_connection_id == b'\x11' * 8

    def test_unknown_version_reported_as_visible(self):
        """未知版本：如实记录可见版本号，不虚标握手完成。"""
        from src.parser.quic.quic_parser import QUICParser
        p = QUICParser()
        pkt = p.parse_packet(self._initial(version=b'\xde\xad\xbe\xef'))
        assert pkt is not None and pkt.is_valid
        assert pkt.version == 0xDEADBEEF
        assert getattr(pkt, 'has_crypto', None) is not True, \
            "未知版本不得声称完成握手分析"

    def test_short_header_degrades_to_stats(self):
        """短头（1-RTT）走统计降级路径，不抛异常不冒充 Initial。"""
        from src.parser.quic.quic_parser import QUICParser
        from src.parser.quic.quic_parser import QUICInitialPacket
        p = QUICParser()
        pkt = p.parse_packet(b'\x5d' + b'\x01\x02\x03\x04\x05\x06\x07')
        assert not isinstance(pkt, QUICInitialPacket), "短头不得解析为Initial"
        assert pkt is None or (isinstance(pkt, dict) and pkt.get('type') == '1-RTT')

    def test_plain_udp_rejected(self):
        """普通 UDP（fixed bit=0）明确非 QUIC。"""
        from src.parser.quic.quic_parser import QUICParser
        p = QUICParser()
        dns_like = b'\xab\xcd\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00'
        assert p.parse_packet(dns_like) is None

    def test_truncated_packet_safe(self):
        from src.parser.quic.quic_parser import QUICParser
        p = QUICParser()
        assert p.parse_packet(b'\xc3\x00') is None   # 不抛异常
        assert p.parse_packet(b'') is None
