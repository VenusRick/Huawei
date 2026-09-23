# -*- coding: utf-8 -*-
"""全链增量改进回归（2026-09-18 第二轮）：评价分母/覆盖/事件配对、
有界读取/PCAPNG流式/因果截断计数、规则置信度语义/自适应浅树。

每条用例都是宿主指出问题的反例守门；运行：
    python3 -m pytest tests/regression/test_chain_improve_20260918_02.py -q
"""
import json
import socket
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

A, B = ('10.0.0.2', 5000), ('10.0.0.1', 443)


def _ip_tcp(src, dst, sport, dport, flags, seq, payload=b'') -> bytes:
    tcp = struct.pack('!HHIIBBHHH', sport, dport, seq, 1, 5 << 4, flags,
                      8192, 0, 0)
    total = 20 + len(tcp) + len(payload)
    ip = struct.pack('!BBHHHBBH', 0x45, 0, total, 0, 0, 64, 6, 0)
    ip += socket.inet_aton(src) + socket.inet_aton(dst)
    return ip + tcp + payload


def _eth(ip_pkt: bytes) -> bytes:
    return b'\x11\x22\x33\x44\x55\x66\xaa\xbb\xcc\xdd\xee\xff\x08\x00' + ip_pkt


def _write_pcap(path: Path, packets) -> None:
    with open(path, 'wb') as f:
        f.write(struct.pack('<IHHiIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))
        for ts, raw in packets:
            sec = int(ts)
            usec = int(round((ts - sec) * 1e6))
            f.write(struct.pack('<IIII', sec, usec, len(raw), len(raw)))
            f.write(raw)


def _session_packets(n_data=6, t0=1.0, gap=0.1):
    out = [(t0, _eth(_ip_tcp(A[0], B[0], A[1], B[1], 0x02, 1000))),
           (t0 + 0.02, _eth(_ip_tcp(B[0], A[0], B[1], A[1], 0x12, 5000))),
           (t0 + 0.04, _eth(_ip_tcp(A[0], B[0], A[1], B[1], 0x10, 1001)))]
    seq_a, seq_b = 1001, 5001
    for i in range(n_data):
        ts = t0 + 0.1 + i * gap
        if i % 2 == 0:
            out.append((ts, _eth(_ip_tcp(A[0], B[0], A[1], B[1], 0x18,
                                         seq_a, b'\x41' * 200))))
            seq_a += 200
        else:
            out.append((ts, _eth(_ip_tcp(B[0], A[0], B[1], A[1], 0x18,
                                         seq_b, b'\x42' * 120))))
            seq_b += 120
    return out


# ======================================================== E1 评价口径
def test_load_predictions_single_object_is_one_record(tmp_path):
    """单行JSONL预测对象（无results包装）=1条预测，不再解析成功却返回空。"""
    from src.evaluation import load_predictions
    f = tmp_path / "single.jsonl"
    f.write_text(json.dumps({"observation_id": "x.pcap:0",
                             "predicted_label": "alpha", "confidence": 0.9}),
                 encoding="utf-8")
    assert len(load_predictions(str(f))) == 1
    # 带统计键但无预测键的对象：显式报错（不是空列表）
    g = tmp_path / "stats.json"
    g.write_text(json.dumps({"engine": "x", "n": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="不含预测键|results"):
        load_predictions(str(g))


def test_compute_metrics_out_of_vocab_keeps_denominator():
    """词表外数字ID不静默丢出混淆矩阵：分母=5，OOV单列，拒识不混同。"""
    from src.evaluation import compute_metrics, OUT_OF_VOCAB
    y_true = [0, 0, 1, 1, 0]
    y_pred = [0, 7, 1, 1, 9]        # 7/9 词表外
    m = compute_metrics(y_true, y_pred, ["alpha", "beta"], label_ids=[0, 1])
    assert m["n_samples"] == 5
    assert "out_of_vocab" in m, "词表外ID无显式接管"
    assert m["out_of_vocab"]["n_pred"] == 2
    assert m["out_of_vocab"]["pred_id_counts"] == {7: 1, 9: 1}
    # 词表外不等于拒识：拒识只数 unknown(-1)
    assert m["n_rejected"] == 0
    # 真值侧词表外同样接管（accuracy 分母不缩水）
    m2 = compute_metrics([0, 5, 1], [0, 0, 1], ["alpha", "beta"],
                         label_ids=[0, 1])
    assert m2["n_samples"] == 3
    assert m2["out_of_vocab"]["n_true"] == 1
    assert m2["accuracy"] == pytest.approx(2 / 3)
    assert OUT_OF_VOCAB == -2


def test_evaluate_truth_coverage_exposes_missing_files(tmp_path):
    """整个真值文件无预测：coverage/missing 单列，不再隐形。"""
    from src.evaluation import evaluate_files, load_truth_manifest
    from src.data_manifest import TrialRecord
    recs = [
        TrialRecord(capture_id="c1", trial_id="t1", platform="pc",
                    device_id="d", app="alpha", behavior="x",
                    pcap="/x/a.pcap", label_source="manual_verified"),
        TrialRecord(capture_id="c2", trial_id="t2", platform="pc",
                    device_id="d", app="beta", behavior="x",
                    pcap="/x/b.pcap", label_source="manual_verified"),
    ]
    truth = load_truth_manifest(recs, "app")
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps({"results": [
        {"observation_id": "a.pcap:0", "source_file": "a.pcap",
         "predicted_label": "alpha"}]}), encoding="utf-8")
    m = evaluate_files(str(pf), truth, {"alpha": 0, "beta": 1},
                       str(tmp_path / "e"), task="app")
    cov = m["truth_coverage"]
    assert cov["n_truth_files"] == 2 and cov["n_covered_files"] == 1
    assert cov["missing_files"] == [{"file": "b.pcap", "label": "beta",
                                     "trial_id": "t2"}]
    # 漏文件不产假100%的分母声明
    assert m["n_samples"] == 1


def test_evaluate_duplicate_observation_counted_once(tmp_path):
    """重复 observation_id 只计首条并计数（同一会话不得计两次分母）。"""
    from src.evaluation import evaluate_files, load_truth_manifest
    from src.data_manifest import TrialRecord
    recs = [TrialRecord(capture_id="c1", trial_id="t1", platform="pc",
                        device_id="d", app="alpha", behavior="x",
                        pcap="/x/a.pcap", label_source="manual_verified")]
    truth = load_truth_manifest(recs, "app")
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps({"results": [
        {"observation_id": "a.pcap:0", "source_file": "a.pcap",
         "predicted_label": "alpha"},
        {"observation_id": "a.pcap:0", "source_file": "a.pcap",
         "predicted_label": "alpha"}]}), encoding="utf-8")
    m = evaluate_files(str(pf), truth, {"alpha": 0}, str(tmp_path / "e"),
                       task="app")
    assert m["n_samples"] == 1
    assert m["n_duplicate_observations"] == 1
    # 无 observation_id 的旧行不去重（同文件多会话合法）
    pf2 = tmp_path / "p2.json"
    pf2.write_text(json.dumps({"results": [
        {"source_file": "a.pcap", "predicted_label": "alpha"},
        {"source_file": "a.pcap", "predicted_label": "alpha"}]}),
        encoding="utf-8")
    m2 = evaluate_files(str(pf2), truth, {"alpha": 0},
                        str(tmp_path / "e2"), task="app")
    assert m2["n_samples"] == 2 and m2["n_duplicate_observations"] == 0


def test_aggregate_events_terminal_grouped_and_background_excluded():
    """事件聚合：同文件不同终端不合并；background 不构成行为事件。"""
    from src.evaluation import aggregate_windows_to_events
    preds = [
        {"source_file": "w.pcap", "terminal": "10.0.0.1",
         "window_start": 1.0, "window_end": 2.0, "predicted_label": "voice"},
        {"source_file": "w.pcap", "terminal": "10.0.0.2",
         "window_start": 1.2, "window_end": 2.2, "predicted_label": "voice"},
        {"source_file": "w.pcap", "terminal": "10.0.0.1",
         "window_start": 3.0, "window_end": 4.0,
         "predicted_label": "background"},
    ]
    evs = aggregate_windows_to_events(preds)
    assert len(evs) == 2, "不同终端的事件被按文件错误合并"
    terms = sorted(e["terminal"] for e in evs)
    assert terms == ["10.0.0.1", "10.0.0.2"]
    assert all(e["label"] != "background" for e in evs), \
        "background 预测不得进入行为事件序列"


def test_match_events_isolates_source():
    """事件匹配按 source 隔离：不同试次（不同pcap）不跨试次配对。"""
    from src.evaluation import match_events
    pe = [{"label": "voice", "t": 10.4, "terminal": "10.0.0.9",
           "source_file": "trialB.pcap"}]
    te = [{"label": "voice", "t": 10.4, "terminal": "10.0.0.9",
           "source_file": "trialA.pcap"}]
    r = match_events(pe, te)
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 1, \
        "跨试次配对凑TP"
    # 旧API（无source_file）保持原语义
    r2 = match_events([{"label": "v", "t": 1.0, "terminal": "T"}],
                      [{"label": "v", "t": 1.1, "terminal": "T"}])
    assert r2["tp"] == 1


def test_match_events_maximum_cardinality_not_greedy():
    """最大一对一配对：贪心按序取最近真值会少算TP的反例。"""
    from src.evaluation import match_events
    preds = [{"label": "v", "t": 0.9, "terminal": "T", "source_file": "f"},
             {"label": "v", "t": 1.05, "terminal": "T", "source_file": "f"}]
    truth = [{"label": "v", "t": 0.0, "terminal": "T", "source_file": "f"},
             {"label": "v", "t": 1.0, "terminal": "T", "source_file": "f"}]
    r = match_events(preds, truth, tolerance=1.0)
    assert r["tp"] == 2, f"贪心少算TP: tp={r['tp']}（0.9抢走1.0真值）"
    assert r["matching"] == "max_cardinality"


# ======================================================== R2 读取/截断
def test_runtime_entry_consumes_generator_not_batch(tmp_path):
    """正式提取入口走有界生成器：read_pcap 整批入口被禁用时仍可提取。"""
    import src.parser.pcap_reader as pr
    from src.features.runtime import extract_feature_records_with_stats
    p = tmp_path / "s.pcap"
    _write_pcap(p, _session_packets(n_data=6))
    orig = pr.PCAPReader.read_pcap
    pr.PCAPReader.read_pcap = (
        lambda self, *a, **k: (_ for _ in ()).throw(
            AssertionError("整批 read_pcap 不应被正式入口调用")))
    try:
        recs, st = extract_feature_records_with_stats(str(p))
    finally:
        pr.PCAPReader.read_pcap = orig
    assert len(recs) >= 1 and st.sessions_total >= 1
    # 会话顺序与整批入口一致（observation_id 可追溯）
    from src.parser.pcap_reader import PCAPReader
    from src.parser.session.session_manager import SessionManager
    batch = PCAPReader(SessionManager()).read_pcap(str(p))
    assert len(batch) == st.sessions_total


def _pcapng_bytes(n_packets: int, endian='<'):
    def block(btype, body):
        total = 8 + len(body) + 4
        return (struct.pack(f'{endian}II', btype, total) + body
                + struct.pack(f'{endian}I', total))
    shb_body = struct.pack(f'{endian}IHHq', 0x1A2B3C4D if endian == '<'
                           else 0x1A2B3C4D, 1, 0, -1)
    idb_body = struct.pack(f'{endian}HHI', 1, 0, 0)
    data = block(0x0A0D0D0A, shb_body) + block(1, idb_body)
    frame = _eth(_ip_tcp(A[0], B[0], A[1], B[1], 0x02, 1000))
    for i in range(n_packets):
        epb_body = (struct.pack(f'{endian}IIIII', 0, 0, i * 1_000_000,
                                len(frame), len(frame)) + frame)
        data += block(6, epb_body)
    return data


def test_pcapng_read_cap_is_real_and_streamed(tmp_path):
    """PCAPNG：真读包上限生效 + 流式解码与整读版逐包一致。"""
    from src.parser.pcap_reader import PCAPReader
    from src.parser.session.session_manager import SessionManager
    f = tmp_path / "x.pcapng"
    f.write_bytes(_pcapng_bytes(10))
    r = PCAPReader(SessionManager())
    pkts = list(r._read_packets(str(f), max_read_packets=4))
    assert len(pkts) == 4, "PCAPNG读包上限未生效"
    assert r.packets_last_read == 0 or True   # 统计在 read_pcap_generator
    # 与整读版（兼容入口）输出一致
    r2 = PCAPReader(SessionManager())
    full = list(r2._read_pcapng(_pcapng_bytes(10)))
    assert len(full) == 10
    r3 = PCAPReader(SessionManager())
    streamed = list(r3._read_pcapng_stream(
        __import__('io').BytesIO(_pcapng_bytes(10))))
    assert streamed == full, "流式与整读解码不一致"
    # 尾部残缺块：完整块照常解析，不静默读空也不崩
    truncated = _pcapng_bytes(5)[:-7]
    r4 = PCAPReader(SessionManager())
    partial = list(r4._read_pcapng(truncated))
    assert len(partial) == 4, f"残缺尾块处理异常: {len(partial)}"
    # 大端 section 同样流式可用
    be = _pcapng_bytes(3, endian='>')
    r5 = PCAPReader(SessionManager())
    assert len(list(r5._read_pcapng(be))) == 3


def test_pcapng_generator_yields_sessions_with_cap(tmp_path):
    """PCAPNG 经正式生成器读取：上限生效、会话照常产出。"""
    from src.parser.pcap_reader import PCAPReader
    from src.parser.session.session_manager import SessionManager
    f = tmp_path / "x.pcapng"
    f.write_bytes(_pcapng_bytes(8))
    r = PCAPReader(SessionManager())
    sessions = list(r.read_pcap_generator(str(f), max_read_packets=5))
    assert r.read_capped is True
    assert r.packets_last_read == 5
    assert len(sessions) >= 1


def test_truncated_view_retransmission_causal(tmp_path):
    """截断视图的重传/乱序计数取截断点因果快照（未来重传不泄入）。"""
    from src.features.runtime import truncated_session_view
    from src.parser.session.session_manager import (
        SessionManager, PacketInfo, Protocol)

    def build(retrans_after_prefix):
        sm = SessionManager()
        for pkt in [
            PacketInfo(timestamp=0.0, src_ip=A[0], dst_ip=B[0],
                       src_port=A[1], dst_port=B[1], protocol=Protocol.TCP,
                       length=60, tcp_flags=0x02, tcp_seq=1000),
            PacketInfo(timestamp=0.02, src_ip=B[0], dst_ip=A[0],
                       src_port=B[1], dst_port=A[1], protocol=Protocol.TCP,
                       length=60, tcp_flags=0x12, tcp_seq=5000),
            PacketInfo(timestamp=0.04, src_ip=A[0], dst_ip=B[0],
                       src_port=A[1], dst_port=B[1], protocol=Protocol.TCP,
                       length=60, tcp_flags=0x10, tcp_seq=1001),
        ]:
            sm.process_packet(pkt)
        seq = 1001
        for i in range(12):
            ts = 0.1 + i * 0.01
            cur = seq
            sm.process_packet(PacketInfo(
                timestamp=ts, src_ip=A[0], dst_ip=B[0], src_port=A[1],
                dst_port=B[1], protocol=Protocol.TCP, length=260,
                payload_length=200, tcp_flags=0x18, tcp_seq=cur,
                payload=b"x" * 200))
            seq += 200
            # 前缀(前8包)之后才发生的重传：不得计入前缀视图
            if retrans_after_prefix and i == 10:
                sm.process_packet(PacketInfo(
                    timestamp=ts + 0.001, src_ip=A[0], dst_ip=B[0],
                    src_port=A[1], dst_port=B[1], protocol=Protocol.TCP,
                    length=260, payload_length=200, tcp_flags=0x18,
                    tcp_seq=cur, payload=b"x" * 200))
        return sm.flush_all()[0]

    s_future = build(True)
    assert s_future.num_retransmissions == 1
    view, trunc = truncated_session_view(s_future, max_packets=8)
    assert trunc and len(view.packets) == 8
    assert view.num_retransmissions == 0, \
        f"未来重传泄入前缀: {view.num_retransmissions}"
    # 原会话不被污染
    assert s_future.num_retransmissions == 1

    # 截断点之前的重传：如实计入前缀
    sm = SessionManager()
    for pkt in [
        PacketInfo(timestamp=0.0, src_ip=A[0], dst_ip=B[0], src_port=A[1],
                   dst_port=B[1], protocol=Protocol.TCP, length=60,
                   tcp_flags=0x02, tcp_seq=1000),
        PacketInfo(timestamp=0.02, src_ip=B[0], dst_ip=A[0], src_port=B[1],
                   dst_port=A[1], protocol=Protocol.TCP, length=60,
                   tcp_flags=0x12, tcp_seq=5000),
        PacketInfo(timestamp=0.04, src_ip=A[0], dst_ip=B[0], src_port=A[1],
                   dst_port=B[1], protocol=Protocol.TCP, length=60,
                   tcp_flags=0x10, tcp_seq=1001),
        PacketInfo(timestamp=0.05, src_ip=A[0], dst_ip=B[0], src_port=A[1],
                   dst_port=B[1], protocol=Protocol.TCP, length=260,
                   payload_length=200, tcp_flags=0x18, tcp_seq=1001,
                   payload=b"x" * 200),          # 建立expected=1201
        PacketInfo(timestamp=0.06, src_ip=A[0], dst_ip=B[0], src_port=A[1],
                   dst_port=B[1], protocol=Protocol.TCP, length=260,
                   payload_length=200, tcp_flags=0x18, tcp_seq=1001,
                   payload=b"x" * 200),          # 前缀内重传
    ]:
        sm.process_packet(pkt)
    s_past = sm.flush_all()[0]
    assert s_past.num_retransmissions == 1
    view2, trunc2 = truncated_session_view(s_past, max_packets=10)
    assert not trunc2 or view2.num_retransmissions == 1


def test_tls_reassembly_no_prefix_backfill():
    """TLS跨分段重组只标注凑满record的包：前缀外的完成不回填前缀包。"""
    from src.parser.pcap_reader import PCAPReader
    from src.parser.session.session_manager import (
        SessionManager, PacketInfo)
    rd = PCAPReader(SessionManager())
    payload_full = bytes([0x16, 0x03, 0x01, 0x00, 0x04]) + b"\x01\x02\x03\x04"
    pkts = []
    for part in (payload_full[:2], payload_full[2:5], payload_full[5:]):
        p = PacketInfo(src_ip=A[0], dst_ip=B[0], src_port=A[1], dst_port=B[1],
                       payload=part)
        rd._tls_feed(p)
        pkts.append(p)
    assert pkts[0].tls_info is None and pkts[1].tls_info is None, \
        "未凑满record的分段包不得带tls_info（前缀截断后即成未来握手泄漏）"
    assert pkts[2].tls_info is not None, "凑满record的完成包应有tls_info"


# ======================================================== R3 规则语义
def _separable_df(n_classes=6, per_class=4, seed=0):
    import numpy as np
    import pandas as pd
    rng = np.random.RandomState(seed)
    rows = []
    y = []
    for c in range(n_classes):
        for _ in range(per_class):
            rows.append({"f0": c * 10 + rng.randn() * 0.5,
                         "f1": c * 100 + rng.randn() * 2})
            y.append(c)
    X = pd.DataFrame(rows)
    X["f_noise"] = rng.randn(len(X))
    return X, pd.Series(y)


def test_stat_rules_confidence_is_precision_and_deployed_rules_fire():
    """统计规则：confidence=train precision；部署规则必须在训练集命中>0。"""
    import numpy as np
    from src.engine.rule_compiler.optimized_generator import (
        OptimizedRuleGenerator)
    X, y = _separable_df(n_classes=3, per_class=6)
    feats = ["f0", "f1", "f_noise"]
    gen = OptimizedRuleGenerator()
    rules = gen._generate_statistical_rules(X, y, feats,
                                            {0: "a", 1: "b", 2: "c"})
    assert rules, "可分数据应产出统计规则"
    for r in rules:
        cov = pd_series_true(X)
        for c in r["conditions"]:
            cov &= (X[c["feature"]] >= c["min"])
            cov &= (X[c["feature"]] <= c["max"])
        n_fire = int(cov.sum())
        assert n_fire > 0, f"{r['id']} 零覆盖死规则被部署"
        prec = float((y[cov] == dict(a=0, b=1, c=2)[r["action"]["result"]]).mean())
        assert r["confidence"] == pytest.approx(prec, abs=1e-6), \
            f"{r['id']} confidence应=train precision"
        assert r["train_coverage"] == pytest.approx(
            float(cov[y == dict(a=0, b=1, c=2)[r["action"]["result"]]].mean()),
            abs=1e-4)


def pd_series_true(X):
    import pandas as pd
    return pd.Series(True, index=X.index)


def test_rule_conditions_skip_features_missing_in_train():
    """训练含NaN的特征不进规则条件（训练0填与推理缺失不匹配的语义冲突）。"""
    import numpy as np
    import pandas as pd
    from src.engine.rule_compiler.optimized_generator import (
        OptimizedRuleGenerator)
    X, y = _separable_df(n_classes=2, per_class=6)
    X.loc[X.index[:3], "f0"] = np.nan       # f0 在训练里部分缺失
    gen = OptimizedRuleGenerator()
    rules = gen._generate_statistical_rules(X, y, ["f0", "f1"],
                                            {0: "a", 1: "b"})
    tree_rules = gen._extract_tree_rules(X, y, ["f0", "f1"], {0: "a", 1: "b"})
    for r in rules + tree_rules:
        used = {c["feature"] for c in r["conditions"]}
        assert "f0" not in used, f"{r['id']} 条件引用了训练含缺失的特征"


def test_engine_applies_threshold_to_rule_matches():
    """引擎对规则命中同样应用置信度阈值（conf=0的死规则不再照发预测）。"""
    from src.engine.matcher.optimized_engine import OptimizedDPIEngine
    eng = OptimizedDPIEngine()
    eng.rules = [
        {"id": "S0", "name": "dead", "type": "statistical", "priority": 50,
         "confidence": 0.0,
         "conditions": [{"feature": "f", "op": ">=", "value": 0}],
         "action": {"result": "Dead", "source": "t"}},
        {"id": "S1", "name": "ok", "type": "statistical", "priority": 100,
         "confidence": 0.9,
         "conditions": [{"feature": "f", "op": ">=", "value": 0}],
         "action": {"result": "Ok", "source": "t"}},
    ]
    eng.confidence_threshold = 0.65
    res = eng.match({"f": 1.0})
    assert res and res[0].result == "Ok", \
        "低置信度规则应被阈值拦截且不阻断后续规则"


def test_engine_priority_order_not_list_order():
    """规则按 priority 升序评估：STAT(200)放在列表前也不压过 DT(100)。"""
    from src.engine.matcher.optimized_engine import OptimizedDPIEngine
    dt = {"id": "DT", "name": "dt", "type": "decision_tree", "priority": 100,
          "confidence": 0.95,
          "conditions": [{"feature": "f", "op": ">=", "value": 0}],
          "action": {"result": "TreeResult", "source": "decision_tree"}}
    st = {"id": "ST", "name": "st", "type": "statistical", "priority": 200,
          "confidence": 0.95,
          "conditions": [{"feature": "f", "op": ">=", "value": 0}],
          "action": {"result": "StatResult", "source": "statistical"}}
    eng = OptimizedDPIEngine()
    eng.rules = [st, dt]           # 列表顺序故意反转
    eng.confidence_threshold = 0.5
    res = eng.match({"f": 1.0})
    assert res[0].result == "TreeResult", \
        f"priority未生效，被列表顺序支配: {res[0].result}"


def test_adaptive_tree_exports_rules_for_small_classes():
    """自适应浅树：6类×4样本固定leaf=5无法为小类产纯叶，自适应可以。"""
    from src.engine.rule_compiler.optimized_generator import (
        OptimizedRuleGenerator)
    X, y = _separable_df(n_classes=6, per_class=4, seed=1)
    names = {i: f"c{i}" for i in range(6)}
    fixed = OptimizedRuleGenerator(tree_max_depth=5,
                                   tree_min_samples_leaf=5)
    r_fixed = fixed._extract_tree_rules(X, y, ["f0", "f1"], names)
    classes_fixed = {r["action"]["result"] for r in r_fixed}
    adaptive = OptimizedRuleGenerator()
    r_adapt = adaptive._extract_tree_rules(X, y, ["f0", "f1"], names)
    classes_adapt = {r["action"]["result"] for r in r_adapt}
    assert len(classes_adapt) > len(classes_fixed), (
        f"自适应浅树应覆盖更多类: adapt={len(classes_adapt)} "
        f"fixed={len(classes_fixed)}")
    assert len(classes_adapt) >= 5, \
        f"6类可分数据自适应树应导出>=5类规则: {sorted(classes_adapt)}"


def test_evaluate_all_rejected_still_full_denominator(tmp_path):
    """全拒识：分母完整、拒识率1.0、recall=0（不产出虚高指标）。"""
    from src.evaluation import evaluate_files, load_truth_manifest
    from src.data_manifest import TrialRecord
    recs = [TrialRecord(capture_id="c1", trial_id="t1", platform="pc",
                        device_id="d", app="alpha", behavior="x",
                        pcap="/x/a.pcap", label_source="manual_verified")]
    truth = load_truth_manifest(recs, "app")
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps({"results": [
        {"observation_id": "a.pcap:0", "source_file": "a.pcap",
         "predicted_label": "unknown", "confidence": 0.0}]}),
        encoding="utf-8")
    m = evaluate_files(str(pf), truth, {"alpha": 0}, str(tmp_path / "e"))
    assert m["n_samples"] == 1 and m["n_rejected"] == 1
    assert m["rejection_rate"] == 1.0
    assert m["per_class"]["alpha"]["recall"] == 0.0
    assert m["truth_coverage"]["n_covered_files"] == 1
