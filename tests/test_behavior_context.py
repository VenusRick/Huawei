# -*- coding: utf-8 -*-
"""M2 行为上下文合成验收（方案 M2 验收七条的合成部分）。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parser.context import (BehaviorContextTracker, WindowFeatureBuilder,
                                DEFAULT_CONTEXT_CONFIG)
from src.data_manifest import TrialRecord, label_window


CFG = {"window_sec": 5.0, "step_sec": 1.0, "max_segments": 3,
       "segment_gap_sec": 2.0, "overflow": {"aggregate_remaining": True}}
TERM = "10.0.0.1"


def _feed(tracker, packets):
    """packets: (ts, src, dst, canon, len)"""
    for (ts, src, dst, canon, ln) in packets:
        tracker.feed_packet(ts, src, dst, canon, ln)


def _collect(packets, terminal=TERM, cfg=CFG):
    tr = BehaviorContextTracker(cfg, terminal=terminal, capture_id="c1")
    out = []
    tr.on_window = out.append
    _feed(tr, packets)
    tr.finalize()
    builder = WindowFeatureBuilder()
    return [(w, builder.build(w)) for w in out]


# 同样的包/字节总量，不同跨流先后结构
_SEQ = [(0.0, TERM, "1.1.1.1", "connA", 100), (0.2, "1.1.1.1", TERM, "connA", 100),
        (0.4, TERM, "1.1.1.1", "connA", 100),
        (2.5, TERM, "2.2.2.2", "connB", 100), (2.7, "2.2.2.2", TERM, "connB", 100)]
_ALT = [(0.0, TERM, "1.1.1.1", "connA", 100), (0.2, TERM, "2.2.2.2", "connB", 100),
        (0.4, "1.1.1.1", TERM, "connA", 100),
        (2.5, "2.2.2.2", TERM, "connB", 100), (2.7, TERM, "1.1.1.1", "connA", 100)]


def test_crossflow_ordering_changes_relation_features():
    """验收1：相同单流统计、不同跨流先后/重叠结构 -> 关系特征不同。"""
    _, f_seq = _collect(_SEQ)[0]
    _, f_alt = _collect(_ALT)[0]
    diffs = [k for k in f_seq if f_seq[k] != f_alt.get(k)]
    assert any("rel" in k or "dir_changes" in k for k in diffs), \
        f"先后结构未引起特征差异: {diffs}"
    # 窗级总量保持一致（不因结构不同漏计）
    assert f_seq["behavior_win_n_pkt"] == f_alt["behavior_win_n_pkt"] == 5.0
    assert f_seq["behavior_win_n_conn"] == f_alt["behavior_win_n_conn"] == 2.0


def test_two_terminal_views_not_merged():
    """验收2：两个终端各持一半模式 -> 两视角特征不同（方向相对终端）。"""
    # 终端A视角：seq 里 TERM 发起 connA
    # 纯双方流量：两终端视角互补
    duplex = [(0.0, TERM, "1.1.1.1", "c", 100), (0.2, "1.1.1.1", TERM, "c", 100),
              (0.4, TERM, "1.1.1.1", "c", 100), (0.6, "1.1.1.1", TERM, "c", 100)]
    _, fa = _collect(duplex, terminal=TERM)[0]
    _, fb = _collect(duplex, terminal="1.1.1.1")[0]
    assert fa["behavior_win_up_ratio"] + fb["behavior_win_up_ratio"] \
        == pytest.approx(1.0), "双方流量两视角应互补，而不是合并成一个模式"
    # 三方流量（含第三方远端）：两视角特征不同，各自只看本端方向
    _, fa3 = _collect(_SEQ, terminal=TERM)[0]
    _, fb3 = _collect(_SEQ, terminal="1.1.1.1")[0]
    assert fa3 != fb3, "不同终端视角的特征不得相同（各自半模式不被合并）"


def test_no_future_packets_in_window():
    """验收3：窗口发射时只含当时已喂的包（不用未来包）。"""
    tr = BehaviorContextTracker(CFG, terminal=TERM, capture_id="c1")
    seen_at_emit = []
    orig = []

    def on_win(w):
        seen_at_emit.append((w["window_start"], w["n_packets"],
                             len(orig)))

    tr.on_window = on_win
    for i in range(30):
        ts = i * 0.5
        _feed(tr, [(ts, TERM, "9.9.9.9", "connC", 60)])
        orig.append(ts)
        # 每喂一个包后检查：已发射窗口的 n_packets 不超过当前已喂总数
        for (_, n_pkts, fed) in seen_at_emit:
            assert n_pkts <= fed, "窗口使用了未来包"


def test_batched_feed_same_result():
    """验收4：分批输入与一次输入结果相同（事件序决定一切）。"""
    packets = []
    t = 0.0
    for i in range(24):
        src = TERM if i % 2 == 0 else "3.3.3.3"
        dst = "3.3.3.3" if i % 2 == 0 else TERM
        packets.append((t, src, dst, f"conn{i % 3}", 50 + 10 * (i % 4)))
        t += 0.45
    one = _collect(packets)
    # 分三批
    tr2 = BehaviorContextTracker(CFG, terminal=TERM, capture_id="c1")
    out2 = []
    tr2.on_window = out2.append
    for chunk in (packets[:8], packets[8:16], packets[16:]):
        _feed(tr2, chunk)
    tr2.finalize()
    b = WindowFeatureBuilder()
    two = [(w, b.build(w)) for w in out2]
    assert len(one) == len(two)
    for (w1, f1), (w2, f2) in zip(one, two):
        assert w1["window_start"] == w2["window_start"]
        assert f1 == f2, \
            f"窗口{w1['window_start']}分批不一致"


def test_stable_ordering_same_start_time():
    """验收4b：相同起始时间的流段按首包序号稳定排序。"""
    tr = BehaviorContextTracker(CFG, terminal=TERM, capture_id="c1")
    out = []
    tr.on_window = out.append
    # 三条连接同一时刻发首包（pkt_index 决定次序）
    _feed(tr, [(0.0, TERM, "a", "cB", 10), (0.0, TERM, "b", "cA", 10),
               (0.0, TERM, "c", "cC", 10)])
    tr.finalize()
    segs = out[0]["segments"]
    order = [s.conn_key for s in segs]
    assert order == ["cB", "cA", "cC"], f"同刻段未按首包序稳定排序: {order}"


def test_feature_names_carry_no_label_leak():
    """验收6（合成部分）：特征键不含文件名/类别目录语义。"""
    _, feats = _collect(_SEQ)[0]
    for k in feats:
        assert "app" not in k.lower() and "label" not in k.lower() and \
            "class" not in k.lower(), f"特征名疑似标签泄漏: {k}"


def test_overflow_segments_aggregated_not_dropped():
    """K=3 槽位 + overflow 聚合：第 4+ 条流段进窗级统计不静默丢。"""
    packets = []
    for i in range(5):   # 5 条连接 > max_segments=3
        packets.append((i * 0.1, TERM, f"s{i}", f"conn{i}", 100))
    tr = BehaviorContextTracker(CFG, terminal=TERM, capture_id="c1")
    out = []
    tr.on_window = out.append
    _feed(tr, packets)
    tr.finalize()
    w = out[0]
    assert w["n_segments"] == 5
    assert len(w["segments"]) == 3            # 槽位上限
    assert w["n_overflow_segments"] == 2      # 不静默丢
    f = WindowFeatureBuilder().build(w)
    assert f["behavior_win_n_seg"] == 5.0     # 窗级统计用全部
    assert f["behavior_win_overflow_n_seg"] == 2.0
    assert f["behavior_seg2_valid"] == 1.0    # 前三槽有效
    assert f["behavior_seg2_pkt"] == 1.0


def test_missing_slot_validity_marker():
    """缺槽位输出 None（有效性标记），不零填。"""
    tr = BehaviorContextTracker(CFG, terminal=TERM, capture_id="c1")
    out = []
    tr.on_window = out.append
    _feed(tr, [(0.0, TERM, "x", "only", 50)])
    tr.finalize()
    f = WindowFeatureBuilder().build(out[0])
    assert f["behavior_seg0_valid"] == 1.0
    assert f["behavior_seg1_valid"] is None
    assert f["behavior_seg1_duration"] is None
    assert f["behavior_rel01_gap"] is None


def test_long_connection_across_windows():
    """验收3b：同一长连接跨多窗，各窗只统计本窗包。"""
    packets = []
    for i in range(40):
        packets.append((i * 0.5, TERM if i % 2 else "5.5.5.5",
                        "5.5.5.5" if i % 2 else TERM, "longconn", 80))
    res = _collect(packets)
    assert len(res) >= 3, "长连接应跨多窗"
    counts = [w["n_packets"] for w, _ in res]
    assert max(counts) <= 10 + 2, f"窗内包数超出窗口容量: {counts}"


def test_label_window_mapping():
    """行为区间 -> 窗口标签映射（重叠率阈值）。"""
    r = TrialRecord(capture_id="c", trial_id="t", platform="pc",
                    device_id="d", app="wechat", behavior="voice_call",
                    pcap="x.pcap", label_source="manual_verified",
                    behavior_start=10.0, behavior_end=30.0)
    assert label_window(r, 12.0, 17.0) == "voice_call"      # 完整落在区间
    assert label_window(r, 0.0, 5.0) == "background"        # 区间外
    assert label_window(r, 8.0, 13.0) == "voice_call"       # 重叠 60%
    assert label_window(r, 27.0, 32.0) == "voice_call"      # 重叠 60%
    assert label_window(r, 28.5, 33.5) == "background"      # 重叠 30% < 50%


# ---------------------------------------------------------------------
# dpi_infer 窗口模式一致性：behavior bundle 下 --pcap 与 --pcap-dir 同参
# （2026-09-18 修复 --pcap 静默退化 app 会话模式后补的守门）
# ---------------------------------------------------------------------
@pytest.mark.skipif(
    not Path(__file__).resolve().parents[1].joinpath(
        "output", "p1_e2e", "mine", "bundle", "rules.json").exists(),
    reason="p1_e2e behavior bundle 未生成")
def test_behavior_bundle_single_pcap_matches_dir_mode(tmp_path):
    """同一 behavior bundle 同一 pcap：单文件窗口推理 == 目录窗口推理。"""
    import json as _json
    import subprocess
    root = Path(__file__).resolve().parents[1]
    bundle = root / "output" / "p1_e2e" / "mine" / "bundle"
    pcap_dir = root / "output" / "fixture_m2" / "test"
    pcaps = sorted(pcap_dir.glob("*.pcap"))
    if not pcaps:
        pytest.skip("fixture_m2 test pcaps 未生成")
    pcap = pcaps[0]

    def run(mode, target, out):
        r = subprocess.run(
            [sys.executable, "-m", "src.engine.dpi_infer",
             "--rules", str(bundle), mode, str(target), "-o", str(out)],
            cwd=str(root), capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-300:]
        return _json.loads(out.read_text())["results"]

    single = run("--pcap", pcap, tmp_path / "single.json")
    batch = [x for x in run("--pcap-dir", pcap_dir, tmp_path / "dir.json")
             if x.get("source_file") == pcap.name]
    # 单文件必须是窗口记录（含 window_start/end），不是会话记录
    assert single and "window_start" in single[0], \
        "behavior bundle 的 --pcap 未走窗口模式"
    assert len(single) == len(batch)
    for a, b in zip(single, batch):
        assert a["observation_id"] == b["observation_id"]
        assert abs(a["window_start"] - b["window_start"]) < 1e-9
        assert abs(a["window_end"] - b["window_end"]) < 1e-9
        assert a["predicted_label"] == b["predicted_label"], \
            f"单文件/目录预测不一致: {a['predicted_label']} vs {b['predicted_label']}"
        assert abs(a["confidence"] - b["confidence"]) < 1e-9
