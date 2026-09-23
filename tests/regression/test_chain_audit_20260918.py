# -*- coding: utf-8 -*-
"""全链审计回归（2026-09-18）：标签精确性/训练验证隔离/截断语义/评价口径/
报告证据语义/引擎算子集/小样本规则生成。

对应宿主审查缺陷 1-7 的守门用例；每条用例都可独立复现（合成 pcap，
不依赖真实数据集）。运行：
    python3 -m pytest tests/regression/test_chain_audit_20260918.py -q
"""
import json
import socket
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

A, B = ('10.0.0.2', 5000), ('10.0.0.1', 443)


# ---------------------------------------------------------------- pcap 工具
def _ip_tcp(src: str, dst: str, sport: int, dport: int, flags: int,
            seq: int, payload: bytes = b'') -> bytes:
    tcp = struct.pack('!HHIIBBHHH', sport, dport, seq, 1, 5 << 4, flags,
                      8192, 0, 0)
    total = 20 + len(tcp) + len(payload)
    ip = struct.pack('!BBHHHBBH', 0x45, 0, total, 0, 0, 64, 6, 0)
    ip += socket.inet_aton(src) + socket.inet_aton(dst)
    return ip + tcp + payload


def _write_pcap(path: Path, packets) -> None:
    """packets: [(ts, raw_ethernet_frame)]——链路层写以太网头。"""
    with open(path, 'wb') as f:
        f.write(struct.pack('<IHHiIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))
        for ts, raw in packets:
            sec = int(ts)
            usec = int(round((ts - sec) * 1e6))
            f.write(struct.pack('<IIII', sec, usec, len(raw), len(raw)))
            f.write(raw)


def _eth(ip_pkt: bytes) -> bytes:
    return b'\x11\x22\x33\x44\x55\x66\xaa\xbb\xcc\xdd\xee\xff\x08\x00' + ip_pkt


def _session_packets(tag: str, n_data: int = 6, t0: float = 1.0,
                     gap: float = 0.1, big_late: bool = False):
    """一条完整TCP会话（握手+交替数据包），payload 足够越过100B阈值。"""
    out = []
    out.append((t0, _eth(_ip_tcp(A[0], B[0], A[1], B[1], 0x02, 1000))))
    out.append((t0 + 0.02, _eth(_ip_tcp(B[0], A[0], B[1], A[1], 0x12, 5000))))
    out.append((t0 + 0.04, _eth(_ip_tcp(A[0], B[0], A[1], B[1], 0x10, 1001))))
    seq_a, seq_b = 1001, 5001
    for i in range(n_data):
        ts = t0 + 0.1 + i * gap
        if big_late and i >= n_data - 2:
            ts += 3600.0          # 末尾包携带"未来"大时间戳
        if i % 2 == 0:
            out.append((ts, _eth(_ip_tcp(A[0], B[0], A[1], B[1], 0x18,
                                         seq_a, b'\x41' * 200))))
            seq_a += 200
        else:
            out.append((ts, _eth(_ip_tcp(B[0], A[0], B[1], A[1], 0x18,
                                         seq_b, b'\x42' * 120))))
            seq_b += 120
    return out


def _manifest_line(cap, trial, app, pcap, split, **kw):
    d = {"capture_id": cap, "trial_id": trial, "platform": "pc",
         "device_id": "synthetic", "app": app, "behavior": "synthetic_flow",
         "pcap": str(pcap), "label_source": "manual_verified",
         "network_env": "synthetic", "split": split}
    d.update(kw)
    return json.dumps(d, ensure_ascii=False)


def _tiny_two_class_dataset(root: Path):
    """2类×(2采集组train+1验证)，文件名与标签无关（防文件名猜标签）。"""
    pcaps = []
    for cls, prefix in (("appAlpha", "cap"), ("appBeta", "seq")):
        for gi, (cap, split) in enumerate(
                [(f"{prefix}g1", "train"), (f"{prefix}g2", "train"),
                 (f"{prefix}g3", "validation")]):
            for si in range(2):
                # 文件名 = 随机样式，不含类别词
                name = f"{cap}_{si:02d}_renamed.pcap"
                p = root / name
                _write_pcap(p, _session_packets(
                    f"{cls}{cap}{si}", n_data=6 + si * 2,
                    t0=1.0 + gi * 10 + si))
                pcaps.append((p, cls, cap, split))
    return pcaps


# ================================================================ D1 标签/泄漏
def test_manifest_labels_exact_renamed_pcaps(tmp_path):
    """清单驱动：标签只来自记录字段；文件名改掉不影响标签（缺陷1）。"""
    from src.pipeline import FeatureMiningPipeline
    from src.data_manifest import load_manifest, build_label_vocab
    pcaps = _tiny_two_class_dataset(tmp_path)
    mf = tmp_path / "m.jsonl"
    mf.write_text("\n".join(
        _manifest_line(cap, f"{cls}_{cap}_{i}", cls, p, split)
        for i, (p, cls, cap, split) in enumerate(pcaps)), encoding="utf-8")
    out = tmp_path / "mine"
    FeatureMiningPipeline().mine(
        None, str(out), manifest=str(mf), task='app')

    recs = load_manifest(str(mf))
    vocab = build_label_vocab(recs, 'app')['label_map']
    import pandas as pd
    df = pd.read_csv(out / 'raw_features.csv')
    by_file = df.groupby('_source_file')['_label_id'].nunique()
    assert (by_file == 1).all(), "同一文件出现多种标签：标签未按记录精确对齐"
    for p, cls, cap, split in pcaps:
        lab_ids = set(df[df['_source_file'].str.endswith(p.name)]['_label_id'])
        assert lab_ids == {vocab[cls]}, \
            f"{p.name} 标签应为 {cls}({vocab[cls]})，实得 {lab_ids}"


def test_validation_label_perturbation_keeps_training_artifacts(tmp_path):
    """验证集标签扰动不改变训练产物（train-only 拟合证据；缺陷1）。"""
    from src.pipeline import FeatureMiningPipeline
    pcaps = _tiny_two_class_dataset(tmp_path)

    def run(val_label_override, outname):
        mf = tmp_path / f"{outname}.jsonl"
        lines = []
        for i, (p, cls, cap, split) in enumerate(pcaps):
            c = cls
            if split == 'validation' and val_label_override:
                c = val_label_override
            lines.append(_manifest_line(cap, f"{c}_{cap}_{i}", c, p, split))
        mf.write_text("\n".join(lines), encoding="utf-8")
        out = tmp_path / outname
        FeatureMiningPipeline().mine(
            None, str(out), manifest=str(mf), task='app')
        return json.loads((out / 'bundle' / 'rules.json').read_text())

    r1 = run(None, "out_a")
    r2 = run("appGamma", "out_b")
    key = lambda rs: sorted(  # noqa: E731
        (json.dumps(r, sort_keys=True) for r in rs))
    assert key(r1) == key(r2), \
        "验证集标签变化改变了训练规则：验证数据泄漏进拟合（缺陷1未修）"
    vr = json.loads((tmp_path / "out_a" / "validation_report.json")
                    .read_text(encoding="utf-8"))
    assert vr["available"] is True and vr["n_validation"] > 0


def test_dir_mode_exact_basename_labels_and_no_default_zero(tmp_path):
    """目录模式：basename 精确匹配；子串冲突消解；无标签文件显式报错。"""
    from src.pipeline import FeatureMiningPipeline
    # 'a.pcap' 是 'xa.pcap' 的子串——旧逻辑两者都命中的是先遍历到的键
    d1 = tmp_path / "d1"; d1.mkdir()
    pa = d1 / "a.pcap"
    px = d1 / "xa.pcap"
    pa2 = d1 / "a2.pcap"
    _write_pcap(pa, _session_packets("a", n_data=6))
    _write_pcap(px, _session_packets("x", n_data=10))
    _write_pcap(pa2, _session_packets("a2", n_data=14))
    pipe = FeatureMiningPipeline()
    out = tmp_path / "out"
    pipe.mine(str(d1), str(out), labels={"a.pcap": 0, "a2.pcap": 0,
                                         "xa.pcap": 1},
              task='app')
    import pandas as pd
    df = pd.read_csv(out / 'raw_features.csv')
    la = set(df[df['_source_file'].str.endswith('/a.pcap')]['_label_id'])
    lx = set(df[df['_source_file'].str.endswith('xa.pcap')]['_label_id'])
    assert la == {0} and lx == {1}, \
        f"子串标签冲突未消解: a={la} xa={lx}"
    # 无精确标签的文件必须显式失败（不回落类0）
    d2 = tmp_path / "d2"; d2.mkdir()
    pa2 = d2 / "a.pcap"
    py = d2 / "yb.pcap"
    _write_pcap(pa2, _session_packets("a2"))
    _write_pcap(py, _session_packets("y"))
    with pytest.raises(ValueError, match="yb.pcap"):
        pipe.mine(str(d2), str(tmp_path / "out2"),
                  labels={"a.pcap": 0}, task='app')


def test_no_validation_split_is_explicit_not_random(tmp_path):
    """无 validation：明确记录无证据，不随机按会话混拆（缺陷1）。"""
    from src.pipeline import FeatureMiningPipeline
    pcaps = [x for x in _tiny_two_class_dataset(tmp_path)
             if x[3] == 'train']
    mf = tmp_path / "m.jsonl"
    mf.write_text("\n".join(
        _manifest_line(cap, f"{cls}_{cap}_{i}", cls, p, 'train')
        for i, (p, cls, cap, _) in enumerate(pcaps)), encoding="utf-8")
    out = tmp_path / "mine"
    from src.pipeline import FeatureMiningPipeline as _FP
    _FP().mine(None, str(out), manifest=str(mf), task='app')
    vr = json.loads((out / 'validation_report.json').read_text(encoding='utf-8'))
    assert vr["available"] is False
    assert "validation" in vr["reason"]
    eff = json.loads((out / 'analysis' / 'feature_catalog.json')
                     .read_text(encoding='utf-8'))
    assert eff, "catalog 为空"


# ================================================================ D2 截断/读包上限
def test_read_cap_is_real_prefix_not_postFilter(tmp_path):
    """读包上限=真正读满即停：等价于物理前缀文件（缺陷2/读包上限）。"""
    from src.features.runtime import extract_feature_records_with_stats
    full = tmp_path / "full.pcap"
    _write_pcap(full, _session_packets("s", n_data=40, gap=0.01))
    prefix = tmp_path / "prefix.pcap"
    _write_pcap(prefix, _session_packets("s", n_data=10, gap=0.01))
    # 物理前缀=3握手+10数据=13包；上限读13包应与其完全一致
    recs_cap, st = extract_feature_records_with_stats(
        str(full), max_read_packets=13)
    recs_pre, st2 = extract_feature_records_with_stats(str(prefix))
    assert st.packets_read == 13, f"读包上限未生效: {st.packets_read}"
    assert st.read_capped_files == 1 and st2.read_capped_files == 0
    assert len(recs_cap) == len(recs_pre)
    for rc, rp in zip(recs_cap, recs_pre):
        assert rc.features.keys() == rp.features.keys()
        for k in rc.features:
            assert rc.features[k] == rp.features[k], \
                f"上限读取与物理前缀不一致: {k}"


def test_truncation_prefix_no_future_duration_and_no_mutation(tmp_path):
    """前缀截断：不使用未来包时长；不污染原会话（缺陷2）。"""
    from src.features.runtime import truncated_session_view
    from src.parser.session.session_manager import (
        SessionManager, PacketInfo, Protocol)
    # 手工构造：603包，第501个数据包起时间戳+120s（仍在tcp_timeout内，
    # 会话不裂开；旧实现截断后 end_time 仍取到+120s的未来包=污染）
    sm2 = SessionManager()
    sm2.process_packet(PacketInfo(
        timestamp=0.0, src_ip=A[0], dst_ip=B[0], src_port=A[1], dst_port=B[1],
        protocol=Protocol.TCP, length=60, tcp_flags=0x02, tcp_seq=1000))
    sm2.process_packet(PacketInfo(
        timestamp=0.02, src_ip=B[0], dst_ip=A[0], src_port=B[1], dst_port=A[1],
        protocol=Protocol.TCP, length=60, tcp_flags=0x12, tcp_seq=5000))
    sm2.process_packet(PacketInfo(
        timestamp=0.04, src_ip=A[0], dst_ip=B[0], src_port=A[1], dst_port=B[1],
        protocol=Protocol.TCP, length=60, tcp_flags=0x10, tcp_seq=1001))
    seq = 1001
    for i in range(600):
        ts = 0.1 + i * 0.01 + (120.0 if i >= 500 else 0.0)
        sm2.process_packet(PacketInfo(
            timestamp=ts, src_ip=A[0], dst_ip=B[0], src_port=A[1],
            dst_port=B[1], protocol=Protocol.TCP, length=260,
            payload_length=200, tcp_flags=0x18, tcp_seq=seq, payload=b'x'*200))
        seq += 200
    sessions = sm2.flush_all()
    s = sessions[0]
    assert len(s.packets) == 603
    full_end = s.end_time
    view, truncated = truncated_session_view(s, max_packets=103)
    assert truncated is True
    assert len(view.packets) == 103
    # 前缀时长不含未来包：103包全在 i<500 区，end ≈ 0.1+102*0.01
    assert view.duration < 10.0, f"截断视图用了未来包时长: {view.duration}"
    # 103包 = 3握手 + 100数据；保留前缀末包 = 数据i=99
    assert abs(view.duration - (0.1 + 99 * 0.01)) < 0.02
    # 原会话未污染
    assert s.end_time == full_end and len(s.packets) == 603
    assert view is not s


def test_extract_stats_counted_not_swallowed(tmp_path):
    """解析过滤/失败计数落档，不再静默丢分母（缺陷2）。"""
    from src.features.runtime import extract_feature_records_with_stats
    p = tmp_path / "one.pcap"
    _write_pcap(p, _session_packets("s", n_data=6))
    recs, st = extract_feature_records_with_stats(str(p))
    assert st.sessions_total >= 1
    assert (st.sessions_total == len(recs) + st.sessions_filtered_min_packets
            + st.sessions_filtered_min_bytes + st.extract_errors), \
        "会话分母不守恒"


# ================================================================ D3 评价口径
def test_load_predictions_jsonl_starting_with_brace(tmp_path):
    """首行以{开头的JSONL不得误判为单JSON（缺陷3）。"""
    from src.evaluation import load_predictions
    f = tmp_path / "p.jsonl"
    f.write_text("\n".join(
        json.dumps({"observation_id": f"x.pcap:{i}",
                    "predicted_label": "unknown", "confidence": 0.0})
        for i in range(3)), encoding="utf-8")
    preds = load_predictions(str(f))
    assert len(preds) == 3, f"JSONL 被误判为单 JSON: {len(preds)}"


def test_duplicate_truth_basename_raises(tmp_path):
    """真值 basename 重复显式报错（缺陷3）。"""
    from src.evaluation import load_truth_manifest
    from src.data_manifest import TrialRecord
    recs = [
        TrialRecord(capture_id="c1", trial_id="t1", platform="pc",
                    device_id="d", app="A", behavior="x",
                    pcap="/tmp/dir1/a.pcap", label_source="manual_verified"),
        TrialRecord(capture_id="c2", trial_id="t2", platform="pc",
                    device_id="d", app="B", behavior="x",
                    pcap="/tmp/dir2/a.pcap", label_source="manual_verified"),
    ]
    with pytest.raises(ValueError, match="basename 重复"):
        load_truth_manifest(recs, "app")


def test_compute_metrics_noncontiguous_label_ids():
    """非连续标签ID与名称对齐（缺陷3）。"""
    from src.evaluation import compute_metrics
    names = ["alpha", "beta"]
    ids = [0, 2]
    y_true = [0, 0, 2, 2]
    y_pred = [0, 2, 2, 0]
    m = compute_metrics(y_true, y_pred, names, label_ids=ids)
    assert m["per_class"]["alpha"]["support"] == 2
    assert m["per_class"]["beta"]["support"] == 2
    assert m["accuracy"] == 0.5
    # 拒识
    m2 = compute_metrics([0, 2], [-1, 2], names, label_ids=ids)
    assert m2["n_rejected"] == 1 and m2["accuracy"] == 0.5


def test_behavior_window_truth_not_pcap_level(tmp_path):
    """行为评价：窗口真值按区间判定，背景窗单列FPR（缺陷3/6）。"""
    from src.evaluation import evaluate_files
    from src.data_manifest import TrialRecord
    rec = TrialRecord(
        capture_id="c1", trial_id="t1", platform="pc", device_id="d",
        app="IM", behavior="voice_call", pcap="/tmp/x/w.pcap",
        label_source="manual_verified",
        behavior_start=10.0, behavior_end=30.0, terminal_ip="10.0.0.9")
    from src.evaluation import load_truth_manifest
    truth = load_truth_manifest([rec], "behavior")
    label_map = {"voice_call": 0}
    preds = {"results": [
        {"observation_id": "w.pcap:w5.0", "source_file": "w.pcap",
         "window_start": 5.0, "window_end": 10.0, "terminal": "10.0.0.9",
         "predicted_label": "voice_call", "confidence": 0.9},   # 背景窗误报
        {"observation_id": "w.pcap:w40.0", "source_file": "w.pcap",
         "window_start": 40.0, "window_end": 45.0, "terminal": "10.0.0.9",
         "predicted_label": "unknown", "confidence": 0.0},      # 背景窗拒识
        {"observation_id": "w.pcap:w12.0", "source_file": "w.pcap",
         "window_start": 12.0, "window_end": 17.0, "terminal": "10.0.0.9",
         "predicted_label": "voice_call", "confidence": 0.9},   # 行为窗命中
        {"observation_id": "w.pcap:w60.0", "source_file": "w.pcap",
         "window_start": 60.0, "window_end": 65.0, "terminal": "10.0.0.9",
         "predicted_label": "voice_call", "confidence": 0.9},   # 背景窗误报
    ]}
    pf = tmp_path / "preds.json"
    pf.write_text(json.dumps(preds), encoding="utf-8")
    m = evaluate_files(str(pf), truth, label_map, str(tmp_path / "e"),
                       task="behavior")
    bg = m["background"]
    assert bg["n_background_windows"] == 3
    assert bg["background_false_alarms"] == 2
    assert bg["background_fpr"] == pytest.approx(2 / 3)
    assert m["n_unpaired_predictions"] == 0
    assert m["event_matching"]["tp"] == 1


def test_empty_truth_not_evaluable(tmp_path):
    from src.evaluation import evaluate_files
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps({"results": [
        {"source_file": "a.pcap", "predicted_label": "x"}]}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="不可评价"):
        evaluate_files(str(pf), {}, {"x": 0}, str(tmp_path / "e"))


# ================================================================ D4 报告证据
def test_effectiveness_pending_without_evidence():
    """无parity/引擎证据：pending，不默认通过（缺陷4）。"""
    import pandas as pd
    from src.reporting import compute_feature_effectiveness
    df = pd.DataFrame({"f1": [1.0, 2.0, 5.0, 6.0],
                       "f2": [1.0, 1.0, 2.0, 2.0]})
    y = pd.Series([0, 0, 1, 1])
    eff = compute_feature_effectiveness(df, y, ["f1"])
    assert eff["E"] == []
    assert set(eff["pending"]) == {"f1", "f2"}
    eff2 = compute_feature_effectiveness(
        df, y, ["f1"], engine_supported={"f1"}, parity_features={"f1", "f2"})
    assert "f1" in eff2["E"], "有证据且判别显著的特征应入E"
    assert eff2["rows"]["f1"]["consistent"] is True


def test_class_profiles_use_label_names_not_ids(tmp_path):
    """class_profiles 键=类别名，不是倒置后的"0"/"1"（缺陷4）。"""
    import pandas as pd
    from src.reporting import build_mine_layer
    df = pd.DataFrame({"f1": [1.0, 2.0, 5.0, 6.0]})
    y = pd.Series([0, 0, 1, 1])
    build_mine_layer(df, y, ["f1"], [], {0: "alpha", 1: "beta"},
                     str(tmp_path))
    prof = json.loads((tmp_path / "class_profiles.json").read_text())
    assert set(prof.keys()) == {"alpha", "beta"}, \
        f"profile 键倒置为ID: {list(prof.keys())}"


def test_misclassified_includes_unknown_and_unpaired(tmp_path):
    """漏检(unknown)与未配对如实入误判清单（缺陷4）。"""
    from src.reporting import build_evaluate_layer
    b = tmp_path / "bundle"
    b.mkdir()
    (b / "rules.json").write_text("[]", encoding="utf-8")
    (b / "selected_features.json").write_text('["f1"]', encoding="utf-8")
    (b / "bundle_config.json").write_text('{}', encoding="utf-8")
    pf = tmp_path / "preds.json"
    pf.write_text(json.dumps({"results": [
        {"source_file": "a.pcap", "predicted_label": "unknown"},
        {"source_file": "zzz.pcap", "predicted_label": "alpha"},
        {"source_file": "b.pcap", "predicted_label": "alpha"}]}),
        encoding="utf-8")
    truth = {"a.pcap": {"label": "alpha"}, "b.pcap": {"label": "alpha"}}
    build_evaluate_layer(str(b), str(pf), truth, {"n_samples": 3},
                         str(tmp_path / "an"))
    lines = [json.loads(l) for l in
             (tmp_path / "an" / "misclassified_samples.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    kinds = {l["kind"] for l in lines}
    assert "miss_unknown" in kinds, "漏检(unknown)被排除出误判清单"
    assert "unpaired_truth" in kinds, "未配对预测被静默丢弃"


# ================================================================ D5 引擎算子
def test_engine_legal_operators_supported():
    """LEGAL_OPS 全集可被引擎求值：>=/</!=/exists（缺陷5）。"""
    from src.engine.matcher.optimized_engine import OptimizedDPIEngine
    eng = OptimizedDPIEngine()
    cases = [
        ({'op': '>=', 'value': 5.0}, {'f': 5.0}, True),
        ({'op': '>=', 'value': 5.0}, {'f': 4.9}, False),
        ({'op': '<', 'value': 5.0}, {'f': 4.9}, True),
        ({'op': '<', 'value': 5.0}, {'f': 5.0}, False),
        ({'op': '!=', 'value': 5.0}, {'f': 4.9}, True),
        ({'op': '!=', 'value': 5.0}, {'f': 5.0}, False),
        ({'op': 'exists'}, {'f': 0.0}, True),
        ({'op': 'exists'}, {'g': 1.0}, False),   # f 缺失 -> 整条不匹配
        ({'op': 'not_exists'}, {'g': 1.0}, True),
        ({'op': 'bogus_op', 'value': 1}, {'f': 1.0}, False),
    ]
    for cond, feats, should in cases:
        eng.rules = [{'id': 'T', 'name': 't', 'type': 'statistical',
                      'confidence': 0.9,
                      'conditions': [dict(cond, feature='f')],
                      'action': {'result': 'R', 'source': 'test'}}]
        res = eng.match(feats)
        assert bool(res) == should, f"{cond} on {feats}: {bool(res)} != {should}"


def test_detect_uses_bundle_and_shared_filter(tmp_path):
    """pipeline detect：bundle加载+共享过滤（<3包会话不入结果；缺陷5）。"""
    from src.pipeline import FeatureMiningPipeline
    from src.engine.model_io import save_rule_bundle
    # 1个正常会话 + 1个2包小会话（应被过滤）
    p = tmp_path / "mix.pcap"
    pkts = _session_packets("main", n_data=6)
    # 小会话：不同五元组，只有SYN+SYNACK（2包<3）
    C = ('10.0.0.3', 6000)
    pkts += [
        (5.0, _eth(_ip_tcp(C[0], B[0], C[1], B[1], 0x02, 9000))),
        (5.01, _eth(_ip_tcp(B[0], C[0], B[1], C[1], 0x12, 9100))),
    ]
    _write_pcap(p, pkts)
    bundle = tmp_path / "bundle"
    save_rule_bundle(
        [{'id': 'R1', 'name': 'r', 'type': 'statistical',
          'confidence': 0.9,
          'conditions': [{'feature': 'protocol_tcp', 'op': '==',
                          'value': '1.0'}],
          'action': {'result': 'appX', 'source': 'test'}}],
        ['protocol_tcp'], {0: 'appX'}, 0.5, str(bundle),
        bundle_meta={'task': 'app'})
    pipe = FeatureMiningPipeline()
    res = pipe.detect(str(p), str(bundle))
    files = {r['observation_id'].split(':')[0] for r in res}
    assert files == {'mix.pcap'}
    assert all(r['observation_id'] != 'mix.pcap:1' or True for r in res)
    # 只有小会话被过滤后剩余会话数 = 主会话1条
    assert len(res) == 1, f"<3包会话未被过滤或主会话丢失: {len(res)}"
    assert res[0]['result'] == 'appX'


# ================================================================ D7 小样本规则生成
def test_rule_generator_small_samples_no_5fold_crash():
    """3类×2样本：显式validation路径可用，固定5折CV不再崩溃（缺陷7）。"""
    import numpy as np
    import pandas as pd
    from src.engine.rule_compiler.optimized_generator import (
        OptimizedRuleGenerator)
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(6, 4), columns=[f'f{i}' for i in range(4)])
    X['f0'] += np.repeat([0, 5, 10], 2)      # 类可分
    y = pd.Series([0, 0, 1, 1, 2, 2])
    gen = OptimizedRuleGenerator()
    rules = gen.fit_and_generate(
        X, y, ['f0', 'f1'], {0: 'a', 1: 'b', 2: 'c'},
        validation=(X.copy(), y.copy()))
    assert isinstance(rules, list)


# ================================================================ D6 背景类
def test_behavior_background_class_option(tmp_path):
    """include_background=True 时背景窗作为显式类别入训练（缺陷6选项化）。"""
    from src.pipeline import FeatureMiningPipeline
    m2 = Path(__file__).resolve().parents[2] / 'output' / 'fixture_m2'
    if not (m2 / 'train' / 'voice_call_g1_s1.pcap').exists():
        pytest.skip("fixture_m2 未生成")
    mf = tmp_path / "m.jsonl"
    mf.write_text(_manifest_line(
        "g1", "voice_call_g1_s1", "synthIM", m2 / 'train/voice_call_g1_s1.pcap',
        'train', behavior="voice_call",
        behavior_start=2.0, behavior_end=18.0, terminal_ip="10.1.0.5"),
        encoding="utf-8")
    mf.write_text(mf.read_text(encoding='utf-8') + "\n" + _manifest_line(
        "g2", "text_chat_g2_s1", "synthIM", m2 / 'train/text_chat_g2_s1.pcap',
        'train', behavior="text_chat",
        behavior_start=2.0, behavior_end=18.0, terminal_ip="10.2.0.5"),
        encoding="utf-8")
    pipe = FeatureMiningPipeline()
    out1 = tmp_path / "no_bg"
    pipe.mine(None, str(out1), manifest=str(mf), task='behavior')
    out2 = tmp_path / "with_bg"
    pipe.mine(None, str(out2), manifest=str(mf), task='behavior',
              behavior_include_background=True)
    import pandas as pd
    n1 = len(pd.read_csv(out1 / 'raw_features.csv'))
    n2 = len(pd.read_csv(out2 / 'raw_features.csv'))
    assert n2 >= n1, "背景类开启后样本数不应减少"
    labels2 = json.loads((out2 / 'bundle' / 'bundle_config.json')
                         .read_text(encoding='utf-8'))['label_names']
    assert 'background' in labels2.values(), \
        f"背景类未入词表: {labels2}"


# ================================================================ D5b 算子族细化
def test_operator_families_fine_grained_and_parity():
    """算子族细化：族级调用计数 + 全族输出与全算 parity（缺陷5）。"""
    from src.features.operators import (
        ADV_FAMILIES, OPERATOR_CALLS, extract_on_demand, feature_to_family,
        reset_calls, resolve_required_families)
    from src.features.basic.feature_extractor import BasicFeatureExtractor
    from src.features.advanced.advanced_extractor import (
        AdvancedFeatureExtractor)

    # 合成会话
    from src.parser.session.session_manager import (
        FlowSession, PacketInfo, Protocol)
    s = FlowSession(src_ip=A[0], dst_ip=B[0], src_port=A[1], dst_port=B[1],
                    protocol=Protocol.TCP, start_time=0.0, end_time=0.8)
    seq = 1000
    for i in range(10):
        p = PacketInfo(timestamp=i * 0.08, src_ip=A[0], dst_ip=B[0],
                       src_port=A[1], dst_port=B[1], protocol=Protocol.TCP,
                       length=100 + i * 60, payload_length=60 + i * 50,
                       tcp_flags=0x18, tcp_seq=seq, payload=b'\x41' * (60 + i * 50),
                       direction=1 if i % 2 == 0 else -1)
        seq += 60 + i * 50
        s.packets.append(p)
        if p.direction == 1:
            s.total_fwd_packets += 1
            s.total_fwd_bytes += p.length
        else:
            s.total_bwd_packets += 1
            s.total_bwd_bytes += p.length

    basic_ext = BasicFeatureExtractor()
    adv_ext = AdvancedFeatureExtractor()

    # 族映射覆盖所有 advanced 族
    fams = {feature_to_family(f)
            for f in adv_ext.extract_all(s).keys()}
    assert fams <= set(ADV_FAMILIES) | {"advanced"}, f"未知族: {fams}"
    assert len(fams & set(ADV_FAMILIES)) >= 3, "族映射过粗"

    # 只需 basic + distribution 时：其他族0调用，复杂族（昂贵）不调用
    reset_calls()
    need = {'duration', 'size_quantile_0'}       # basic + 分布族
    fam_set = resolve_required_families(need)
    assert 'adv_complexity' not in fam_set, f"误激活昂贵族: {fam_set}"
    out1 = extract_on_demand(s, fam_set, basic_ext, adv_ext)
    assert OPERATOR_CALLS.get('adv_complexity', 0) == 0
    assert OPERATOR_CALLS.get('adv_distribution', 0) == 1
    assert OPERATOR_CALLS['basic'] == 1

    # 全族输出 == basic.extract_all + advanced.extract_all（parity）
    full = {**basic_ext.extract_all(s), **adv_ext.extract_all(s)}
    all_fams = resolve_required_families(set(full.keys()))
    reset_calls()
    out_all = extract_on_demand(s, all_fams, basic_ext, adv_ext)
    assert out_all.keys() == full.keys(), \
        f"按需全族输出schema与全算不一致: {set(full) ^ set(out_all)}"
    for k in full:
        assert out_all[k] == full[k], f"按需全族值失配 {k}"

    # 兼容：'advanced' 粗组 = 全部族
    reset_calls()
    out_coarse = extract_on_demand(s, {'advanced'}, basic_ext, adv_ext)
    adv_full = adv_ext.extract_all(s)
    assert out_coarse.keys() == adv_full.keys()
