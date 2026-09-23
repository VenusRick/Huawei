# -*- coding: utf-8 -*-
"""快速特征profile回归（2026-09-18 第三轮 fast_dpi）：
profile与全量同名特征逐值一致、按名提取不调用未请求族、算子组调度、
bundle默认profile/前缀包数、观测级评价（固定清单=唯一分母）。

运行：python3 -m pytest tests/regression/test_fast_dpi_20260918_03.py -q
"""
import json
import random
import socket
import struct
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

A, B = ('10.0.0.2', 5000), ('10.0.0.1', 443)


def _synthetic_session(n=40, seed=7):
    from src.parser.session.session_manager import (
        FlowSession, PacketInfo, Protocol)
    s = FlowSession(src_ip=A[0], dst_ip=B[0], src_port=A[1], dst_port=B[1],
                    protocol=Protocol.TCP, start_time=0.0, end_time=0.9)
    rng = random.Random(seed)
    seq = 1000
    for i in range(n):
        pl = rng.randint(20, 400)
        p = PacketInfo(timestamp=i * 0.03 + rng.random() * 0.02,
                       src_ip=A[0], dst_ip=B[0], src_port=A[1], dst_port=B[1],
                       protocol=Protocol.TCP, length=pl + 40,
                       payload_length=pl, tcp_flags=0x18, tcp_seq=seq,
                       payload=b"\x41" * pl, direction=1 if i % 2 == 0 else -1)
        seq += pl
        s.packets.append(p)
        if p.direction == 1:
            s.total_fwd_packets += 1
            s.total_fwd_bytes += p.length
        else:
            s.total_bwd_packets += 1
            s.total_bwd_bytes += p.length
    return s


def test_fast_profile_parity_with_full_extraction():
    """fast16/fast32 与全量 extract_all 同名特征逐值一致（公式单一来源）。"""
    from src.features.basic.feature_extractor import BasicFeatureExtractor
    from src.features.fast_profile import FAST16, FAST32
    s = _synthetic_session()
    b = BasicFeatureExtractor()
    full = b.extract_all(s)
    f16 = b.extract(FAST16, s)
    f32 = b.extract(FAST32, s)
    assert set(f16) == set(FAST16), f"fast16名字失配: {set(f16) ^ set(FAST16)}"
    assert set(f32) == set(FAST32), f"fast32名字失配: {set(f32) ^ set(FAST32)}"
    for k in FAST32:
        assert f32[k] == full[k], f"fast/full失配 {k}: {f32[k]} != {full[k]}"


def test_extract_requested_missing_name_not_fabricated():
    """请求不存在的名字不产出该键（缺失语义交给调用方，不伪造0）。"""
    from src.features.basic.feature_extractor import BasicFeatureExtractor
    b = BasicFeatureExtractor()
    out = b.extract(['total_packets', 'no_such_feature'], _synthetic_session(6))
    assert 'total_packets' in out and 'no_such_feature' not in out


def test_extract_requested_skips_unneeded_families():
    """只请求 flow_stats/session_info 名字时，其余族方法一次不调用。"""
    from src.features.basic.feature_extractor import BasicFeatureExtractor
    from src.features.fast_profile import FAST16
    b = BasicFeatureExtractor()
    calls = []

    def _wrap(method_name):
        orig = getattr(b, method_name)

        def _hook(session):
            calls.append(method_name)
            return orig(session)
        return _hook

    for m in BasicFeatureExtractor.FAMILY_METHODS.values():
        setattr(b, m, _wrap(m))
    out = b.extract(FAST16, _synthetic_session(6))   # 仅3族
    needed = {'_extract_session_info', '_extract_flow_stats',
              '_extract_timing_features'}
    assert set(calls) <= needed, f"未请求族被调用: {set(calls) - needed}"
    assert needed <= set(calls)
    assert set(out) == set(FAST16)


def test_on_demand_fast_group_zero_advanced_calls():
    """fast 组调度：fast32 计数=1，advanced/昂贵族全 0。"""
    from src.features.operators import (OPERATOR_CALLS, extract_on_demand,
                                        reset_calls)
    from src.features.basic.feature_extractor import BasicFeatureExtractor
    from src.features.advanced.advanced_extractor import (
        AdvancedFeatureExtractor)
    from src.features.fast_profile import FAST32
    reset_calls()
    out = extract_on_demand(_synthetic_session(6), {'fast32'},
                            BasicFeatureExtractor(),
                            AdvancedFeatureExtractor())
    assert set(out) == set(FAST32)
    assert OPERATOR_CALLS.get('fast32') == 1
    assert all(OPERATOR_CALLS.get(k, 0) == 0 for k in
               ('basic', 'advanced', 'adv_complexity', 'adv_multiscale',
                'adv_distribution', 'adv_cross', 'adv_direction'))


def test_profile_of_features_and_resolve():
    """覆盖判断：fast16子集→fast16；含TLS名→fast32；含高级名→None；
    族映射保持宿主契约（resolve_required_families 只做族映射）。"""
    from src.features.fast_profile import profile_of_features
    from src.features.operators import resolve_required_families
    assert profile_of_features({'total_packets', 'iat_mean'}) == 'fast16'
    assert profile_of_features({'tls_has_sni'}) == 'fast32'
    assert profile_of_features(set()) is None
    assert profile_of_features({'pkt_size_entropy'}) is None
    assert resolve_required_families(['total_packets']) == {'basic'}


def _write_pcap(path: Path, variant: int = 0):
    def ip_tcp(src, dst, sport, dport, flags, seq, payload=b''):
        tcp = struct.pack('!HHIIBBHHH', sport, dport, seq, 1, 5 << 4, flags,
                          8192, 0, 0)
        total = 20 + len(tcp) + len(payload)
        ip = struct.pack('!BBHHHBBH', 0x45, 0, total, 0, 0, 64, 6, 0)
        ip += socket.inet_aton(src) + socket.inet_aton(dst)
        return ip + tcp + payload

    def eth(ip_pkt):
        return (b'\x11\x22\x33\x44\x55\x66\xaa\xbb\xcc\xdd\xee\xff\x08\x00'
                + ip_pkt)

    fwd_len = 200 + variant * 180          # 类间可分：包长/节奏不同
    bwd_len = 150 + variant * 30
    gap = 0.05 + variant * 0.12
    pkts = []
    pkts.append((1.0, eth(ip_tcp(A[0], B[0], A[1], B[1], 0x02, 1000))))
    pkts.append((1.02, eth(ip_tcp(B[0], A[0], B[1], A[1], 0x12, 5000))))
    pkts.append((1.04, eth(ip_tcp(A[0], B[0], A[1], B[1], 0x10, 1001))))
    seq = 1001
    for i in range(40):
        ts = 1.1 + i * gap
        if i % 2 == 0:
            pkts.append((ts, eth(ip_tcp(A[0], B[0], A[1], B[1], 0x18,
                                        seq, b'\x41' * fwd_len))))
            seq += fwd_len
        else:
            pkts.append((ts, eth(ip_tcp(B[0], A[0], B[1], A[1], 0x18,
                                        5001 + i, b'\x42' * bwd_len))))
    with open(path, 'wb') as f:
        f.write(struct.pack('<IHHiIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))
        for ts, raw in pkts:
            sec = int(ts)
            usec = int(round((ts - sec) * 1e6))
            f.write(struct.pack('<IIII', sec, usec, len(raw), len(raw)))
            f.write(raw)
    return pkts


def test_bundle_profile_defaults_used_by_cli(tmp_path):
    """bundle 携带 feature_profile/max_packets_per_session：CLI 默认读取
    （前缀32包、只算profile特征；禁止训练32包CLI悄悄500包/319维）。"""
    from src.engine.model_io import save_rule_bundle
    pcap = tmp_path / "s.pcap"
    pkts = _write_pcap(pcap)
    bundle = tmp_path / "bundle"
    save_rule_bundle(
        [{'id': 'R1', 'name': 'r', 'type': 'statistical', 'priority': 100,
          'confidence': 1.0,
          'conditions': [{'feature': 'total_packets', 'op': '>=', 'value': 0}],
          'action': {'result': 'siteA', 'source': 'statistical'}}],
        ['total_packets', 'duration'], {0: 'siteA'}, 0.5, str(bundle),
        bundle_meta={'task': 'app', 'feature_profile': 'fast16',
                     'max_packets_per_session': 32})
    out = tmp_path / "pred.json"
    r = subprocess.run(
        [sys.executable, '-m', 'src.engine.dpi_infer',
         '--rules', str(bundle), '--pcap', str(pcap),
         '-o', str(out), '--on-demand'],
        capture_output=True, text=True, cwd=str(Path(__file__).parents[2]),
        timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'fast16' in r.stdout, "CLI未按bundle默认profile提取"
    assert '会话前缀包数(bundle默认): 32' in r.stdout, "CLI未按bundle前缀包数"
    data = json.loads(out.read_text())
    assert data['operator_calls'].get('fast16', 0) >= 1
    assert data['operator_calls'].get('basic', 0) == 0
    assert data['operator_calls'].get('adv_complexity', 0) == 0
    assert data['results'], "应产出预测"
    assert data['results'][0]['predicted_label'] == 'siteA'


def test_mine_profile_fast32_rows_and_bundle_meta(tmp_path):
    """mine --profile fast32：行内特征=profile名集；bundle携带profile与前缀。"""
    from src.pipeline import FeatureMiningPipeline
    pcap = tmp_path / "s.pcap"
    _write_pcap(pcap)
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.pcap").write_bytes(pcap.read_bytes())
    _write_pcap(pcap, variant=1)          # 类间可分
    (d / "b.pcap").write_bytes(pcap.read_bytes())
    pipe = FeatureMiningPipeline()
    out = tmp_path / "mine"
    pipe.mine(str(d), str(out), labels={"a.pcap": 0, "b.pcap": 1},
              task='app', profile='fast32')
    import pandas as pd
    df = pd.read_csv(out / 'raw_features.csv')
    feat_cols = [c for c in df.columns if not str(c).startswith('_')]
    from src.features.fast_profile import FAST32
    assert set(feat_cols) == set(FAST32), \
        f"fast32行特征失配: {set(feat_cols) ^ set(FAST32)}"
    cfg = json.loads((out / 'bundle' / 'bundle_config.json').read_text())
    assert cfg['feature_profile'] == 'fast32'
    assert cfg['max_packets_per_session'] == 32


def test_evaluate_observations_fixed_denominator(tmp_path):
    """观测级评价：固定清单=唯一分母；缺预测行计错不冒充拒识；
    清单外行单列；重复清单observation_id显式报错。"""
    from src.evaluation import evaluate_observations, load_observation_list
    obs = tmp_path / "obs.jsonl"
    obs.write_text("\n".join(json.dumps(o) for o in [
        {"observation_id": "f.pcap:0", "label": "siteA"},
        {"observation_id": "f.pcap:1", "label": "siteA"},
        {"observation_id": "f.pcap:2", "label": "siteB"},
    ]), encoding="utf-8")
    pf = tmp_path / "pred.json"
    pf.write_text(json.dumps({"results": [
        {"observation_id": "f.pcap:0", "predicted_label": "siteA"},
        # f.pcap:1 无预测行 -> missing（计错）
        {"observation_id": "f.pcap:2", "predicted_label": "siteA"},  # 错分
        {"observation_id": "f.pcap:9", "predicted_label": "siteA"},  # 清单外
        {"observation_id": "f.pcap:0", "predicted_label": "siteA"},  # 重复
    ]}), encoding="utf-8")
    observations = load_observation_list(str(obs))
    m = evaluate_observations(str(pf), observations,
                              str(tmp_path / "e"))
    assert m["n_observations"] == 3
    assert m["n_samples"] == 3
    assert m["n_missing_predictions"] == 1
    assert m["missing_prediction_ids"] == ["f.pcap:1"]
    assert m["n_extra_predictions"] == 1
    assert m["n_duplicate_predictions"] == 1
    # 缺行+错分：siteA recall = 1/2（missing 不能被静默剔除）
    assert m["per_class"]["siteA"]["recall"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(1 / 3)
    assert m["evaluation_scope"]["status"] == "incomplete"
    # 清单重复 -> 显式报错
    obs2 = tmp_path / "obs2.jsonl"
    obs2.write_text("\n".join(json.dumps(o) for o in [
        {"observation_id": "f.pcap:0", "label": "siteA"},
        {"observation_id": "f.pcap:0", "label": "siteA"},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_observation_list(str(obs2))
