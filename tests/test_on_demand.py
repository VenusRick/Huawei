# -*- coding: utf-8 -*-
"""按需计算验收（G08）：算子组调度、调用计数留证、输出与全算一致。"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.operators import (OPERATOR_CALLS, extract_on_demand,
                                    feature_to_group,
                                    required_features_of_rules,
                                    resolve_required_operators, reset_calls)

FIXTURE = Path(__file__).resolve().parents[1] / "output" / "fixture_m1"
ROOT = Path(__file__).resolve().parents[1]


def test_feature_group_mapping():
    assert feature_to_group("duration") == "basic"
    assert feature_to_group("fwd_bytes") == "basic"
    assert feature_to_group("max_burst_size") == "advanced"
    assert feature_to_group("mean_iat") in ("basic", "advanced")  # 前缀判定
    assert feature_to_group("packet_bow_11") == "advanced"


def test_resolve_only_basic():
    rules = [{"conditions": [{"feature": "duration"}, {"feature": "fwd_bytes"}]}]
    feats = required_features_of_rules(rules, ["duration"])
    assert resolve_required_operators(feats) == {"basic"}


def test_resolve_mixed():
    rules = [{"conditions": [{"feature": "duration"},
                             {"feature": "max_burst_size"}]}]
    feats = required_features_of_rules(rules, [])
    assert resolve_required_operators(feats) == {"basic", "advanced"}


class _FakeExt:
    def __init__(self, tag):
        self.tag = tag

    def extract_all(self, session):
        return {self.tag: 1.0}


def test_call_count_only_active_called():
    reset_calls()
    out = extract_on_demand(object(), {"basic"}, _FakeExt("b"), _FakeExt("a"))
    assert out == {"b": 1.0}
    assert OPERATOR_CALLS == {"basic": 1, "advanced": 0}
    reset_calls()
    out = extract_on_demand(object(), {"advanced"}, _FakeExt("b"), _FakeExt("a"))
    assert out == {"a": 1.0}
    assert OPERATOR_CALLS == {"basic": 0, "advanced": 1}
    reset_calls()


@pytest.mark.skipif(not (FIXTURE / "test" / "appA").exists(),
                    reason="fixture 未生成")
def test_on_demand_output_parity(tmp_path):
    """全算 vs --on-demand：同 bundle 同 pcap 预测语义一致。

    输出路径用 tmp_path，避免固定 /tmp 路径被他进程遗留文件占用
    （stale 文件导致 PermissionError 且测试互相污染）。
    """
    bundle = ROOT / "output" / "p0_e2e" / "mine" / "bundle"
    if not bundle.exists():
        pytest.skip("p0_e2e bundle 不存在")
    pcap = sorted((FIXTURE / "test" / "appA").glob("*.pcap"))[0]
    out_path = tmp_path / "od_par.json"

    def run(extra):
        r = subprocess.run(
            [sys.executable, "-m", "src.engine.dpi_infer",
             "--rules", str(bundle), "--pcap", str(pcap),
             "-o", str(out_path)] + extra,
            cwd=str(ROOT), capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-300:]
        return json.loads(out_path.read_text())["results"]

    full = run([])
    ondm = run(["--on-demand"])
    assert len(full) == len(ondm)
    for a, b in zip(full, ondm):
        assert a["predicted_label"] == b["predicted_label"], \
            f"输出不一致: {a['predicted_label']} vs {b['predicted_label']}"
        assert abs(a.get("confidence", 0) - b.get("confidence", 0)) < 1e-9
