# -*- coding: utf-8 -*-
import json
import subprocess
import sys

from src.engine.model_io import load_rule_bundle, save_rule_bundle
from src.parser.context import TunnelWindowFeatureBuilder
from src.parser.context import BehaviorFlowWindowFeatureBuilder
from src.parser.session.session_manager import PacketInfo
from src.profile_specs import (filter_forbidden_features, get_profile_spec,
                               temporal_vote_rows)


def test_profile_contracts_and_legacy_behavior():
    app = get_profile_spec("application64", "app")
    assert app["max_packets"] == 64
    assert app["feature_budget"] == 64
    assert app["observation_unit"] == "flow_prefix"
    beh = get_profile_spec("behavior15", "behavior")
    assert beh["context"]["window_sec"] == 15.0
    assert beh["context"]["step_sec"] == 15.0
    assert beh["observation_unit"] == "flow_window"
    tun = get_profile_spec("tunnel15", "tool")
    assert tun["temporal_vote"] == 3
    assert tun["observation_unit"] == "tunnel_flow_window"
    # Historical behavior --profile full must remain window based.
    assert get_profile_spec("full", "behavior")["observation_unit"] == "behavior_window"


def test_identifier_policy_filters_known_shortcuts():
    spec = get_profile_spec("application64", "app")
    feats = ["duration", "src_port", "dst_port", "tls_has_sni",
             "tls_sni_length", "tls_ja3_hash_prefix", "pkt_size_mean"]
    assert filter_forbidden_features(feats, spec) == ["duration", "pkt_size_mean"]


def test_tunnel_builder_is_fixed_64_and_identifier_free():
    events = []
    for i in range(20):
        events.append((i * 0.5, "10.0.0.1", "10.0.0.2", "tcp-key",
                       [40, 583, 1500, 1064][i % 4], i, 1 if i % 3 else 0))
    win = {
        "window_start": 0.0, "window_end": 15.0,
        "n_connections": 1, "n_segments": 1, "n_overflow_segments": 0,
        "overflow": [], "segments": [], "events": events,
        "truncated_prefix": False,
    }
    f = TunnelWindowFeatureBuilder().build(win)
    assert len(f) == 64
    assert all(k.startswith("tunnel_") for k in f)
    forbidden = ("ip", "port", "sni", "ja3", "ja4", "useragent")
    assert not any(any(x in k.lower() for x in forbidden) for k in f)
    assert f["tunnel_size_eq40_ratio"] > 0
    assert f["tunnel_size_560_610_ratio"] > 0
    assert f["tunnel_size_ge1450_ratio"] > 0


def test_temporal_vote_three_windows():
    rows = [
        {"predicted_label": "Tor", "confidence": .9, "window_start": 0, "window_end": 15},
        {"predicted_label": "NonTor", "confidence": .6, "window_start": 15, "window_end": 30},
        {"predicted_label": "Tor", "confidence": .8, "window_start": 30, "window_end": 45},
        {"predicted_label": "NonTor", "confidence": .7, "window_start": 45, "window_end": 60},
    ]
    out = temporal_vote_rows(rows, 3)
    assert len(out) == 1  # one complete/usable block; one-window tail is not emitted
    assert out[0]["predicted_label"] == "Tor"
    assert out[0]["window_start"] == 0
    assert out[0]["window_end"] == 45
    assert out[0]["vote_size"] == 3


def test_temporal_vote_never_crosses_sessions():
    rows = [
        {"session_index": 1, "predicted_label": "Tor", "confidence": .8,
         "window_start": 0, "window_end": 15},
        {"session_index": 2, "predicted_label": "NonTor", "confidence": .9,
         "window_start": 0, "window_end": 15},
        {"session_index": 1, "predicted_label": "Tor", "confidence": .7,
         "window_start": 15, "window_end": 30},
    ]
    out = temporal_vote_rows(rows, 3)
    assert len(out) == 1
    assert out[0]["session_index"] == 1
    assert out[0]["predicted_label"] == "Tor"
    assert out[0]["vote_size"] == 2


def test_behavior_flow_window_builder_fixed_64():
    ps = []
    for i in range(12):
        p = PacketInfo(timestamp=i * .2, length=100 + i * 20,
                       payload_length=40 + i * 10,
                       direction=1 if i % 2 == 0 else -1)
        ps.append(p)
    f = BehaviorFlowWindowFeatureBuilder().build(ps)
    assert len(f) == 64
    assert all(k.startswith("behavior_flow_") for k in f)
    assert f["behavior_flow_n_packets"] == 12
    assert f["behavior_flow_fwd_packets"] == 6
    assert f["behavior_flow_bwd_packets"] == 6


def test_rule_bundle_roundtrip_and_independent_cli(tmp_path):
    bundle = tmp_path / "bundle"
    spec = get_profile_spec("tunnel15", "tool")
    rules = [{
        "id": "R1", "name": "tor-test", "type": "statistical",
        "priority": 1, "confidence": .99,
        "conditions": [{"feature": "tunnel_n_pkt", "op": ">=", "value": 1}],
        "action": {"result": "Tor", "confidence": .99, "source": "unit"},
    }]
    save_rule_bundle(rules, ["tunnel_n_pkt"], {0: "Tor"}, .7,
                     str(bundle), bundle_meta={
                         "task": "tool", "feature_profile": "tunnel15",
                         "profile_spec": spec, "context": spec["context"],
                         "temporal_vote": 3,
                     })
    _, feats, labels, cfg = load_rule_bundle(str(bundle))
    assert feats == ["tunnel_n_pkt"]
    assert labels["0"] == "Tor"
    assert cfg["profile_spec"]["temporal_vote"] == 3

    fv = tmp_path / "features.json"
    out = tmp_path / "pred.json"
    fv.write_text(json.dumps([{"tunnel_n_pkt": 10.0}]))
    p = subprocess.run(
        [sys.executable, "-m", "src.engine.dpi_infer", "--rules", str(bundle),
         "--features", str(fv), "-o", str(out)],
        text=True, capture_output=True, check=True)
    assert "Tor" in p.stdout
    pred = json.loads(out.read_text())
    assert pred["results"][0]["predicted_label"] == "Tor"
