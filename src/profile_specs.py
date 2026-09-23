# -*- coding: utf-8 -*-
"""Runtime profile contract shared by mine/detect/independent inference.

Profiles only describe observation/extraction/deployment semantics. They do not
contain fitted thresholds or test-derived choices. Keeping this contract in the
RuleBundle prevents training/inference from silently using different packet or
window budgets.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Optional


_IDENTIFIER_FEATURES = {
    "src_ip", "dst_ip", "src_port", "dst_port",
    "is_well_known_src_port", "is_well_known_dst_port",
    "protocol_tcp", "protocol_udp",
    "tls_has_sni", "tls_sni_length", "tls_sni",
    "tls_ja3_hash_prefix", "tls_ja4_hash_prefix",
    "tls_version",
    "quic_sni", "quic_useragent", "quic_version",
}


PROFILE_SPECS: Dict[str, Dict] = {
    # Legacy profiles remain intentionally permissive for backwards compatibility.
    "full": {
        "profile_name": "full",
        "observation_unit": "flow",
        "feature_budget": None,
        "temporal_vote": 1,
        "forbidden_features": [],
    },
    "fast16": {
        "profile_name": "fast16",
        "observation_unit": "flow_prefix",
        "max_packets": 32,
        "feature_budget": 16,
        "temporal_vote": 1,
        "forbidden_features": [],
    },
    "fast32": {
        "profile_name": "fast32",
        "observation_unit": "flow_prefix",
        "max_packets": 32,
        "feature_budget": 32,
        "temporal_vote": 1,
        "forbidden_features": [],
    },
    "application64": {
        "profile_name": "application64",
        "task": "app",
        "observation_unit": "flow_prefix",
        "max_packets": 64,
        "feature_budget": 64,
        "temporal_vote": 1,
        "identifier_policy": "exclude",
        "forbidden_features": sorted(_IDENTIFIER_FEATURES),
    },
    "behavior15": {
        "profile_name": "behavior15",
        "task": "behavior",
        "observation_unit": "flow_window",
        "feature_budget": 64,
        "temporal_vote": 1,
        "identifier_policy": "exclude",
        "forbidden_features": sorted(_IDENTIFIER_FEATURES),
        "context": {
            "window_sec": 15.0,
            "step_sec": 15.0,
            "max_segments": 3,
            "ordering": "start_time_then_first_packet_index",
            "segment_gap_sec": 2.0,
            "overflow": {"aggregate_remaining": True},
            "min_window_packets": 5,
        },
    },
    "tunnel15": {
        "profile_name": "tunnel15",
        "task": "tool",
        "observation_unit": "tunnel_flow_window",
        "feature_budget": 64,
        "temporal_vote": 3,
        "max_windows_per_file": 200,
        "identifier_policy": "exclude",
        "forbidden_features": sorted(_IDENTIFIER_FEATURES),
        "context": {
            "window_sec": 15.0,
            "step_sec": 15.0,
            "max_segments": 3,
            "ordering": "start_time_then_first_packet_index",
            "segment_gap_sec": 2.0,
            "overflow": {"aggregate_remaining": True},
            "min_window_packets": 5,
        },
    },
}


def profile_names() -> List[str]:
    return list(PROFILE_SPECS)


def get_profile_spec(name: str, task: Optional[str] = None) -> Dict:
    """Return a defensive copy and validate a profile/task combination."""
    if name not in PROFILE_SPECS:
        raise ValueError(
            f"未知特征profile: {name}（可选 {' / '.join(profile_names())}）")
    spec = deepcopy(PROFILE_SPECS[name])
    required = spec.get("task")
    if required and task and required != task:
        raise ValueError(
            f"profile={name} 仅用于 task={required}，当前 task={task}")
    if task:
        spec.setdefault("task", task)
    # Historical behavior bundles used profile=full/fast but still ran the
    # window runtime. Preserve that contract when upgrading old CLI calls.
    if task == "behavior" and name in ("full", "fast16", "fast32"):
        spec["observation_unit"] = "behavior_window"
    return spec


def filter_forbidden_features(features: Iterable[str], spec: Dict) -> List[str]:
    """Apply the profile identifier policy without mutating source feature data."""
    blocked = set(spec.get("forbidden_features") or ())
    if not blocked:
        return list(features)
    return [f for f in features if f not in blocked]


def temporal_vote_rows(rows: List[Dict], vote_size: int,
                       label_key: str = "predicted_label") -> List[Dict]:
    """Non-overlapping causal temporal vote used by tunnel bundles.

    A partial tail with fewer than two windows is not emitted. Ties are resolved
    by summed confidence, then first occurrence, making replay deterministic.
    """
    n = max(1, int(vote_size or 1))
    if n <= 1:
        return [dict(r) for r in rows]
    # If a flow/session identifier is present, never vote across flows. This is
    # the frozen ISCXTor semantics; crossing sessions can fabricate context.
    groups: List[List[Dict]] = []
    if any("session_index" in r for r in rows):
        by = {}
        order = []
        for r in rows:
            key = r.get("session_index")
            if key not in by:
                by[key] = []
                order.append(key)
            by[key].append(r)
        for key in order:
            groups.append(sorted(by[key], key=lambda r: float(r.get("window_start", 0) or 0)))
    else:
        groups = [rows]

    out: List[Dict] = []
    for seq in groups:
        for start in range(0, len(seq), n):
            block = seq[start:start + n]
            if len(block) < 2:
                continue
            labels = [str(r.get(label_key, "unknown")) for r in block]
            counts: Dict[str, int] = {}
            confs: Dict[str, float] = {}
            first: Dict[str, int] = {}
            for i, (lab, row) in enumerate(zip(labels, block)):
                counts[lab] = counts.get(lab, 0) + 1
                confs[lab] = confs.get(lab, 0.0) + float(row.get("confidence", 0.0) or 0.0)
                first.setdefault(lab, i)
            winner = max(counts, key=lambda lab: (counts[lab], confs[lab], -first[lab]))
            chosen = next((dict(r) for r in block if str(r.get(label_key, "unknown")) == winner),
                          dict(block[0]))
            chosen[label_key] = winner
            chosen["confidence"] = confs[winner] / max(counts[winner], 1)
            chosen["vote_size"] = len(block)
            chosen["vote_labels"] = labels
            if "window_start" in block[0]:
                chosen["window_start"] = block[0].get("window_start")
            if "window_end" in block[-1]:
                chosen["window_end"] = block[-1].get("window_end")
            out.append(chosen)
    return out
