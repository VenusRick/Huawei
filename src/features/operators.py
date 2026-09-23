# -*- coding: utf-8 -*-
"""特征算子组与按需计算调度（M4/G08；2026-09-18 审计细化到特征族）。

依赖链：RuleBundle -> required_features -> required_operators -> 运行时只
激活所需算子组。

两层调度（审计 2026-09-18 细化）：
- 粗组（兼容层）：basic / advanced——test_on_demand 与旧bundle口径；
- 特征族（细化层）：advanced 内部按 AdvancedFeatureExtractor 的5个族
  方法再切分（multiscale/direction/distribution/complexity/cross），
  昂贵族（complexity：近似熵/Hurst）只在需要时调用。
  特征名->族的映射由"每个族方法在合成会话上实际产出的特征名"动态
  推导（防硬编码名漂移），进程内缓存。

不先算 319 维再取 3 维：规则所需特征落在哪些族就只调哪些族，
调用计数（OPERATOR_CALLS，含族级）留证。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

# 粗组定义（兼容层）
_ADVANCED_MARKERS = (
    "burst", "iit", "iat_", "entropy", "ngram", "gapped", "bow_",
    "dir_2gram", "packet_bow", "tcp_window", "byte_dist", "var_",
)
OPERATOR_GROUPS: Dict[str, Set[str]] = {
    "basic_counter": {"basic"},
    "advanced_behavioral": {"advanced"},
}

# 特征族 -> AdvancedFeatureExtractor 族方法（细化层）
ADV_FAMILIES: Dict[str, str] = {
    "adv_multiscale": "_extract_multi_scale_features",
    "adv_direction": "_extract_directional_patterns",
    "adv_distribution": "_extract_distribution_features",
    "adv_complexity": "_extract_complexity_features",
    "adv_cross": "_extract_cross_features",
}
# 族成本标注（复杂度族含近似熵/Hurst，最昂贵）
FAMILY_COST: Dict[str, str] = {
    "basic": "low",
    "adv_multiscale": "medium",
    "adv_direction": "low",
    "adv_distribution": "medium",
    "adv_complexity": "high",
    "adv_cross": "medium",
}

# 调用计数（进程内累计，供验收取证；含族级键）
OPERATOR_CALLS: Dict[str, int] = {"basic": 0, "advanced": 0}


def feature_to_group(feature: str) -> str:
    """特征名 -> 粗组（basic | advanced）。"""
    f = feature.lower()
    return "advanced" if any(m in f for m in _ADVANCED_MARKERS) else "basic"


_FAMILY_INDEX: Optional[Dict[str, str]] = None


def _build_family_index() -> Dict[str, str]:
    """合成会话上跑 basic + 每个族方法一次，记录特征名->族（动态防漂移）。

    basic extractor 的产出先入表（防 marker 误判：如 tcp_window 按前缀
    像advanced实为basic产出）；advanced 族名后入（不覆盖）。
    """
    from src.features.advanced.advanced_extractor import (
        AdvancedFeatureExtractor)
    from src.features.basic.feature_extractor import BasicFeatureExtractor
    from src.parser.session.session_manager import (
        FlowSession, PacketInfo, Protocol)
    s = FlowSession(src_ip="10.0.0.2", dst_ip="10.0.0.1", src_port=5000,
                    dst_port=443, protocol=Protocol.TCP,
                    start_time=0.0, end_time=1.0)
    seq = 1000
    for i in range(8):
        p = PacketInfo(timestamp=i * 0.1, src_ip="10.0.0.2",
                       dst_ip="10.0.0.1", src_port=5000, dst_port=443,
                       protocol=Protocol.TCP, length=100 + i * 50,
                       payload_length=50 + i * 40, tcp_flags=0x18,
                       tcp_seq=seq, payload=b"\x41" * (50 + i * 40),
                       direction=1 if i % 2 == 0 else -1)
        seq += 50 + i * 40
        s.packets.append(p)
    s.total_fwd_packets = sum(1 for p in s.packets if p.direction == 1)
    s.total_bwd_packets = sum(1 for p in s.packets if p.direction == -1)
    s.total_fwd_bytes = sum(p.length for p in s.packets if p.direction == 1)
    s.total_bwd_bytes = sum(p.length for p in s.packets if p.direction == -1)
    basic_ext = BasicFeatureExtractor()
    ext = AdvancedFeatureExtractor()
    idx: Dict[str, str] = {}
    try:
        for n in basic_ext.extract_all(s).keys():
            idx[n] = "basic"
    except Exception:  # noqa: BLE001
        pass
    for fam, method in ADV_FAMILIES.items():
        try:
            names = set(getattr(ext, method)(s).keys())
        except Exception:  # noqa: BLE001
            names = set()
        for n in names:
            idx.setdefault(n, fam)   # basic 已占名不覆盖
    return idx


def feature_to_family(feature: str) -> str:
    """特征名 -> 特征族（basic | adv_*；未知 advanced 特征归 adv_cross）。"""
    global _FAMILY_INDEX
    if _FAMILY_INDEX is None:
        _FAMILY_INDEX = _build_family_index()
    if feature in _FAMILY_INDEX:
        return _FAMILY_INDEX[feature]
    return "advanced" if feature_to_group(feature) == "advanced" else "basic"


def required_features_of_rules(rules: Iterable[dict],
                               selected_features: Iterable[str]) -> List[str]:
    """纯规则运行时只依赖条件引用；候选特征目录不等于计算依赖。

    selected_features 保留为兼容参数。它是训练侧的候选/编辑目录，
    未被部署规则引用的特征不应触发算子。模型对照模式不走本函数。
    """
    used: Set[str] = set()
    for r in rules:
        if r.get("type") == "ensemble_classifier":
            continue  # 独立规则引擎不执行训练模型占位项
        for c in r.get("conditions", []):
            feat = c.get("feature")
            if feat:
                used.add(feat)
    return sorted(used)


def resolve_required_operators(features: Iterable[str]) -> Set[str]:
    """特征集合 -> 粗组集合（兼容层：basic | advanced）。"""
    return {feature_to_group(f) for f in features}


def resolve_required_families(features: Iterable[str]) -> Set[str]:
    """特征集合 -> 特征族集合（细化层：basic | adv_*）。

    fast profile 判定不在本函数（宿主契约：依赖解析只做族映射）——
    调用方（如 dpi_infer）用 fast_profile.profile_of_features 先判
    能否整组落入 fast16/fast32，可以则激活 fast 组（同名特征同一批
    族方法计算，逐值一致），否则用本函数的族映射。
    """
    out: Set[str] = set()
    for f in features:
        fam = feature_to_family(f)
        out.add("advanced" if fam == "advanced" else fam)
    return out


def reset_calls() -> None:
    OPERATOR_CALLS.clear()
    OPERATOR_CALLS.update({"basic": 0, "advanced": 0})


def extract_on_demand(session, active_groups: Set[str],
                      basic_extractor, advanced_extractor) -> Dict:
    """按需提取：只调用激活组/族的 extractor（计数留证）。

    active_groups 接受粗组名（basic/advanced）、族名（adv_*）或
    fast profile 名（fast16/fast32——basic extractor 的按名提取，
    未请求族一次不调用，2026-09-18 第三轮）。
    返回合并特征 dict；未激活族的特征不产出（引擎也用不到）。
    """
    from src.features.fast_profile import FAST_PROFILES
    feats: Dict = {}
    for prof, names in FAST_PROFILES.items():
        if prof in active_groups:
            OPERATOR_CALLS[prof] = OPERATOR_CALLS.get(prof, 0) + 1
            feats.update(basic_extractor.extract(names, session))
    if "basic" in active_groups:
        OPERATOR_CALLS["basic"] = OPERATOR_CALLS.get("basic", 0) + 1
        feats.update(basic_extractor.extract_all(session))
    if "advanced" in active_groups:
        # 粗组（兼容旧bundle语义）：整个 advanced extractor 一次调用
        OPERATOR_CALLS["advanced"] = OPERATOR_CALLS.get("advanced", 0) + 1
        feats.update(advanced_extractor.extract_all(session))
    for fam, method in ADV_FAMILIES.items():
        if fam in active_groups:
            OPERATOR_CALLS[fam] = OPERATOR_CALLS.get(fam, 0) + 1
            feats.update(getattr(advanced_extractor, method)(session))
    return feats
