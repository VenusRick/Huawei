# -*- coding: utf-8 -*-
"""特征算子组与按需计算调度（M4/G08）。

依赖链：RuleBundle -> required_features -> required_operators -> 运行时只
激活所需算子组。首版两组调度（basic/advanced），组内一起计算（方案允许），
但必须实测减少哪些组的调用（OPERATOR_CALLS 计数器）。

不先算 319 维再取 3 维：当规则所需特征全部落在 basic 组时，
AdvancedFeatureExtractor 不被调用（反之亦然），调用计数留证。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Set

# 算子组定义：组名 -> 该组产出的特征（前缀/全名混合判定）
# basic 组：会话级统计计数/包长/方向/端口等（BasicFeatureExtractor 产出）
# advanced 组：burst/IAT/熵/ngram/方向二元组等（AdvancedFeatureExtractor 产出）
_ADVANCED_MARKERS = (
    "burst", "iit", "iat_", "entropy", "ngram", "gapped", "bow_",
    "dir_2gram", "packet_bow", "tcp_window", "byte_dist", "var_",
)
OPERATOR_GROUPS: Dict[str, Set[str]] = {
    "basic_counter": {"basic"},
    "advanced_behavioral": {"advanced"},
}

# 调用计数（进程内累计，供验收取证）
OPERATOR_CALLS: Dict[str, int] = {"basic": 0, "advanced": 0}


def feature_to_group(feature: str) -> str:
    """特征名 -> 所属算子组（basic | advanced）。"""
    f = feature.lower()
    return "advanced" if any(m in f for m in _ADVANCED_MARKERS) else "basic"


def required_features_of_rules(rules: Iterable[dict],
                               selected_features: Iterable[str]) -> List[str]:
    """bundle -> 规则实际引用的特征（并上 selected，保持与引擎 schema 一致）。"""
    used: Set[str] = set()
    for r in rules:
        for c in r.get("conditions", []):
            feat = c.get("feature")
            if feat:
                used.add(feat)
    used |= set(selected_features)
    return sorted(used)


def resolve_required_operators(features: Iterable[str]) -> Set[str]:
    """特征集合 -> 需要激活的算子组集合。"""
    return {feature_to_group(f) for f in features}


def reset_calls() -> None:
    for k in OPERATOR_CALLS:
        OPERATOR_CALLS[k] = 0


def extract_on_demand(session, active_groups: Set[str],
                      basic_extractor, advanced_extractor) -> Dict:
    """按需提取：只调用激活组的 extractor（计数留证）。

    返回合并特征 dict；未激活组的特征不产出（引擎也用不到）。
    """
    feats: Dict = {}
    if "basic" in active_groups:
        OPERATOR_CALLS["basic"] += 1
        feats.update(basic_extractor.extract_all(session))
    if "advanced" in active_groups:
        OPERATOR_CALLS["advanced"] += 1
        feats.update(advanced_extractor.extract_all(session))
    return feats
