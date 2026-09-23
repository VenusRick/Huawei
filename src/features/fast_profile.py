# -*- coding: utf-8 -*-
"""快速特征 profile（2026-09-18 第三轮 fast_dpi）：机器可读的少量低成本特征集。

设计约束（执行计划 S1）：
- 候选池 ≤32 个低成本特征，首选 16 维版；只含包数/字节/时长/方向比/
  包长统计/少量IAT/突发计数/TLS可见标记——不含近似熵/Hurst/频谱/BoW；
- 不复制特征公式：特征一律由 BasicFeatureExtractor 的既有族方法计算，
  本模块只定义"要哪些名字"，由 extractor.extract(requested) 按需计算；
- 不把源IP、源临时端口、文件/目录名当类别特征；
- fast32 ⊇ fast16（同一批族，可叠加），缺名时由 extractor 不产出（缺失
  语义与全量路径一致：FeatureRecord.missing / 规则缺失即不匹配）。

profile 与会话前缀包数是两个正交参数；bundle 携带
feature_profile + max_packets_per_session，训练端与独立CLI读同一配置。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# 默认会话前缀包数（执行计划 S1.4：首轮32包，验证不足才试64）
FAST_DEFAULT_PREFIX_PACKETS = 32
FAST_FALLBACK_PREFIX_PACKETS = 64

# 16 维低成本版：flow_stats + timing + session_info 三族即可覆盖
FAST16: Tuple[str, ...] = (
    'duration',                 # session_info
    'total_packets',            # flow_stats（计数器 O(1)）
    'total_bytes',
    'total_fwd_packets',
    'total_bwd_packets',
    'total_fwd_bytes',
    'total_bwd_bytes',
    'fwd_bwd_byte_ratio',
    'pkt_size_mean',
    'pkt_size_std',
    'pkt_size_max',
    'pkt_size_min',
    'fwd_pkt_size_mean',
    'bwd_pkt_size_mean',
    'iat_mean',                 # timing（单遍时间戳）
    'iat_std',
)

# 32 维版：+16（IAT补充/速率/方向比/突发/TLS可见标记/前缀因果重传）
FAST32: Tuple[str, ...] = FAST16 + (
    'iat_max',
    'fwd_iat_mean',
    'bwd_iat_mean',
    'num_bursts',
    'pkts_first_0.5s',
    'packet_rate',
    'byte_rate',
    'down_up_byte_ratio',
    'fwd_pkt_size_max',
    'bwd_pkt_size_std',
    'tls_has_sni',              # protocol 族（首个tls_info即停）
    'tls_sni_length',
    'tls_num_cipher_suites',
    'tls_num_extensions',
    'tls_has_tls13_ciphers',
    'num_retransmissions',      # flags 族（前缀因果快照计数）
)

FAST_PROFILES: Dict[str, Tuple[str, ...]] = {
    'fast16': FAST16,
    'fast32': FAST32,
}


def resolve_profile(name: Optional[str]) -> Tuple[str, Tuple[str, ...]]:
    """profile 名 -> (名字, 特征名元组)；'full'/None 除外（全量319维路径）。

    'full' 不是 fast profile——返回空元组表示"不裁剪"，
    由调用方走既有全量提取（兼容回归，非本轮默认训练路径）。
    """
    if name in (None, '', 'full'):
        return 'full', ()
    if name not in FAST_PROFILES:
        raise ValueError(
            f"未知特征profile: {name}（可选: full / "
            f"{' / '.join(FAST_PROFILES)}）")
    return name, FAST_PROFILES[name]


def profile_of_features(features) -> Optional[str]:
    """特征名集合能被哪个最小 fast profile 覆盖（None=需要全量路径）。

    规则依赖解析用：部署规则只引用 fast16 名字时激活 fast16 组，
    不触发全量 basic/advanced。
    """
    fs = set(features)
    if not fs:
        return None
    if fs <= set(FAST16):
        return 'fast16'
    if fs <= set(FAST32):
        return 'fast32'
    return None


def default_prefix_packets(profile: Optional[str]) -> int:
    """profile -> 默认会话前缀包数（bundle 未显式携带时的兜底）。

    fast profile 首轮一律 32 包；64 包是验证不足时的回退实验参数
    （FAST_FALLBACK_PREFIX_PACKETS），由调用方显式指定，不是独立 profile。
    """
    return FAST_DEFAULT_PREFIX_PACKETS
