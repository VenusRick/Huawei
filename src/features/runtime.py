# -*- coding: utf-8 -*-
"""统一观测与特征运行时（契约层 P0）。

训练侧与独立推理侧共用同一个提取入口，保证 parity（M1 已验证
319 维特征零失配）；本模块把该路径固化为正式服务，供
    - tests/test_generic_pipeline.py（薄包装后的调试入口）
    - src/pipeline.py mine
    - src/engine/dpi_infer.py --rules
三方调用，全工程只维护一份解析与一份特征实现。

提取路径与 dpi_infer 原实现逐字一致（勿改过滤阈值/截断逻辑，
否则 parity 回退）：
    SessionManager(tcp_timeout=300, udp_timeout=60)
    -> PCAPReader.read_pcap
    -> 过滤 total_packets<3 / total_bytes<100
    -> len(packets)>max_packets 截断并重算 total_*
    -> {**basic.extract_all, **advanced.extract_all}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.parser.pcap_reader import PCAPReader
from src.parser.session.session_manager import SessionManager
from src.features.basic.feature_extractor import BasicFeatureExtractor
from src.features.advanced.advanced_extractor import AdvancedFeatureExtractor

DEFAULT_MAX_PACKETS = 500
DEFAULT_TCP_TIMEOUT = 300
DEFAULT_UDP_TIMEOUT = 60
# 与训练/推理两侧一致的会话过滤阈值（勿改）
MIN_SESSION_PACKETS = 3
MIN_SESSION_BYTES = 100


@dataclass
class ObservationSpec:
    """一次观测的元数据（observation_id 全链唯一，评价/报告据此追溯）。"""
    observation_id: str
    source_file: str
    session_index: int = 0
    capture_id: str = ""
    terminal: str = ""            # 可见终端（M2 行为窗口用；单流路径为空）
    window_start: Optional[float] = None   # M2：窗口起点（相对 pcap 首包秒）
    window_end: Optional[float] = None
    # R30：截断前缀标记——窗口起点晚于会话首包时为 True，此时会话时长
    # 是"窗口内观测时长"而非完整会话时长，不虚构前缀。单流路径恒 False。
    truncated_prefix: bool = False


@dataclass
class FeatureSpec:
    """特征定义（特征目录的机器可读形态，M3 报告与 P3 按需计算共用）。"""
    name: str
    layer: str                    # basic | advanced | window | segment | relation
    unit: str = ""
    formula: str = ""
    missing_semantics: str = ""   # 缺失含义（如"无下行包"）
    cost: str = "low"             # low | medium | high
    operator_group: str = ""      # P3 按需计算的算子组（basic_counter 等）


@dataclass
class FeatureRecord:
    """一条观测的特征向量 + 缺失状态（缺失不零填，R22 语义）。"""
    observation: ObservationSpec
    features: Dict[str, float] = field(default_factory=dict)
    missing: Set[str] = field(default_factory=set)

    def get(self, name: str) -> Optional[float]:
        if name in self.missing or name not in self.features:
            return None
        return self.features[name]


def extract_feature_records(pcap_path: str,
                            max_packets: int = DEFAULT_MAX_PACKETS,
                            capture_id: str = "",
                            tcp_timeout: int = DEFAULT_TCP_TIMEOUT,
                            udp_timeout: int = DEFAULT_UDP_TIMEOUT
                            ) -> List[FeatureRecord]:
    """从单个 pcap 提取全部会话的特征记录（共享入口，勿复制此逻辑）。

    observation_id 规则："{文件名}:{session_index}"，与 M1 parity 一致。
    """
    src_file = str(Path(pcap_path))
    file_tag = Path(pcap_path).name
    sm = SessionManager(tcp_timeout=tcp_timeout, udp_timeout=udp_timeout)
    reader = PCAPReader(sm)
    sessions = reader.read_pcap(pcap_path)
    basic_ext = BasicFeatureExtractor()
    advanced_ext = AdvancedFeatureExtractor()
    records: List[FeatureRecord] = []
    for idx, session in enumerate(sessions):
        if session.total_packets < MIN_SESSION_PACKETS:
            continue
        if session.total_bytes < MIN_SESSION_BYTES:
            continue
        # 大会话截断（与训练流水线一致：截断后重算包/字节统计）
        if len(session.packets) > max_packets:
            session.packets = session.packets[:max_packets]
            session.total_fwd_packets = sum(
                1 for p in session.packets if p.direction == 1)
            session.total_bwd_packets = sum(
                1 for p in session.packets if p.direction == -1)
            session.total_fwd_bytes = sum(
                p.length for p in session.packets if p.direction == 1)
            session.total_bwd_bytes = sum(
                p.length for p in session.packets if p.direction == -1)
        try:
            feats = {**basic_ext.extract_all(session),
                     **advanced_ext.extract_all(session)}
        except Exception:
            continue
        obs = ObservationSpec(
            observation_id=f"{file_tag}:{idx}",
            source_file=src_file,
            session_index=idx,
            capture_id=capture_id or file_tag,
        )
        records.append(FeatureRecord(observation=obs, features=feats))
    return records


def feature_names(records: List[FeatureRecord]) -> List[str]:
    """特征 schema（按首次出现顺序；R16：单包流不早退保证 schema 稳定）。"""
    seen: Dict[str, None] = {}
    for r in records:
        for k in r.features:
            seen.setdefault(k, None)
    return list(seen)


def records_to_matrix(records: List[FeatureRecord],
                      columns: Optional[List[str]] = None
                      ) -> "tuple[list, list]":
    """特征记录 -> (matrix, columns)；缺失值记 None（不零填），由调用方
    决定训练填充策略（训练侧填充不污染推理语义，R24）。"""
    cols = columns or feature_names(records)
    matrix = []
    for r in records:
        row = [r.get(c) for c in cols]
        matrix.append(row)
    return matrix, cols


def extract_behavior_feature_records(pcap_path: str,
                                     context_config: Optional[dict] = None,
                                     terminal: Optional[str] = None,
                                     capture_id: str = "",
                                     tcp_timeout: int = DEFAULT_TCP_TIMEOUT,
                                     udp_timeout: int = DEFAULT_UDP_TIMEOUT
                                     ) -> List[FeatureRecord]:
    """从 pcap 提取行为窗口特征记录（M2：终端窗口/连接/流段/关系）。

    训练侧与独立推理侧共用本入口（bundle 内 context 配置驱动同参数），
    窗口发射即产出、不用未来包；R30 截断前缀写入 ObservationSpec。
    """
    from src.parser.context import BehaviorContextTracker, WindowFeatureBuilder
    tracker = BehaviorContextTracker(context_config, terminal=terminal,
                                     capture_id=capture_id or Path(pcap_path).name)
    builder = WindowFeatureBuilder()
    records: List[FeatureRecord] = []

    def _on_window(win: dict) -> None:
        feats = builder.build(win)
        obs = ObservationSpec(
            observation_id=(f"{Path(pcap_path).name}:w{win['window_start']:.1f}"),
            source_file=str(pcap_path),
            session_index=int(win["window_start"]),
            capture_id=win["capture_id"],
            terminal=win["terminal"],
            window_start=win["window_start"],
            window_end=win["window_end"],
            truncated_prefix=bool(win["truncated_prefix"]),
        )
        records.append(FeatureRecord(observation=obs, features=feats))

    tracker.on_window = _on_window

    def _listener(pkt) -> None:
        tracker.feed_packet(
            float(getattr(pkt, "timestamp", 0.0)),
            getattr(pkt, "src_ip", ""),
            getattr(pkt, "dst_ip", ""),
            str(getattr(pkt, "canonical_tuple", "")),
            int(getattr(pkt, "length", getattr(pkt, "packet_size", 0)) or 0))

    sm = SessionManager(tcp_timeout=tcp_timeout, udp_timeout=udp_timeout,
                        packet_listener=_listener)
    reader = PCAPReader(sm)
    reader.read_pcap(pcap_path)
    tracker.finalize()
    return records


def extract_features_from_sessions(sessions,
                                   max_packets: int = DEFAULT_MAX_PACKETS
                                   ) -> List[Dict]:
    """会话列表 -> 特征 dict 列表（共享提取序列，单一实现）。

    过滤/截断/提取与 extract_feature_records 完全一致；
    tests/test_generic_pipeline.py 等旧入口应薄包装本函数，
    不得复制此逻辑（防双实现漂移）。
    """
    from src.features.basic.feature_extractor import BasicFeatureExtractor
    from src.features.advanced.advanced_extractor import (
        AdvancedFeatureExtractor)
    basic_ext = BasicFeatureExtractor()
    advanced_ext = AdvancedFeatureExtractor()
    out: List[Dict] = []
    for session in sessions:
        if session.total_packets < MIN_SESSION_PACKETS:
            continue
        if session.total_bytes < MIN_SESSION_BYTES:
            continue
        if len(session.packets) > max_packets:
            session.packets = session.packets[:max_packets]
            session.total_fwd_packets = sum(
                1 for p in session.packets if p.direction == 1)
            session.total_bwd_packets = sum(
                1 for p in session.packets if p.direction == -1)
            session.total_fwd_bytes = sum(
                p.length for p in session.packets if p.direction == 1)
            session.total_bwd_bytes = sum(
                p.length for p in session.packets if p.direction == -1)
        try:
            out.append({**basic_ext.extract_all(session),
                        **advanced_ext.extract_all(session)})
        except Exception:
            continue
    return out
