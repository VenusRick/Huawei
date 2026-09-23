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

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
class ExtractStats:
    """一次提取的解析/过滤/截断/失败计数（审计证据，不静默丢分母）。"""
    files: int = 0
    packets_read: int = 0
    read_capped_files: int = 0
    sessions_total: int = 0
    sessions_filtered_min_packets: int = 0
    sessions_filtered_min_bytes: int = 0
    sessions_truncated: int = 0
    extract_errors: int = 0
    error_details: List[str] = field(default_factory=list)
    # 资源上限证据（会话管理器计数；正常为0，非0说明有会话被上限影响）
    sessions_evicted: int = 0
    sessions_overflow_dropped: int = 0

    def to_dict(self) -> Dict:
        return dict(self.__dict__, error_details=self.error_details[:20])


def truncated_session_view(session, max_packets: int) -> Tuple:
    """前缀截断的会话视图：(视图, 是否截断)。

    红线（审计 2026-09-18）：
    - 不污染原 Session（浅拷贝 + 新包列表，R24 语义延伸到截断）；
    - 不使用未来包：end_time/duration 取前缀最后一个包的时间戳，
      TLS/QUIC 指纹只来自前缀内包的 tls_info（握手在前缀之后完成则
      特征如实呈现"无握手"）；
    - TCP state/is_closed 按前缀内标志重推（未见前缀外 FIN/RST）；
    - num_retransmissions/num_out_of_order 取前缀最后入列包的因果快照
      （SessionManager._append_packet 记录的 ingest 时刻累计值）——
      截断点之后观测到的重传/乱序不再泄入前缀特征（2026-09-18 第二轮，
      替代上一轮"保留全会话计数"的已知边界；手工构造的无快照会话回退
      保留原值，不伪造归零）。
    """
    if len(session.packets) <= max_packets:
        return session, False
    s = copy.copy(session)
    s.packets = list(session.packets[:max_packets])
    s.total_fwd_packets = sum(1 for p in s.packets if p.direction == 1)
    s.total_bwd_packets = sum(1 for p in s.packets if p.direction == -1)
    s.total_fwd_bytes = sum(p.length for p in s.packets if p.direction == 1)
    s.total_bwd_bytes = sum(p.length for p in s.packets if p.direction == -1)
    if s.packets:
        last = s.packets[-1]
        s.end_time = last.timestamp
        s.last_activity_time = s.end_time
        # 因果重传/乱序计数：截断点快照（无快照属性的手工会话保留原值）
        if hasattr(last, "cum_retransmissions"):
            s.num_retransmissions = int(last.cum_retransmissions)
        if hasattr(last, "cum_out_of_order"):
            s.num_out_of_order = int(last.cum_out_of_order)
        _recompute_prefix_tcp_state(s)
    s.is_closed = any(p.tcp_flags & 0x05 for p in s.packets) if s.packets else False
    return s, True


def _recompute_prefix_tcp_state(s) -> None:
    """按前缀内 TCP 标志重推会话状态（不引用前缀外事件）。"""
    from src.parser.session.session_manager import Protocol, TCPState
    if s.protocol != Protocol.TCP or not s.packets:
        return
    syn = syn_ack = fin = False
    for p in s.packets:
        if p.tcp_flags & 0x04:            # RST
            s.state = TCPState.CLOSED
            return
        if (p.tcp_flags & 0x02) and (p.tcp_flags & 0x10):
            syn_ack = True
        elif (p.tcp_flags & 0x02):
            syn = True
        if p.tcp_flags & 0x01:
            fin = True
    if fin:
        s.state = TCPState.FIN_WAIT
    elif syn and syn_ack:
        s.state = TCPState.ESTABLISHED
    elif syn:
        s.state = TCPState.SYN_SENT
    else:
        s.state = TCPState.ESTABLISHED


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


def extract_feature_records_with_stats(pcap_path: str,
                                       max_packets: int = DEFAULT_MAX_PACKETS,
                                       capture_id: str = "",
                                       tcp_timeout: int = DEFAULT_TCP_TIMEOUT,
                                       udp_timeout: int = DEFAULT_UDP_TIMEOUT,
                                       max_read_packets: Optional[int] = None,
                                       active_groups=None,
                                       ) -> Tuple[List[FeatureRecord], ExtractStats]:
    """从单个 pcap 提取全部会话的特征记录 + 解析/过滤/截断计数。

    全工程唯一实现（训练/推理/独立CLI 共用；勿复制此逻辑）。
    observation_id 规则："{文件名}:{session_index}"，与 M1 parity 一致。
    max_read_packets 是真正的读包上限（读满即停）。
    active_groups 非空时按算子组按需提取（P3，与 dpi_infer --on-demand 同路径）。
    有界消费（2026-09-18 第二轮）：迭代 read_pcap_generator 逐会话
    提取，已关闭会话即取即弃（峰值内存≈活跃会话+特征记录，不再囤积
    全文件会话对象）；会话顺序与整批 read_pcap 按构造一致（同一生成器）。
    """
    src_file = str(Path(pcap_path))
    file_tag = Path(pcap_path).name
    stats = ExtractStats(files=1)
    sm = SessionManager(tcp_timeout=tcp_timeout, udp_timeout=udp_timeout)
    reader = PCAPReader(sm)
    packets_before = reader.total_packets
    basic_ext = BasicFeatureExtractor()
    advanced_ext = AdvancedFeatureExtractor()
    records: List[FeatureRecord] = []
    for idx, session in enumerate(reader.read_pcap_generator(
            pcap_path, max_read_packets=max_read_packets)):
        stats.sessions_total += 1
        if session.total_packets < MIN_SESSION_PACKETS:
            stats.sessions_filtered_min_packets += 1
            continue
        if session.total_bytes < MIN_SESSION_BYTES:
            stats.sessions_filtered_min_bytes += 1
            continue
        # 大会话前缀截断（不污染原 Session；不使用前缀外时长/握手）
        view, truncated = truncated_session_view(session, max_packets)
        if truncated:
            stats.sessions_truncated += 1
        try:
            if active_groups is None:
                feats = {**basic_ext.extract_all(view),
                         **advanced_ext.extract_all(view)}
            else:
                from src.features.operators import extract_on_demand
                feats = extract_on_demand(view, active_groups,
                                          basic_ext, advanced_ext)
        except Exception as e:  # noqa: BLE001
            stats.extract_errors += 1
            stats.error_details.append(
                f"{file_tag}:{idx}: {type(e).__name__}: {e}")
            continue
        obs = ObservationSpec(
            observation_id=f"{file_tag}:{idx}",
            source_file=src_file,
            session_index=idx,
            capture_id=capture_id or file_tag,
        )
        records.append(FeatureRecord(observation=obs, features=feats))
    stats.packets_read = reader.total_packets - packets_before
    stats.read_capped_files = 1 if reader.read_capped else 0
    # 资源上限证据：非0说明有会话被驱逐/堆积丢弃（不静默）
    stats.sessions_evicted = sm.evicted_sessions
    stats.sessions_overflow_dropped = sm.closed_overflow_dropped
    return records, stats


def extract_feature_records(pcap_path: str,
                            max_packets: int = DEFAULT_MAX_PACKETS,
                            capture_id: str = "",
                            tcp_timeout: int = DEFAULT_TCP_TIMEOUT,
                            udp_timeout: int = DEFAULT_UDP_TIMEOUT,
                            max_read_packets: Optional[int] = None
                            ) -> List[FeatureRecord]:
    """兼容入口：只返回特征记录（计数见 extract_feature_records_with_stats）。"""
    records, _ = extract_feature_records_with_stats(
        pcap_path, max_packets=max_packets, capture_id=capture_id,
        tcp_timeout=tcp_timeout, udp_timeout=udp_timeout,
        max_read_packets=max_read_packets)
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
                                     udp_timeout: int = DEFAULT_UDP_TIMEOUT,
                                     max_read_packets: Optional[int] = None
                                     ) -> List[FeatureRecord]:
    """从 pcap 提取行为窗口特征记录（M2：终端窗口/连接/流段/关系）。

    训练侧与独立推理侧共用本入口（bundle 内 context 配置驱动同参数），
    窗口发射即产出、不用未来包；R30 截断前缀写入 ObservationSpec。
    有界消费：迭代 read_pcap_generator 逐会话丢弃（只需包级事件流，
    会话对象不囤积；2026-09-18 第二轮）。
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
    for _session in reader.read_pcap_generator(
            pcap_path, max_read_packets=max_read_packets):
        pass            # 会话即取即弃：窗口经 packet_listener 已产出
    tracker.finalize()
    return records


def extract_tunnel_feature_records(pcap_path: str,
                                   context_config: Optional[dict] = None,
                                   terminal: Optional[str] = None,
                                   capture_id: str = "",
                                   tcp_timeout: int = DEFAULT_TCP_TIMEOUT,
                                   udp_timeout: int = DEFAULT_UDP_TIMEOUT,
                                   max_read_packets: Optional[int] = None
                                   ) -> List[FeatureRecord]:
    """Extract identifier-free fixed-window tunnel observations.

    Training and independent inference share this exact packet-listener path.
    The returned feature dictionary is always the 64-dimension tunnel contract.
    """
    from src.parser.context import BehaviorContextTracker, TunnelWindowFeatureBuilder
    tracker = BehaviorContextTracker(context_config, terminal=terminal,
                                     capture_id=capture_id or Path(pcap_path).name)
    builder = TunnelWindowFeatureBuilder()
    records: List[FeatureRecord] = []

    def _on_window(win: dict) -> None:
        feats = builder.build(win)
        obs = ObservationSpec(
            observation_id=(f"{Path(pcap_path).name}:t{win['window_start']:.1f}"),
            source_file=str(pcap_path),
            session_index=int(win["window_start"]),
            capture_id=win["capture_id"], terminal=win["terminal"],
            window_start=win["window_start"], window_end=win["window_end"],
            truncated_prefix=bool(win["truncated_prefix"]),
        )
        records.append(FeatureRecord(observation=obs, features=feats))

    tracker.on_window = _on_window

    def _listener(pkt) -> None:
        tracker.feed_packet(
            float(getattr(pkt, "timestamp", 0.0)),
            getattr(pkt, "src_ip", ""), getattr(pkt, "dst_ip", ""),
            str(getattr(pkt, "canonical_tuple", "")),
            int(getattr(pkt, "length", getattr(pkt, "packet_size", 0)) or 0))

    sm = SessionManager(tcp_timeout=tcp_timeout, udp_timeout=udp_timeout,
                        packet_listener=_listener)
    reader = PCAPReader(sm)
    for _session in reader.read_pcap_generator(
            pcap_path, max_read_packets=max_read_packets):
        pass
    tracker.finalize()
    return records


def extract_flow_window_feature_records(pcap_path: str,
                                        context_config: Optional[dict] = None,
                                        feature_kind: str = "behavior",
                                        capture_id: str = "",
                                        tcp_timeout: int = DEFAULT_TCP_TIMEOUT,
                                        udp_timeout: int = DEFAULT_UDP_TIMEOUT,
                                        max_read_packets: Optional[int] = None
                                        ) -> List[FeatureRecord]:
    """Split each long reconstructed flow into non-overlapping fixed windows.

    Window timestamps are relative to the PCAP's first packet, so manifest
    behavior intervals keep their existing semantics. Only packets visible in
    the slice are used; background flows remain separate observations.
    """
    from src.parser.context import (BehaviorFlowWindowFeatureBuilder,
                                    TunnelWindowFeatureBuilder)
    cfg = dict(context_config or {})
    win_sec = float(cfg.get("window_sec", 15.0))
    min_pkts = int(cfg.get("min_window_packets", 5))
    if win_sec <= 0:
        raise ValueError("window_sec must be positive")
    builder = (BehaviorFlowWindowFeatureBuilder() if feature_kind == "behavior"
               else TunnelWindowFeatureBuilder())
    sm = SessionManager(tcp_timeout=tcp_timeout, udp_timeout=udp_timeout)
    reader = PCAPReader(sm)
    records: List[FeatureRecord] = []
    file_tag = Path(pcap_path).name
    for session_index, session in enumerate(reader.read_pcap_generator(
            pcap_path, max_read_packets=max_read_packets)):
        ps = sorted(session.packets, key=lambda x: float(x.timestamp))
        if not ps or float(ps[-1].timestamp - ps[0].timestamp) < win_sec:
            continue
        origin = float(reader.first_timestamp if reader.first_timestamp is not None
                       else ps[0].timestamp)
        start = float(ps[0].timestamp)
        end = float(ps[-1].timestamp)
        k = 0
        while start + k * win_sec <= end:
            a = start + k * win_sec
            b = a + win_sec
            wp = [pkt for pkt in ps if a <= float(pkt.timestamp) < b]
            if len(wp) >= min_pkts:
                w0, w1 = a - origin, b - origin
                if feature_kind == "behavior":
                    feats = builder.build(wp)
                    tag = "bw"
                else:
                    # Reuse the tunnel builder by constructing the same event
                    # tuple contract as BehaviorContextTracker, but for one flow.
                    evs = []
                    for idx, pkt in enumerate(wp):
                        rel = float(pkt.timestamp) - origin
                        up = 1 if int(getattr(pkt, "direction", 1) or 1) > 0 else 0
                        evs.append((rel, getattr(pkt, "src_ip", ""),
                                    getattr(pkt, "dst_ip", ""),
                                    f"session:{session_index}", int(pkt.length), idx, up))
                    win = {
                        "window_start": w0, "window_end": w1,
                        "n_connections": 1, "n_segments": 1,
                        "n_overflow_segments": 0, "segments": [], "overflow": [],
                        "events": evs, "truncated_prefix": k > 0,
                    }
                    feats = builder.build(win)
                    tag = "tw"
                obs = ObservationSpec(
                    observation_id=f"{file_tag}:{tag}{session_index}:{k}",
                    source_file=str(pcap_path), session_index=session_index,
                    capture_id=capture_id or file_tag,
                    window_start=w0, window_end=w1,
                    truncated_prefix=(k > 0),
                )
                records.append(FeatureRecord(observation=obs, features=feats))
            k += 1
    return records


def extract_features_from_sessions(sessions,
                                   max_packets: int = DEFAULT_MAX_PACKETS
                                   ) -> List[Dict]:
    """会话列表 -> 特征 dict 列表（共享提取序列，单一实现）。

    过滤/截断/提取与 extract_feature_records_with_stats 完全一致；
    tests/test_generic_pipeline.py 等旧入口应薄包装本函数，
    不得复制此逻辑（防双实现漂移）。
    """
    basic_ext = BasicFeatureExtractor()
    advanced_ext = AdvancedFeatureExtractor()
    out: List[Dict] = []
    for session in sessions:
        if session.total_packets < MIN_SESSION_PACKETS:
            continue
        if session.total_bytes < MIN_SESSION_BYTES:
            continue
        view, _truncated = truncated_session_view(session, max_packets)
        try:
            out.append({**basic_ext.extract_all(view),
                        **advanced_ext.extract_all(view)})
        except Exception:
            continue
    return out
