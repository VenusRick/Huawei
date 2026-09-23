"""
会话重建模块
支持TCP乱序重排、重传检测与处理、UDP会话划分。

核心功能：
- 基于五元组的会话追踪
- TCP流重组（处理乱序、重传）
- UDP会话超时管理
- 每个会话的包序列输出

参考：Iris的连接状态机模型
"""
import struct
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict


class Protocol(Enum):
    TCP = 6
    UDP = 17
    OTHER = 0


class TCPState(Enum):
    """TCP连接状态机（参考Iris）"""
    INIT = 0
    SYN_SENT = 1
    SYN_ACK_RECEIVED = 2
    ESTABLISHED = 3
    FIN_WAIT = 4
    CLOSED = 5


@dataclass
class PacketInfo:
    """解析后的包信息"""
    timestamp: float = 0.0
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: Protocol = Protocol.OTHER
    length: int = 0              # 整包长度（含头部）
    payload_length: int = 0      # 载荷长度
    payload: bytes = b""

    # TCP特有
    tcp_flags: int = 0
    tcp_seq: int = 0
    tcp_ack: int = 0
    tcp_window: int = 0
    tcp_urgent_ptr: int = 0

    # 方向（相对于会话发起方）
    direction: int = 0           # 1=前向(发起方→响应方), -1=后向

    # IP层
    ip_ttl: int = 0
    ip_tos: int = 0
    ip_id: int = 0
    ip_flags: int = 0

    # 是否为重传
    is_retransmission: bool = False
    # 是否为乱序
    is_out_of_order: bool = False

    # TLS信息（如果可解析）
    tls_info: Optional[dict] = None

    @property
    def five_tuple(self) -> Tuple[str, str, int, int, int]:
        """返回五元组"""
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol.value)

    @property
    def canonical_tuple(self) -> Tuple[str, str, int, int, int]:
        """规范化的五元组（小IP:小端口在前）"""
        if (self.src_ip, self.src_port) <= (self.dst_ip, self.dst_port):
            return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol.value)
        else:
            return (self.dst_ip, self.src_ip, self.dst_port, self.src_port, self.protocol.value)

    @property
    def tcp_flag_names(self) -> List[str]:
        """TCP标志名称列表"""
        flags = []
        if self.tcp_flags & 0x01: flags.append("FIN")
        if self.tcp_flags & 0x02: flags.append("SYN")
        if self.tcp_flags & 0x04: flags.append("RST")
        if self.tcp_flags & 0x08: flags.append("PSH")
        if self.tcp_flags & 0x10: flags.append("ACK")
        if self.tcp_flags & 0x20: flags.append("URG")
        if self.tcp_flags & 0x40: flags.append("ECE")
        if self.tcp_flags & 0x80: flags.append("CWR")
        return flags


@dataclass
class FlowSession:
    """流量会话"""
    # 五元组
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: Protocol = Protocol.OTHER

    # 状态
    state: TCPState = TCPState.INIT
    start_time: float = 0.0
    end_time: float = 0.0
    last_activity_time: float = 0.0

    # 包序列（按时间排序，已去重乱序处理）
    packets: List[PacketInfo] = field(default_factory=list)

    # TCP序列追踪
    fwd_next_seq: int = 0
    bwd_next_seq: int = 0

    # 统计
    total_fwd_packets: int = 0
    total_bwd_packets: int = 0
    total_fwd_bytes: int = 0
    total_bwd_bytes: int = 0
    num_retransmissions: int = 0
    num_out_of_order: int = 0

    # TLS信息
    tls_handshake_complete: bool = False
    tls_version: int = 0
    sni: str = ""
    ja3_hash: str = ""
    ja3s_hash: str = ""

    # 是否已终止
    is_closed: bool = False

    @property
    def five_tuple(self) -> Tuple[str, str, int, int, int]:
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol.value)

    @property
    def duration(self) -> float:
        # 零起点时间戳(start_time=0.0)合法，直接计算差值
        return self.end_time - self.start_time

    @property
    def total_packets(self) -> int:
        return self.total_fwd_packets + self.total_bwd_packets

    @property
    def total_bytes(self) -> int:
        return self.total_fwd_bytes + self.total_bwd_bytes

    def get_directional_packets(self, direction: int) -> List[PacketInfo]:
        """获取指定方向的包"""
        return [p for p in self.packets if p.direction == direction]


class SessionManager:
    """会话管理器"""

    def __init__(self, tcp_timeout: int = 300, udp_timeout: int = 60,
                 tcp_establish_timeout: int = 5, max_sessions: int = 1000000,
                 packet_listener=None, max_closed_sessions: int = 100000):
        self.tcp_timeout = tcp_timeout
        # M4 资源上限与证据计数（达到上限可统计，不伪装完整）
        self.max_closed_sessions = max_closed_sessions
        self.evicted_sessions = 0          # 活跃会话超限被驱逐
        self.closed_overflow_dropped = 0   # 已关闭会话堆积超限被丢弃
        # M2: 逐包事件转发（供 context.py 行为窗口聚合；None 时零开销）
        self.packet_listener = packet_listener
        self.udp_timeout = udp_timeout
        self.tcp_establish_timeout = tcp_establish_timeout
        self.max_sessions = max_sessions

        # 活跃会话表: canonical_tuple -> FlowSession
        self.active_sessions: Dict[Tuple, FlowSession] = {}
        # 已关闭会话
        self.closed_sessions: List[FlowSession] = []

        # TCP序列号追踪：five_tuple -> {expected_seq, packets_buffer}
        self._tcp_tracker: Dict[Tuple, dict] = {}

    def process_packet(self, pkt: PacketInfo) -> Optional[FlowSession]:
        """处理一个包，返回完整的会话（如果会话关闭）"""
        if self.packet_listener is not None:
            self.packet_listener(pkt)   # M2: 事件转发先于会话状态更新
        canonical = pkt.canonical_tuple
        # R04: ingest时应用空闲超时：同五元组旧会话超时先关闭，新包按新会话处理
        old = self.active_sessions.get(canonical)
        if old is not None:
            _timeout = self.tcp_timeout if old.protocol == Protocol.TCP else self.udp_timeout
            if pkt.timestamp - old.last_activity_time > _timeout:
                del self.active_sessions[canonical]
                old.is_closed = True
                self._append_closed(old)
                self._tcp_tracker.pop(canonical, None)
        is_new = canonical not in self.active_sessions

        if is_new:
            # 检查会话数量限制
            if len(self.active_sessions) >= self.max_sessions:
                self._evict_oldest()
            session = self._create_session(pkt)
            self.active_sessions[canonical] = session
        else:
            session = self.active_sessions[canonical]

        # 更新最后活动时间
        session.last_activity_time = pkt.timestamp

        # TCP状态机处理
        if pkt.protocol == Protocol.TCP:
            completed_session = self._process_tcp_packet(session, pkt, canonical)
            if completed_session:
                return completed_session
        else:
            # UDP / 其他协议，直接追加
            self._append_packet(session, pkt)

        return None

    def flush_all(self) -> List[FlowSession]:
        """关闭所有活跃会话，返回已关闭会话列表"""
        # 未flush的乱序缓冲包按seq序补入会话（避免丢包），再统一关闭
        for canonical, session in list(self.active_sessions.items()):
            tracker = self._tcp_tracker.get(canonical)
            if not tracker:
                continue
            for buf_key in ('fwd_buffer', 'bwd_buffer'):
                buf = tracker.get(buf_key) or {}
                for seq in sorted(buf):
                    self._append_packet(session, buf[seq])
                buf.clear()
        self._tcp_tracker.clear()
        all_sessions = list(self.active_sessions.values())
        self.active_sessions.clear()
        self.closed_sessions.extend(all_sessions)
        for s in all_sessions:
            s.is_closed = True
        return all_sessions

    def cleanup_expired(self, current_time: float) -> List[FlowSession]:
        """清理过期会话"""
        expired = []
        timeout_map = {
            Protocol.TCP: self.tcp_timeout,
            Protocol.UDP: self.udp_timeout,
            Protocol.OTHER: self.udp_timeout,
        }

        for key in list(self.active_sessions.keys()):
            session = self.active_sessions[key]
            timeout = timeout_map.get(session.protocol, self.udp_timeout)
            if current_time - session.last_activity_time > timeout:
                expired.append(session)
                del self.active_sessions[key]
                session.is_closed = True

        self.closed_sessions.extend(expired)
        return expired

    def _create_session(self, pkt: PacketInfo) -> FlowSession:
        """创建新会话"""
        canonical = pkt.canonical_tuple
        # R01: 会话端点取首包真实发起方/响应方，不按canonical字典序
        session = FlowSession(
            src_ip=pkt.src_ip,
            dst_ip=pkt.dst_ip,
            src_port=pkt.src_port,
            dst_port=pkt.dst_port,
            protocol=pkt.protocol,
            start_time=pkt.timestamp,
        )

        if pkt.protocol == Protocol.TCP:
            self._tcp_tracker[pkt.canonical_tuple] = {
                'fwd_expected_seq': None,
                'bwd_expected_seq': None,
                'fwd_buffer': {},
                'bwd_buffer': {},
                'syn_seen': False,
                'syn_ack_seen': False,
                'fin_fwd': False,
                'fin_bwd': False,
            }

        return session

    def _process_tcp_packet(self, session: FlowSession, pkt: PacketInfo,
                            canonical: Tuple) -> Optional[FlowSession]:
        """TCP状态机处理"""
        tracker = self._tcp_tracker.get(canonical)
        if tracker is None:
            return None

        # 判断方向
        is_fwd = (pkt.src_ip == session.src_ip and pkt.src_port == session.src_port)

        # SYN包
        if pkt.tcp_flags & 0x02 and not (pkt.tcp_flags & 0x10):
            # 纯SYN
            if not tracker['syn_seen']:
                tracker['syn_seen'] = True
                tracker['fwd_expected_seq' if is_fwd else 'bwd_expected_seq'] = pkt.tcp_seq + 1
                session.state = TCPState.SYN_SENT
                self._append_packet(session, pkt)
                return None
            else:
                # 重传SYN
                pkt.is_retransmission = True
                session.num_retransmissions += 1
                return None

        # SYN-ACK包
        if (pkt.tcp_flags & 0x02) and (pkt.tcp_flags & 0x10):
            if session.state == TCPState.SYN_SENT:
                tracker['syn_ack_seen'] = True
                tracker['fwd_expected_seq' if is_fwd else 'bwd_expected_seq'] = pkt.tcp_seq + 1
                session.state = TCPState.SYN_ACK_RECEIVED
                self._append_packet(session, pkt)
                return None
            else:
                pkt.is_retransmission = True
                session.num_retransmissions += 1
                return None

        # ACK包（三次握手完成）
        if pkt.tcp_flags & 0x10 and session.state == TCPState.SYN_ACK_RECEIVED:
            session.state = TCPState.ESTABLISHED
            session.tls_handshake_complete = False

        # FIN包：双向FIN均出现则正常关闭并回收会话
        if pkt.tcp_flags & 0x01:
            fin_key = 'fin_fwd' if is_fwd else 'fin_bwd'
            other_key = 'fin_bwd' if is_fwd else 'fin_fwd'
            if tracker.get(other_key):
                session.state = TCPState.CLOSED
                session.is_closed = True
                self._append_packet(session, pkt)
                del self.active_sessions[canonical]
                self._append_closed(session)
                self._tcp_tracker.pop(canonical, None)
                return session
            tracker[fin_key] = True
            session.state = TCPState.FIN_WAIT

        # RST包
        if pkt.tcp_flags & 0x04:
            session.state = TCPState.CLOSED
            session.is_closed = True
            self._append_packet(session, pkt)
            del self.active_sessions[canonical]
            self._append_closed(session)
            return session

        # 数据包：检测重传和乱序
        if session.state == TCPState.ESTABLISHED or session.state == TCPState.FIN_WAIT:
            self._check_reorder_retransmit(session, pkt, tracker, is_fwd)

        # 正常追加（乱序包已入buffer待按序flush，不在此追加避免双计）
        if not pkt.is_retransmission and not pkt.is_out_of_order:
            self._append_packet(session, pkt)

        return None

    def _check_reorder_retransmit(self, session: FlowSession, pkt: PacketInfo,
                                   tracker: dict, is_fwd: bool):
        """检测重传和乱序"""
        expected_key = 'fwd_expected_seq' if is_fwd else 'bwd_expected_seq'
        buffer_key = 'fwd_buffer' if is_fwd else 'bwd_buffer'

        expected_seq = tracker[expected_key]
        seq = pkt.tcp_seq

        if expected_seq is None:
            # 零载荷包(纯ACK等)不推进expected，避免污染序号基准
            if pkt.payload_length > 0:
                tracker[expected_key] = seq + pkt.payload_length
            return

        if pkt.payload_length == 0:
            return

        if seq < expected_seq:
            # 可能是重传
            pkt.is_retransmission = True
            session.num_retransmissions += 1
        elif seq > expected_seq:
            # 乱序：缓存等待
            pkt.is_out_of_order = True
            session.num_out_of_order += 1
            tracker[buffer_key][seq] = pkt
        else:
            # 按序到达
            tracker[expected_key] = seq + pkt.payload_length
            # 检查缓存中是否有后续包
            self._flush_buffer(session, tracker, buffer_key, expected_key)

    def _flush_buffer(self, session: FlowSession, tracker: dict,
                      buffer_key: str, expected_key: str):
        """刷新乱序缓冲区"""
        buf = tracker[buffer_key]
        while tracker[expected_key] in buf:
            buffered_pkt = buf.pop(tracker[expected_key])
            buffered_pkt.is_out_of_order = False
            self._append_packet(session, buffered_pkt)
            tracker[expected_key] += buffered_pkt.payload_length

    def _append_packet(self, session: FlowSession, pkt: PacketInfo):
        """追加包到会话"""
        # 确定方向
        is_fwd = (pkt.src_ip == session.src_ip and pkt.src_port == session.src_port)
        pkt.direction = 1 if is_fwd else -1

        # 因果快照（审计 2026-09-18 第二轮）：记录"本包入列时刻"的
        # 重传/乱序累计值，供前缀截断视图按截断点重放（不用未来事件）。
        # 语义：截断到任一位置，计数=该位置最后入列包的快照值——重传/
        # 乱序事件的观测(ingest)先于该包入列则计入；乱序包本体若因等待
        # 重排落在列表后段，其"到达事件"仍按到达时刻因果计入。
        pkt.cum_retransmissions = session.num_retransmissions
        pkt.cum_out_of_order = session.num_out_of_order

        session.packets.append(pkt)
        session.end_time = pkt.timestamp
        session.last_activity_time = pkt.timestamp

        if is_fwd:
            session.total_fwd_packets += 1
            session.total_fwd_bytes += pkt.length
        else:
            session.total_bwd_packets += 1
            session.total_bwd_bytes += pkt.length

    def _append_closed(self, session) -> None:
        """已关闭会话入列；堆积超上限丢弃最老并计数（证据不伪装）。"""
        self.closed_sessions.append(session)
        if len(self.closed_sessions) > self.max_closed_sessions:
            self.closed_sessions.pop(0)
            self.closed_overflow_dropped += 1

    def take_closed_sessions(self):
        """取出并清空已关闭会话（流式消费即释放，防内存囤积）。"""
        out = self.closed_sessions
        self.closed_sessions = []
        return out

    def _evict_oldest(self):
        """淘汰最旧的会话"""
        if not self.active_sessions:
            return
        oldest_key = min(self.active_sessions,
                        key=lambda k: self.active_sessions[k].last_activity_time)
        session = self.active_sessions.pop(oldest_key)
        session.is_closed = True
        session._evicted = True   # 证据状态：被上限驱逐，非自然关闭
        self.evicted_sessions += 1
        self._append_closed(session)

    def get_all_closed_sessions(self) -> List[FlowSession]:
        """获取所有已关闭会话"""
        return self.closed_sessions.copy()

    def get_active_session_count(self) -> int:
        return len(self.active_sessions)

    def get_closed_session_count(self) -> int:
        return len(self.closed_sessions)

    def reset(self):
        self.evicted_sessions = 0
        self.closed_overflow_dropped = 0
        """重置所有状态"""
        self.active_sessions.clear()
        self.closed_sessions.clear()
        self._tcp_tracker.clear()


def parse_ip_header(data: bytes, offset: int = 0) -> Optional[dict]:
    """解析IP头（支持IPv4）"""
    if len(data) - offset < 20:
        return None

    version_ihl = data[offset]
    version = (version_ihl >> 4) & 0x0F
    if version != 4:
        # IPv6支持
        if version == 6 and len(data) - offset >= 40:
            return _parse_ipv6_header(data, offset)
        return None

    ihl = (version_ihl & 0x0F) * 4
    total_length = struct.unpack('!H', data[offset+2:offset+4])[0]
    ttl = data[offset+8]
    protocol = data[offset+9]
    src_ip = '.'.join(str(b) for b in data[offset+12:offset+16])
    dst_ip = '.'.join(str(b) for b in data[offset+16:offset+20])

    return {
        'version': version,
        'header_length': ihl,
        'total_length': total_length,
        'ttl': ttl,
        'protocol': protocol,
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'tos': data[offset+1],
        'ip_id': struct.unpack('!H', data[offset+4:offset+6])[0],
        'ip_flags': (data[offset+6] >> 5) & 0x07,
        'payload_offset': offset + ihl,
    }


def _parse_ipv6_header(data: bytes, offset: int) -> Optional[dict]:
    """解析IPv6头"""
    if len(data) - offset < 40:
        return None

    payload_length = struct.unpack('!H', data[offset+4:offset+6])[0]
    next_header = data[offset+6]
    hop_limit = data[offset+7]

    # IPv6地址格式化
    src_bytes = data[offset+8:offset+24]
    dst_bytes = data[offset+24:offset+40]
    src_ip = ':'.join(f'{src_bytes[i]:02x}{src_bytes[i+1]:02x}' for i in range(0, 16, 2))
    dst_ip = ':'.join(f'{dst_bytes[i]:02x}{dst_bytes[i+1]:02x}' for i in range(0, 16, 2))

    # 简化：映射到IPv4兼容格式
    # 实际使用时可能需要完整IPv6支持
    return {
        'version': 6,
        'header_length': 40,
        'total_length': payload_length + 40,
        'ttl': hop_limit,
        'protocol': next_header,
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'tos': 0,
        'ip_id': 0,
        'ip_flags': 0,
        'payload_offset': offset + 40,
    }


def parse_tcp_header(data: bytes, offset: int = 0) -> Optional[dict]:
    """解析TCP头"""
    if len(data) - offset < 20:
        return None

    src_port = struct.unpack('!H', data[offset:offset+2])[0]
    dst_port = struct.unpack('!H', data[offset+2:offset+4])[0]
    seq = struct.unpack('!I', data[offset+4:offset+8])[0]
    ack = struct.unpack('!I', data[offset+8:offset+12])[0]
    data_offset = ((data[offset+12] >> 4) & 0x0F) * 4
    flags = data[offset+13]
    window = struct.unpack('!H', data[offset+14:offset+16])[0]
    urgent_ptr = struct.unpack('!H', data[offset+18:offset+20])[0]

    payload_offset = offset + data_offset
    payload_length = len(data) - payload_offset if payload_offset < len(data) else 0

    return {
        'src_port': src_port,
        'dst_port': dst_port,
        'seq': seq,
        'ack': ack,
        'data_offset': data_offset,
        'flags': flags,
        'window': window,
        'urgent_ptr': urgent_ptr,
        'payload_offset': payload_offset,
        'payload_length': payload_length,
    }


def parse_udp_header(data: bytes, offset: int = 0) -> Optional[dict]:
    """解析UDP头"""
    if len(data) - offset < 8:
        return None

    src_port = struct.unpack('!H', data[offset:offset+2])[0]
    dst_port = struct.unpack('!H', data[offset+2:offset+4])[0]
    length = struct.unpack('!H', data[offset+4:offset+6])[0]

    return {
        'src_port': src_port,
        'dst_port': dst_port,
        'length': length,
        'payload_offset': offset + 8,
        'payload_length': length - 8,
    }
