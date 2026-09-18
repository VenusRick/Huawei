"""
PCAP文件读取器
整合以太网/IP/TCP|UDP/TLS/QUIC解析，输出完整的会话数据。

处理流程：
PCAP → 以太网帧 → IP包 → TCP/UDP段 → TLS/QUIC解析 → 会话重建
"""
import struct
import time
from pathlib import Path
from typing import List, Optional, Generator

from src.parser.session.session_manager import (
    SessionManager, PacketInfo, FlowSession, Protocol,
    parse_ip_header, parse_tcp_header, parse_udp_header
)
from src.parser.tls.tls_parser import TLSParser, TLSHandshakeRecord
from src.parser.quic.quic_parser import QUICParser, is_quic_packet, is_quic_port


# 以太网帧类型
ETHERTYPE_IPV4 = 0x0800
ETHERTYPE_IPV6 = 0x86DD
ETHERTYPE_ARP = 0x0806
ETHERTYPE_VLAN = 0x8100


class PCAPReader:
    """PCAP文件读取器"""

    MAX_TLS_BUFFER_TOTAL = 64 * 1024 * 1024   # 全局TLS重组缓冲上限

    def __init__(self, session_manager: Optional[SessionManager] = None):
        self.session_manager = session_manager or SessionManager()
        self.tls_parser = TLSParser()
        self.quic_parser = QUICParser()

        # 统计
        self.total_packets = 0
        self.total_tcp_packets = 0
        self.total_udp_packets = 0
        self.total_tls_records = 0
        self.total_quic_packets = 0
        self.parse_errors = 0
        self.tls_buffer_evicted = 0

        # TCP异常统计（从会话累加）
        self.total_retransmissions = 0
        self.total_out_of_order = 0
        self.sessions_with_retransmissions = 0
        self.sessions_with_out_of_order = 0

        # TLS record跨分段重组缓冲: (src_ip,src_port,dst_ip,dst_port) -> bytes
        self._tls_buffers: dict = {}

    def _read_packets(self, file_path: str):
        """
        通用包生成器，自动识别PCAP/PCAPNG格式。
        每次 yield (pkt_data, timestamp, link_type)。
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PCAP文件不存在: {file_path}")

        with open(file_path, 'rb') as f:
            data = f.read()

        if len(data) < 8:
            return

        magic = struct.unpack('<I', data[0:4])[0]

        if magic in (0xa1b2c3d4, 0xd4c3b2a1, 0xa1b23c4d, 0x4d3cb2a1):
            # ── PCAP 格式 ──
            endian = '<' if magic in (0xa1b2c3d4, 0xa1b23c4d) else '>'
            # 纳秒精度由文件头magic直接判定：0xa1b23c4d(小端)/0x4d3cb2a1(大端)
            tsresol = 1_000_000_000.0 if magic in (0xa1b23c4d, 0x4d3cb2a1) else 1_000_000.0
            link_type = struct.unpack(f'{endian}I', data[20:24])[0]
            pos = 24

            while pos + 16 <= len(data):
                ts_sec = struct.unpack(f'{endian}I', data[pos:pos+4])[0]
                ts_frac = struct.unpack(f'{endian}I', data[pos+4:pos+8])[0]
                incl_len = struct.unpack(f'{endian}I', data[pos+8:pos+12])[0]
                pos += 16
                if pos + incl_len > len(data):
                    break
                pkt_data = data[pos:pos+incl_len]
                pos += incl_len
                timestamp = ts_sec + ts_frac / tsresol
                yield pkt_data, timestamp, link_type

        elif magic == 0x0a0d0d0a:
            # ── PCAPNG 格式 ──
            yield from self._read_pcapng(data)

        else:
            raise ValueError(f"不支持的文件格式: magic=0x{magic:08x}")

    def _read_pcapng(self, data: bytes):
        """解析PCAPNG格式，yield (pkt_data, timestamp, link_type)
        支持小端/大端section（由SHB的BOM判定）；BOM无效时报错而非静默读空。
        """
        pos = 0
        endian = '<'  # 当前section字节序
        link_type = 1  # 默认以太网
        tsresol = 1_000_000.0  # 默认微秒
        ifaces = []  # 接口描述列表 [{link_type, tsresol}]

        while pos + 12 <= len(data):
            # SHB块类型0x0a0d0d0a为回文，不受字节序影响；BOM在绝对偏移pos+8
            if data[pos:pos+4] == b'\x0a\x0d\x0d\x0a':
                bom = struct.unpack('<I', data[pos+8:pos+12])[0]
                if bom == 0x1A2B3C4D:
                    endian = '<'
                elif bom == 0x4D3C2B1A:
                    endian = '>'
                else:
                    raise ValueError(f"PCAPNG字节序标记(BOM)无效: 0x{bom:08x}")
            block_type = struct.unpack(f'{endian}I', data[pos:pos+4])[0]
            block_len = struct.unpack(f'{endian}I', data[pos+4:pos+8])[0]
            if block_len < 12 or pos + block_len > len(data):
                break

            body = data[pos+8:pos+block_len-4]
            # 块尾部的Total Length（用于校验）
            tail_len = struct.unpack(f'{endian}I', data[pos+block_len-4:pos+block_len])[0]
            if tail_len != block_len:
                break

            if block_type == 0x0a0d0d0a:
                # Section Header Block：BOM已在循环头处理
                pass
            elif block_type == 0x00000001:
                # Interface Description Block
                if len(body) >= 8:
                    lt = struct.unpack(f'{endian}H', body[0:2])[0]
                    # Options中查找tsresol（optcode=9）
                    opt_tsresol = 1_000_000.0
                    opt_pos = 8
                    while opt_pos + 4 <= len(body):
                        opt_code = struct.unpack(f'{endian}H', body[opt_pos:opt_pos+2])[0]
                        opt_len = struct.unpack(f'{endian}H', body[opt_pos+2:opt_pos+4])[0]
                        if opt_code == 0:  # opt_endofopt
                            break
                        if opt_code == 9 and opt_len >= 1:  # if_tsresol
                            resol_byte = body[opt_pos+4]
                            if resol_byte & 0x80:
                                opt_tsresol = 2.0 ** (resol_byte & 0x7f)
                            else:
                                opt_tsresol = 10.0 ** resol_byte
                        opt_pos += 4 + ((opt_len + 3) & ~3)  # 4字节对齐
                    ifaces.append({'link_type': lt, 'tsresol': opt_tsresol})
            elif block_type == 0x00000006:
                # Enhanced Packet Block
                if len(body) >= 20:
                    iface_id = struct.unpack(f'{endian}I', body[0:4])[0]
                    ts_high = struct.unpack(f'{endian}I', body[4:8])[0]
                    ts_low = struct.unpack(f'{endian}I', body[8:12])[0]
                    cap_len = struct.unpack(f'{endian}I', body[12:16])[0]
                    if iface_id < len(ifaces):
                        iface = ifaces[iface_id]
                        lt = iface['link_type']
                        resol = iface['tsresol']
                    else:
                        lt = link_type
                        resol = tsresol
                    ts_raw = (ts_high << 32) | ts_low
                    timestamp = ts_raw / resol
                    pkt_offset = 20
                    if pkt_offset + cap_len <= len(body):
                        pkt_data = body[pkt_offset:pkt_offset+cap_len]
                        yield pkt_data, timestamp, lt
            elif block_type == 0x00000003:
                # Simple Packet Block
                if len(body) >= 4:
                    orig_len = struct.unpack(f'{endian}I', body[0:4])[0]
                    cap_len = min(orig_len, block_len - 16)  # 块头+尾共12字节+4字节orig_len
                    if 4 + cap_len <= len(body):
                        pkt_data = body[4:4+cap_len]
                        yield pkt_data, 0.0, link_type  # SPB无时间戳
            pos += block_len

    def _tls_feed(self, pkt):
        """按四元组方向缓冲TCP载荷，凑满完整TLS握手record后解析（支持跨分段ClientHello）"""
        key = (pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port)
        buf = self._tls_buffers.get(key, b'') + pkt.payload
        while len(buf) >= 5 and buf[0] == 0x16:
            rec_len = struct.unpack('!H', buf[3:5])[0]
            if len(buf) < 5 + rec_len:
                break
            if pkt.tls_info is None:
                record = self.tls_parser.parse_record(buf[:5+rec_len])
                if record:
                    pkt.tls_info = self.tls_parser.extract_features(record)
                    self.total_tls_records += 1
            buf = buf[5+rec_len:]
        # 只保留仍在等待续传的握手record；非握手残留或超限缓冲直接丢弃
        if not buf or buf[0] != 0x16 or len(buf) > 65536:
            buf = b''
        self._tls_buffers[key] = buf
        # M4: TLS 缓冲全局上限（超限丢最大缓冲并计数，证据可统计）
        if sum(len(v) for v in self._tls_buffers.values()) > self.MAX_TLS_BUFFER_TOTAL:
            for k2 in sorted(self._tls_buffers,
                             key=lambda k2: len(self._tls_buffers[k2]),
                             reverse=True)[:max(1, len(self._tls_buffers) // 4)]:
                self._tls_buffers.pop(k2, None)
                self.tls_buffer_evicted += 1

    def read_pcap(self, file_path: str) -> List[FlowSession]:
        """读取PCAP/PCAPNG文件，返回所有会话"""
        self.session_manager.reset()
        self._tls_buffers = {}

        for pkt_data, timestamp, link_type in self._read_packets(file_path):
            try:
                self._process_raw_packet(pkt_data, timestamp, link_type)
            except Exception:
                self.parse_errors += 1
            self.total_packets += 1

        # 关闭所有剩余会话并累加TCP异常统计
        closed_sessions = self.session_manager.flush_all()
        for session in closed_sessions:
            self._accumulate_tcp_anomalies(session)

        return self.session_manager.get_all_closed_sessions()

    def _accumulate_tcp_anomalies(self, session: FlowSession):
        """从会话中累加重传/乱序统计"""
        if session.num_retransmissions > 0:
            self.total_retransmissions += session.num_retransmissions
            self.sessions_with_retransmissions += 1
        if session.num_out_of_order > 0:
            self.total_out_of_order += session.num_out_of_order
            self.sessions_with_out_of_order += 1

    def read_pcap_generator(self, file_path: str) -> Generator[FlowSession, None, None]:
        """生成器方式读取PCAP/PCAPNG，边解析边产出已完成会话"""
        self.session_manager.reset()
        self._tls_buffers = {}

        for pkt_data, timestamp, link_type in self._read_packets(file_path):
            self._process_raw_packet(pkt_data, timestamp, link_type)
            # M4: 全部关闭路径（返回值/FIN/超时/驱逐）统一从 closed 交付，
            # 交付即释放（SM 不囤积、不重复交付）
            for s in self.session_manager.take_closed_sessions():
                self._accumulate_tcp_anomalies(s)
                yield s
            self.total_packets += 1

        # 产出所有剩余会话并累加统计
        for session in self.session_manager.flush_all():
            self._accumulate_tcp_anomalies(session)
            yield session

    def _process_raw_packet(self, pkt_data: bytes, timestamp: float,
                            link_type: int) -> Optional[FlowSession]:
        """处理一个原始包"""
        ip_offset = 0

        # 以太网帧解封
        if link_type == 1:  # LINKTYPE_ETHERNET
            if len(pkt_data) < 14:
                return None
            ethertype = struct.unpack('!H', pkt_data[12:14])[0]
            ip_offset = 14

            # VLAN标签
            if ethertype == ETHERTYPE_VLAN:
                if len(pkt_data) < 18:
                    return None
                ethertype = struct.unpack('!H', pkt_data[16:18])[0]
                ip_offset = 18

            if ethertype != ETHERTYPE_IPV4 and ethertype != ETHERTYPE_IPV6:
                return None

        elif link_type == 113:  # LINKTYPE_LINUX_SLL
            if len(pkt_data) < 16:
                return None
            ethertype = struct.unpack('!H', pkt_data[14:16])[0]
            ip_offset = 16

        elif link_type == 101:  # LINKTYPE_RAW (IP directly)
            ip_offset = 0
        else:
            ip_offset = 0

        # IP解析
        ip_info = parse_ip_header(pkt_data, ip_offset)
        if ip_info is None:
            return None

        protocol_num = ip_info['protocol']
        payload_offset = ip_info['payload_offset']

        # TCP/UDP解析
        pkt = PacketInfo()
        pkt.timestamp = timestamp
        pkt.src_ip = ip_info['src_ip']
        pkt.dst_ip = ip_info['dst_ip']
        pkt.length = ip_info['total_length']
        pkt.ip_ttl = ip_info['ttl']
        pkt.ip_tos = ip_info.get('tos', 0)
        pkt.ip_id = ip_info.get('ip_id', 0)
        pkt.ip_flags = ip_info.get('ip_flags', 0)

        if protocol_num == 6:  # TCP
            tcp_info = parse_tcp_header(pkt_data, payload_offset)
            if tcp_info is None:
                return None
            pkt.protocol = Protocol.TCP
            pkt.src_port = tcp_info['src_port']
            pkt.dst_port = tcp_info['dst_port']
            pkt.tcp_flags = tcp_info['flags']
            pkt.tcp_seq = tcp_info['seq']
            pkt.tcp_ack = tcp_info['ack']
            pkt.tcp_window = tcp_info['window']
            pkt.tcp_urgent_ptr = tcp_info['urgent_ptr']
            pkt.payload_length = tcp_info['payload_length']
            if tcp_info['payload_length'] > 0:
                # 按IP total_length截断，排除以太网帧padding
                ip_end = ip_offset + ip_info['total_length']
                pkt.payload = pkt_data[tcp_info['payload_offset']:min(len(pkt_data), ip_end)]
                pkt.payload_length = len(pkt.payload)
            self.total_tcp_packets += 1

            # TLS解析（跨TCP分段重组：按方向缓冲，凑满完整record再解析）
            if pkt.payload:
                self._tls_feed(pkt)
            # RST/FIN时清理该方向重组缓冲
            if pkt.tcp_flags & 0x04 or pkt.tcp_flags & 0x01:
                self._tls_buffers.pop((pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port), None)

        elif protocol_num == 17:  # UDP
            udp_info = parse_udp_header(pkt_data, payload_offset)
            if udp_info is None:
                return None
            pkt.protocol = Protocol.UDP
            pkt.src_port = udp_info['src_port']
            pkt.dst_port = udp_info['dst_port']
            pkt.payload_length = udp_info['payload_length']
            if udp_info['payload_length'] > 0 and udp_info['payload_offset'] < len(pkt_data):
                # 按IP total_length截断，排除以太网帧padding
                ip_end = ip_offset + ip_info['total_length']
                pkt.payload = pkt_data[udp_info['payload_offset']:min(len(pkt_data), ip_end)]
                pkt.payload_length = len(pkt.payload)
            self.total_udp_packets += 1

            # QUIC检测（解析失败仅降级为普通UDP，不影响包统计与会话）
            if is_quic_port(pkt.src_port, pkt.dst_port):
                if pkt.payload and is_quic_packet(pkt.payload):
                    try:
                        quic_feats = self.quic_parser.extract_features_from_udp_payload(pkt.payload)
                        if quic_feats:
                            pkt.tls_info = quic_feats
                            self.total_quic_packets += 1
                    except Exception:
                        pass
        else:
            pkt.protocol = Protocol.OTHER

        # 送入会话管理器
        return self.session_manager.process_packet(pkt)

    def get_statistics(self) -> dict:
        """获取解析统计信息"""
        return {
            'total_packets': self.total_packets,
            'total_tcp_packets': self.total_tcp_packets,
            'total_udp_packets': self.total_udp_packets,
            'total_tls_records': self.total_tls_records,
            'total_quic_packets': self.total_quic_packets,
            'parse_errors': self.parse_errors,
            'active_sessions': self.session_manager.get_active_session_count(),
            'closed_sessions': self.session_manager.get_closed_session_count(),
            'total_retransmissions': self.total_retransmissions,
            'total_out_of_order': self.total_out_of_order,
            'sessions_with_retransmissions': self.sessions_with_retransmissions,
            'sessions_with_out_of_order': self.sessions_with_out_of_order,
        }
