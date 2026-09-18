"""
QUIC协议解析器
解析QUIC Initial包，提取握手信息。

QUIC特点：
- 基于UDP，端口443为主
- Initial包有固定的包头格式
- TLS 1.3集成在QUIC中（TLS in QUIC）
- Initial包使用初始密钥可解密

核心提取字段：
- QUIC版本
- 连接ID (Source/Destination CID)
- Token
- Initial包中的CRYPTO帧（包含TLS ClientHello/ServerHello）
- 传输参数（从TLS扩展中提取）
"""
import struct
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class QUICVersion:
    """QUIC版本定义"""
    VERSION_1 = 0x00000001         # RFC 9000
    VERSION_2 = 0x6b3343cf         # RFC 9369
    DRAFT_29 = 0xff00001d          # Draft 29
    DRAFT_32 = 0xff000020          # Draft 32
    GOOGLE_QUIC = 0x51303430       # Google Q040
    MICROSOFT_QUIC = 0x00000001    # MS-Quic


@dataclass
class QUICInitialPacket:
    """QUIC Initial包解析结果"""
    # 包头
    header_form: int = 0           # 1=Long Header
    fixed_bit: int = 0
    long_packet_type: int = 0      # 0=Initial, 1=0-RTT, 2=Handshake, 3=Retry
    version: int = 0
    dest_connection_id: bytes = b""
    src_connection_id: bytes = b""
    token_length: int = 0
    token: bytes = b""
    packet_number: int = 0
    packet_length: int = 0

    # 解析状态
    is_valid: bool = False
    is_version_specific: bool = False

    # 统计信息
    total_bytes: int = 0


@dataclass
class QUICConnection:
    """QUIC连接信息"""
    dest_cid: bytes = b""
    src_cid: bytes = b""
    version: int = 0
    initial_packets: List[QUICInitialPacket] = field(default_factory=list)
    handshake_complete: bool = False

    # 提取的TLS信息（从Initial包中解密后）
    client_hello: Optional[dict] = None
    server_hello: Optional[dict] = None

    # 统计
    total_initial_bytes: int = 0
    num_initial_packets: int = 0


class QUICParser:
    """QUIC协议解析器"""

    # QUIC版本映射
    KNOWN_VERSIONS = {
        0x00000001: "QUIC v1 (RFC 9000)",
        0x6b3343cf: "QUIC v2 (RFC 9369)",
        0xff00001d: "Draft-29",
        0xff000020: "Draft-32",
        0x51303430: "Google QUIC Q040",
    }

    # 长包头类型
    PACKET_TYPES = {
        0: "Initial",
        1: "0-RTT",
        2: "Handshake",
        3: "Retry",
    }

    def __init__(self):
        self.connections: Dict[str, QUICConnection] = {}

    def parse_packet(self, data: bytes, offset: int = 0) -> Optional[QUICInitialPacket]:
        """解析一个QUIC包"""
        if len(data) - offset < 7:
            return None

        first_byte = data[offset]

        # 检查是否为长包头 (Header Form bit)
        header_form = (first_byte >> 7) & 1
        if header_form != 1:
            # 短包头（1-RTT），不是Initial包
            return self._parse_short_header(data, offset)

        # 长包头解析
        pkt = QUICInitialPacket()
        pkt.header_form = header_form
        pkt.fixed_bit = (first_byte >> 6) & 1
        pkt.long_packet_type = (first_byte >> 4) & 3
        pkt.total_bytes = len(data) - offset

        # 只关注Initial包
        if pkt.long_packet_type != 0:
            return None

        pos = offset + 1

        # 版本 (4字节)
        if pos + 4 > len(data):
            return None
        pkt.version = struct.unpack('!I', data[pos:pos+4])[0]
        pos += 4

        # Destination Connection ID
        if pos >= len(data):
            return None
        dcid_len = data[pos]
        pos += 1
        if pos + dcid_len > len(data):
            return None
        pkt.dest_connection_id = data[pos:pos+dcid_len]
        pos += dcid_len

        # Source Connection ID
        if pos >= len(data):
            return None
        scid_len = data[pos]
        pos += 1
        if pos + scid_len > len(data):
            return None
        pkt.src_connection_id = data[pos:pos+scid_len]
        pos += scid_len

        # Token Length (变长整数)
        if pos >= len(data):
            return None
        pkt.token_length, consumed = self._read_varint(data, pos)
        pos += consumed
        if pos + pkt.token_length > len(data):
            return None
        pkt.token = data[pos:pos+pkt.token_length]
        pos += pkt.token_length

        # Packet Length (变长整数)
        if pos >= len(data):
            return None
        pkt.packet_length, consumed = self._read_varint(data, pos)
        pos += consumed

        # Packet Number (1-4字节，取决于长度位)
        pn_length = (first_byte & 0x03) + 1
        if pos + pn_length > len(data):
            return None
        pkt.packet_number = int.from_bytes(data[pos:pos+pn_length], 'big')
        pos += pn_length

        pkt.is_valid = True

        # 记录连接信息
        self._track_connection(pkt)

        return pkt

    def _parse_short_header(self, data: bytes, offset: int) -> Optional[dict]:
        """解析短包头（1-RTT），用于统计"""
        if len(data) - offset < 5:
            return None
        first_byte = data[offset]
        # §5.3: QUIC 包 fixed bit 必须为 1（RFC 9000），否则是普通 UDP
        if ((first_byte >> 6) & 1) != 1:
            return None
        return {
            'type': '1-RTT',
            'spin_bit': (first_byte >> 5) & 1,
            'key_phase': (first_byte >> 2) & 1,
            'packet_number_length': (first_byte & 0x03) + 1,
            'total_bytes': len(data) - offset,
        }

    def _read_varint(self, data: bytes, offset: int) -> Tuple[int, int]:
        """读取QUIC变长整数 (RFC 9000 Section 16)"""
        if offset >= len(data):
            return 0, 0

        first = data[offset]
        length = 1 << ((first >> 6) & 0x03)

        if offset + length > len(data):
            return 0, 0

        value = int.from_bytes(data[offset:offset+length], 'big')
        # 清除长度指示位
        if length == 1:
            value &= 0x3F
        elif length == 2:
            value &= 0x3FFF
        elif length == 4:
            value &= 0x3FFFFFFF
        elif length == 8:
            value &= 0x3FFFFFFFFFFFFFFF

        return value, length

    def _track_connection(self, pkt: QUICInitialPacket):
        """追踪QUIC连接"""
        cid_key = pkt.dest_connection_id.hex() + "_" + pkt.src_connection_id.hex()

        if cid_key not in self.connections:
            self.connections[cid_key] = QUICConnection(
                dest_cid=pkt.dest_connection_id,
                src_cid=pkt.src_connection_id,
                version=pkt.version
            )

        conn = self.connections[cid_key]
        conn.initial_packets.append(pkt)
        conn.num_initial_packets += 1
        conn.total_initial_bytes += pkt.packet_length

    # ===================== 特征提取接口 =====================

    def extract_features(self, conn: QUICConnection) -> Dict[str, object]:
        """从QUIC连接提取特征字典"""
        features = {}

        # 基础信息
        features['quic_version'] = conn.version
        features['quic_version_name'] = self.KNOWN_VERSIONS.get(
            conn.version, f"Unknown (0x{conn.version:08x})"
        )
        features['quic_dest_cid_length'] = len(conn.dest_cid)
        features['quic_src_cid_length'] = len(conn.src_cid)
        features['quic_num_initial_packets'] = conn.num_initial_packets
        features['quic_total_initial_bytes'] = conn.total_initial_bytes

        # Token特征
        if conn.initial_packets:
            first_pkt = conn.initial_packets[0]
            features['quic_has_token'] = int(first_pkt.token_length > 0)
            features['quic_token_length'] = first_pkt.token_length
            features['quic_token_hash'] = hashlib.md5(
                first_pkt.token
            ).hexdigest()[:16] if first_pkt.token else ""

        # 版本特征
        features['quic_is_v1'] = int(conn.version == 0x00000001)
        features['quic_is_v2'] = int(conn.version == 0x6b3343cf)
        features['quic_is_draft'] = int((conn.version >> 24) == 0xFF)
        features['quic_is_google'] = int(
            conn.version in [0x51303430, 0x51303431, 0x51303432, 0x51303433]
        )

        # CID指纹（用于关联同一连接的多个包）
        features['quic_cid_hash'] = hashlib.md5(
            conn.dest_cid + conn.src_cid
        ).hexdigest()[:16]

        return features

    def extract_features_from_udp_payload(self, data: bytes) -> Optional[Dict[str, object]]:
        """从UDP载荷直接提取QUIC特征（快捷方法）"""
        pkt = self.parse_packet(data)
        # 短包头返回dict（无is_valid属性），仅Initial包可提取特征
        if pkt and not isinstance(pkt, dict) and pkt.is_valid:
            cid_key = pkt.dest_connection_id.hex() + "_" + pkt.src_connection_id.hex()
            if cid_key in self.connections:
                return self.extract_features(self.connections[cid_key])
        return None


def is_quic_packet(data: bytes) -> bool:
    """启发式判断是否为QUIC包"""
    if len(data) < 7:
        return False

    first_byte = data[0]
    header_form = (first_byte >> 7) & 1

    if header_form == 1:
        # 长包头
        long_packet_type = (first_byte >> 4) & 3
        if long_packet_type > 3:
            return False
        # 检查版本
        if len(data) >= 5:
            version = struct.unpack('!I', data[1:5])[0]
            if version == 0:
                return False  # 版本0不是QUIC
            if version in QUICParser.KNOWN_VERSIONS:
                return True
            # 其他可能的版本
            if (version >> 8) == 0xFF0000:
                return True  # Draft版本
        return True
    else:
        # 短包头，更难判断
        # 检查第一个字节的模式
        # QUIC短包头: 01KSPNNN (K=Key Phase, S=Spin Bit, P=Reserved, NNN=PN Length-1)
        if (first_byte & 0x40) == 0:  # Fixed bit must be 0 in short header
            return False
        return True  # 不确定，假定是


def is_quic_port(src_port: int, dst_port: int) -> bool:
    """基于端口判断是否可能是QUIC"""
    QUIC_PORTS = {443, 8443, 7890, 853, 8853}
    return src_port in QUIC_PORTS or dst_port in QUIC_PORTS
