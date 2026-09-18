"""
TLS 1.3 解析器
解析TLS握手消息，提取ClientHello/ServerHello中的特征字段。
参考：PacketPrint的加密流量指纹方法

核心提取字段：
- JA3/JA4指纹
- SNI (Server Name Indication)
- ALPN (Application-Layer Protocol Negotiation)
- 支持的密码套件列表
- 扩展列表及顺序
- TLS版本
- 证书信息
"""
import hashlib
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import IntEnum


# RFC 8701 16位GREASE精确集合（§5.2：位掩码判定会误吞 0x1A2A 这类非保留值）
GREASE_16 = frozenset(0x0A0A + 0x1010 * i for i in range(16))


class TLSContentType(IntEnum):
    CHANGE_CIPHER_SPEC = 20
    ALERT = 21
    HANDSHAKE = 22
    APPLICATION_DATA = 23
    HEARTBEAT = 24


class TLSHandshakeType(IntEnum):
    CLIENT_HELLO = 1
    SERVER_HELLO = 2
    NEW_SESSION_TICKET = 4
    ENCRYPTED_EXTENSIONS = 8
    CERTIFICATE = 11
    CERTIFICATE_VERIFY = 15
    FINISHED = 20


class TLSExtensionType(IntEnum):
    SERVER_NAME = 0           # SNI
    MAX_FRAGMENT_LENGTH = 1
    CLIENT_CERTIFICATE_URL = 2
    TRUSTED_CA_KEYS = 3
    STATUS_REQUEST = 5
    SUPPORTED_GROUPS = 10
    EC_POINT_FORMATS = 11
    SIGNATURE_ALGORITHMS = 13
    USE_SRTP = 14
    HEARTBEAT = 15
    APPLICATION_LAYER_PROTOCOL_NEGOTIATION = 16  # ALPN
    SIGNED_CERTIFICATE_TIMESTAMP = 18
    CLIENT_CERTIFICATE_TYPE = 19
    SERVER_CERTIFICATE_TYPE = 20
    PADDING = 21
    EXTENDED_MASTER_SECRET = 23
    COMPRESS_CERTIFICATE = 27
    SESSION_TICKET = 35
    PRE_SHARED_KEY = 41
    EARLY_DATA = 42
    SUPPORTED_VERSIONS = 43
    COOKIE = 44
    PSK_KEY_EXCHANGE_MODES = 45
    CERTIFICATE_AUTHORITIES = 47
    POST_HANDSHAKE_AUTH = 49
    KEY_SHARE = 51


@dataclass
class TLSExtension:
    """TLS扩展"""
    ext_type: int
    ext_name: str = ""
    data: bytes = b""
    parsed_value: Optional[object] = None


@dataclass
class TLSClientHello:
    """ClientHello解析结果"""
    tls_version: int = 0
    random: bytes = b""
    session_id: bytes = b""
    cipher_suites: List[int] = field(default_factory=list)
    compression_methods: List[int] = field(default_factory=list)
    extensions: List[TLSExtension] = field(default_factory=list)

    # 解析后的关键字段
    sni: str = ""
    alpn_protocols: List[str] = field(default_factory=list)
    supported_versions: List[int] = field(default_factory=list)
    supported_groups: List[int] = field(default_factory=list)
    signature_algorithms: List[int] = field(default_factory=list)
    ec_point_formats: List[int] = field(default_factory=list)

    # 指纹
    ja3_str: str = ""
    ja3_hash: str = ""
    ja4_str: str = ""
    ja4_hash: str = ""

    # 扩展指纹
    extension_types: List[int] = field(default_factory=list)
    extension_order_hash: str = ""


@dataclass
class TLSServerHello:
    """ServerHello解析结果"""
    tls_version: int = 0
    random: bytes = b""
    session_id: bytes = b""
    cipher_suite: int = 0
    compression_method: int = 0
    extensions: List[TLSExtension] = field(default_factory=list)

    # 解析后的关键字段
    selected_alpn: str = ""
    supported_version: int = 0

    # 指纹
    ja3s_str: str = ""
    ja3s_hash: str = ""
    ja4s_str: str = ""
    ja4s_hash: str = ""


@dataclass
class TLSCertificate:
    """证书信息"""
    cert_data: bytes = b""
    cert_length: int = 0
    issuer: str = ""
    subject: str = ""
    serial_number: bytes = b""
    signature_algorithm: int = 0
    not_before: str = ""
    not_after: str = ""


@dataclass
class TLSHandshakeRecord:
    """完整的TLS握手记录"""
    content_type: int = 0
    tls_version: int = 0
    record_length: int = 0

    client_hello: Optional[TLSClientHello] = None
    server_hello: Optional[TLSServerHello] = None
    certificates: List[TLSCertificate] = field(default_factory=list)

    # 统计信息
    handshake_messages_count: int = 0
    total_handshake_bytes: int = 0
    has_encrypted_extensions: bool = False
    has_certificate: bool = False
    has_certificate_verify: bool = False
    has_finished: bool = False


class TLSParser:
    """TLS协议解析器"""

    # 常用TLS版本号映射
    TLS_VERSIONS = {
        0x0301: "TLS 1.0",
        0x0302: "TLS 1.1",
        0x0303: "TLS 1.2",
        0x0304: "TLS 1.3",
    }

    # 常见密码套件名称（部分）
    CIPHER_SUITE_NAMES = {
        0x1301: "TLS_AES_128_GCM_SHA256",
        0x1302: "TLS_AES_256_GCM_SHA384",
        0x1303: "TLS_CHACHA20_POLY1305_SHA256",
        0x002F: "TLS_RSA_WITH_AES_128_CBC_SHA",
        0x0035: "TLS_RSA_WITH_AES_256_CBC_SHA",
        0x009C: "TLS_RSA_WITH_AES_128_GCM_SHA256",
        0x009D: "TLS_RSA_WITH_AES_256_GCM_SHA384",
        0xC02B: "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
        0xC02C: "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
        0xC02F: "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
        0xC030: "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
        0xCCA8: "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
        0xCCA9: "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256",
    }

    # 扩展类型名称映射
    EXTENSION_NAMES = {v.value: v.name for v in TLSExtensionType}

    def __init__(self):
        self.records: List[TLSHandshakeRecord] = []

    def parse_record(self, data: bytes, offset: int = 0) -> Optional[TLSHandshakeRecord]:
        """解析一个TLS记录"""
        if len(data) - offset < 5:
            return None

        record = TLSHandshakeRecord()
        record.content_type = data[offset]
        record.tls_version = struct.unpack('!H', data[offset+1:offset+3])[0]
        record.record_length = struct.unpack('!H', data[offset+3:offset+5])[0]

        if record.content_type != TLSContentType.HANDSHAKE:
            return None

        payload = data[offset+5:offset+5+record.record_length]
        if len(payload) < 4:
            return None
        # R13: record头声明长度未凑满=截断，不得视为完整记录
        if len(payload) < record.record_length:
            return None

        self._parse_handshake_messages(record, payload)
        return record

    def _parse_handshake_messages(self, record: TLSHandshakeRecord, payload: bytes):
        """解析握手中的所有握手消息"""
        pos = 0
        while pos + 4 <= len(payload):
            hs_type = payload[pos]
            hs_length = struct.unpack('!I', b'\x00' + payload[pos+1:pos+4])[0]
            msg_data = payload[pos+4:pos+4+hs_length]

            record.handshake_messages_count += 1
            record.total_handshake_bytes += 4 + hs_length

            if hs_type == TLSHandshakeType.CLIENT_HELLO:
                record.client_hello = self._parse_client_hello(msg_data)
            elif hs_type == TLSHandshakeType.SERVER_HELLO:
                record.server_hello = self._parse_server_hello(msg_data)
            elif hs_type == TLSHandshakeType.CERTIFICATE:
                record.has_certificate = True
                record.certificates = self._parse_certificates(msg_data)
            elif hs_type == TLSHandshakeType.ENCRYPTED_EXTENSIONS:
                record.has_encrypted_extensions = True
            elif hs_type == TLSHandshakeType.CERTIFICATE_VERIFY:
                record.has_certificate_verify = True
            elif hs_type == TLSHandshakeType.FINISHED:
                record.has_finished = True

            pos += 4 + hs_length

    def _parse_client_hello(self, data: bytes) -> TLSClientHello:
        """解析ClientHello消息"""
        ch = TLSClientHello()

        if len(data) < 34:
            return ch

        ch.tls_version = struct.unpack('!H', data[0:2])[0]
        ch.random = data[2:34]

        pos = 34
        # Session ID
        if pos < len(data):
            session_id_len = data[pos]
            pos += 1
            ch.session_id = data[pos:pos+session_id_len]
            pos += session_id_len

        # Cipher Suites
        if pos + 2 <= len(data):
            cs_len = struct.unpack('!H', data[pos:pos+2])[0]
            pos += 2
            for i in range(0, cs_len, 2):
                if pos + i + 2 <= len(data):
                    cs = struct.unpack('!H', data[pos+i:pos+i+2])[0]
                    ch.cipher_suites.append(cs)
            pos += cs_len

        # Compression Methods
        if pos < len(data):
            cm_len = data[pos]
            pos += 1
            ch.compression_methods = list(data[pos:pos+cm_len])
            pos += cm_len

        # Extensions
        if pos + 2 <= len(data):
            ext_len = struct.unpack('!H', data[pos:pos+2])[0]
            pos += 2
            self._parse_extensions(ch, data[pos:pos+ext_len])

        # 生成指纹
        self._compute_ja3(ch)
        self._compute_ja4(ch)
        self._compute_extension_fingerprint(ch)

        return ch

    def _parse_server_hello(self, data: bytes) -> TLSServerHello:
        """解析ServerHello消息"""
        sh = TLSServerHello()

        if len(data) < 34:
            return sh

        sh.tls_version = struct.unpack('!H', data[0:2])[0]
        sh.random = data[2:34]

        pos = 34
        # Session ID
        if pos < len(data):
            session_id_len = data[pos]
            pos += 1
            sh.session_id = data[pos:pos+session_id_len]
            pos += session_id_len

        # Cipher Suite
        if pos + 2 <= len(data):
            sh.cipher_suite = struct.unpack('!H', data[pos:pos+2])[0]
            pos += 2

        # Compression Method
        if pos < len(data):
            sh.compression_method = data[pos]
            pos += 1

        # Extensions
        if pos + 2 <= len(data):
            ext_len = struct.unpack('!H', data[pos:pos+2])[0]
            pos += 2
            self._parse_server_extensions(sh, data[pos:pos+ext_len])

        # 生成指纹
        self._compute_ja3s(sh)
        self._compute_ja4s(sh)

        return sh

    def _parse_extensions(self, ch: TLSClientHello, ext_data: bytes):
        """解析ClientHello扩展"""
        pos = 0
        while pos + 4 <= len(ext_data):
            ext_type = struct.unpack('!H', ext_data[pos:pos+2])[0]
            ext_len = struct.unpack('!H', ext_data[pos+2:pos+4])[0]
            ext_value = ext_data[pos+4:pos+4+ext_len]

            ext = TLSExtension(
                ext_type=ext_type,
                ext_name=self.EXTENSION_NAMES.get(ext_type, f"unknown_{ext_type}"),
                data=ext_value
            )

            # 解析特定扩展
            if ext_type == TLSExtensionType.SERVER_NAME:
                self._parse_sni(ext, ext_value, ch)
            elif ext_type == TLSExtensionType.APPLICATION_LAYER_PROTOCOL_NEGOTIATION:
                self._parse_alpn(ext, ext_value, ch)
            elif ext_type == TLSExtensionType.SUPPORTED_VERSIONS:
                self._parse_supported_versions(ext, ext_value, ch)
            elif ext_type == TLSExtensionType.SUPPORTED_GROUPS:
                self._parse_supported_groups(ext, ext_value, ch)
            elif ext_type == TLSExtensionType.SIGNATURE_ALGORITHMS:
                self._parse_signature_algorithms(ext, ext_value, ch)
            elif ext_type == TLSExtensionType.EC_POINT_FORMATS:
                self._parse_ec_point_formats(ext, ext_value, ch)

            ch.extensions.append(ext)
            ch.extension_types.append(ext_type)
            pos += 4 + ext_len

    def _parse_server_extensions(self, sh: TLSServerHello, ext_data: bytes):
        """解析ServerHello扩展"""
        pos = 0
        while pos + 4 <= len(ext_data):
            ext_type = struct.unpack('!H', ext_data[pos:pos+2])[0]
            ext_len = struct.unpack('!H', ext_data[pos+2:pos+4])[0]
            ext_value = ext_data[pos+4:pos+4+ext_len]

            ext = TLSExtension(ext_type=ext_type, data=ext_value)

            if ext_type == TLSExtensionType.APPLICATION_LAYER_PROTOCOL_NEGOTIATION:
                if len(ext_value) >= 2:
                    alpn_len = struct.unpack('!H', ext_value[0:2])[0]
                    if len(ext_value) >= 2 + alpn_len:
                        proto_len = ext_value[2]
                        sh.selected_alpn = ext_value[3:3+proto_len].decode('ascii', errors='ignore')
            elif ext_type == TLSExtensionType.SUPPORTED_VERSIONS:
                if len(ext_value) >= 2:
                    sh.supported_version = struct.unpack('!H', ext_value[0:2])[0]

            sh.extensions.append(ext)
            pos += 4 + ext_len

    def _parse_sni(self, ext: TLSExtension, data: bytes, ch: TLSClientHello):
        """解析SNI扩展"""
        if len(data) < 5:
            return
        # SNI list length (2) + name type (1) + name length (2)
        name_len = struct.unpack('!H', data[3:5])[0]
        if len(data) >= 5 + name_len:
            ch.sni = data[5:5+name_len].decode('ascii', errors='ignore')

    def _parse_alpn(self, ext: TLSExtension, data: bytes, ch: TLSClientHello):
        """解析ALPN扩展"""
        if len(data) < 2:
            return
        alpn_list_len = struct.unpack('!H', data[0:2])[0]
        pos = 2
        while pos < 2 + alpn_list_len and pos < len(data):
            proto_len = data[pos]
            if pos + 1 + proto_len <= len(data):
                proto = data[pos+1:pos+1+proto_len].decode('ascii', errors='ignore')
                ch.alpn_protocols.append(proto)
            pos += 1 + proto_len

    def _parse_supported_versions(self, ext: TLSExtension, data: bytes, ch: TLSClientHello):
        """解析支持的TLS版本扩展"""
        if len(data) < 1:
            return
        versions_len = data[0]
        pos = 1
        while pos + 2 <= 1 + versions_len and pos + 2 <= len(data):
            version = struct.unpack('!H', data[pos:pos+2])[0]
            ch.supported_versions.append(version)
            pos += 2

    def _parse_supported_groups(self, ext: TLSExtension, data: bytes, ch: TLSClientHello):
        """解析支持的椭圆曲线组"""
        if len(data) < 2:
            return
        groups_len = struct.unpack('!H', data[0:2])[0]
        pos = 2
        while pos + 2 <= 2 + groups_len and pos + 2 <= len(data):
            group = struct.unpack('!H', data[pos:pos+2])[0]
            ch.supported_groups.append(group)
            pos += 2

    def _parse_signature_algorithms(self, ext: TLSExtension, data: bytes, ch: TLSClientHello):
        """解析签名算法"""
        if len(data) < 2:
            return
        sa_len = struct.unpack('!H', data[0:2])[0]
        pos = 2
        while pos + 2 <= 2 + sa_len and pos + 2 <= len(data):
            sa = struct.unpack('!H', data[pos:pos+2])[0]
            ch.signature_algorithms.append(sa)
            pos += 2

    def _parse_ec_point_formats(self, ext: TLSExtension, data: bytes, ch: TLSClientHello):
        """解析EC点格式"""
        if len(data) < 1:
            return
        formats_len = data[0]
        ch.ec_point_formats = list(data[1:1+formats_len])

    def _parse_certificates(self, data: bytes) -> List[TLSCertificate]:
        """解析证书消息"""
        certs = []
        if len(data) < 3:
            return certs
        # TLS 1.3证书消息有3字节请求上下文前缀
        pos = 3
        while pos + 3 <= len(data):
            cert_len = struct.unpack('!I', b'\x00' + data[pos:pos+3])[0]
            pos += 3
            if pos + cert_len <= len(data):
                cert = TLSCertificate(
                    cert_data=data[pos:pos+cert_len],
                    cert_length=cert_len
                )
                certs.append(cert)
            pos += cert_len
            # 跳过扩展
            if pos + 2 <= len(data):
                ext_len = struct.unpack('!H', data[pos:pos+2])[0]
                pos += 2 + ext_len
        return certs

    # ===================== JA3 指纹计算 =====================

    @staticmethod
    def _is_grease(value: int) -> bool:
        """RFC 8701 GREASE精确判定：16位保留值集合（位掩码会误判 0x1A2A 等）"""
        return value in GREASE_16

    def _compute_ja3(self, ch: TLSClientHello):
        """计算JA3指纹字符串和哈希（按RFC 8701剔除GREASE值）"""
        # JA3 = TLSVersion,Ciphers,Extensions,EllipticCurves,EllipticCurvePointFormats
        version = ch.tls_version
        ciphers = "-".join(str(c) for c in ch.cipher_suites if not self._is_grease(c))
        extensions = "-".join(str(e) for e in ch.extension_types if not self._is_grease(e))
        curves = "-".join(str(g) for g in ch.supported_groups if not self._is_grease(g))
        formats = "-".join(str(f) for f in ch.ec_point_formats if not self._is_grease(f))

        ch.ja3_str = f"{version},{ciphers},{extensions},{curves},{formats}"
        ch.ja3_hash = hashlib.md5(ch.ja3_str.encode()).hexdigest()

    def _compute_ja3s(self, sh: TLSServerHello):
        """计算JA3S指纹"""
        ext_types = [e.ext_type for e in sh.extensions if not self._is_grease(e.ext_type)]
        extensions = "-".join(str(e) for e in ext_types)

        sh.ja3s_str = f"{sh.tls_version},{sh.cipher_suite},{extensions}"
        sh.ja3s_hash = hashlib.md5(sh.ja3s_str.encode()).hexdigest()

    # ===================== JA4 指纹计算 =====================

    def _compute_ja4(self, ch: TLSClientHello):
        """计算JA4指纹 (简化版)"""
        # JA4格式: t{TLS_ver}{SNI}{num_ciphers}{num_extensions}_{hash1}_{hash2}
        sni_flag = 'd' if ch.sni else 'i'  # d=domain, i=IP
        tls_ver = "13" if 0x0304 in ch.supported_versions else (
            "12" if 0x0303 in ch.supported_versions else f"{ch.tls_version & 0xFF:02d}"
        )

        num_ciphers = f"{len(ch.cipher_suites):02d}"
        num_extensions = f"{len(ch.extensions):02d}"

        # 第一部分
        ja4_a = f"t{tls_ver}{sni_flag}{num_ciphers}{num_extensions}"

        # 第二部分：密码套件哈希（排序后取前32字符）
        sorted_ciphers = sorted(ch.cipher_suites)
        cipher_str = ",".join(f"{c:04x}" for c in sorted_ciphers)
        ja4_b = hashlib.sha256(cipher_str.encode()).hexdigest()[:12]

        # 第三部分：扩展哈希
        ext_str = ",".join(f"{e:04x}" for e in ch.extension_types)
        ja4_c = hashlib.sha256(ext_str.encode()).hexdigest()[:12]

        ch.ja4_str = f"{ja4_a}_{ja4_b}_{ja4_c}"
        ch.ja4_hash = hashlib.sha256(ch.ja4_str.encode()).hexdigest()

    def _compute_ja4s(self, sh: TLSServerHello):
        """计算JA4S指纹"""
        ext_types = [e.ext_type for e in sh.extensions]
        tls_ver = "13" if sh.supported_version == 0x0304 else "12"

        ja4s_a = f"t{tls_ver}{sh.cipher_suite:04x}{len(ext_types):02d}"
        ext_str = ",".join(f"{e:04x}" for e in ext_types)
        ja4s_b = hashlib.sha256(ext_str.encode()).hexdigest()[:12]

        sh.ja4s_str = f"{ja4s_a}_{ja4s_b}"
        sh.ja4s_hash = hashlib.sha256(sh.ja4s_str.encode()).hexdigest()

    def _compute_extension_fingerprint(self, ch: TLSClientHello):
        """计算扩展顺序指纹"""
        ext_str = "-".join(str(e) for e in ch.extension_types)
        ch.extension_order_hash = hashlib.sha256(ext_str.encode()).hexdigest()[:16]

    # ===================== 特征提取接口 =====================

    def extract_features(self, record: TLSHandshakeRecord) -> Dict[str, object]:
        """从TLS握手记录提取特征字典"""
        features = {}

        # 基础信息
        features['tls_version'] = record.tls_version
        features['handshake_messages_count'] = record.handshake_messages_count
        features['total_handshake_bytes'] = record.total_handshake_bytes
        features['has_encrypted_extensions'] = int(record.has_encrypted_extensions)
        features['has_certificate'] = int(record.has_certificate)

        # ClientHello特征
        if record.client_hello:
            ch = record.client_hello
            features['ch_tls_version'] = ch.tls_version
            features['ch_sni'] = ch.sni
            features['ch_sni_length'] = len(ch.sni)
            features['ch_num_cipher_suites'] = len(ch.cipher_suites)
            features['ch_num_extensions'] = len(ch.extensions)
            features['ch_num_supported_versions'] = len(ch.supported_versions)
            features['ch_num_supported_groups'] = len(ch.supported_groups)
            features['ch_num_signature_algorithms'] = len(ch.signature_algorithms)
            features['ch_session_id_length'] = len(ch.session_id)
            features['ch_compression_methods'] = ch.compression_methods

            # 指纹
            features['ja3_hash'] = ch.ja3_hash
            features['ja3_str'] = ch.ja3_str
            features['ja4_hash'] = ch.ja4_hash
            features['ja4_str'] = ch.ja4_str
            features['extension_order_hash'] = ch.extension_order_hash

            # ALPN
            features['alpn_protocols'] = ch.alpn_protocols
            features['has_h2'] = int('h2' in ch.alpn_protocols)
            features['has_http11'] = int('http/1.1' in ch.alpn_protocols)

            # 密码套件特征
            tls13_ciphers = [c for c in ch.cipher_suites if 0x1301 <= c <= 0x1303]
            features['has_tls13_ciphers'] = int(len(tls13_ciphers) > 0)
            features['num_tls13_ciphers'] = len(tls13_ciphers)

            # 扩展列表（有序）
            features['extension_types'] = ch.extension_types

        # ServerHello特征
        if record.server_hello:
            sh = record.server_hello
            features['sh_tls_version'] = sh.tls_version
            features['sh_cipher_suite'] = sh.cipher_suite
            features['sh_num_extensions'] = len(sh.extensions)
            features['sh_selected_alpn'] = sh.selected_alpn
            features['ja3s_hash'] = sh.ja3s_hash
            features['ja3s_str'] = sh.ja3s_str
            features['ja4s_hash'] = sh.ja4s_hash
            features['ja4s_str'] = sh.ja4s_str

        # 证书特征
        features['num_certificates'] = len(record.certificates)
        if record.certificates:
            cert = record.certificates[0]
            features['cert_length'] = cert.cert_length

        return features


def parse_tls_from_pcap(pcap_data: bytes) -> List[TLSHandshakeRecord]:
    """从原始PCAP数据解析所有TLS握手记录"""
    parser = TLSParser()
    records = []

    # 跳过PCAP全局头(24字节)
    pos = 24
    while pos + 16 <= len(pcap_data):
        # 读取包头
        ts_sec = struct.unpack('<I', pcap_data[pos:pos+4])[0]
        incl_len = struct.unpack('<I', pcap_data[pos+8:pos+12])[0]
        orig_len = struct.unpack('<I', pcap_data[pos+12:pos+16])[0]
        pos += 16

        if pos + incl_len > len(pcap_data):
            break

        pkt_data = pcap_data[pos:pos+incl_len]
        pos += incl_len

        # 跳过以太网头(14字节) + IP头(至少20字节) + TCP头(至少20字节)
        # 简化处理：搜索TLS记录头
        for offset in range(14, min(len(pkt_data) - 5, 100)):
            if pkt_data[offset] == 0x16:  # TLS Handshake
                record = parser.parse_record(pkt_data, offset)
                if record and record.client_hello:
                    records.append(record)
                break

    return records
