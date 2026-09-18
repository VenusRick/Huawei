"""
生成模拟PCAP数据集
用 scapy 生成具有不同流量模式的模拟PCAP文件，用于验证完整流水线。

每个场景生成不同特征模式：
- benign: 正常HTTPS浏览，短连接为主
- wechat_voice: 中等持续时间，对称流量
- wechat_video: 长持续时间，下行远大于上行
- whatsapp_voice: 类似微信语音但参数不同
- whatsapp_video: 类似微信视频但参数不同
- tor: 高熵、无SNI、固定包大小
- instagram_post: 短连接，上行较大（上传图片）
- facebook_post: 短连接，上行较大
- twitter_post: 短连接，上行较大
"""
import os
import random
import struct
from pathlib import Path

# 使用 scapy 生成PCAP
try:
    from scapy.all import (
        IP, TCP, UDP, Raw, wrpcap, Ether,
        RandShort, RandIP
    )
except ImportError:
    print("需要安装 scapy: pip install scapy")
    exit(1)


def random_ip():
    """生成随机IP"""
    return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def generate_tls_client_hello(sni=""):
    """生成模拟的TLS ClientHello载荷"""
    payload = bytearray()
    # TLS Record Header
    payload.append(0x16)  # Handshake
    payload.extend(b'\x03\x01')  # TLS 1.0 record version
    # Record length (placeholder)
    record_len_pos = len(payload)
    payload.extend(b'\x00\x00')

    # ClientHello
    payload.append(1)  # ClientHello
    # Handshake length (placeholder)
    hs_len_pos = len(payload)
    payload.extend(b'\x00\x00\x00')

    # TLS version
    payload.extend(b'\x03\x03')  # TLS 1.2

    # Random (32 bytes)
    payload.extend(os.urandom(32))

    # Session ID
    payload.append(0)

    # Cipher Suites
    payload.extend(b'\x00\x04')
    payload.extend(b'\x13\x01')  # TLS_AES_128_GCM_SHA256
    payload.extend(b'\x13\x02')  # TLS_AES_256_GCM_SHA384

    # Compression
    payload.append(1)
    payload.append(0)

    # Extensions
    ext_data = bytearray()

    # SNI extension (如果提供了SNI)
    if sni:
        sni_bytes = sni.encode('ascii')
        ext_data.extend(b'\x00\x00')
        ext_data.extend((5 + len(sni_bytes)).to_bytes(2, 'big'))
        ext_data.extend((3 + len(sni_bytes)).to_bytes(2, 'big'))
        ext_data.append(0)
        ext_data.extend(len(sni_bytes).to_bytes(2, 'big'))
        ext_data.extend(sni_bytes)

    # Supported Versions
    ext_data.extend(b'\x00\x2b\x00\x03\x02\x03\x04')

    # 计算并填充长度
    payload.extend(len(ext_data).to_bytes(2, 'big'))
    payload.extend(ext_data)

    # 填充handshake长度
    hs_len = len(payload) - hs_len_pos - 3
    payload[hs_len_pos] = (hs_len >> 16) & 0xFF
    payload[hs_len_pos + 1] = (hs_len >> 8) & 0xFF
    payload[hs_len_pos + 2] = hs_len & 0xFF

    # 填充record长度
    record_len = len(payload) - record_len_pos - 2
    payload[record_len_pos] = (record_len >> 8) & 0xFF
    payload[record_len_pos + 1] = record_len & 0xFF

    return bytes(payload)


def generate_tls_server_hello():
    """生成模拟的TLS ServerHello载荷"""
    payload = bytearray()
    payload.append(0x16)
    payload.extend(b'\x03\x03')
    payload.extend(b'\x00\x3e')  # record length

    payload.append(2)  # ServerHello
    payload.extend(b'\x00\x00\x3a')  # hs length
    payload.extend(b'\x03\x03')  # TLS 1.2
    payload.extend(os.urandom(32))  # random
    payload.append(0)  # session id
    payload.extend(b'\x13\x01')  # cipher suite
    payload.append(0)  # compression

    # Extensions
    payload.extend(b'\x00\x00')  # no extensions
    return bytes(payload)


def make_pkt(src_ip, dst_ip, src_port, dst_port, size, direction, ts, proto='tcp'):
    """构造一个数据包"""
    if proto == 'tcp':
        flags = 'PA' if direction == 1 else 'A'
        pkt = (
            IP(src=src_ip, dst=dst_ip) /
            TCP(sport=src_port, dport=dst_port, flags=flags,
                seq=random.randint(1000, 999999),
                ack=random.randint(1000, 999999)) /
            Raw(load=os.urandom(max(0, size - 54)))
        )
    else:
        pkt = (
            IP(src=src_ip, dst=dst_ip) /
            UDP(sport=src_port, dport=dst_port) /
            Raw(load=os.urandom(max(0, size - 42)))
        )
    pkt.time = ts
    return pkt


def generate_session(config, start_time):
    """
    生成一个完整的会话（含TLS握手 + 数据传输）

    config: {
        'src_ip', 'dst_ip', 'sni', 'duration', 'fwd_pkt_count',
        'bwd_pkt_count', 'fwd_size_range', 'bwd_size_range',
        'iat_range', 'include_tls', 'proto'
    }
    """
    pkts = []
    ts = start_time
    src_port = random.randint(32768, 65535)
    dst_port = 443
    src_ip = config.get('src_ip', random_ip())
    dst_ip = config.get('dst_ip', random_ip())
    proto = config.get('proto', 'tcp')

    # TCP 三次握手
    if proto == 'tcp':
        pkts.append(make_pkt(src_ip, dst_ip, src_port, dst_port, 66, 1, ts))
        ts += 0.01
        pkts.append(make_pkt(dst_ip, src_ip, dst_port, src_port, 66, -1, ts))
        ts += 0.005
        pkts.append(make_pkt(src_ip, dst_ip, src_port, dst_port, 54, 1, ts))
        ts += 0.01

    # TLS 握手
    if config.get('include_tls', True):
        sni = config.get('sni', '')
        client_hello = generate_tls_client_hello(sni)
        pkts.append(make_pkt(src_ip, dst_ip, src_port, dst_port,
                            54 + len(client_hello), 1, ts))
        ts += 0.02

        server_hello = generate_tls_server_hello()
        pkts.append(make_pkt(dst_ip, src_ip, dst_port, src_port,
                            54 + len(server_hello), -1, ts))
        ts += 0.05

    # 数据传输
    duration = config.get('duration', 10.0)
    fwd_count = config.get('fwd_pkt_count', 50)
    bwd_count = config.get('bwd_pkt_count', 50)
    fwd_size = config.get('fwd_size_range', (100, 1400))
    bwd_size = config.get('bwd_size_range', (100, 1400))
    iat_range = config.get('iat_range', (0.01, 0.5))

    total_count = fwd_count + bwd_count
    if total_count == 0:
        return pkts

    interval = duration / total_count

    for i in range(total_count):
        if i < fwd_count:
            direction = 1
            size = random.randint(*fwd_size)
            pkt = make_pkt(src_ip, dst_ip, src_port, dst_port, size, direction, ts, proto)
        else:
            direction = -1
            size = random.randint(*bwd_size)
            pkt = make_pkt(dst_ip, src_ip, dst_port, src_port, size, direction, ts, proto)

        pkts.append(pkt)
        ts += interval + random.uniform(*iat_range)

    return pkts


def generate_dataset(output_dir: str, sessions_per_class: int = 20):
    """生成完整数据集"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 各场景的流量模式配置
    # 参考论文：PacketPrint的App流量特征 + NTLFlowLyzer的行为画像
    scenarios = {
        'benign': {
            'sni': 'www.google.com',
            'duration': 3.0,
            'fwd_pkt_count': 15,
            'bwd_pkt_count': 20,
            'fwd_size_range': (200, 800),
            'bwd_size_range': (500, 1400),
            'iat_range': (0.05, 0.3),
            'include_tls': True,
        },
        'wechat_voice': {
            'sni': '',  # 微信通话不一定有标准SNI
            'duration': 60.0,
            'fwd_pkt_count': 100,
            'bwd_pkt_count': 100,
            'fwd_size_range': (80, 300),    # 语音通话包较小
            'bwd_size_range': (80, 300),
            'iat_range': (0.01, 0.05),      # 语音通话IAT较小
            'include_tls': True,
        },
        'wechat_video': {
            'sni': '',
            'duration': 120.0,
            'fwd_pkt_count': 80,
            'bwd_pkt_count': 300,           # 视频下行远多于上行
            'fwd_size_range': (200, 800),   # 上行较小（摄像头）
            'bwd_size_range': (800, 1400),  # 下行大（视频流）
            'iat_range': (0.005, 0.03),
            'include_tls': True,
        },
        'whatsapp_voice': {
            'sni': '',
            'duration': 45.0,
            'fwd_pkt_count': 80,
            'bwd_pkt_count': 80,
            'fwd_size_range': (60, 250),
            'bwd_size_range': (60, 250),
            'iat_range': (0.01, 0.06),
            'include_tls': True,
        },
        'whatsapp_video': {
            'sni': '',
            'duration': 90.0,
            'fwd_pkt_count': 60,
            'bwd_pkt_count': 250,
            'fwd_size_range': (150, 600),
            'bwd_size_range': (600, 1400),
            'iat_range': (0.005, 0.04),
            'include_tls': True,
        },
        'tor': {
            'sni': '',  # Tor没有标准SNI
            'duration': 30.0,
            'fwd_pkt_count': 60,
            'bwd_pkt_count': 60,
            'fwd_size_range': (500, 520),   # Tor cell固定512字节
            'bwd_size_range': (500, 520),
            'iat_range': (0.05, 0.5),
            'include_tls': True,
        },
        'instagram_post': {
            'sni': 'instagram.com',
            'duration': 5.0,
            'fwd_pkt_count': 40,            # 发帖上行多（上传图片）
            'bwd_pkt_count': 15,
            'fwd_size_range': (500, 1400),
            'bwd_size_range': (100, 500),
            'iat_range': (0.01, 0.2),
            'include_tls': True,
        },
        'facebook_post': {
            'sni': 'facebook.com',
            'duration': 4.0,
            'fwd_pkt_count': 35,
            'bwd_pkt_count': 12,
            'fwd_size_range': (400, 1400),
            'bwd_size_range': (100, 400),
            'iat_range': (0.01, 0.15),
            'include_tls': True,
        },
        'twitter_post': {
            'sni': 'twitter.com',
            'duration': 3.5,
            'fwd_pkt_count': 30,
            'bwd_pkt_count': 10,
            'fwd_size_range': (300, 1200),
            'bwd_size_range': (100, 400),
            'iat_range': (0.01, 0.12),
            'include_tls': True,
        },
    }

    total_generated = 0

    for scenario_name, config in scenarios.items():
        scenario_dir = output_path / scenario_name
        scenario_dir.mkdir(exist_ok=True)

        all_pkts = []
        ts = 1700000000.0  # 固定起始时间戳

        print(f"  生成 {scenario_name} ({sessions_per_class} 个会话)...")

        for i in range(sessions_per_class):
            # 每个会话使用不同的IP
            cfg = config.copy()
            cfg['src_ip'] = random_ip()
            cfg['dst_ip'] = random_ip()

            # 添加随机扰动（模拟真实流量的多样性）
            cfg['duration'] = config['duration'] * random.uniform(0.7, 1.3)
            cfg['fwd_pkt_count'] = max(5, int(config['fwd_pkt_count'] * random.uniform(0.8, 1.2)))
            cfg['bwd_pkt_count'] = max(5, int(config['bwd_pkt_count'] * random.uniform(0.8, 1.2)))

            session_pkts = generate_session(cfg, ts)
            all_pkts.extend(session_pkts)

            # 下一个会话间隔
            ts += cfg['duration'] + random.uniform(5.0, 30.0)

        # 写入PCAP
        pcap_file = scenario_dir / f"{scenario_name}.pcap"
        wrpcap(str(pcap_file), all_pkts)
        total_generated += sessions_per_class

        size_mb = pcap_file.stat().st_size / (1024 * 1024)
        print(f"    → {pcap_file} ({size_mb:.2f} MB, {len(all_pkts)} 个包)")

    print(f"\n  总计: {total_generated} 个会话, {len(scenarios)} 个场景")
    return str(output_path)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'data', 'pcaps')
    print("=" * 50)
    print("  生成模拟PCAP数据集")
    print("=" * 50)
    generate_dataset(output_dir, sessions_per_class=20)
