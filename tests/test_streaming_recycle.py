# -*- coding: utf-8 -*-
"""流式与资源回收验收（M4）：上限证据计数、消费即释放、TLS缓冲上限。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parser.pcap_reader import PCAPReader
from src.parser.session.session_manager import SessionManager
from tests.regression.test_review_regressions import P, Protocol


def _mk(i, seq=1, ts=1.0, plen=100):
    return P("10.0.0.1", 1000 + i, "10.0.0.2", 443, 0x18, seq, ts, plen,
             payload=b"x" * 60)


def test_evict_oldest_counts_evidence():
    sm = SessionManager(max_sessions=2)
    for i in range(3):
        sm.process_packet(_mk(i, ts=1.0 + i))
    assert sm.evicted_sessions == 1
    evicted = [s for s in sm.closed_sessions if getattr(s, "_evicted", False)]
    assert len(evicted) == 1, "被驱逐会话须带证据标记"


def test_closed_overflow_drops_oldest_with_count():
    sm = SessionManager(max_closed_sessions=2)
    for i in range(4):
        sm.process_packet(_mk(i))
        # 手动关闭：喂 FIN 双向
        fin1 = P("10.0.0.1", 1000 + i, "10.0.0.2", 443, 0x11, 999, 9.0, 60)
        fin2 = P("10.0.0.2", 443, "10.0.0.1", 1000 + i, 0x11, 999, 9.1, 60)
        sm.process_packet(fin1)
        sm.process_packet(fin2)
    assert sm.closed_overflow_dropped >= 1, "超限丢弃须留计数"
    assert len(sm.closed_sessions) <= 2


def test_take_closed_clears():
    sm = SessionManager()
    sm.process_packet(_mk(0))
    out = sm.take_closed_sessions()
    assert isinstance(out, list)
    sm.closed_sessions.append("x")
    assert sm.take_closed_sessions() == ["x"]
    assert sm.closed_sessions == []


def _build_pcap(tmp_path, n_flows=3):
    from scapy.all import Ether, IP, TCP, wrpcap
    E0 = 1790000000.0
    pkts = []
    for i in range(n_flows):
        sport = 20000 + i
        pkts.append(Ether() / IP(src="10.1.0.1", dst="10.1.0.2") /
                    TCP(sport=sport, dport=443, flags="PA", seq=1) /
                    (b"d" * 100))
        pkts[-1].time = E0 + i * 10
        pkts.append(Ether() / IP(src="10.1.0.1", dst="10.1.0.2") /
                    TCP(sport=sport, dport=443, flags="FA", seq=200) /
                    (b"e" * 40))
        pkts[-1].time = E0 + i * 10 + 1
        pkts.append(Ether() / IP(src="10.1.0.2", dst="10.1.0.1") /
                    TCP(sport=443, dport=sport, flags="FA", seq=500) /
                    (b"f" * 40))
        pkts[-1].time = E0 + i * 10 + 2
    f = tmp_path / "stream.pcap"
    wrpcap(str(f), pkts)
    return str(f)


def test_generator_releases_closed(tmp_path):
    """流式读取：会话交付后 SM 不再囤积（closed 清空）。"""
    sm = SessionManager(tcp_timeout=300, udp_timeout=60)
    r = PCAPReader(sm)
    delivered = 0
    after_take_sizes = []
    for s in r.read_pcap_generator(_build_pcap(tmp_path)):
        delivered += 1
        after_take_sizes.append(len(sm.closed_sessions))
    assert delivered >= 1
    assert all(n == 0 for n in after_take_sizes), \
        f"流式交付后 SM 仍囤积: {after_take_sizes}"


def test_tls_buffer_global_cap(tmp_path):
    """TLS 缓冲全局超限：丢弃并计数（证据可统计）。"""
    sm = SessionManager()
    r = PCAPReader(sm)
    r.MAX_TLS_BUFFER_TOTAL = 200   # 调小上限触发
    for i in range(6):
        p = P("10.0.0.1", 3000 + i, "10.0.0.2", 443, 0x18, 1, 1.0, 100,
              payload=b"\x16\x03\x01" + b"\xff" * 60)   # 握手半包囤积
        r._tls_feed(p)
    assert r.tls_buffer_evicted >= 1, "超限须丢弃并计数"
