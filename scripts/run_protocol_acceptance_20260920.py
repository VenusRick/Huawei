# -*- coding: utf-8 -*-
"""Formal protocol/session acceptance for Huawei Cup topic 4.

This runner intentionally separates protocol-processing evidence from
classification evidence:

* QUIC: real VisQUIC PCAP processing + parser-state fixtures.
* TCP/session: deterministic semantic probes + real-PCAP anomaly statistics.

No training/test split is changed by this script.
"""
from __future__ import annotations

import json
import resource
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from src.parser.pcap_reader import PCAPReader
from src.parser.quic.quic_parser import QUICInitialPacket, QUICParser, is_quic_packet
from src.parser.session.session_manager import (
    PacketInfo,
    Protocol,
    SessionManager,
    parse_ip_header,
    parse_udp_header,
)


ROOT = Path("/workspace/Huawei")
OUT = ROOT / "output/protocol_and_system_acceptance_20260920"
QUIC_OUT = OUT / "quic"
TCP_OUT = OUT / "tcp_session"


def _rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _quic_initial(version: int = 1) -> bytes:
    first = 0xC0  # long header + fixed bit + Initial + PN len 1
    dcid = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    scid = b"\x11\x12\x13\x14\x15\x16\x17\x18"
    return (
        bytes([first])
        + struct.pack("!I", version)
        + bytes([len(dcid)]) + dcid
        + bytes([len(scid)]) + scid
        + b"\x00" + b"\x01" + b"\x01"
    )


def _protocol_fixtures() -> Dict[str, object]:
    qp = QUICParser()
    v1 = qp.parse_packet(_quic_initial(1))
    unknown = qp.parse_packet(_quic_initial(0x1A2A3A4A))
    short = qp.parse_packet(bytes([0x40]) + b"\x00" * 12)
    plain_udp = b"\x00" * 32
    checks = {
        "v1_initial_valid": bool(isinstance(v1, QUICInitialPacket) and v1.is_valid and v1.version == 1),
        "unknown_long_header_visible": bool(isinstance(unknown, QUICInitialPacket) and unknown.is_valid),
        "short_header_degrades_without_crash": bool(isinstance(short, dict) and short.get("type") == "1-RTT"),
        "plain_udp_negative": bool(qp.parse_packet(plain_udp) is None and not is_quic_packet(plain_udp)),
        "truncated_negative": bool(qp.parse_packet(b"\xC0\x00\x00") is None),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _udp_payload(pkt_data: bytes, link_type: int) -> Tuple[int, int, bytes] | None:
    ip_offset = 0
    if link_type == 1:
        if len(pkt_data) < 14:
            return None
        ethertype = struct.unpack("!H", pkt_data[12:14])[0]
        ip_offset = 14
        if ethertype == 0x8100:
            if len(pkt_data) < 18:
                return None
            ethertype = struct.unpack("!H", pkt_data[16:18])[0]
            ip_offset = 18
        if ethertype not in (0x0800, 0x86DD):
            return None
    elif link_type == 113:
        if len(pkt_data) < 16:
            return None
        ip_offset = 16
    ip = parse_ip_header(pkt_data, ip_offset)
    if not ip or ip["protocol"] != 17:
        return None
    udp = parse_udp_header(pkt_data, ip["payload_offset"])
    if not udp:
        return None
    ip_end = min(len(pkt_data), ip_offset + ip["total_length"])
    payload = pkt_data[udp["payload_offset"]:ip_end]
    return int(udp["src_port"]), int(udp["dst_port"]), payload


def run_quic_real() -> Dict[str, object]:
    roots = [
        Path("/workspace/datasets/VisQUIC/repo-samples/dataset-samples/VisQUIC"),
        Path("/workspace/datasets/VisQUIC/data/VisQUIC"),
    ]
    chosen: List[Path] = []
    per_app = Counter()
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*quic_anonymized_filtered.pcap")):
            try:
                rel = p.relative_to(root)
                app = rel.parts[0] if rel.parts else "unknown"
            except Exception:
                app = "unknown"
            if per_app[app] >= 10:
                continue
            chosen.append(p)
            per_app[app] += 1
            if len(chosen) >= 30:
                break
        if len(chosen) >= 30:
            break
    if not chosen:
        raise RuntimeError("No real VisQUIC PCAP files found")

    t0 = time.perf_counter()
    bytes_total = packets = udp_packets = reader_initial = parse_errors = sessions = 0
    raw_quic_like = raw_initial = raw_short = raw_other_long = raw_plain_udp = 0
    versions = Counter()
    files = []
    for p in chosen:
        bytes_total += p.stat().st_size
        raw_reader = PCAPReader(SessionManager())
        qp = QUICParser()
        local = Counter()
        for pkt_data, _ts, link_type in raw_reader._read_packets(str(p), max_read_packets=50000):
            packets += 1
            u = _udp_payload(pkt_data, link_type)
            if u is None:
                continue
            udp_packets += 1
            _sport, _dport, payload = u
            if not payload:
                continue
            if is_quic_packet(payload):
                raw_quic_like += 1
                local["quic_like"] += 1
                parsed = qp.parse_packet(payload)
                if isinstance(parsed, QUICInitialPacket) and parsed.is_valid:
                    raw_initial += 1
                    local["initial"] += 1
                    versions[f"0x{parsed.version:08x}"] += 1
                elif isinstance(parsed, dict):
                    raw_short += 1
                    local["short"] += 1
                else:
                    raw_other_long += 1
                    local["other_long_or_encrypted"] += 1
            else:
                raw_plain_udp += 1

        rd = PCAPReader(SessionManager())
        ss = list(rd.read_pcap_generator(str(p), max_read_packets=50000))
        st = rd.get_statistics()
        sessions += len(ss)
        reader_initial += int(st["total_quic_packets"])
        parse_errors += int(st["parse_errors"])
        files.append({
            "file": str(p), "bytes": p.stat().st_size, "sessions": len(ss),
            "packets": int(st["packets_last_read"]), "udp_packets": int(st["total_udp_packets"]),
            "reader_quic_initial_packets": int(st["total_quic_packets"]),
            "parse_errors": int(st["parse_errors"]), "raw_states": dict(local),
        })
    wall = time.perf_counter() - t0
    return {
        "source": "real VisQUIC PCAP", "n_files": len(chosen), "app_file_counts": dict(per_app),
        "input_bytes": bytes_total, "packets_scanned_direct": packets, "udp_packets_direct": udp_packets,
        "raw_quic_like_packets": raw_quic_like, "raw_initial_packets": raw_initial,
        "raw_short_header_packets": raw_short, "raw_other_long_or_encrypted_packets": raw_other_long,
        "raw_plain_udp_packets": raw_plain_udp, "versions": dict(versions),
        "pcapreader_sessions": sessions, "pcapreader_quic_initial_packets": reader_initial,
        "parse_errors": parse_errors, "wall_sec": wall,
        "packets_per_sec": packets / max(wall, 1e-9),
        "mib_per_sec": (bytes_total / (1024.0 * 1024.0)) / max(wall, 1e-9),
        "peak_rss_mib": _rss_mib(), "files": files,
    }


def _pkt(ts: float, seq: int, payload: bytes = b"", flags: int = 0x10,
         fwd: bool = True, sport: int = 12345, dport: int = 443) -> PacketInfo:
    if fwd:
        src, dst, sp, dp = "10.0.0.1", "10.0.0.2", sport, dport
    else:
        src, dst, sp, dp = "10.0.0.2", "10.0.0.1", dport, sport
    return PacketInfo(timestamp=ts, src_ip=src, dst_ip=dst, src_port=sp, dst_port=dp,
                      protocol=Protocol.TCP, length=40 + len(payload), payload_length=len(payload),
                      payload=payload, tcp_flags=flags, tcp_seq=seq, tcp_ack=0, tcp_window=65535)


def _session_semantics() -> Dict[str, object]:
    checks: Dict[str, bool] = {}
    sm = SessionManager(tcp_timeout=30, udp_timeout=10)
    sm.process_packet(_pkt(0.0, 1000, flags=0x02))
    sm.process_packet(_pkt(0.1, 5000, flags=0x12, fwd=False))
    sm.process_packet(_pkt(0.2, 1001, flags=0x10))
    sm.process_packet(_pkt(0.3, 1001, payload=b"A" * 100))
    sm.process_packet(_pkt(0.4, 1201, payload=b"C" * 100))
    sess = next(iter(sm.active_sessions.values()))
    checks["out_of_order_detected"] = sess.num_out_of_order == 1
    pre = sess.total_packets
    sm.process_packet(_pkt(0.5, 1101, payload=b"B" * 100))
    checks["gap_fill_flushes_buffer"] = sess.total_packets == pre + 2
    pre = sess.total_packets
    sm.process_packet(_pkt(0.6, 1101, payload=b"B" * 100))
    checks["retransmission_deduplicated"] = sess.num_retransmissions == 1 and sess.total_packets == pre
    canonical = next(iter(sm.active_sessions))
    expected_before = sm._tcp_tracker[canonical]["fwd_expected_seq"]
    sm.process_packet(_pkt(0.7, 999999, payload=b"", flags=0x10))
    expected_after = sm._tcp_tracker[canonical]["fwd_expected_seq"]
    checks["pure_ack_does_not_advance_seq"] = expected_before == expected_after
    sm.process_packet(_pkt(0.8, 5001, payload=b"R" * 50, fwd=False))
    sm.process_packet(_pkt(0.9, 1301, flags=0x11))
    closed = sm.process_packet(_pkt(1.0, 5051, flags=0x11, fwd=False))
    checks["bidirectional_fin_closes"] = bool(closed is not None and closed.is_closed)

    sm2 = SessionManager()
    sm2.process_packet(_pkt(0.0, 10, flags=0x02))
    rst = sm2.process_packet(_pkt(0.1, 11, flags=0x04))
    checks["rst_closes"] = bool(rst is not None and rst.is_closed)

    sm3 = SessionManager(tcp_timeout=1, udp_timeout=1)
    sm3.process_packet(_pkt(0.0, 1, flags=0x02))
    old_obj = next(iter(sm3.active_sessions.values()))
    sm3.process_packet(_pkt(2.0, 100, flags=0x02))
    new_obj = next(iter(sm3.active_sessions.values()))
    checks["tcp_idle_timeout_new_session"] = old_obj is not new_obj and old_obj.is_closed

    def udp(ts: float) -> PacketInfo:
        return PacketInfo(timestamp=ts, src_ip="1.1.1.1", dst_ip="2.2.2.2",
                          src_port=1111, dst_port=2222, protocol=Protocol.UDP,
                          length=100, payload_length=72, payload=b"x" * 72)
    sm4 = SessionManager(udp_timeout=1)
    sm4.process_packet(udp(0.0))
    u1 = next(iter(sm4.active_sessions.values()))
    sm4.process_packet(udp(2.0))
    u2 = next(iter(sm4.active_sessions.values()))
    checks["udp_idle_timeout_new_session"] = u1 is not u2 and u1.is_closed

    sm5 = SessionManager()
    sm5.process_packet(_pkt(0.0, 1000, flags=0x02))
    sm5.process_packet(_pkt(0.1, 5000, flags=0x12, fwd=False))
    sm5.process_packet(_pkt(0.2, 1001, flags=0x10))
    sm5.process_packet(_pkt(0.3, 1201, payload=b"C" * 100))
    before = next(iter(sm5.active_sessions.values())).total_packets
    flushed = sm5.flush_all()[0]
    checks["flush_all_keeps_buffered_packet"] = flushed.total_packets == before + 1
    return {"checks": checks, "passed": all(checks.values())}


def run_tcp_real() -> Dict[str, object]:
    root = Path("/workspace/cz-华为杯/data/all_data")
    files = [root / f"USTC-TFC2016-{x}.pcap" for x in ["FTP", "Gmail", "MySQL", "WorldOfWarcraft"]]
    files = [p for p in files if p.exists()]
    if not files:
        files = sorted(root.glob("CSTNET-TLS1.3-*.pcap"))[:4]
    if not files:
        raise RuntimeError("No real TCP PCAP available")
    t0 = time.perf_counter()
    total_bytes = total_packets = total_tcp = sessions = retrans = ooo = sess_retrans = sess_ooo = parse_errors = 0
    per_file = []
    for p in files:
        total_bytes += p.stat().st_size
        rd = PCAPReader(SessionManager())
        ss = list(rd.read_pcap_generator(str(p), max_read_packets=100000))
        st = rd.get_statistics()
        total_packets += int(st["packets_last_read"]); total_tcp += int(st["total_tcp_packets"])
        sessions += len(ss); retrans += int(st["total_retransmissions"]); ooo += int(st["total_out_of_order"])
        sess_retrans += int(st["sessions_with_retransmissions"]); sess_ooo += int(st["sessions_with_out_of_order"])
        parse_errors += int(st["parse_errors"])
        per_file.append({"file": str(p), "bytes": p.stat().st_size, "packets": int(st["packets_last_read"]),
                         "tcp_packets": int(st["total_tcp_packets"]), "sessions": len(ss),
                         "retransmissions": int(st["total_retransmissions"]), "out_of_order": int(st["total_out_of_order"]),
                         "parse_errors": int(st["parse_errors"])})
    wall = time.perf_counter() - t0
    return {"source": "real USTC/CSTNET PCAP", "n_files": len(files), "input_bytes": total_bytes,
            "packets": total_packets, "tcp_packets": total_tcp, "sessions": sessions,
            "retransmissions": retrans, "out_of_order": ooo,
            "sessions_with_retransmissions": sess_retrans, "sessions_with_out_of_order": sess_ooo,
            "retransmission_rate_per_tcp_packet": retrans/max(total_tcp,1),
            "out_of_order_rate_per_tcp_packet": ooo/max(total_tcp,1), "parse_errors": parse_errors,
            "wall_sec": wall, "packets_per_sec": total_packets/max(wall,1e-9),
            "mib_per_sec": (total_bytes/(1024.0*1024.0))/max(wall,1e-9), "peak_rss_mib": _rss_mib(),
            "files": per_file}


def main() -> None:
    QUIC_OUT.mkdir(parents=True, exist_ok=True); TCP_OUT.mkdir(parents=True, exist_ok=True)
    fixture = _protocol_fixtures(); quic_real = run_quic_real()
    quic = {"fixture": fixture, "real": quic_real,
            "acceptance_pass": bool(fixture["passed"] and quic_real["parse_errors"] == 0 and quic_real["raw_initial_packets"] > 0)}
    (QUIC_OUT/"acceptance.json").write_text(json.dumps(quic,indent=2,ensure_ascii=False))
    semantics = _session_semantics(); tcp_real = run_tcp_real()
    tcp = {"semantic_fixture": semantics, "real": tcp_real,
           "acceptance_pass": bool(semantics["passed"] and tcp_real["parse_errors"] == 0 and tcp_real["sessions"] > 0)}
    (TCP_OUT/"acceptance.json").write_text(json.dumps(tcp,indent=2,ensure_ascii=False))
    print("QUIC",json.dumps({k:v for k,v in quic_real.items() if k!="files"},ensure_ascii=False))
    print("QUIC_FIXTURE",fixture)
    print("TCP",json.dumps({k:v for k,v in tcp_real.items() if k!="files"},ensure_ascii=False))
    print("TCP_SEMANTIC",semantics)


if __name__ == "__main__":
    main()
