# -*- coding: utf-8 -*-
"""Regression guards for protocol/session acceptance added 2026-09-20."""
from scripts.run_protocol_acceptance_20260920 import _protocol_fixtures, _session_semantics


def test_quic_protocol_fixture_states():
    result = _protocol_fixtures()
    assert result["passed"], result
    assert all(result["checks"].values())


def test_tcp_session_semantics():
    result = _session_semantics()
    assert result["passed"], result
    required = {
        "out_of_order_detected", "gap_fill_flushes_buffer", "retransmission_deduplicated",
        "pure_ack_does_not_advance_seq", "bidirectional_fin_closes", "rst_closes",
        "tcp_idle_timeout_new_session", "udp_idle_timeout_new_session", "flush_all_keeps_buffered_packet",
    }
    assert required.issubset(result["checks"])
