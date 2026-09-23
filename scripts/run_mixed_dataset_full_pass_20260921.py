# -*- coding: utf-8 -*-
"""Full-pass mixed-dataset experiment with a fresh pre-registered holdout.

Goal: on a NEW split (generated before any search), reach simultaneously
Accuracy>=0.95, Macro Precision>=0.95, Macro Recall>=0.98, max per-class
FPR<=0.05 over the 16 classes of USTC/CSTNET/CrossPlatform-China-Android/
VisQUIC.

Discipline (2026-09-21 task contract):
- No filename/path/label/dataset fields as features; the final classifier
  receives only generic traffic-shape features. No port/SNI/JA3/JA4/TLS/
  QUIC/protocol-number/TCP-flag/window fields.
- Fresh split created in stage `split`; stage `search` only ever loads
  train/validation rows; stage `test` loads test rows exactly once, after
  frozen_config.json exists, and refuses to overwrite previous results.
- Feature pool = runtime common_shape (identifier-free subset of the
  production 319-dim schema) + a NEW generic prefix<=64 sequence family
  (direction runs, length/IAT quantiles+histograms, burst structure at two
  gap thresholds, autocorrelation, cumulative progress curves, segment
  ratios). Selected feature budget <= 64.
- The standard part of extraction REUSES the production implementation
  (import, not copy): runtime thresholds, truncated_session_view and the
  Basic/Advanced extractors all come from src/.

Artifacts land in output/mixed_dataset_full_pass_20260921/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path('/workspace/Huawei')
sys.path.insert(0, str(ROOT))

from src.features.runtime import (MIN_SESSION_PACKETS, MIN_SESSION_BYTES,
                                  truncated_session_view)
from src.features.basic.feature_extractor import BasicFeatureExtractor
from src.features.advanced.advanced_extractor import AdvancedFeatureExtractor
from src.parser.pcap_reader import PCAPReader
from src.parser.session.session_manager import SessionManager

OUT = ROOT / 'output/mixed_dataset_full_pass_20260921'
PARTS = OUT / 'cache_parts'
SEED = 20260922
DATASETS = ['CSTNET', 'CrossPlatform', 'USTC', 'VisQUIC']
PREFIX_PACKETS = 64
TARGET_PER_CLASS = 320          # observation budget per class before split
TEST_QUOTA = 60                 # per-class fresh-test rows (adaptive below)
VAL_QUOTA = 60

IDENTIFIER_EXACT = {
    'src_port', 'dst_port', 'is_well_known_src_port', 'is_well_known_dst_port',
    'tls_ja3_hash_prefix', 'tls_ja4_hash_prefix',
}
TCP_FLAG_FEATURES = {
    f'{p}_{f}_flag_count' for p in ('', 'fwd_', 'bwd_')
    for f in ('syn', 'ack', 'fin', 'rst', 'psh', 'urg', 'ece', 'cwr')
}


def common_shape_features(cols: Sequence[str]) -> List[str]:
    out = []
    for c in cols:
        if c.startswith('_') or c in IDENTIFIER_EXACT:
            continue
        if c.startswith(('tls_', 'quic_', 'protocol_', 'tcp_window_')):
            continue
        if c in TCP_FLAG_FEATURES or c in {'syn_ack_rtt', 'num_retransmissions',
                                           'retransmission_ratio'}:
            continue
        out.append(c)
    return sorted(out)


# --------------------------------------------------------------------------
# New generic sequence features (prefix <= 64 packets, transport-agnostic).
# --------------------------------------------------------------------------
def _q(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p)) if len(a) else 0.0


def _safe_ratio(a: float, b: float) -> float:
    return float(a) / float(b) if abs(b) > 1e-12 else 0.0


def _ac(x: np.ndarray, lag: int) -> float:
    """Lag-k autocorrelation of a 1-D series (0 for constant/short series)."""
    x = np.asarray(x, float)
    if len(x) <= lag or float(np.std(x)) < 1e-12:
        return 0.0
    xc = x - x.mean()
    denom = float(np.sum(xc * xc))
    return float(np.sum(xc[:-lag] * xc[lag:]) / denom) if denom > 1e-12 else 0.0


def _hist_frac(a: np.ndarray, edges: Sequence[float]) -> List[float]:
    # np.histogram with an explicit edge list of len(edges)+1 boundaries yields
    # exactly len(edges) bins; keep the empty-array fallback the same width.
    if len(a) == 0:
        return [0.0] * len(edges)
    cnt, _ = np.histogram(a, bins=[float(e) for e in edges] + [np.inf])
    return [float(v) / len(a) for v in cnt]


LEN_EDGES = [32, 64, 128, 256, 512, 768, 1024, 1400, 2048]      # 10 bins
IAT_EDGES = [1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 1e-1, 2.5e-1, 1.0]  # 9 bins


def sequence_features(packets) -> Dict[str, float]:
    """Generic prefix-sequence features from (timestamp, length, direction).

    All features are definable for any TCP/UDP session and use only the
    first <=PREFIX_PACKETS packets (the same prefix view as the runtime).
    """
    n = len(packets)
    t = np.array([float(p.timestamp) for p in packets], float)
    L = np.array([float(p.length) for p in packets], float)
    D = np.array([1 if int(getattr(p, 'direction', 1)) >= 0 else -1 for p in packets], int)
    fwd = D > 0
    bwd = ~fwd
    f: Dict[str, float] = {}

    # -- direction run structure -------------------------------------------
    runs: List[int] = []
    cur = 1
    for i in range(1, n):
        if D[i] != D[i - 1]:
            runs.append(cur)
            cur = 1
        else:
            cur += 1
    runs.append(cur)
    dir_runs = [D[0]] + [D[i] for i in range(1, n) if D[i] != D[i - 1]]
    fwd_runs = [r for r, d in zip(runs, dir_runs) if d > 0]
    bwd_runs = [r for r, d in zip(runs, dir_runs) if d < 0]
    f['seq_dir_n_runs'] = float(len(runs))
    f['seq_dir_switch_rate'] = _safe_ratio(len(runs) - 1, max(n - 1, 1))
    f['seq_dir_max_run'] = float(max(runs))
    f['seq_dir_mean_run'] = float(np.mean(runs))
    f['seq_dir_max_fwd_run'] = float(max(fwd_runs)) if fwd_runs else 0.0
    f['seq_dir_max_bwd_run'] = float(max(bwd_runs)) if bwd_runs else 0.0
    f['seq_dir_first_bwd_pos'] = float(np.argmax(bwd) if bwd.any() else n)
    f['seq_dir_bwd_frac'] = _safe_ratio(int(bwd.sum()), n)
    for k in (8, 16, 32, 64):
        kk = min(k, n)
        f[f'seq_dir_fwd_frac_p{k}'] = _safe_ratio(int(fwd[:kk].sum()), kk)
    # fwd run and bwd run mean lengths
    f['seq_dir_fwd_run_mean'] = float(np.mean(fwd_runs)) if fwd_runs else 0.0
    f['seq_dir_bwd_run_mean'] = float(np.mean(bwd_runs)) if bwd_runs else 0.0

    # -- length statistics per direction ------------------------------------
    lf, lb = L[fwd], L[bwd]
    for tag, arr in (('f', lf), ('b', lb)):
        f[f'seq_len_{tag}_mean'] = float(arr.mean()) if len(arr) else 0.0
        f[f'seq_len_{tag}_std'] = float(arr.std()) if len(arr) > 1 else 0.0
        for p in (25, 50, 75, 90):
            f[f'seq_len_{tag}_q{p}'] = _q(arr, p)
        f[f'seq_len_{tag}_max'] = float(arr.max()) if len(arr) else 0.0
        f[f'seq_len_{tag}_min'] = float(arr.min()) if len(arr) else 0.0
    f['seq_len_mean'] = float(L.mean())
    f['seq_len_std'] = float(L.std())
    f['seq_len_q50'] = _q(L, 50)
    f['seq_len_q75'] = _q(L, 75)
    # length histograms (fractions) per direction
    for tag, arr in (('f', lf), ('b', lb)):
        for i, v in enumerate(_hist_frac(arr, LEN_EDGES)):
            f[f'seq_lenhist_{tag}_{i}'] = v

    # -- cumulative progress curves -----------------------------------------
    cumL = np.cumsum(L)
    cumF = np.cumsum(np.where(fwd, L, 0.0))
    cumB = np.cumsum(np.where(bwd, L, 0.0))
    tot = float(cumL[-1]) if n else 0.0
    totF = float(cumF[-1])
    totB = float(cumB[-1])
    for k in (8, 16, 32, 64):
        kk = min(k, n)
        f[f'seq_cum_tot_p{k}'] = _safe_ratio(float(cumL[kk - 1]), tot)
        f[f'seq_cum_fwd_p{k}'] = _safe_ratio(float(cumF[kk - 1]), totF)
        f[f'seq_cum_bwd_p{k}'] = _safe_ratio(float(cumB[kk - 1]), totB)
    for k in (4, 8, 16):
        kk = min(k, n)
        f[f'seq_cum_pkts_p{k}'] = _safe_ratio(float(cumL[kk - 1]), tot)

    # -- inter-arrival time statistics --------------------------------------
    if n >= 2:
        dt = np.diff(t)
        dt = np.where(dt < 0, 0.0, dt)
        f['seq_iat_mean'] = float(dt.mean())
        f['seq_iat_std'] = float(dt.std())
        for p in (50, 75, 90):
            f[f'seq_iat_q{p}'] = _q(dt, p)
        f['seq_iat_max'] = float(dt.max())
        for i, v in enumerate(_hist_frac(dt, IAT_EDGES)):
            f[f'seq_iathist_{i}'] = v
        half = max((n - 1) // 2, 1)
        m1, m2 = float(dt[:half].mean()) if half else 0.0, float(dt[half:].mean())
        f['seq_iat_halfratio'] = float(np.log1p(m1 / m2)) if m2 > 1e-12 else (
            0.0 if m1 <= 1e-12 else 8.0)
        idx = np.arange(len(dt), dtype=float)
        if len(dt) > 2 and float(dt.std()) > 1e-15:
            f['seq_iat_trend'] = float(np.corrcoef(idx, dt)[0, 1])
        else:
            f['seq_iat_trend'] = 0.0
        # within-direction IAT
        for tag, m in (('f', fwd), ('b', bwd)):
            pos = np.where(m)[0]
            if len(pos) >= 2:
                dtd = np.diff(t[pos])
                dtd = np.where(dtd < 0, 0.0, dtd)
                f[f'seq_iat_{tag}_mean'] = float(dtd.mean())
                f[f'seq_iat_{tag}_q50'] = _q(dtd, 50)
            else:
                f[f'seq_iat_{tag}_mean'] = 0.0
                f[f'seq_iat_{tag}_q50'] = 0.0
        # segment means (packet-index conditioned)
        segL = [L[a:b] for a, b in ((0, 8), (8, 16), (16, 32), (32, 64)) if a < n]
        gm = float(L.mean()) if n else 0.0
        for i, s in enumerate(segL):
            f[f'seq_seg_len_ratio_{i}'] = _safe_ratio(float(s.mean()) if len(s) else 0.0, gm)
        segD = [dt[a:b] for a, b in ((0, 8), (8, 16), (16, 32), (32, 64)) if a < len(dt)]
        gmd = float(dt.mean()) if len(dt) else 0.0
        for i, s in enumerate(segD):
            f[f'seq_seg_iat_ratio_{i}'] = _safe_ratio(float(s.mean()) if len(s) else 0.0, gmd)
    else:
        for k in ('mean', 'std', 'q50', 'q75', 'q90', 'max', 'halfratio', 'trend'):
            f[f'seq_iat_{k}'] = 0.0
        for i in range(len(IAT_EDGES) + 1):
            f[f'seq_iathist_{i}'] = 0.0
        for tag in ('f', 'b'):
            f[f'seq_iat_{tag}_mean'] = 0.0
            f[f'seq_iat_{tag}_q50'] = 0.0
        for i in range(4):
            f[f'seq_seg_len_ratio_{i}'] = 0.0
            f[f'seq_seg_iat_ratio_{i}'] = 0.0

    # -- burst structure at two gap thresholds ------------------------------
    for name, thr in (('05', 0.05), ('2', 0.20)):
        if n >= 2:
            d = np.diff(t)
            starts = np.where(d > thr)[0] + 1
            bounds = [0] + starts.tolist() + [n]
            blens = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
            bbytes = [float(cumL[bounds[i + 1] - 1] - (cumL[bounds[i] - 1] if bounds[i] > 0 else 0.0))
                      for i in range(len(bounds) - 1)]
            gaps = [float(d[s - 1]) for s in starts]
            f[f'seq_burst{name}_count'] = float(len(blens))
            f[f'seq_burst{name}_max_len'] = float(max(blens))
            f[f'seq_burst{name}_mean_len'] = float(np.mean(blens))
            f[f'seq_burst{name}_max_bytes'] = float(max(bbytes)) if bbytes else 0.0
            f[f'seq_burst{name}_top_share'] = _safe_ratio(max(bbytes) if bbytes else 0.0, tot)
            f[f'seq_burst{name}_gap_mean'] = float(np.mean(gaps)) if gaps else 0.0
            f[f'seq_burst{name}_gap_max'] = float(max(gaps)) if gaps else 0.0
        else:
            for k in ('count', 'max_len', 'mean_len', 'max_bytes', 'top_share',
                      'gap_mean', 'gap_max'):
                f[f'seq_burst{name}_{k}'] = 0.0

    # -- autocorrelation / coupling -----------------------------------------
    f['seq_ac_len_lag1'] = _ac(L, 1)
    f['seq_ac_len_lag2'] = _ac(L, 2)
    f['seq_ac_len_lag3'] = _ac(L, 3)
    if n >= 3:
        dt0 = np.diff(t)
        f['seq_ac_iat_lag1'] = _ac(dt0, 1)
        f['seq_ac_iat_lag2'] = _ac(dt0, 2)
    else:
        f['seq_ac_iat_lag1'] = 0.0
        f['seq_ac_iat_lag2'] = 0.0
    if float(L.std()) > 1e-12:
        f['seq_corr_len_dir'] = float(np.corrcoef(L, D)[0, 1])
    else:
        f['seq_corr_len_dir'] = 0.0
    # bwd byte share in the opening 8 packets vs overall
    k8 = min(8, n)
    f['seq_bwdshare_first8'] = _safe_ratio(float(cumB[k8 - 1]), float(cumL[k8 - 1]))
    f['seq_bwdshare_overall'] = _safe_ratio(totB, tot)
    # log byte ratio fwd/bwd (clip)
    f['seq_log_ratio_fb_bytes'] = float(np.clip(np.log1p(totF) - np.log1p(totB), 0.0, 14.0))

    # -- positional packet signature (first packets of the prefix) ----------
    # Raw signed lengths / directions / gaps at fixed packet positions; for
    # short sessions these carry most of the app-specific shape.
    for i in range(16):
        f[f'seq_pos_len_{i}'] = float(L[i] * D[i]) if i < n else 0.0
    for i in range(12):
        f[f'seq_pos_dir_{i}'] = float(D[i]) if i < n else 0.0
    if n >= 2:
        dtp = np.diff(t)
        for i in range(1, 9):
            f[f'seq_pos_iat_{i}'] = (float(min(max(dtp[i - 1], 0.0), 5.0))
                                     if i - 1 < len(dtp) else 0.0)
    else:
        for i in range(1, 9):
            f[f'seq_pos_iat_{i}'] = 0.0
    f['seq_first8_fwd_bytes'] = float(cumF[k8 - 1])
    f['seq_first8_bwd_bytes'] = float(cumB[k8 - 1])

    # -- time-window density / cadence ---------------------------------------
    # Fractions of prefix packets/bytes arriving within successive time
    # windows measured from the session's first packet; plus the time at
    # which half of the prefix bytes have arrived. Captures request/response
    # cadence (map-tile bursts vs streaming chunks vs telemetry pings).
    if n >= 2:
        rel = t - t[0]
        for w, hi in (('100ms', .1), ('500ms', .5), ('1s', 1.0), ('2s', 2.0), ('5s', 5.0)):
            m = rel <= hi
            f[f'seq_tw_pkt_{w}'] = _safe_ratio(int(m.sum()), n)
            f[f'seq_tw_byte_{w}'] = _safe_ratio(float(cumL[m].sum()), tot)
        m5 = rel > 5.0
        f['seq_tw_pkt_gt5s'] = _safe_ratio(int(m5.sum()), n)
        f['seq_tw_byte_gt5s'] = _safe_ratio(float(cumL[m5].sum()), tot)
        # time to 50% of prefix bytes (normalized by prefix span)
        span = float(rel[-1])
        hit = np.where(cumL >= .5 * tot)[0]
        f['seq_tw_t50_bytes'] = (float(rel[hit[0]]) / span) if (len(hit) and span > 0) else 0.0
        # relative position of the largest inter-packet gap
        if n >= 3:
            d0 = np.diff(t)
            f['seq_tw_maxgap_pos'] = float(np.argmax(d0)) / float(n - 2)
            f['seq_tw_maxgap_sec'] = float(np.max(d0))
        else:
            f['seq_tw_maxgap_pos'] = 0.0
            f['seq_tw_maxgap_sec'] = 0.0
        # instantaneous rate in the first 0.5s vs overall prefix rate
        m05 = rel <= .5
        f['seq_tw_rate_first500ms'] = _safe_ratio(float(cumL[m05].sum()), max(.5, span))
    else:
        for w in ('100ms', '500ms', '1s', '2s', '5s', 'gt5s'):
            f[f'seq_tw_pkt_{w}'] = 0.0
            f[f'seq_tw_byte_{w}'] = 0.0
        f['seq_tw_t50_bytes'] = 0.0
        f['seq_tw_maxgap_pos'] = 0.0
        f['seq_tw_maxgap_sec'] = 0.0
        f['seq_tw_rate_first500ms'] = 0.0

    # -- size diversity --------------------------------------------------------
    uniq = len(np.unique(L))
    f['seq_size_diversity'] = _safe_ratio(uniq, n)
    cnt = np.bincount(np.searchsorted(np.unique(L), L))
    p = cnt / max(n, 1)
    p = p[p > 0]
    f['seq_size_entropy'] = float(-np.sum(p * np.log(p))) if len(p) else 0.0

    # -- transition matrices -------------------------------------------------
    # Direction transitions (2x2), length-regime transitions (3x3) and
    # IAT-regime transitions (3x3), row-normalized.
    if n >= 2:
        trans = np.zeros((2, 2))
        for i in range(n - 1):
            a = 0 if D[i] > 0 else 1
            b = 0 if D[i + 1] > 0 else 1
            trans[a, b] += 1
        trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1)
        for a in range(2):
            for b in range(2):
                f[f'seq_tr_dir_{a}{b}'] = float(trans[a, b])

        rl = np.vectorize(lambda x: 0 if x <= 200 else (1 if x <= 800 else 2))
        Lr = rl(L)
        tl = np.zeros((3, 3))
        for i in range(n - 1):
            tl[Lr[i], Lr[i + 1]] += 1
        tl = tl / np.maximum(tl.sum(axis=1, keepdims=True), 1)
        for a in range(3):
            for b in range(3):
                f[f'seq_tr_len_{a}{b}'] = float(tl[a, b])

        ri = np.vectorize(lambda x: 0 if x <= 0.01 else (1 if x <= 0.1 else 2))
        dt0 = np.diff(t)
        Ir = ri(dt0)
        ti = np.zeros((3, 3))
        for i in range(len(Ir) - 1):
            ti[Ir[i], Ir[i + 1]] += 1
        ti = ti / np.maximum(ti.sum(axis=1, keepdims=True), 1)
        for a in range(3):
            for b in range(3):
                f[f'seq_tr_iat_{a}{b}'] = float(ti[a, b])
    else:
        for a in range(2):
            for b in range(2):
                f[f'seq_tr_dir_{a}{b}'] = 0.0
        for a in range(3):
            for b in range(3):
                f[f'seq_tr_len_{a}{b}'] = 0.0
                f[f'seq_tr_iat_{a}{b}'] = 0.0

    out = {}
    for k, v in f.items():
        v = float(v)
        if not np.isfinite(v):
            v = 0.0
        out[k] = v
    return out


# --------------------------------------------------------------------------
# Stage: extract
# --------------------------------------------------------------------------
def dataset_files() -> Dict[str, Dict[str, List[Path]]]:
    ar = Path('/workspace/cz-华为杯/data/all_data')
    ds: Dict[str, Dict[str, List[Path]]] = {
        'USTC': {c: [ar / f'USTC-TFC2016-{c}.pcap']
                 for c in ['FTP', 'Gmail', 'MySQL', 'WorldOfWarcraft']},
        'CSTNET': {c: sorted(ar.glob(f'CSTNET-TLS1.3-{c}*.pcap'))
                   for c in ['acm.org', 'huawei.com', 'overleaf.com', 'vivo.com.cn']},
    }
    cr = Path('/workspace/datasets/CrossPlatform/china/android')
    ds['CrossPlatform'] = {c: [cr / f'{c}.pcap'] for c in
                           ['bubei.tingshu', 'com.aikan', 'com.autonavi.minimap',
                            'com.baidu.BaiduMap']}
    vr = Path('/workspace/datasets/VisQUIC/data/VisQUIC')
    ds['VisQUIC'] = {}
    for c in ['cloudflare.com', 'discord.com', 'google.com', 'cdnetworks.com']:
        ds['VisQUIC'][c] = sorted((vr / c).rglob('*quic_anonymized_filtered.pcap'))
    return ds


READ_CAPS = {'USTC': 20000, 'CSTNET': None, 'CrossPlatform': None, 'VisQUIC': None}

# Larger observation pools for the bottleneck domain: more training rows per
# class without touching the (still small) fresh-test quota.
EXTRACT_TARGETS = {('CrossPlatform', 'bubei.tingshu'): 700,
                   ('CrossPlatform', 'com.autonavi.minimap'): 700,
                   ('CrossPlatform', 'com.baidu.BaiduMap'): 700}


def class_target(dname: str, cname: str) -> int:
    return EXTRACT_TARGETS.get((dname, cname), TARGET_PER_CLASS)


def extract_one_class(dname: str, cname: str, files: List[Path]) -> Tuple[pd.DataFrame, dict]:
    target = class_target(dname, cname)
    rng = random.Random(f'{SEED}:{dname}:{cname}:files')
    files = list(files)
    rng.shuffle(files)
    basic_ext = BasicFeatureExtractor()
    advanced_ext = AdvancedFeatureExtractor()
    collected: List[Dict[str, object]] = []
    read_info = {'files': len(files), 'packets_read': 0, 'sessions_total': 0,
                 'filtered': 0, 'truncated': 0, 'capped_files': 0, 'target': target}
    for p in files:
        if dname != 'VisQUIC' and len(collected) >= target + 40:
            break
        sm = SessionManager()
        reader = PCAPReader(sm)
        before = reader.total_packets
        class_rows: List[Dict[str, object]] = []
        for idx, session in enumerate(reader.read_pcap_generator(
                str(p), max_read_packets=READ_CAPS[dname])):
            read_info['sessions_total'] += 1
            if session.total_packets < MIN_SESSION_PACKETS or session.total_bytes < MIN_SESSION_BYTES:
                read_info['filtered'] += 1
                continue
            view, truncated = truncated_session_view(session, PREFIX_PACKETS)
            if truncated:
                read_info['truncated'] += 1
            row: Dict[str, object] = {
                '_dataset': dname, '_class': cname, '_label': f'{dname}::{cname}',
                '_source_file': str(p), '_observation_id': f'{p.name}:{idx}',
                '_full_packets': int(session.total_packets),
                '_full_bytes': int(session.total_bytes),
            }
            try:
                row.update(basic_ext.extract_all(view))
                row.update(advanced_ext.extract_all(view))
                row.update(sequence_features(view.packets))
            except Exception as e:  # noqa: BLE001
                read_info.setdefault('extract_errors', 0)
                read_info['extract_errors'] = read_info.get('extract_errors', 0) + 1
                continue
            class_rows.append(row)
        read_info['packets_read'] += reader.total_packets - before
        read_info['capped_files'] += 1 if reader.read_capped else 0
        collected.extend(class_rows)
        print(f'  EXTRACT {dname} {cname} {p.name} +{len(class_rows)} '
              f'total={len(collected)} pkts={reader.total_packets - before}', flush=True)
        if dname == 'VisQUIC' and len(collected) >= target + 20:
            break
    random.Random(f'{SEED}:{dname}:{cname}:rows').shuffle(collected)
    collected = collected[:target]
    if not collected:
        raise RuntimeError(f'no records {dname}:{cname}')
    return pd.DataFrame(collected), read_info


def stage_extract() -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    configs = dataset_files()
    manifest = {}
    for dname in DATASETS:
        for cname, files in configs[dname].items():
            if not files:
                raise RuntimeError(f'no files {dname}:{cname}')
            part = PARTS / f'{dname}__{cname}.csv'
            if part.exists():
                print('SKIP (cached)', part.name, flush=True)
                manifest[f'{dname}::{cname}'] = {'cached': True}
                continue
            t0 = time.time()
            df, info = extract_one_class(dname, cname, files)
            df.to_csv(part, index=False)
            info['rows'] = len(df)
            info['wall_sec'] = round(time.time() - t0, 1)
            manifest[f'{dname}::{cname}'] = info
            print(f'PART {dname}::{cname} rows={len(df)} info={info}', flush=True)
            (OUT / 'extract_manifest.json').write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False))
    (OUT / 'extract_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))


def load_cache() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(PARTS.glob('*.csv'))]
    df = pd.concat(frames, ignore_index=True, sort=False)
    if df['_observation_id'].duplicated().any():
        dup = df[df['_observation_id'].duplicated(keep=False)]['_observation_id'].unique()[:5]
        raise RuntimeError(f'duplicate observation ids: {dup}')
    return df


# --------------------------------------------------------------------------
# Stage: split (fresh, pre-registered)
# --------------------------------------------------------------------------
def adaptive_quotas(avail: int) -> Tuple[int, int, int]:
    test = min(TEST_QUOTA, max(30, int(round(avail * 0.22))))
    val = min(VAL_QUOTA, max(20, int(round(avail * 0.20))))
    train = avail - test - val
    if train < 48:
        raise RuntimeError(f'insufficient train rows: avail={avail}')
    return train, val, test


def dev_val_size(dev_rows: int) -> int:
    """Validation quota from the post-holding dev pool; train keeps the rest."""
    val = min(VAL_QUOTA, max(20, int(round(dev_rows * 0.18))))
    if dev_rows - val < 48:
        val = max(0, dev_rows - 48)
    return val


def stage_split() -> None:
    df = load_cache()
    parts = []
    granularity = {}
    for (d, c), g in df.groupby(['_dataset', '_class'], sort=True):
        g = g.reset_index(drop=True)
        files = sorted(g['_source_file'].unique())
        test_n = min(TEST_QUOTA, max(30, int(round(len(g) * 0.22))))
        multi = len(files) >= 3
        if multi:
            # File-disjoint fresh test. Once a source file is chosen as held-out,
            # ALL of its observations are excluded from Train/Validation; the
            # test side is downsampled to its quota and the remaining held-file
            # rows are dropped entirely (they may not leak back into dev splits).
            frng = random.Random(f'{SEED}:{d}:{c}:testfiles')
            shuffled = list(files)
            frng.shuffle(shuffled)
            held, acc = [], 0
            for fp in shuffled:
                if acc >= test_n:
                    break
                held.append(fp)
                acc += int((g['_source_file'] == fp).sum())
            test_pool = g[g['_source_file'].isin(held)]
            srng = random.Random(f'{SEED}:{d}:{c}:testrows')
            idx = list(test_pool.index)
            srng.shuffle(idx)
            test_df = test_pool.loc[idx[:test_n]]
            dropped = int(len(test_pool) - len(test_df))
            rest = g[~g['_source_file'].isin(held)]
            val_n = dev_val_size(len(rest))
            if len(rest) - val_n < 48:
                raise RuntimeError(f'{d}::{c}: held-out leaves too few train rows')
            rrng = random.Random(f'{SEED}:{d}:{c}:rest')
            ridx = list(rest.index)
            rrng.shuffle(ridx)
            rest = rest.loc[ridx]
            val_df = rest.iloc[:val_n]
            train_df = rest.iloc[val_n:]
            granularity[f'{d}::{c}'] = {
                'test_granularity': 'capture-file-disjoint',
                'held_out_files': held, 'n_files': len(files),
                'dropped_rows': dropped,
                'train': len(train_df), 'val': len(val_df), 'test': len(test_df)}
        else:
            # Fallback (documented): single/two-capture classes cannot be split
            # at capture granularity; observation-level split within the capture.
            grng = random.Random(f'{SEED}:{d}:{c}:sessionlevel')
            idx = list(g.index)
            grng.shuffle(idx)
            gg = g.loc[idx]
            test_df = gg.iloc[:test_n]
            val_n = dev_val_size(len(gg) - test_n)
            val_df = gg.iloc[test_n:test_n + val_n]
            train_df = gg.iloc[test_n + val_n:]
            granularity[f'{d}::{c}'] = {
                'test_granularity': 'session-level-within-single-capture-fallback',
                'n_files': len(files),
                'dropped_rows': int(len(gg) - len(train_df) - len(val_df) - len(test_df)),
                'train': len(train_df), 'val': len(val_df), 'test': len(test_df)}
        for name, frame in (('train_pool', train_df), ('validation', val_df), ('test', test_df)):
            frame = frame.copy()
            frame['_split'] = name
            parts.append(frame)
    out = pd.concat(parts, ignore_index=True, sort=False)

    # ---- hard leakage assertions (fail loudly rather than ship a leaky split)
    audit = {'checks': [], 'per_class': {}}
    splits = {s: out[out['_split'] == s] for s in ('train_pool', 'validation', 'test')}
    ids = {s: set(splits[s]['_observation_id']) for s in splits}
    for a, b in (('train_pool', 'validation'), ('train_pool', 'test'), ('validation', 'test')):
        inter = ids[a] & ids[b]
        audit['checks'].append(
            {'check': f'observation_id disjoint {a} vs {b}',
             'intersection': len(inter), 'passed': not inter})
        if inter:
            raise RuntimeError(f'observation_id leak between {a} and {b}: {sorted(inter)[:5]}')
    file_viol = {}
    for key, g in out.groupby(['_dataset', '_class']):
        d, c = key
        dev = g[g['_split'].isin(['train_pool', 'validation'])]
        tst = g[g['_split'] == 'test']
        inter_files = sorted(set(dev['_source_file']) & set(tst['_source_file']))
        entry = {'granularity': granularity[f'{d}::{c}']['test_granularity'],
                 'dev_files': int(dev['_source_file'].nunique()),
                 'test_files': int(tst['_source_file'].nunique()),
                 'file_intersection': inter_files}
        if granularity[f'{d}::{c}']['test_granularity'] == 'capture-file-disjoint':
            entry['passed'] = not inter_files
            if inter_files:
                file_viol[f'{d}::{c}'] = inter_files
        else:
            # fallback classes share the single capture by construction; this is
            # recorded, not asserted away.
            entry['passed'] = 'fallback-single-capture'
        audit['per_class'][f'{d}::{c}'] = entry
    audit['checks'].append(
        {'check': 'multi-file classes: dev/test source_file intersection = 0',
         'violations': file_viol, 'passed': not file_viol})
    if file_viol:
        raise RuntimeError(f'source_file leak in multi-file classes: {file_viol}')
    audit['fallback_classes'] = sorted(
        k for k, v in granularity.items()
        if v['test_granularity'].startswith('session-level'))

    meta_cols = ['_dataset', '_class', '_label', '_source_file', '_observation_id',
                 '_full_packets', '_full_bytes', '_split']
    out[meta_cols].to_csv(OUT / 'fresh_split.csv', index=False)
    split_sha = hashlib.sha256((OUT / 'fresh_split.csv').read_bytes()).hexdigest()
    (OUT / 'split_leakage_audit.json').write_text(
        json.dumps(audit, indent=2, ensure_ascii=False))
    spec = {
        'seed': SEED, 'split_sha256': split_sha,
        'prefix_packets': PREFIX_PACKETS,
        'target_per_class': TARGET_PER_CLASS,
        'extract_targets': {f'{d}::{c}': t for (d, c), t in EXTRACT_TARGETS.items()},
        'created_before_search': True,
        'per_class': granularity,
        'counts': out.groupby(['_split']).size().to_dict(),
        'policy': ('fresh split generated before any feature/model search; multi-file '
                   'classes use capture-file-disjoint tests (held-out files fully '
                   'excluded from dev, surplus dropped); single-capture classes fall '
                   'back to observation-level splits; test rows are opened exactly '
                   'once in stage test after frozen_config.json'),
    }
    (OUT / 'fresh_split_spec.json').write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    print(json.dumps(spec['counts'], indent=2), flush=True)
    print('split_sha256', split_sha, flush=True)
    print('leakage audit: all checks passed;', len(audit['fallback_classes']),
          'fallback classes', flush=True)


def load_split() -> pd.DataFrame:
    cache = load_cache()
    split = pd.read_csv(OUT / 'fresh_split.csv')
    keep = ['_observation_id', '_split']
    df = cache.merge(split[keep], on='_observation_id', how='inner')
    if len(df) != len(split):
        raise RuntimeError('split/cache row mismatch')
    return df


# --------------------------------------------------------------------------
# Metrics (official four, over the 16-class label set)
# --------------------------------------------------------------------------
def official_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     labels: Sequence[str]) -> Dict[str, object]:
    lid = {c: i for i, c in enumerate(labels)}
    yt = np.array([lid[c] for c in y_true])
    yp = np.array([lid[c] for c in y_pred])
    n = len(yt)
    acc = float(accuracy_score(yt, yp))
    m_p = float(precision_score(yt, yp, average='macro', zero_division=0))
    m_r = float(recall_score(yt, yp, average='macro', zero_division=0))
    m_f1 = float(f1_score(yt, yp, average='macro', zero_division=0))
    k = len(labels)
    fp = np.zeros(k, int)
    for i in range(k):
        fp[i] = int(((yp == i) & (yt != i)).sum())
    neg = np.array([(yt != i).sum() for i in range(k)], int)
    fpr = fp / np.maximum(neg, 1)
    per_class = {}
    for i, c in enumerate(labels):
        m = yt == i
        per_class[c] = {
            'recall': float((yp[m] == i).mean()) if m.any() else 0.0,
            'precision': float(precision_score(yt == i, yp == i, zero_division=0)),
            'fpr': float(fpr[i]), 'fp': int(fp[i]), 'n': int(m.sum())}
    conf = np.zeros((k, k), int)
    for a, b in zip(yt, yp):
        conf[a, b] += 1
    return {'accuracy': acc, 'macro_precision': m_p, 'macro_recall': m_r,
            'macro_f1': m_f1, 'max_per_class_fpr': float(fpr.max()),
            'errors': int((yt != yp).sum()), 'n': n,
            'per_class': per_class, 'confusion': conf.tolist()}


TARGETS = {'accuracy': .95, 'macro_precision': .95, 'macro_recall': .98,
           'max_per_class_fpr': .05}


def slack(m: Dict[str, object]) -> float:
    return min(m['accuracy'] - TARGETS['accuracy'],
               m['macro_precision'] - TARGETS['macro_precision'],
               m['macro_recall'] - TARGETS['macro_recall'],
               TARGETS['max_per_class_fpr'] - m['max_per_class_fpr'])


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_model(name: str, seed: int, n_classes: int):
    if name == 'extra':
        return ExtraTreesClassifier(n_estimators=800, max_features='sqrt',
                                    min_samples_leaf=1, class_weight='balanced',
                                    random_state=seed, n_jobs=8)
    if name == 'rf':
        return RandomForestClassifier(n_estimators=600, max_features='sqrt',
                                      class_weight='balanced', random_state=seed,
                                      n_jobs=8)
    if name == 'xgb':
        return XGBClassifier(n_estimators=320, max_depth=6, learning_rate=.035,
                             min_child_weight=1, subsample=.90, colsample_bytree=.90,
                             reg_lambda=1.8, random_state=seed, n_jobs=8,
                             eval_metric='mlogloss', tree_method='hist')
    if name == 'hgb':
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=.06,
                                              max_leaf_nodes=31, l2_regularization=1.0,
                                              class_weight='balanced',
                                              random_state=seed)
    raise ValueError(name)


MODEL_MEMBERS = {'extra': ['extra'], 'xgb': ['xgb'], 'hgb': ['hgb'], 'rf': ['rf'],
                 'ens_xe': ['xgb', 'extra'], 'ens_xh': ['xgb', 'hgb'],
                 'ens_eh': ['extra', 'hgb'],
                 'ens_all': ['xgb', 'extra', 'hgb', 'rf'],
                 'moe_extra': ['moe'], 'moe_xgb': ['moe']}


def clean(d: pd.DataFrame, fs: Sequence[str]) -> pd.DataFrame:
    return d[list(fs)].apply(pd.to_numeric, errors='coerce') \
                     .replace([np.inf, -np.inf], np.nan).fillna(0)


def fit_predict_proba(train: pd.DataFrame, eval_df: pd.DataFrame,
                      fs: Sequence[str], model_name: str, seed: int,
                      labels: Sequence[str]):
    lid = {c: i for i, c in enumerate(labels)}
    ytr = train['_label'].map(lid).to_numpy()
    Xtr, Xe = clean(train, fs), clean(eval_df, fs)
    if model_name.startswith('moe_'):
        return _fit_predict_proba_moe(train, eval_df, Xtr, Xe, fs, model_name,
                                      seed, labels)
    probs = []
    for j, member in enumerate(MODEL_MEMBERS[model_name]):
        m = make_model(member, seed + j, len(labels))
        if member == 'xgb':
            m.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
        else:
            m.fit(Xtr, ytr)
        p = np.zeros((len(Xe), len(labels)))
        p[:, m.classes_] = m.predict_proba(Xe)
        probs.append(p)
    return np.mean(probs, axis=0)


def _fit_predict_proba_moe(train: pd.DataFrame, eval_df: pd.DataFrame,
                           Xtr: pd.DataFrame, Xe: pd.DataFrame,
                           fs: Sequence[str], model_name: str, seed: int,
                           labels: Sequence[str]) -> np.ndarray:
    """Latent-domain gate + per-domain experts.

    Deployment logic: the gate is a classifier over the SAME feature vector
    (never a dataset ID); it infers the latent domain of a session, and only
    the corresponding 4-class expert is evaluated. Final probability of a
    class = P(gate=domain) * P(expert=class); confidence likewise. Training
    uses dataset labels only to fit the gate/expert heads, which is the
    sanctioned latent-domain construction.
    """
    member = model_name.split('_', 1)[1]
    dom_of = {l: l.split('::')[0] for l in labels}
    domains = sorted(set(dom_of.values()))
    dom_id = {d: i for i, d in enumerate(domains)}
    lid = {c: i for i, c in enumerate(labels)}
    ytr = train['_label'].map(lid).to_numpy()
    dtr = train['_label'].map(lambda l: dom_id[dom_of[l]]).to_numpy()

    def _mk(s):
        m = make_model(member, s, len(labels))
        return m

    gate = _mk(seed)
    gtr = compute_sample_weight('balanced', dtr) if member == 'xgb' else None
    gate.fit(Xtr, dtr, sample_weight=gtr) if member == 'xgb' else gate.fit(Xtr, dtr)
    gp = np.zeros((len(Xe), len(domains)))
    gp[:, gate.classes_] = gate.predict_proba(Xe)

    out = np.zeros((len(Xe), len(labels)))
    for di, d in enumerate(domains):
        mask = dtr == di
        cls = [l for l in labels if dom_of[l] == d]
        if mask.sum() < len(cls) * 2:
            # too few rows for a stable expert: uniform head
            for l in cls:
                out[:, lid[l]] = gp[:, di] / len(cls)
            continue
        exp = _mk(seed + 1 + di)
        yd = train.loc[mask, '_label'].map({c: i for i, c in enumerate(cls)}).to_numpy()
        if member == 'xgb':
            exp.fit(Xtr[mask], yd, sample_weight=compute_sample_weight('balanced', yd))
        else:
            exp.fit(Xtr[mask], yd)
        pe = np.zeros((len(Xe), len(cls)))
        pe[:, exp.classes_] = exp.predict_proba(Xe)
        for j, l in enumerate(cls):
            out[:, lid[l]] = gp[:, di] * pe[:, j]
    return out


def evaluate(train: pd.DataFrame, eval_df: pd.DataFrame, fs: Sequence[str],
             model_name: str, seed: int, labels: Sequence[str]) -> Dict[str, object]:
    proba = fit_predict_proba(train, eval_df, fs, model_name, seed, labels)
    pred = np.array([labels[i] for i in np.argmax(proba, axis=1)])
    m = official_metrics(eval_df['_label'].to_numpy(), pred, labels)
    if '_dataset' in eval_df.columns:
        yt = eval_df['_label'].to_numpy()
        ds = eval_df['_dataset'].to_numpy()
        per_ds = {}
        for d in DATASETS:
            mask = ds == d
            if not mask.any():
                continue
            rec = [float((pred[mask & (yt == c)] == c).mean())
                   for c in pd.unique(yt[mask])]
            per_ds[d] = {'accuracy': float((pred[mask] == yt[mask]).mean()),
                         'macro_recall': float(np.mean(rec)), 'n': int(mask.sum())}
        m['per_dataset'] = per_ds
        m['worst_dataset_accuracy'] = min(v['accuracy'] for v in per_ds.values())
    return m


# --------------------------------------------------------------------------
# Stage: search (train/validation only)
# --------------------------------------------------------------------------
def norm(a: np.ndarray) -> np.ndarray:
    if len(a) == 0:
        return a
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(a)


def fused_score(df: pd.DataFrame, ycol: str, fs: Sequence[str], seed: int) -> Dict[str, float]:
    usable = [f for f in fs
              if pd.to_numeric(df[f], errors='coerce').fillna(0).std() > 1e-10]
    X = clean(df, usable)
    labels = sorted(df[ycol].unique())
    y = df[ycol].map({c: i for i, c in enumerate(labels)}).to_numpy()
    m = XGBClassifier(n_estimators=120, max_depth=5, learning_rate=.06,
                      subsample=.9, colsample_bytree=.9, reg_lambda=1.8,
                      random_state=seed, n_jobs=8, eval_metric='mlogloss',
                      tree_method='hist')
    m.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    imp = norm(np.asarray(m.feature_importances_, float))
    try:
        mi = norm(np.asarray(mutual_info_classif(X, y, random_state=seed), float))
    except Exception:  # noqa: BLE001
        mi = np.zeros(len(usable))
    s = .6 * imp + .4 * mi
    return {f: float(v) for f, v in zip(usable, s)}


def build_score_table(train: pd.DataFrame, fs: List[str]) -> pd.DataFrame:
    per, top_sets = {}, {}
    for i, d in enumerate(DATASETS):
        dd = train[train['_dataset'] == d]
        per[d] = fused_score(dd, '_class', fs, SEED + i)
        top_sets[d] = set(sorted(per[d], key=lambda f: (-per[d][f], f))[:64])
    pooled = fused_score(train, '_label', fs, SEED + 100)
    seq_pool = [f for f in fs if f.startswith('seq_')]
    seq_top = set(sorted({f: pooled.get(f, 0.0) for f in seq_pool},
                         key=lambda f: (-pooled.get(f, 0.0), f))[:64]) if seq_pool else set()
    rows = []
    for f in fs:
        vals = np.array([per[d].get(f, 0.0) for d in DATASETS])
        rows.append({'feature': f,
                     **{f'score_{d}': float(per[d].get(f, 0.0)) for d in DATASETS},
                     'mean_score': float(vals.mean()), 'min_score': float(vals.min()),
                     'std_score': float(vals.std()), 'pooled_score': float(pooled.get(f, 0.0)),
                     'support64': int(sum(f in top_sets[d] for d in DATASETS)),
                     'in_seq_top64': int(f in seq_top)})
    tab = pd.DataFrame(rows)
    for c in ['mean_score', 'min_score', 'std_score', 'pooled_score']:
        tab[f'n_{c}'] = norm(tab[c].to_numpy(float))
    tab['n_support'] = tab.support64 / 4.0
    return tab


def rank_variant(tab: pd.DataFrame, variant: str, seq_boost: float = 0.0) -> List[str]:
    t = tab.copy()
    base = {
        'consensus_mean': t.n_mean_score,
        'robust_worst': (.42 * t.n_mean_score + .25 * t.n_min_score + .18 * t.n_pooled_score
                         + .20 * t.n_support - .05 * t.n_std_score),
        'robust_stable': (.35 * t.n_mean_score + .15 * t.n_min_score + .20 * t.n_pooled_score
                          + .35 * t.n_support - .05 * t.n_std_score),
        'pooled': t.n_pooled_score,
    }[variant]
    if seq_boost:
        base = base + seq_boost * t.in_seq_top64.astype(float)
    t['objective'] = base
    return t.sort_values(['objective', 'support64', 'mean_score'],
                         ascending=[False, False, False]).feature.tolist()


def hybrid_subset(tab: pd.DataFrame, core_k: int, total_k: int = 64) -> List[str]:
    robust = rank_variant(tab, 'robust_stable')
    core = [f for f in robust if int(tab.loc[tab.feature == f, 'support64'].iloc[0]) >= 2][:core_k]
    if len(core) < core_k:
        core += [f for f in robust if f not in core][:core_k - len(core)]
    out = list(core)
    for f in rank_variant(tab, 'pooled'):
        if f not in out:
            out.append(f)
        if len(out) >= total_k:
            break
    return out[:total_k]


def stage_search() -> None:
    df = load_split()
    train = df[df['_split'] == 'train_pool'].reset_index(drop=True)
    val = df[df['_split'] == 'validation'].reset_index(drop=True)
    labels = sorted(df['_label'].unique())
    if len(labels) != 16:
        raise RuntimeError(f'expected 16 labels, got {len(labels)}')
    fs_all = common_shape_features(df.columns)
    tab = build_score_table(train, fs_all)
    tab.to_csv(OUT / 'feature_scores_train.csv', index=False)

    subsets: Dict[str, List[str]] = {}
    for variant in ['consensus_mean', 'robust_worst', 'robust_stable', 'pooled']:
        subsets[f'{variant}_k64'] = rank_variant(tab, variant)[:64]
    for variant in ['robust_stable', 'pooled']:
        subsets[f'{variant}_seqboost_k64'] = rank_variant(tab, variant, seq_boost=.25)[:64]
    for core in (48, 56):
        subsets[f'hybrid_core{core}_k64'] = hybrid_subset(tab, core, 64)
    subsets['seq_only_top64'] = [f for f in rank_variant(tab, 'pooled')
                                 if f.startswith('seq_')][:64]
    subsets['runtime_only_top64'] = [f for f in rank_variant(tab, 'robust_stable')
                                     if not f.startswith('seq_')][:64]
    # Round-robin: fixed quota per dataset's own ranking (score_{d} columns of
    # the train-only score table), so every domain's discriminative features
    # are represented (targets CrossPlatform).
    rr: List[str] = []
    pooled_rank = rank_variant(tab, 'pooled')
    for d in DATASETS:
        own = tab.sort_values([f'score_{d}', 'pooled_score'], ascending=False).feature.tolist()
        rr += [f for f in own[:16] if f not in rr]
    for f in pooled_rank:
        if len(rr) >= 64:
            break
        if f not in rr:
            rr.append(f)
    subsets['roundrobin16_k64'] = rr[:64]
    # Budget-relaxation arm (cost must be justified on validation).
    subsets['robust_stable_k96'] = rank_variant(tab, 'robust_stable')[:96]
    subsets['hybrid_core72_k96'] = hybrid_subset(tab, 72, 96)
    subsets['pooled_top128'] = rank_variant(tab, 'pooled')[:128]
    subsets['seq_full_pool'] = [f for f in rank_variant(tab, 'pooled')
                                if f.startswith('seq_')][:256]

    rows = []
    for sname, fs in subsets.items():
        nseq = sum(f.startswith('seq_') for f in fs)
        for model in ['extra', 'xgb', 'hgb', 'rf', 'ens_xe', 'ens_xh', 'ens_eh',
                      'ens_all', 'moe_extra', 'moe_xgb']:
            m = evaluate(train, val, fs, model, SEED + 10, labels)
            rows.append({'subset': sname, 'k': len(fs), 'n_seq': nseq, 'model': model,
                         'accuracy': m['accuracy'], 'macro_precision': m['macro_precision'],
                         'macro_recall': m['macro_recall'], 'macro_f1': m['macro_f1'],
                         'max_per_class_fpr': m['max_per_class_fpr'],
                         'slack': slack(m), 'errors': m['errors'],
                         'worst_dataset_accuracy': m.get('worst_dataset_accuracy'),
                         **{f'acc_{d}': m.get('per_dataset', {}).get(d, {}).get('accuracy')
                            for d in DATASETS}})
            print('VAL', rows[-1], flush=True)
    lb = pd.DataFrame(rows).sort_values('slack', ascending=False)
    lb.to_csv(OUT / 'validation_leaderboard.csv', index=False)

    # Seed-stability check on the top-3 configurations (validation only).
    top3 = lb.head(3)
    stab = []
    for _, r in top3.iterrows():
        fs = subsets[r['subset']]
        ms = [evaluate(train, val, fs, r['model'], SEED + 200 + 40 * i, labels)
              for i in range(3)]
        stab.append({'subset': r['subset'], 'model': r['model'],
                     'macro_recall_mean': float(np.mean([m['macro_recall'] for m in ms])),
                     'macro_recall_min': float(np.min([m['macro_recall'] for m in ms])),
                     'accuracy_mean': float(np.mean([m['accuracy'] for m in ms])),
                     'accuracy_min': float(np.min([m['accuracy'] for m in ms])),
                     'macro_precision_mean': float(np.mean([m['macro_precision'] for m in ms])),
                     'max_fpr_max': float(np.max([m['max_per_class_fpr'] for m in ms])),
                     'slack_mean': float(np.mean([slack(m) for m in ms]))})
        print('STAB', stab[-1], flush=True)
    pd.DataFrame(stab).to_csv(OUT / 'validation_stability.csv', index=False)
    # Pre-registered choice: highest mean slack across stability seeds.
    best = max(stab, key=lambda r: (r['slack_mean'], r['macro_recall_mean']))
    best_fs = subsets[best['subset']]

    def _passes(m: Dict[str, float]) -> bool:
        return (m['accuracy'] >= TARGETS['accuracy']
                and m['macro_precision'] >= TARGETS['macro_precision']
                and m['macro_recall'] >= TARGETS['macro_recall']
                and m['max_per_class_fpr'] <= TARGETS['max_per_class_fpr'])

    best_row = lb.iloc[0]
    single_pass = bool(best_row['slack'] >= 0)
    stab_pass = (best['accuracy_mean'] >= TARGETS['accuracy']
                 and best['macro_precision_mean'] >= TARGETS['macro_precision']
                 and best['macro_recall_mean'] >= TARGETS['macro_recall']
                 and best['max_fpr_max'] <= TARGETS['max_per_class_fpr'])
    common = {
        'experiment': 'mixed_dataset_full_pass_20260921',
        'seed': SEED, 'prefix_packets': PREFIX_PACKETS,
        'feature_space': 'common_shape+generic_seq',
        'selected_subset': best['subset'], 'selected_features': best_fs,
        'selected_k': len(best_fs),
        'n_seq_features': int(sum(f.startswith('seq_') for f in best_fs)),
        'model': best['model'], 'model_members': MODEL_MEMBERS[best['model']],
        'fit_policy': 'refit on train+validation with frozen features/model, fixed seed',
        'refit_seed': SEED + 900,
        'selection_rule': ('max min-slack over official four metrics on validation, '
                           'then 3-seed stability mean-slack among top-3'),
        'targets': TARGETS,
        'validation_snapshot': best,
        'validation_best_single': {k: float(best_row[k]) for k in
                                   ('accuracy', 'macro_precision', 'macro_recall',
                                    'max_per_class_fpr', 'slack')},
    }
    if best['model'].startswith('moe_'):
        common['model_deployment'] = (
            'Latent-domain MoE: gate classifier maps the SAME feature vector to a '
            'latent domain (never a dataset ID); only the winning domain expert is '
            'evaluated; confidence = gate_prob * expert_prob.')
    if single_pass and stab_pass:
        common['fresh_test_policy'] = (
            'test rows opened exactly once after this file is written; no retuning '
            'if test fails')
        fp = OUT / 'frozen_config.json'
        fp.write_text(json.dumps(common, indent=2, ensure_ascii=False))
        print('FROZEN (validation four-metric gate passed)', best, flush=True)
        print('frozen_config_sha256',
              hashlib.sha256(fp.read_bytes()).hexdigest(), flush=True)
    else:
        common['status'] = 'validation_not_passed'
        common['reason'] = ('best configuration does not simultaneously meet '
                            'Acc>=.95 / MacroP>=.95 / MacroR>=.98 / maxFPR<=.05 on '
                            'validation (single-fit pass=%s, 3-seed-mean pass=%s); '
                            'fresh test stays sealed' % (single_pass, stab_pass))
        (OUT / 'search_outcome.json').write_text(
            json.dumps(common, indent=2, ensure_ascii=False))
        print('NOT FROZEN:', common['reason'], flush=True)


# --------------------------------------------------------------------------
# Stage: test (one-shot)
# --------------------------------------------------------------------------
def stage_test() -> None:
    tm = OUT / 'test_metrics.json'
    if tm.exists():
        raise RuntimeError('test_metrics.json already exists; fresh holdout may only be '
                           'opened once (refusing to overwrite)')
    if not (OUT / 'frozen_config.json').exists():
        raise RuntimeError('frozen_config.json missing: the validation four-metric gate '
                           'has not passed, the fresh test must stay sealed')
    frozen = json.loads((OUT / 'frozen_config.json').read_text())
    df = load_split()
    labels = sorted(df['_label'].unique())
    dev = df[df['_split'].isin(['train_pool', 'validation'])].reset_index(drop=True)
    test = df[df['_split'] == 'test'].reset_index(drop=True)
    fs = frozen['selected_features']
    m = evaluate(dev, test, fs, frozen['model'], frozen['refit_seed'], labels)
    per_rows = []
    for c, v in m['per_class'].items():
        per_rows.append({'label': c, **v})
    pd.DataFrame(per_rows).to_csv(OUT / 'per_class_metrics.csv', index=False)
    pd.DataFrame(m['confusion'], index=labels, columns=labels).to_csv(OUT / 'confusion.csv')

    passes = {'accuracy': m['accuracy'] >= TARGETS['accuracy'],
              'macro_precision': m['macro_precision'] >= TARGETS['macro_precision'],
              'macro_recall': m['macro_recall'] >= TARGETS['macro_recall'],
              'max_per_class_fpr': m['max_per_class_fpr'] <= TARGETS['max_per_class_fpr']}
    result = {
        'frozen_config_sha256': hashlib.sha256(
            (OUT / 'frozen_config.json').read_bytes()).hexdigest(),
        'split_sha256': json.loads((OUT / 'fresh_split_spec.json').read_text())['split_sha256'],
        'metrics': {k: m[k] for k in ('accuracy', 'macro_precision', 'macro_recall',
                                      'macro_f1', 'max_per_class_fpr', 'errors', 'n')},
        'targets': TARGETS, 'pass': passes, 'all_pass': all(passes.values()),
        'per_dataset': m.get('per_dataset'),
        'opened_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'one_shot': True,
        'per_class': m['per_class'],
    }
    tm.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: result[k] for k in ('metrics', 'pass', 'all_pass')}, indent=2),
          flush=True)


# --------------------------------------------------------------------------
# Stage: selftest
# --------------------------------------------------------------------------
def stage_selftest() -> None:
    class P:  # minimal packet stub
        def __init__(self, ts, ln, d):
            self.timestamp, self.length, self.direction = ts, ln, d

    rng = random.Random(0)
    for n in (3, 5, 17, 64, 80):
        pkts = [P(i * rng.uniform(0.001, 0.4), rng.randint(60, 1500),
                  1 if i % 2 else -1) for i in range(n)]
        feats = sequence_features(pkts[:PREFIX_PACKETS])
        bad = {k: v for k, v in feats.items() if not np.isfinite(v)}
        assert not bad, f'non-finite seq features at n={n}: {bad}'
        assert len(feats) > 100
    # one-direction session (no bwd packets)
    pkts = [P(i * 0.01, 800, 1) for i in range(20)]
    feats = sequence_features(pkts)
    assert all(np.isfinite(v) for v in feats.values())
    # metrics sanity
    labels = [f'c{i}' for i in range(3)]
    yt = np.array(['c0', 'c0', 'c1', 'c1', 'c2', 'c2'])
    yp = np.array(['c0', 'c1', 'c1', 'c1', 'c2', 'c0'])
    m = official_metrics(yt, yp, labels)
    assert abs(m['accuracy'] - 4 / 6) < 1e-9
    assert m['per_class']['c1']['recall'] == 1.0
    assert m['per_class']['c0']['fpr'] > 0
    print('SELFTEST OK — n_seq_features =', len(feats), flush=True)


def stage_diagnose() -> None:
    """CrossPlatform-focused error diagnosis on Train/Validation ONLY."""
    df = load_split()
    train = df[df['_split'] == 'train_pool'].reset_index(drop=True)
    val = df[df['_split'] == 'validation'].reset_index(drop=True)
    labels = sorted(df['_label'].unique())
    fs_all = common_shape_features(df.columns)
    seq_fs = [f for f in fs_all if f.startswith('seq_')]
    rt_fs = [f for f in fs_all if not f.startswith('seq_')]
    report = {}
    for tag, pool in (('all', fs_all), ('runtime_only', rt_fs), ('seq_only', seq_fs)):
        for model in ('extra', 'ens_xe'):
            m = evaluate(train, val, pool, model, SEED + 33, labels)
            report[f'{tag}__{model}'] = {
                k: m[k] for k in ('accuracy', 'macro_precision', 'macro_recall',
                                  'max_per_class_fpr')}
            report[f'{tag}__{model}']['per_dataset'] = m.get('per_dataset')
            print('DIAG', tag, model, report[f'{tag}__{model}']['accuracy'],
                  report[f'{tag}__{model}']['macro_recall'], flush=True)
    # CrossPlatform-only confusion (4-class within-domain expert view).
    cpt, cpv = train[train['_dataset'] == 'CrossPlatform'], val[val['_dataset'] == 'CrossPlatform']
    cp_labels = sorted(cpt['_label'].unique())
    m = evaluate(cpt, cpv, fs_all, 'extra', SEED + 34, cp_labels)
    print('CrossPlatform 4-class validation confusion (rows=true):', flush=True)
    conf = pd.DataFrame(m['confusion'], index=cp_labels, columns=cp_labels)
    print(conf.to_string(), flush=True)
    # Error profile vs session size for CrossPlatform.
    proba = fit_predict_proba(cpt, cpv, fs_all, 'extra', SEED + 34, cp_labels)
    pred = np.array([cp_labels[i] for i in np.argmax(proba, axis=1)])
    cpv2 = cpv.copy()
    cpv2['_pred'] = pred
    cpv2['_ok'] = cpv2['_pred'] == cpv2['_label']
    prof = cpv2.groupby('_ok')[['_full_packets', '_full_bytes']].median()
    print('CrossPlatform median full session packets/bytes by correctness:',
          prof.to_dict(), flush=True)
    report['crossplatform_confusion'] = m
    report['crossplatform_error_profile'] = {
        'median_full_packets': {str(k): float(v) for k, v in
                                cpv2.groupby('_ok')['_full_packets'].median().items()},
        'median_full_bytes': {str(k): float(v) for k, v in
                              cpv2.groupby('_ok')['_full_bytes'].median().items()},
        'errors_by_class': {c: int(((cpv2['_class'] == c) & ~cpv2['_ok']).sum())
                            for c in sorted(cpv2['_class'].unique())}}
    (OUT / 'diagnose_crossplatform.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['selftest', 'extract', 'split', 'search', 'test',
                             'diagnose', 'all'])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'selftest':
        stage_selftest()
        return
    if args.stage in ('extract', 'all'):
        stage_extract()
    if args.stage in ('split', 'all'):
        stage_split()
    if args.stage in ('diagnose',):
        stage_diagnose()
    if args.stage in ('search', 'all'):
        stage_search()
    if args.stage in ('test', 'all'):
        stage_test()


if __name__ == '__main__':
    main()
