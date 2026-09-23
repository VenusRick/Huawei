# -*- coding: utf-8 -*-
"""Scenario B (2026-09-22): Behavior15 raw-PCAP four-metric push.

Previous rounds (ledger 26.3/27.3): single-dominant-flow 15s windows failed the
official four metrics under every raw-PCAP split (file-disjoint 84.75/74.82/
87.43/12.10; session-disjoint 84.57/84.33/80.75/15.22; window-stratified
95.31/93.08/95.02/4.37). The official 15s ARFF feature-vector benchmark passed
but is a different, feature-vector input regime.

This round changes the observation unit instead of re-tuning the same one:
- multi-flow aggregate windows: every observation aggregates ALL reconstructed
  sessions of a capture that are active in [t0+i*step, t0+i*step+W), for
  W in {30s, 45s, 60s} with optional 50% overlap (a real DPI engine watching a
  terminal sees exactly this aggregate stream; no inner/label information is
  used);
- feature pool = the generic 197-dim sequence family (direction runs, size/IAT
  quantiles+hists, bursts, autocorrelation, progress curves, cadence) over the
  aggregate packet stream + multi-flow context counts (n sessions, top-flow
  shares, new-flow starts, remote endpoint diversity);
- three split granularities, all PRE-REGISTERED before any model search
  (seed 20260922): window-stratified (sessions may cross splits),
  session-disjoint (whole sessions in one split), capture-disjoint (whole
  files in one split);
- gate identical to the other scenarios: validation four metrics pass
  (single-fit AND 3-seed stability mean) -> frozen_config -> open that
  granularity's test exactly once.

Blindness note: the 21 ISCXVPN2016 captures themselves have been explored in
earlier rounds (their window-level difficulty is known), so any PASS here is
"fresh pre-registered split over previously-explored captures", not a fresh
dataset claim. This is stated in every artifact.

Stages: extract | split | search | test --granularity <name>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path('/workspace/Huawei')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_mixed_dataset_full_pass_20260921 import sequence_features  # noqa: E402
from src.parser.pcap_reader import PCAPReader  # noqa: E402
from src.parser.session.session_manager import SessionManager  # noqa: E402
from src.features.runtime import MIN_SESSION_PACKETS, MIN_SESSION_BYTES  # noqa: E402

OUT = ROOT / 'output/all_scenarios_metric_push_20260922/behavior_rawpcap'
CACHE = OUT / 'cache_sessions.pkl'
SEED = 20260922
LABELS = ['chat', 'audio', 'video']
LID = {c: i for i, c in enumerate(LABELS)}
FILES = [
    ('chat', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPS-01/vpn_aim_chat1a.pcap'),
    ('chat', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPS-01/vpn_aim_chat1b.pcap'),
    ('chat', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPS-01/vpn_facebook_chat1a.pcap'),
    ('chat', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPS-01/vpn_facebook_chat1b.pcap'),
    ('chat', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPS-01/vpn_hangouts_chat1a.pcap'),
    ('chat', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPS-01/vpn_hangouts_chat1b.pcap'),
    ('chat', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPs-02/vpn_icq_chat1a.pcap'),
    ('chat', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPs-02/vpn_icq_chat1b.pcap'),
    ('chat', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPs-02/vpn_skype_chat1a.pcap'),
    ('chat', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPs-02/vpn_skype_chat1b.pcap'),
    ('audio', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPS-01/vpn_facebook_audio2.pcap'),
    ('audio', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPS-01/vpn_hangouts_audio1.pcap'),
    ('audio', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPS-01/vpn_hangouts_audio2.pcap'),
    ('audio', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPs-02/vpn_skype_audio1.pcap'),
    ('audio', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPs-02/vpn_skype_audio2.pcap'),
    ('audio', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPs-02/vpn_voipbuster1a.pcap'),
    ('audio', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPs-02/vpn_voipbuster1b.pcap'),
    ('video', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPs-02/vpn_netflix_A.pcap'),
    ('video', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPs-02/vpn_vimeo_A.pcap'),
    ('video', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPs-02/vpn_vimeo_B.pcap'),
    ('video', '/workspace/datasets/ISCXVPN2016/pcap/VPN-PCAPs-02/vpn_youtube_A.pcap'),
]
READ_CAP = 50000
WINDOW_CONFIGS = {  # name -> (W seconds, step fraction of W)
    'w30': (30.0, 1.0), 'w30ov': (30.0, 0.5),
    'w45': (45.0, 1.0), 'w45ov': (45.0, 0.5),
    'w60': (60.0, 1.0), 'w60ov': (60.0, 0.5),
}
MIN_WINDOW_PACKETS = 20
TARGETS = {'accuracy': .95, 'macro_precision': .95, 'macro_recall': .98,
           'max_per_class_fpr': .05}


# ---------------------------------------------------------------------------
# Stage extract: one read pass -> per-file session timelines
# ---------------------------------------------------------------------------
class _P:  # tiny packet record for cached timelines
    __slots__ = ('timestamp', 'length', 'direction')

    def __init__(self, t: float, L: int, d: int):
        self.timestamp = t
        self.length = L
        self.direction = d


def stage_extract() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for lab, path in FILES:
        sm = SessionManager()
        reader = PCAPReader(sm)
        sessions = []
        for idx, session in enumerate(reader.read_pcap_generator(path,
                                                                 max_read_packets=READ_CAP)):
            if session.total_packets < MIN_SESSION_PACKETS or \
                    session.total_bytes < MIN_SESSION_BYTES:
                continue
            key = (session.src_ip, session.dst_ip, session.src_port,
                   session.dst_port, str(session.protocol))
            pkts = [(float(p.timestamp), int(p.length), int(getattr(p, 'direction', 1)))
                    for p in session.packets]
            sessions.append({'key': key, 'sid': f'{Path(path).stem}:{idx}',
                             'pkts': pkts})
        data[Path(path).name] = {'label': lab, 'sessions': sessions,
                                 'read_capped': bool(reader.read_capped),
                                 'packets_read': int(reader.total_packets)}
        print(f'EXTRACT {Path(path).name} label={lab} sessions={len(sessions)} '
              f'pkts={reader.total_packets} capped={reader.read_capped}', flush=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(data, f)
    meta = {k: {'label': v['label'], 'sessions': len(v['sessions']),
                'packets_read': v['packets_read'], 'read_capped': v['read_capped']}
            for k, v in data.items()}
    (OUT / 'extraction_meta.json').write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Windowing + features
# ---------------------------------------------------------------------------
def _gini(counts: np.ndarray) -> float:
    if len(counts) == 0 or counts.sum() == 0:
        return 0.0
    p = np.sort(counts / counts.sum())
    n = len(p)
    return float(1.0 - ((2 * np.arange(1, n + 1) - n - 1) * p).sum())


def build_windows(cfg: str) -> pd.DataFrame:
    with open(CACHE, 'rb') as f:
        data = pickle.load(f)
    W, step_frac = WINDOW_CONFIGS[cfg]
    step = W * step_frac
    rows = []
    for fname, rec in data.items():
        lab = rec['label']
        all_ts = np.array([t for s in rec['sessions'] for (t, _, _) in s['pkts']])
        if len(all_ts) == 0:
            continue
        t0 = float(all_ts.min())
        t_end = float(all_ts.max())
        n_win = int(np.ceil((t_end - t0 - W) / step)) + 1 if t_end - t0 > W else 1
        for i in range(max(1, n_win)):
            lo, hi = t0 + i * step, t0 + i * step + W
            agg_pkts, flow_pk, flow_by, flow_new, endpoints = [], {}, {}, set(), set()
            for s in rec['sessions']:
                started = False
                for (t, L, d) in s['pkts']:
                    if lo <= t < hi:
                        agg_pkts.append(_P(t, L, d))
                        flow_pk[s['sid']] = flow_pk.get(s['sid'], 0) + 1
                        flow_by[s['sid']] = flow_by.get(s['sid'], 0) + L
                        started = True
                        endpoints.add(s['key'][2] if d < 0 else s['key'][3])
                if started:
                    flow_new.add(s['sid'])
            if len(agg_pkts) < MIN_WINDOW_PACKETS:
                continue
            agg_pkts.sort(key=lambda p: p.timestamp)
            if len(agg_pkts) > 4096:  # keep feature cost bounded, deterministic
                agg_pkts = agg_pkts[:4096]
            pk = np.array([flow_pk[s] for s in flow_pk])
            by = np.array([flow_by[s] for s in flow_by])
            top_pk = float(pk.max() / pk.sum()) if pk.sum() else 0.0
            top_by = float(by.max() / by.sum()) if by.sum() else 0.0
            row = {
                '_observation_id': f'{fname}:{cfg}:{i}',
                '_source_file': fname, '_label': lab, '_label_id': LID[lab],
                '_win_cfg': cfg, '_win_index': i, '_win_seconds': W,
                '_n_packets_total': len(agg_pkts),
                '_n_sessions_active': len(flow_pk),
                '_n_sessions_started': len(flow_new),
                '_n_remote_endpoints': len(endpoints),
                '_top_flow_pkt_share': top_pk,
                '_top_flow_byte_share': top_by,
                '_flow_pkt_gini': _gini(pk.astype(float)),
                '_flow_byte_gini': _gini(by.astype(float)),
            }
            row.update(sequence_features(agg_pkts))
            rows.append(row)
    df = pd.DataFrame(rows)
    # per-session id set for session-disjoint splitting
    return df


def window_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not c.startswith('_')]


# ---------------------------------------------------------------------------
# Stage split: pre-register all granularities BEFORE any model work
# ---------------------------------------------------------------------------
GRANULARITIES = ['win_stratified', 'session_disjoint', 'capture_disjoint']


def _assign(per_label_items: Dict[str, List[str]], frac=(0.6, 0.2, 0.2)) -> Dict[str, str]:
    out = {}
    for lab, items in per_label_items.items():
        rng = random.Random(f'{SEED}:{lab}')
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        nt = max(1, int(round(n * frac[0])))
        nv = max(1, int(round(n * frac[1])))
        if n - nt - nv < 1:
            nt, nv = max(1, n - 2), 1
        for x in items[:nt]:
            out[x] = 'train'
        for x in items[nt:nt + nv]:
            out[x] = 'validation'
        for x in items[nt + nv:]:
            out[x] = 'test'
    return out


def stage_split() -> None:
    sp = OUT / 'splits'
    sp.mkdir(parents=True, exist_ok=True)
    manifest = {'seed': SEED,
                'registered_before_search': True,
                'registered_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'granularities': {}}
    for cfg in WINDOW_CONFIGS:
        df = build_windows(cfg)
        df.to_csv(OUT / f'windows_{cfg}.csv', index=False)
        # window-stratified: unit = window
        w = _assign({lab: df[df._label == lab]._observation_id.tolist()
                     for lab in LABELS})
        # session-disjoint: unit = reconstructed session. Window inherits the
        # split of its most-active session (ties -> lowest session index).
        with open(CACHE, 'rb') as f:
            data = pickle.load(f)
        sess_assign = {}
        for lab in LABELS:
            units = []
            for fname, rec in data.items():
                if rec['label'] != lab:
                    continue
                for s in rec['sessions']:
                    units.append(s['sid'])
            sess_assign.update(_assign({lab: units}))
        # window -> dominant session split
        w_sess = {}
        for fname, rec in data.items():
            Wlen = WINDOW_CONFIGS[cfg][0]
            step = WINDOW_CONFIGS[cfg][1] * Wlen
            all_ts = [t for s in rec['sessions'] for (t, _, _) in s['pkts']]
            if not all_ts:
                continue
            t0 = min(all_ts)
            t_end = max(all_ts)
            n_win = int(np.ceil((t_end - t0 - Wlen) / step)) + 1 if t_end - t0 > Wlen else 1
            for i in range(max(1, n_win)):
                lo, hi = t0 + i * step, t0 + i * step + Wlen
                counts: Dict[str, int] = {}
                for s in rec['sessions']:
                    c = sum(1 for (t, _, _) in s['pkts'] if lo <= t < hi)
                    if c:
                        counts[s['sid']] = c
                if counts:
                    dom = max(counts.items(), key=lambda kv: (kv[1], -int(kv[0].split(':')[-1])))[0]
                    w_sess[f'{fname}:{cfg}:{i}'] = sess_assign[dom]
        # capture-disjoint: unit = file
        cap = _assign({lab: [f for f, r in data.items() if r['label'] == lab]
                       for lab in LABELS})
        rows = []
        for _, r in df.iterrows():
            oid = r['_observation_id']
            rows.append({'_observation_id': oid, '_label': r['_label'],
                         '_split_win': w.get(oid, 'train'),
                         '_split_sess': w_sess.get(oid),
                         '_split_cap': cap[r['_source_file']]})
        sdf = pd.DataFrame(rows)
        sdf.to_csv(sp / f'split_{cfg}.csv', index=False)
        sha = hashlib.sha256((sp / f'split_{cfg}.csv').read_bytes()).hexdigest()
        manifest['granularities'][cfg] = {
            'windows': int(len(df)),
            'per_label': df.groupby('_label').size().to_dict(),
            'split_sha256': sha,
            'unit_notes': {
                'win_stratified': 'random per-label window assignment; sessions '
                                  'may cross splits (competition-oriented)',
                'session_disjoint': 'window inherits the split of its dominant '
                                    'reconstructed session',
                'capture_disjoint': 'whole capture (file) in one split',
            },
            'counts': {g: sdf[c].value_counts().to_dict() for g, c in
                       (('win_stratified', '_split_win'),
                        ('session_disjoint', '_split_sess'),
                        ('capture_disjoint', '_split_cap'))},
        }
        print(f'SPLIT {cfg} windows={len(df)} sha={sha[:12]}', flush=True)
    (OUT / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))


def load_windowed(cfg: str, granularity: str) -> pd.DataFrame:
    df = pd.read_csv(OUT / f'windows_{cfg}.csv')
    sp = pd.read_csv(OUT / 'splits' / f'split_{cfg}.csv')
    col = {'win_stratified': '_split_win', 'session_disjoint': '_split_sess',
           'capture_disjoint': '_split_cap'}[granularity]
    df = df.merge(sp[['_observation_id', col]], on='_observation_id')
    df['_split'] = df[col]
    if df['_split'].isna().any():
        raise RuntimeError('unassigned windows')
    return df


# ---------------------------------------------------------------------------
# Metrics / models
# ---------------------------------------------------------------------------
def metric(y: np.ndarray, p: np.ndarray) -> Dict[str, object]:
    ids = list(range(3))
    P, R, F1, s = precision_recall_fscore_support(y, p, labels=ids, zero_division=0)
    cm = confusion_matrix(y, p, labels=ids)
    fprs = []
    for i in ids:
        fp = cm[:, i].sum() - cm[i, i]
        neg = cm.sum() - cm[i, :].sum()
        fprs.append(fp / neg if neg else 0.0)
    r = {'accuracy': float(accuracy_score(y, p)),
         'macro_precision': float(P.mean()), 'macro_recall': float(R.mean()),
         'macro_f1': float(F1.mean()), 'max_class_fpr': float(max(fprs)),
         'errors': int((p != y).sum()),
         'per_class': {LABELS[i]: {'P': float(P[i]), 'R': float(R[i]),
                                   'FPR': float(fprs[i]), 'N': int(s[i])}
                       for i in ids},
         'cm': cm.tolist()}
    r['target_pass'] = bool(r['accuracy'] >= TARGETS['accuracy']
                            and r['macro_precision'] >= TARGETS['macro_precision']
                            and r['macro_recall'] >= TARGETS['macro_recall']
                            and r['max_class_fpr'] <= TARGETS['max_per_class_fpr'])
    return r


def make_model(name: str, seed: int):
    if name == 'xgb':
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=.04,
                             min_child_weight=1, subsample=.9, colsample_bytree=.9,
                             reg_lambda=1.5, random_state=seed, n_jobs=8,
                             eval_metric='mlogloss', tree_method='hist')
    if name == 'extra':
        return ExtraTreesClassifier(n_estimators=600, max_features='sqrt',
                                    class_weight='balanced', random_state=seed,
                                    n_jobs=8)
    raise ValueError(name)


def fit_predict(train: pd.DataFrame, eval_df: pd.DataFrame, fs: Sequence[str],
                model: str, seed: int) -> np.ndarray:
    Xtr = train[list(fs)].replace([np.inf, -np.inf], np.nan).fillna(0)
    ytr = train['_label_id'].astype(int).to_numpy()
    Xe = eval_df[list(fs)].replace([np.inf, -np.inf], np.nan).fillna(0)
    members = ['xgb', 'extra'] if model == 'ens_xe' else [model]
    probs = []
    for j, mem in enumerate(members):
        m = make_model(mem, seed + j)
        if mem == 'xgb':
            m.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
        else:
            m.fit(Xtr, ytr)
        p = np.zeros((len(Xe), 3))
        p[:, m.classes_] = m.predict_proba(Xe)
        probs.append(p)
    return np.argmax(np.mean(probs, axis=0), axis=1)


def rank_features(train: pd.DataFrame, fs: Sequence[str]) -> List[str]:
    X = train[list(fs)].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train['_label_id'].astype(int).to_numpy()
    m = XGBClassifier(n_estimators=160, max_depth=5, learning_rate=.06,
                      subsample=.9, colsample_bytree=.9, reg_lambda=1.5,
                      random_state=SEED, n_jobs=8, eval_metric='mlogloss',
                      tree_method='hist')
    m.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    order = np.argsort(m.feature_importances_)[::-1]
    return [fs[i] for i in order]


# ---------------------------------------------------------------------------
def stage_search(granularity: str) -> None:
    rows, frozen_all = [], {}
    for cfg in WINDOW_CONFIGS:
        df = load_windowed(cfg, granularity)
        tr = df[df._split == 'train'].reset_index(drop=True)
        va = df[df._split == 'validation'].reset_index(drop=True)
        if len(tr) < 30 or len(va) < 10:
            print(f'SKIP {cfg} (too few rows: train={len(tr)} val={len(va)})', flush=True)
            continue
        fs_all = window_feature_columns(df)
        rank = rank_features(tr, fs_all)
        for k in (32, 64, 0):  # 0 = full pool
            fs = rank[:k] if k else list(fs_all)
            for model in ('xgb', 'extra', 'ens_xe'):
                p = fit_predict(tr, va, fs, model, SEED + 10)
                m = metric(va['_label_id'].astype(int).to_numpy(), p)
                rows.append({'granularity': granularity, 'win_cfg': cfg,
                             'k': k if k else len(fs), 'model': model,
                             **{x: m[x] for x in ('accuracy', 'macro_precision',
                                                  'macro_recall', 'macro_f1',
                                                  'max_class_fpr', 'errors')},
                             'target_pass': m['target_pass'],
                             'n_train': len(tr), 'n_val': len(va)})
                print('VAL', rows[-1], flush=True)
    lb = pd.DataFrame(rows)
    lb.to_csv(OUT / f'validation_leaderboard_{granularity}.csv', index=False)
    if not len(lb):
        (OUT / f'search_outcome_{granularity}.json').write_text(json.dumps(
            {'status': 'no_evaluable_configuration'}, indent=2))
        return
    # stability of the top config
    top = lb.sort_values(['target_pass', 'macro_recall', 'accuracy'],
                         ascending=False).iloc[0]
    cfg, k, model = top['win_cfg'], int(top['k']), top['model']
    df = load_windowed(cfg, granularity)
    tr = df[df._split == 'train'].reset_index(drop=True)
    va = df[df._split == 'validation'].reset_index(drop=True)
    fs_all = window_feature_columns(df)
    rank = rank_features(tr, fs_all)
    fs = rank[:k] if k <= len(rank) and top['k'] in (32, 64) else list(fs_all)
    ms = [metric(va['_label_id'].astype(int).to_numpy(),
                 fit_predict(tr, va, fs, model, SEED + 200 + 40 * i))
          for i in range(3)]
    stab = {'win_cfg': cfg, 'k': len(fs), 'model': model,
            'macro_recall_mean': float(np.mean([m['macro_recall'] for m in ms])),
            'accuracy_mean': float(np.mean([m['accuracy'] for m in ms])),
            'macro_precision_min': float(np.min([m['macro_precision'] for m in ms])),
            'max_fpr_max': float(np.max([m['max_class_fpr'] for m in ms])),
            'all_pass_seeds': int(sum(m['target_pass'] for m in ms))}
    pd.DataFrame([stab]).to_csv(OUT / f'stability_{granularity}.csv', index=False)
    single_pass = bool(top['target_pass'])
    stab_pass = (stab['accuracy_mean'] >= TARGETS['accuracy']
                 and stab['macro_precision_min'] >= TARGETS['macro_precision']
                 and stab['macro_recall_mean'] >= TARGETS['macro_recall']
                 and stab['max_fpr_max'] <= TARGETS['max_per_class_fpr'])
    outcome = {
        'scenario': 'B_behavior_rawpcap_multiflow',
        'granularity': granularity,
        'blinding': 'fresh pre-registered split (seed 20260922) over 21 '
                    'previously-explored ISCXVPN2016 captures',
        'win_cfg': cfg, 'window_seconds': WINDOW_CONFIGS[cfg][0],
        'step_seconds': WINDOW_CONFIGS[cfg][1] * WINDOW_CONFIGS[cfg][0],
        'selected_features': fs, 'k': len(fs), 'model': model,
        'refit_seed': SEED + 900,
        'validation_best': {x: float(top[x]) for x in
                            ('accuracy', 'macro_precision', 'macro_recall',
                             'max_class_fpr')},
        'validation_stability': stab,
        'targets': TARGETS,
    }
    if single_pass and stab_pass:
        fp = OUT / f'frozen_config_{granularity}.json'
        outcome['fresh_test_policy'] = ('this granularity test rows open exactly '
                                        'once, next, with refit on train+validation')
        fp.write_text(json.dumps(outcome, indent=2, ensure_ascii=False))
        print('FROZEN', granularity, stab, flush=True)
    else:
        outcome['status'] = 'validation_not_passed'
        outcome['reason'] = (f'best single pass={single_pass}, stability pass='
                             f'{stab_pass}; test stays sealed')
        (OUT / f'search_outcome_{granularity}.json').write_text(
            json.dumps(outcome, indent=2, ensure_ascii=False))
        print('NOT FROZEN', granularity, outcome['reason'], flush=True)


def stage_test(granularity: str) -> None:
    tm = OUT / f'test_metrics_{granularity}.json'
    if tm.exists():
        raise RuntimeError(f'{tm.name} already exists: one-shot rule')
    fp = OUT / f'frozen_config_{granularity}.json'
    if not fp.exists():
        raise RuntimeError(f'frozen_config_{granularity}.json missing: gate not passed')
    frozen = json.loads(fp.read_text())
    cfg = frozen['win_cfg']
    df = load_windowed(cfg, granularity)
    dev = df[df._split.isin(['train', 'validation'])].reset_index(drop=True)
    te = df[df._split == 'test'].reset_index(drop=True)
    p = fit_predict(dev, te, frozen['selected_features'], frozen['model'],
                    frozen['refit_seed'])
    m = metric(te['_label_id'].astype(int).to_numpy(), p)
    passes = {'accuracy': m['accuracy'] >= TARGETS['accuracy'],
              'macro_precision': m['macro_precision'] >= TARGETS['macro_precision'],
              'macro_recall': m['macro_recall'] >= TARGETS['macro_recall'],
              'max_per_class_fpr': m['max_class_fpr'] <= TARGETS['max_per_class_fpr']}
    result = {
        'frozen_config_sha256': hashlib.sha256(fp.read_bytes()).hexdigest(),
        'granularity': granularity, 'win_cfg': cfg,
        'metrics': {x: m[x] for x in ('accuracy', 'macro_precision',
                                      'macro_recall', 'macro_f1',
                                      'max_class_fpr', 'errors')},
        'n_test': int(len(te)),
        'targets': TARGETS, 'pass': passes, 'all_pass': all(passes.values()),
        'per_class': m['per_class'], 'cm': m['cm'],
        'opened_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'one_shot': True,
    }
    tm.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    pc = pd.DataFrame([{'label': c, **v} for c, v in m['per_class'].items()])
    pc.to_csv(OUT / f'per_class_metrics_{granularity}.csv', index=False)
    print(json.dumps({k: result[k] for k in ('metrics', 'pass', 'all_pass')},
                     indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['extract', 'split', 'search', 'test'])
    ap.add_argument('--granularity', default='win_stratified')
    args = ap.parse_args()
    if args.stage == 'extract':
        stage_extract()
    elif args.stage == 'split':
        stage_split()
    elif args.stage == 'search':
        stage_search(args.granularity)
    else:
        stage_test(args.granularity)


if __name__ == '__main__':
    main()
