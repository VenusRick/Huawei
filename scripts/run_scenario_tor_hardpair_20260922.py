# -*- coding: utf-8 -*-
"""Scenario C (2026-09-22): Tor hard-pair engineering stress improvements.

Baseline (ledger 27.4.2, post-exploration matched-pair frozen test):
Acc 91.39 / Macro-P 94.83 / Macro-R 83.02 / MaxFPR 33.96, with Vimeo Tor
recall 33.33% and Skype file-transfer Tor recall 5.88%. The ordinary
file-disjoint Tor RuleBundle PASS (99.14/99.12/98.51/2.70) is untouched.

All matched business pairs were exposed in earlier exploration, so this round
is explicitly a POST-EXPLORATION ENGINEERING STRESS on the same frozen hard
split (train pairs / validation pairs / hard test pairs). No fresh/blind
claim. No unused pairs remain to form a fresh pair-holdout (all 24 pairs are
assigned in the frozen split).

Observation-unit upgrades tried this round (error diagnosis first: bulk
encrypted transfers are only separable through longer context):
1. session-level aggregation: all 15s windows of one reconstructed session ->
   mean/std/max/min of the 64-dim stats + window count / span features, one
   prediction per session;
2. causal multi-window vote at k in {3,5,7} on the window-level model;
3. capture-context features: whole-capture aggregates appended to each window
   (labeled post-hoc context, needs the full capture, not deployable online);
4. decision-threshold calibration on the validation pairs.

Stages: run (writes leaderboard + test metrics + per-pair table)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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

SRC = ROOT / 'output/final_profile_runtime_20260919/tunnel'
OUT = ROOT / 'output/all_scenarios_metric_push_20260922/tor_hardpair'
SEED = 20260922
LABELS = ['NonTor', 'Tor']
LID = {x: i for i, x in enumerate(LABELS)}
PAT = re.compile(r':tw(\d+):(\d+)$')
TARGETS = {'accuracy': .95, 'macro_precision': .95, 'macro_recall': .98,
           'max_per_class_fpr': .05}
VAL_KEYS = ('accuracy', 'macro_precision', 'macro_recall', 'max_class_fpr')

PAIRS = [
    ('ssl_browse', 'BROWSING_gate_SSL_Browsing.pcap', 'SSL_Browsing.pcap'),
    ('aim_chat', 'CHAT_gate_AIM_chat.pcap', 'AIM_Chat.pcap'),
    ('icq_chat', 'CHAT_gate_ICQ_chat.pcap', 'ICQ_Chat.pcap'),
    ('facebook_chat', 'CHAT_gate_facebook_chat.pcap', 'facebook_chat.pcap'),
    ('hangout_chat', 'CHAT_gate_hangout_chat.pcap', 'hangout_chat.pcap'),
    ('skype_chat', 'CHAT_gate_skype_chat.pcap', 'skype_chat.pcap'),
    ('sftp', 'FILE-TRANSFER_gate_SFTP_filetransfer.pcap', 'SFTP_filetransfer.pcap'),
    ('skype_transfer', 'FILE-TRANSFER_tor_skype_transfer.pcap', 'skype_transfer.pcap'),
    ('imap', 'MAIL_gate_Email_IMAP_filetransfer.pcap', 'Email_IMAP_filetransfer.pcap'),
    ('pop', 'MAIL_gate_POP_filetransfer.pcap', 'POP_filetransfer.pcap'),
    ('thunderbird_imap', 'MAIL_Gateway_Thunderbird_Imap.pcap', 'Workstation_Thunderbird_Imap.pcap'),
    ('thunderbird_pop', 'MAIL_Gateway_Thunderbird_POP.pcap', 'Workstation_Thunderbird_POP.pcap'),
    ('p2p_multi', 'P2P_tor_p2p_multipleSpeed.pcap', 'p2p_multipleSpeed.pcap'),
    ('p2p_vuze', 'P2P_tor_p2p_vuze.pcap', 'p2p_vuze.pcap'),
    ('vimeo', 'VIDEO_Vimeo_Gateway.pcap', 'Vimeo_Workstation.pcap'),
    ('youtube_flash', 'VIDEO_Youtube_Flash_Gateway.pcap', 'Youtube_Flash_Workstation.pcap'),
    ('youtube_html5', 'VIDEO_Youtube_HTML5_Gateway.pcap', 'Youtube_HTML5_Workstation.pcap'),
    ('fb_voice', 'VOIP_Facebook_Voice_Gateway.pcap', 'Facebook_Voice_Workstation.pcap'),
    ('hangouts_voice', 'VOIP_Hangouts_voice_Gateway.pcap', 'Hangouts_voice_Workstation.pcap'),
    ('skype_voice', 'VOIP_Skype_Voice_Gateway.pcap', 'Skype_Voice_Workstation.pcap'),
    ('skype_audio', 'VOIP_gate_Skype_Audio.pcap', 'Skype_Audio.pcap'),
    ('facebook_audio', 'VOIP_gate_facebook_Audio.pcap', 'facebook_Audio.pcap'),
    ('hangout_audio', 'VOIP_gate_hangout_audio.pcap', 'Hangout_audio.pcap'),
]
TEST_PAIRS = {'sftp', 'skype_transfer', 'skype_audio', 'vimeo', 'facebook_chat'}
VAL_PAIRS = {'aim_chat', 'imap', 'p2p_vuze', 'youtube_html5', 'hangouts_voice'}


def met(y: np.ndarray, p: np.ndarray) -> Dict[str, object]:
    P, R, F1, s = precision_recall_fscore_support(y, p, labels=[0, 1],
                                                  zero_division=0)
    cm = confusion_matrix(y, p, labels=[0, 1])
    fprs = []
    for i in [0, 1]:
        fp = cm[:, i].sum() - cm[i, i]
        neg = cm.sum() - cm[i, :].sum()
        fprs.append(fp / neg if neg else 0.0)
    r = {'accuracy': float(accuracy_score(y, p)),
         'macro_precision': float(P.mean()), 'macro_recall': float(R.mean()),
         'macro_f1': float(F1.mean()), 'max_class_fpr': float(max(fprs)),
         'errors': int((p != y).sum()),
         'per_class': {LABELS[i]: {'P': float(P[i]), 'R': float(R[i]),
                                   'FPR': float(fprs[i]), 'N': int(s[i])}
                       for i in [0, 1]},
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
                             eval_metric='logloss', tree_method='hist')
    if name == 'extra':
        return ExtraTreesClassifier(n_estimators=600, max_features='sqrt',
                                    class_weight='balanced', random_state=seed,
                                    n_jobs=8)
    raise ValueError(name)


def fit_proba(train: pd.DataFrame, eval_df: pd.DataFrame, fs: Sequence[str],
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
        p = np.zeros((len(Xe), 2))
        p[:, m.classes_] = m.predict_proba(Xe)
        probs.append(p)
    return np.mean(probs, axis=0)


def load_base() -> pd.DataFrame:
    df = pd.read_csv(SRC / 'mine/raw_features.csv')
    base_to_pair = {}
    for pair, t, n in PAIRS:
        base_to_pair[t] = pair
        base_to_pair[n] = pair
    df['_pair'] = [base_to_pair.get(Path(x).name, '') for x in df._source_file.astype(str)]
    df = df[df._pair != ''].copy()
    if not TEST_PAIRS.issubset(set(df._pair.unique())):
        raise RuntimeError('missing mandatory Tor hard pairs')
    df['_split'] = df['_pair'].map(
        lambda p: 'test' if p in TEST_PAIRS else
        ('validation' if p in VAL_PAIRS else 'train'))
    m = df['_observation_id'].astype(str).str.extract(PAT)
    df['_session_index'] = m[0].astype(int)
    df['_window_index'] = m[1].astype(int)
    return df


def session_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all windows of one (file, session) into one observation."""
    feats = [c for c in df.columns if not c.startswith('_')]
    rows = []
    for (f, s), g in df.groupby(['_source_file', '_session_index']):
        if len(g) < 1:
            continue
        agg = {c: float(g[c].replace([np.inf, -np.inf], np.nan).fillna(0).mean())
               for c in feats}
        for c in feats:
            v = g[c].replace([np.inf, -np.inf], np.nan).fillna(0)
            agg[f'{c}__std'] = float(v.std())
            agg[f'{c}__max'] = float(v.max())
        agg.update({
            '_observation_id': f'{Path(f).stem}:sess{s}',
            '_source_file': f, '_label_id': int(g['_label_id'].iloc[0]),
            '_label': ('Tor' if '/Tor/' in str(f) else 'NonTor'),
            '_pair': g['_pair'].iloc[0], '_split': g['_split'].iloc[0],
            '_session_index': int(s), '_n_windows': int(len(g)),
            '_span_windows': int(g['_window_index'].max() - g['_window_index'].min() + 1),
        })
        rows.append(agg)
    return pd.DataFrame(rows)


def add_capture_context(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in df.columns if not c.startswith('_')]
    df = df.copy()
    ctx = df.groupby('_source_file')[feats]. \
        transform(lambda g: g.replace([np.inf, -np.inf], np.nan).fillna(0).mean())
    ctx.columns = [f'{c}__capmean' for c in feats]
    return pd.concat([df, ctx], axis=1)


def causal_vote(pred: Sequence[int], order_idx: Sequence[int], k: int) -> List[int]:
    """Causal majority over the last k windows within an ordered group."""
    out = [int(x) for x in pred]
    run: List[List[int]] = []
    for i in np.argsort(order_idx):
        run.append([int(pred[i])])
        if len(run) > k:
            run.pop(0)
        counts = np.bincount([v for window in run for v in window], minlength=2)
        out[i] = int(np.argmax(counts))
    return out


def per_pair(te: pd.DataFrame, yhat: np.ndarray) -> List[Dict[str, object]]:
    rows = []
    for pair in sorted(te['_pair'].unique()):
        m = (te['_pair'] == pair).to_numpy()
        yt = te['_label_id'].astype(int).to_numpy()[m]
        yp = yhat[m]
        tor_mask = yt == 1
        rows.append({
            'pair': pair, 'n': int(m.sum()),
            'accuracy': float((yp == yt).mean()),
            'tor_recall': float((yp[tor_mask] == 1).mean()) if tor_mask.any() else None,
            'n_tor': int(tor_mask.sum()),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_base()
    feats = [c for c in df.columns if not c.startswith('_')]
    tr = df[df._split == 'train'].reset_index(drop=True)
    va = df[df._split == 'validation'].reset_index(drop=True)
    te = df[df._split == 'test'].reset_index(drop=True)

    # ---------------- error diagnosis on the failing pairs ----------------
    diag = []
    for pair in ('skype_transfer', 'vimeo'):
        g = te[te._pair == pair]
        for lab in (0, 1):
            gg = g[g._label_id == lab]
            diag.append({'pair': pair, 'label': LABELS[lab],
                         'n_windows': int(len(gg)),
                         'mean_bytes': float(gg['tunnel_total_bytes'].mean()) if len(gg) else 0,
                         'mean_pkts': float(gg['tunnel_n_pkt'].mean()) if len(gg) else 0})
    (OUT / 'diagnosis.json').write_text(json.dumps(diag, indent=2))

    results = []

    # ---- A. window-level model (baseline reproduction this round) ----
    for model in ('xgb', 'extra', 'ens_xe'):
        pv = np.argmax(fit_proba(tr, va, feats, model, SEED), axis=1)
        pt = np.argmax(fit_proba(tr, te, feats, model, SEED), axis=1)
        mv, mt = met(va['_label_id'].astype(int).to_numpy(), pv), \
            met(te['_label_id'].astype(int).to_numpy(), pt)
        results.append({'unit': 'window', 'variant': model,
                        'val': {k: mv[k] for k in VAL_KEYS},
                        'test': mt, 'per_pair': per_pair(te, pt)})
        print('RUN', results[-1]['unit'], results[-1]['variant'],
              'test', mt['accuracy'], mt['macro_recall'], mt['max_class_fpr'],
              flush=True)

    # ---- B. causal vote k=3/5/7 on the best window model (by val macro-R) ----
    best_w = max((r for r in results if r['unit'] == 'window'),
                 key=lambda r: r['val']['macro_recall'])
    model = best_w['variant']
    pv = fit_proba(tr, va, feats, model, SEED)
    pt = fit_proba(tr, te, feats, model, SEED)
    for k in (3, 5, 7):
        vhat = causal_vote(np.argmax(pv, axis=1), va['_window_index'].to_numpy(), k)
        that = causal_vote(np.argmax(pt, axis=1), te['_window_index'].to_numpy(), k)
        mv, mt = met(va['_label_id'].astype(int).to_numpy(), np.array(vhat)), \
            met(te['_label_id'].astype(int).to_numpy(), np.array(that))
        results.append({'unit': f'window_vote{k}', 'variant': model,
                        'val': {k2: mv[k2] for k2 in VAL_KEYS},
                        'test': mt, 'per_pair': per_pair(te, np.array(that))})
        print('RUN vote', k, mt['accuracy'], mt['macro_recall'],
              mt['max_class_fpr'], flush=True)

    # ---- C. session-level aggregation ----
    sdf = session_frame(df)
    sfeats = [c for c in sdf.columns if not c.startswith('_')]
    usable = [c for c in sfeats if sdf[c].replace([np.inf, -np.inf], np.nan).fillna(0).std() > 0]
    str_, sva, ste = (sdf[sdf._split == s].reset_index(drop=True)
                      for s in ('train', 'validation', 'test'))
    for model in ('xgb', 'ens_xe'):
        pv = np.argmax(fit_proba(str_, sva, usable, model, SEED + 1), axis=1)
        pt = np.argmax(fit_proba(str_, ste, usable, model, SEED + 1), axis=1)
        mv, mt = met(sva['_label_id'].astype(int).to_numpy(), pv), \
            met(ste['_label_id'].astype(int).to_numpy(), pt)
        results.append({'unit': 'session', 'variant': model,
                        'val': {k: mv[k] for k in VAL_KEYS},
                        'test': mt, 'per_pair': per_pair(ste, pt),
                        'n_sessions': {'train': len(str_), 'val': len(sva),
                                       'test': len(ste)}})
        print('RUN session', model, mt['accuracy'], mt['macro_recall'],
              mt['max_class_fpr'], flush=True)

    # ---- D. capture-context windows (post-hoc, labeled) ----
    cdf = add_capture_context(df)
    cfeats = [c for c in cdf.columns if not c.startswith('_')]
    ctr, cva, cte = (cdf[cdf._split == s].reset_index(drop=True)
                     for s in ('train', 'validation', 'test'))
    for model in ('xgb', 'ens_xe'):
        pv = np.argmax(fit_proba(ctr, cva, cfeats, model, SEED + 2), axis=1)
        pt = np.argmax(fit_proba(ctr, cte, cfeats, model, SEED + 2), axis=1)
        mv, mt = met(cva['_label_id'].astype(int).to_numpy(), pv), \
            met(cte['_label_id'].astype(int).to_numpy(), pt)
        results.append({'unit': 'window_capturectx', 'variant': model,
                        'val': {k: mv[k] for k in VAL_KEYS},
                        'test': mt, 'per_pair': per_pair(cte, pt),
                        'context_note': 'whole-capture mean appended to each '
                                        'window: post-hoc, not online-deployable'})
        print('RUN capctx', model, mt['accuracy'], mt['macro_recall'],
              mt['max_class_fpr'], flush=True)

    # ---- E. threshold calibration on validation pairs (window model) ----
    pv = fit_proba(tr, va, feats, best_w['variant'], SEED)
    yv = va['_label_id'].astype(int).to_numpy()
    best_thr, best_key, best_val = 0.5, None, None
    for thr in np.arange(0.20, 0.81, 0.05):
        m = met(yv, (pv[:, 1] >= thr).astype(int))
        k = (m['macro_recall'], m['accuracy'], -m['max_class_fpr'])
        if best_key is None or k > best_key:
            best_key, best_thr, best_val = k, float(thr), m
    pt = fit_proba(tr, te, feats, best_w['variant'], SEED)
    that = (pt[:, 1] >= best_thr).astype(int)
    mt = met(te['_label_id'].astype(int).to_numpy(), that)
    results.append({'unit': 'window_thr', 'variant': best_w['variant'],
                    'threshold': best_thr,
                    'val': {k: best_val[k] for k in VAL_KEYS},
                    'test': mt, 'per_pair': per_pair(te, that)})
    print('RUN thr', best_thr, mt['accuracy'], mt['macro_recall'],
          mt['max_class_fpr'], flush=True)

    for r in results:
        r['test'] = {k: r['test'][k] for k in
                     ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
                      'max_class_fpr', 'errors', 'per_class', 'target_pass')}
    summary = {
        'scenario': 'C_tor_hardpair_engineering',
        'blinding': 'post-exploration engineering stress on the frozen matched-'
                    'pair hard split (all pairs previously exposed)',
        'frozen_split': {'test_pairs': sorted(TEST_PAIRS),
                         'validation_pairs': sorted(VAL_PAIRS)},
        'targets': TARGETS, 'results': results,
        'baseline_20260920': {'accuracy': 0.9139, 'macro_precision': 0.9483,
                              'macro_recall': 0.8302, 'max_class_fpr': 0.3396},
        'ordinary_file_disjoint_pass_untouched':
            'output/final_profile_runtime_20260919/tunnel RuleBundle PASS '
            '(Acc 99.14 / P 99.12 / R 98.51 / FPR 2.70) is not modified',
    }
    (OUT / 'engineering_stress_results.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    rows = [{'unit': r['unit'], 'variant': r['variant'],
             **{f'test_{k}': r['test'][k] for k in
                ('accuracy', 'macro_precision', 'macro_recall', 'max_class_fpr')},
             'test_pass': r['test']['target_pass'],
             **{f'val_{k}': r['val'][k] for k in VAL_KEYS}}
            for r in results]
    pd.DataFrame(rows).to_csv(OUT / 'leaderboard.csv', index=False)
    pp_rows = [{'unit': r['unit'], **p} for r in results for p in r['per_pair']]
    pd.DataFrame(pp_rows).to_csv(OUT / 'per_pair.csv', index=False)
    sha = hashlib.sha256((OUT / 'engineering_stress_results.json').read_bytes()).hexdigest()
    (OUT / 'SHA256SUMS').write_text(
        f'{sha}  engineering_stress_results.json\n')
    print('DONE', sha[:12], flush=True)


if __name__ == '__main__':
    main()
