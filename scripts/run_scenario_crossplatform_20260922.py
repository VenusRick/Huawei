# -*- coding: utf-8 -*-
"""Scenario E (2026-09-22): CrossPlatform China Android<->iOS mapping closure.

Prior state: no formal Android<->iOS experiment existed. This round:
1. builds an explicit app mapping between china/android (package-name pcaps)
   and china/ios (app-name pcaps) — only apps whose BOTH captures are real
   (many pcaps in the release are 92-byte stubs) are kept;
2. extracts per-session features with the production runtime machinery
   (prefix 64 packets, basic+advanced+generic-sequence families, same as the
   mixed-16 cache);
3. runs platform-transfer evaluation: train Android sessions -> test iOS
   sessions (and the reverse) over the 9 common apps. Because each
   (app, platform) has exactly one capture, platform-disjoint IS
   capture-disjoint here; the label set is the 9 app classes;
4. runs a pooled within-platform control (random session split; single
   capture per app-platform means sessions of one capture may cross splits —
   disclosed, competition-oriented only) to show the head-room;
5. two feature pools: identifier-free common_shape (same discipline as the
   mixed-16 generalization claim) and competition-oriented hs_all (ports/
   TLS/QUIC/protocol/TCP visible fields), reported separately.

Scope note (ledger): if the competition's "cross-platform" means the TOOL
runs on both OSes, data-level cross-OS generalization is NOT a formal target;
this experiment reports honest numbers for both readings and never forces a
PASS.

Stages: extract | run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

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

from run_mixed_dataset_full_pass_20260921 import (  # noqa: E402
    common_shape_features, sequence_features)
from src.features.basic.feature_extractor import BasicFeatureExtractor  # noqa: E402
from src.features.advanced.advanced_extractor import AdvancedFeatureExtractor  # noqa: E402
from src.features.runtime import (MIN_SESSION_BYTES, MIN_SESSION_PACKETS,  # noqa: E402
                                  truncated_session_view)
from src.parser.pcap_reader import PCAPReader  # noqa: E402
from src.parser.session.session_manager import SessionManager  # noqa: E402

OUT = ROOT / 'output/all_scenarios_metric_push_20260922/crossplatform'
PARTS = OUT / 'cache_parts'
SEED = 20260922
PREFIX = 64
READ_CAP = 100000        # packets per file (bounded; recorded per file)
MAX_SESSIONS = 300       # per (app, platform), balanced downstream
TARGETS = {'accuracy': .95, 'macro_precision': .95, 'macro_recall': .98,
           'max_per_class_fpr': .05}

# Confident manual mapping (android package -> ios capture, same product).
# 19 further name pairs were rejected: at least one side is a 92-byte stub
# or an empty file (see mapping_audit.json).
APP_MAP = {
    'baiduwenku': ('com.baidu.wenku', 'baiduwenku'),
    'bilibili': ('tv.danmaku.bili', 'bilibili'),
    'aiqiyi': ('com.qiyi.video', 'aiqiyi'),
    'youku': ('com.youku.phone', 'youku'),
    'weibo': ('com.sina.weibo', 'weibo'),
    'zhihu': ('com.zhihu.android', 'zhihu'),
    'qq': ('com.tencent.mobileqq', 'qq'),
    'qzone': ('com.qzone', 'qqkongjian'),
    'tengxunshipin': ('com.tencent.qqlive', 'tengxunshipin'),
    'renren': ('com.renren.mobile.android', 'renren'),
    'kaixinxiaoxiaole': ('com.happyelements.AndroidAnimal.qq', 'kaixinxiaoxiaole'),
}
A_DIR = Path('/workspace/datasets/CrossPlatform/china/android')
I_DIR = Path('/workspace/datasets/CrossPlatform/china/ios')


def stage_extract() -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    audit = {}
    basic_ext = BasicFeatureExtractor()
    advanced_ext = AdvancedFeatureExtractor()
    for app, (a_name, i_name) in sorted(APP_MAP.items()):
        for platform, fname in (('android', a_name), ('ios', i_name)):
            part = PARTS / f'{app}__{platform}.csv'
            if part.exists():
                continue
            path = (A_DIR if platform == 'android' else I_DIR) / f'{fname}.pcap'
            sm = SessionManager()
            reader = PCAPReader(sm)
            rows, idx = [], 0
            for session in reader.read_pcap_generator(str(path),
                                                      max_read_packets=READ_CAP):
                i = idx
                idx += 1
                if (session.total_packets < MIN_SESSION_PACKETS
                        or session.total_bytes < MIN_SESSION_BYTES):
                    continue
                view, _ = truncated_session_view(session, PREFIX)
                try:
                    row = {'_app': app, '_platform': platform,
                           '_label': f'{app}::{platform}',
                           '_source_file': path.name,
                           '_observation_id': f'{path.name}:{i}',
                           '_full_packets': int(session.total_packets)}
                    row.update(basic_ext.extract_all(view))
                    row.update(advanced_ext.extract_all(view))
                    row.update(sequence_features(view.packets))
                except Exception:  # noqa: BLE001
                    continue
                rows.append(row)
                if len(rows) >= MAX_SESSIONS:
                    break
            df = pd.DataFrame(rows)
            df.to_csv(part, index=False)
            audit[f'{app}/{platform}'] = {
                'file': path.name, 'bytes': path.stat().st_size,
                'sessions_kept': len(df), 'read_capped': bool(reader.read_capped),
                'packets_read': int(reader.total_packets)}
            print(f'EXTRACT {app}/{platform} kept={len(df)} '
                  f'capped={reader.read_capped}', flush=True)
    (OUT / 'mapping_audit.json').write_text(json.dumps(audit, indent=2))


def load_cache() -> pd.DataFrame:
    frames = []
    for p in sorted(PARTS.glob('*.csv')):
        if p.stat().st_size <= 1:  # empty side (stub capture): drop with note
            print(f'SKIP empty part {p.name}', flush=True)
            continue
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True, sort=False)
    if df['_observation_id'].duplicated().any():
        raise RuntimeError('duplicate observation ids')
    return df


def pools(df: pd.DataFrame) -> Dict[str, List[str]]:
    shape = common_shape_features(df.columns)
    tls = [c for c in df.columns if c.startswith(('tls_', 'quic_', 'protocol_',
                                                  'tcp_window_', 'src_port',
                                                  'dst_port', 'is_well_known'))
           and 'ja3' not in c and 'ja4' not in c]
    flags = [c for c in df.columns if c.endswith('_flag_count')
             or c in ('syn_ack_rtt', 'num_retransmissions',
                      'retransmission_ratio')]
    return {'shape_identifier_free': sorted(shape),
            'hs_visible': sorted(set(shape) | set(tls) | set(flags))}


def met(y: np.ndarray, p: np.ndarray, labels: List[str]) -> Dict[str, object]:
    ids = list(range(len(labels)))
    P, R, F1, s = precision_recall_fscore_support(y, p, labels=ids, zero_division=0)
    cm = confusion_matrix(y, p, labels=ids)
    fprs = []
    for i in ids:
        fp = cm[:, i].sum() - cm[i, i]
        neg = cm.sum() - cm[i, :].sum()
        fprs.append(fp / neg if neg else 0.0)
    return {'accuracy': float(accuracy_score(y, p)),
            'macro_precision': float(P.mean()), 'macro_recall': float(R.mean()),
            'macro_f1': float(F1.mean()), 'max_class_fpr': float(max(fprs)),
            'errors': int((p != y).sum()),
            'per_class': {labels[i]: {'P': float(P[i]), 'R': float(R[i]),
                                      'FPR': float(fprs[i]), 'N': int(s[i])}
                          for i in ids},
            'four_pass': bool(accuracy_score(y, p) >= TARGETS['accuracy']
                              and P.mean() >= TARGETS['macro_precision']
                              and R.mean() >= TARGETS['macro_recall']
                              and max(fprs) <= TARGETS['max_per_class_fpr'])}


def fit_predict(train: pd.DataFrame, eval_df: pd.DataFrame, fs: List[str],
                model: str, seed: int, labels: List[str]) -> np.ndarray:
    lid = {c: i for i, c in enumerate(labels)}
    Xtr = train[fs].replace([np.inf, -np.inf], np.nan).fillna(0)
    ytr = train['_app'].map(lid).to_numpy()
    Xe = eval_df[fs].replace([np.inf, -np.inf], np.nan).fillna(0)
    members = ['xgb', 'extra'] if model == 'ens_xe' else [model]
    probs = []
    for j, mem in enumerate(members):
        m = (XGBClassifier(n_estimators=300, max_depth=6, learning_rate=.04,
                           subsample=.9, colsample_bytree=.9, reg_lambda=1.5,
                           random_state=seed + j, n_jobs=8, eval_metric='mlogloss',
                           tree_method='hist') if mem == 'xgb' else
             ExtraTreesClassifier(n_estimators=600, max_features='sqrt',
                                  class_weight='balanced', random_state=seed + j,
                                  n_jobs=8))
        if mem == 'xgb':
            m.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
        else:
            m.fit(Xtr, ytr)
        p = np.zeros((len(Xe), len(labels)))
        p[:, m.classes_] = m.predict_proba(Xe)
        probs.append(p)
    return np.argmax(np.mean(probs, axis=0), axis=1)


def stage_run() -> None:
    df = load_cache()
    # drop (app, platform) sides with too few sessions to be a real class side
    counts = df.groupby(['_app', '_platform']).size().unstack(fill_value=0)
    good_apps = [a for a in counts.index
                 if counts.loc[a].get('android', 0) >= 20
                 and counts.loc[a].get('ios', 0) >= 20]
    dropped = {a: counts.loc[a].to_dict() for a in counts.index if a not in good_apps}
    df = df[df._app.isin(good_apps)].reset_index(drop=True)
    labels = sorted(good_apps)
    ps = pools(df)
    results = {'scenario': 'E_crossplatform_android_ios',
               'apps': labels, 'dropped_apps': dropped,
               'sessions': df.groupby(['_app', '_platform']).size().unstack().to_dict(),
               'targets': TARGETS, 'evaluations': []}

    for pool_name, fs in ps.items():
        fs = [f for f in fs if df[f].replace([np.inf, -np.inf], np.nan)
              .fillna(0).std() > 0]
        for src, dst in (('android', 'ios'), ('ios', 'android')):
            tr = df[df._platform == src].reset_index(drop=True)
            te = df[df._platform == dst].reset_index(drop=True)
            for model in ('xgb', 'ens_xe'):
                p = fit_predict(tr, te, fs, model, SEED + 10, labels)
                m = met(te['_app'].map({c: i for i, c in enumerate(labels)}).to_numpy(),
                        p, labels)
                results['evaluations'].append({
                    'pool': pool_name, 'train_platform': src, 'test_platform': dst,
                    'model': model, 'n_train': len(tr), 'n_test': len(te),
                    'capture_disjoint': True, 'metrics': m})
                print(f'EVAL {pool_name} {src}->{dst} {model}',
                      {k: m[k] for k in ('accuracy', 'macro_precision',
                                         'macro_recall', 'max_class_fpr')},
                      flush=True)

        # pooled within-platform control (sessions of one capture may cross
        # splits: single capture per app-platform; competition-oriented only)
        rng = random.Random(SEED)
        assign = {}
        for app in labels:
            ids = df[df._app == app]._observation_id.tolist()
            rng.shuffle(ids)
            n = len(ids)
            nt = int(n * 0.6)
            nv = int(n * 0.2)
            for x in ids[:nt]:
                assign[x] = 'train'
            for x in ids[nt:nt + nv]:
                assign[x] = 'validation'
            for x in ids[nt + nv:]:
                assign[x] = 'test'
        df['_pool_split'] = df['_observation_id'].map(assign)
        tr = df[df._pool_split == 'train'].reset_index(drop=True)
        va = df[df._pool_split == 'validation'].reset_index(drop=True)
        te = df[df._pool_split == 'test'].reset_index(drop=True)
        lid = {c: i for i, c in enumerate(labels)}
        for model in ('xgb', 'ens_xe'):
            pv = fit_predict(tr, va, fs, model, SEED + 10, labels)
            pt = fit_predict(tr, te, fs, model, SEED + 10, labels)
            mv = met(va['_app'].map(lid).to_numpy(), pv, labels)
            mt = met(te['_app'].map(lid).to_numpy(), pt, labels)
            results['evaluations'].append({
                'pool': pool_name, 'setting': 'pooled_within_platform_control',
                'model': model, 'n_train': len(tr), 'n_val': len(va),
                'n_test': len(te), 'capture_disjoint': False,
                'split_note': 'random session split; one capture per '
                              '(app,platform) so captures cross splits; '
                              'competition-oriented control only',
                'validation': mv, 'metrics': mt})
            print(f'CTRL {pool_name} {model} test',
                  {k: mt[k] for k in ('accuracy', 'macro_precision',
                                      'macro_recall', 'max_class_fpr')}, flush=True)

    best_cross = {}
    for pool_name in ps:
        cross = [e for e in results['evaluations']
                 if e.get('capture_disjoint') and e['pool'] == pool_name]
        if cross:
            b = max(cross, key=lambda e: (e['metrics']['macro_recall'],
                                          e['metrics']['accuracy']))
            best_cross[pool_name] = {
                'train->test': f"{b['train_platform']}->{b['test_platform']}",
                'model': b['model'], **{k: b['metrics'][k] for k in
                                        ('accuracy', 'macro_precision',
                                         'macro_recall', 'max_class_fpr')}}
    results['best_cross_platform'] = best_cross
    results['scope_note'] = (
        'If the competition means the TOOL runs on both OSes, this data-level '
        'cross-OS transfer is not a formal target; numbers reported as-is.')
    (OUT / 'crossplatform_results.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    rows = []
    for e in results['evaluations']:
        m = e['metrics']
        rows.append({'pool': e['pool'],
                     'setting': (e['setting'] if 'setting' in e else
                                 f"{e['train_platform']}->{e['test_platform']}"),
                     'model': e['model'], 'disjoint': e['capture_disjoint'],
                     **{k: m[k] for k in ('accuracy', 'macro_precision',
                                          'macro_recall', 'max_class_fpr')},
                     'four_pass': m['four_pass']})
    pd.DataFrame(rows).to_csv(OUT / 'leaderboard.csv', index=False)
    sha = hashlib.sha256((OUT / 'crossplatform_results.json').read_bytes()).hexdigest()
    (OUT / 'SHA256SUMS').write_text(f'{sha}  crossplatform_results.json\n')
    print('DONE', sha[:12], 'apps=', labels, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['extract', 'run'])
    args = ap.parse_args()
    if args.stage == 'extract':
        stage_extract()
    else:
        stage_run()


if __name__ == '__main__':
    main()
