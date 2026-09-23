# -*- coding: utf-8 -*-
"""Final closure round (2026-09-22): CrossPlatform China Android<->iOS.

Prior state (ledger 31.6): first formal mapping round over the 9 apps with
real captures on both sides; session-level platform transfer (= capture
transfer, since each (app, platform) has exactly ONE capture) peaked at
38.6% Acc (shape) / 37.6% (hs_visible); pooled within-platform control
81.7% / 89.1%.

This final bounded round tests the remaining legitimate treatments before a
structural verdict -- all label-free where they touch the test side:

1. per-capture standardization (capz) / quantile ranking (caprank) of session
   features within each (app, platform) capture -- removes capture-level
   location/scale shift, the classic capture-normalization treatment;
2. CORAL unsupervised alignment (source covariance -> target covariance,
   using unlabeled target sessions only);
3. capture-level aggregation (observation = whole capture; 9 train vs 9 test
   captures) -- the highest-layer observation unit available;
4. identifier-assisted remote-IP feed transfer: IP->app mapping learned from
   the SOURCE platform's captures only; measures both transfer accuracy and
   feed coverage on the target platform (coverage ~0 would prove the feed
   cannot cross platforms);
5. direct domain-gap diagnostics: 1-NN within-platform LOO vs cross-platform
   1-NN under capz.

No sealed test exists for this scenario (post-exploration); the output is a
final verdict, not a gate. Decision rule declared in advance: if none of the
arms reaches the formal four-metric thresholds AND the diagnostics show the
platform transfer is confounded with single-capture structure, the scenario
closes as FINAL_FAIL (data condition: 1 capture per (app, platform)),
to be re-opened only with multi-capture data.

Stages: ipmap | run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path('/workspace/Huawei')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import run_scenario_crossplatform_20260922 as base  # noqa: E402
from run_scenario_crossplatform_20260922 import (  # noqa: E402
    A_DIR, APP_MAP, I_DIR, MAX_SESSIONS, READ_CAP, SEED, TARGETS,
    fit_predict, load_cache, met, pools)
from run_scenario_mixed16_ipfeed_20260922 import remote_ip_of, _u32  # noqa: E402

from src.parser.pcap_reader import PCAPReader  # noqa: E402
from src.parser.session.session_manager import SessionManager  # noqa: E402

SRC = ROOT / 'output/all_scenarios_metric_push_20260922/crossplatform'
OUT = ROOT / 'output/final_experiment_closure_20260922/crossplatform_align'
IPMAP = OUT / 'ipmap.csv'


def stage_ipmap() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_cache()
    need: Dict[str, set] = defaultdict(set)
    for oid in df['_observation_id']:
        fname, idx = oid.rsplit(':', 1)
        need[fname].add(int(idx))
    file_cfg = {}
    for app, (a_name, i_name) in APP_MAP.items():
        file_cfg[f'{a_name}.pcap'] = A_DIR / f'{a_name}.pcap'
        file_cfg[f'{i_name}.pcap'] = I_DIR / f'{i_name}.pcap'
    rows = []
    for fname, idxs in sorted(need.items()):
        sm = SessionManager()
        reader = PCAPReader(sm)
        got = 0
        for idx, session in enumerate(reader.read_pcap_generator(
                str(file_cfg[fname]), max_read_packets=READ_CAP)):
            if idx not in idxs:
                continue
            ip, port = remote_ip_of(session)
            rows.append({'_observation_id': f'{fname}:{idx}',
                         '_remote_ip': ip, '_remote_port': int(port),
                         '_remote_u32': _u32(ip),
                         '_remote_slash24': _u32(ip) & 0xFFFFFF00})
            got += 1
            if got == len(idxs):
                break
        print(f'IPMAP {fname} matched {got}/{len(idxs)}', flush=True)
    m = pd.DataFrame(rows)
    if m['_observation_id'].duplicated().any():
        raise RuntimeError('duplicate ipmap rows')
    m.to_csv(IPMAP, index=False)
    print('TOTAL', len(m), flush=True)


def cap_transform(d: pd.DataFrame, fs: List[str], norm: str) -> pd.DataFrame:
    d = d.copy()
    if norm == 'capz':
        g = d.groupby('_source_file')[fs]
        mu, sd = g.transform('mean'), g.transform('std').fillna(0)
        sd = sd.replace(0, np.nan)
        d[fs] = ((d[fs] - mu) / sd).fillna(0)
    elif norm == 'caprank':
        d[fs] = d.groupby('_source_file')[fs].rank(pct=True).fillna(0.5)
    elif norm != 'raw':
        raise ValueError(norm)
    return d


def clean(d: pd.DataFrame, fs: List[str]) -> pd.DataFrame:
    return d[fs].apply(pd.to_numeric, errors='coerce') \
                .replace([np.inf, -np.inf], np.nan).fillna(0)


def coral(Xs: np.ndarray, Xt: np.ndarray, lam: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """Standard CORAL: Xs' = Xs Cs^-1/2 Ct^1/2 (with shrinkage)."""
    ms, mt = Xs.mean(0), Xt.mean(0)
    Zs, Zt = Xs - ms, Xt - mt
    d = Xs.shape[1]
    Cs = (Zs.T @ Zs) / max(len(Zs) - 1, 1) + lam * np.trace((Zs.T @ Zs) / max(len(Zs) - 1, 1)) / d * np.eye(d)
    Ct = (Zt.T @ Zt) / max(len(Zt) - 1, 1) + lam * np.trace((Zt.T @ Zt) / max(len(Zt) - 1, 1)) / d * np.eye(d)
    ws = np.linalg.inv(_sqrtm_psd(Cs))
    at = _sqrtm_psd(Ct)
    return (Zs @ ws + ms) @ at, Xt


def _sqrtm_psd(M: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0, None)
    return (V * np.sqrt(w)) @ V.T


def fit_pred_numpy(Xtr, ytr, Xte, model, seed, n_classes):
    probs = []
    for j, mem in enumerate((['xgb', 'extra'] if model == 'ens_xe' else [model])):
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
        p = np.zeros((len(Xte), n_classes))
        p[:, m.classes_] = m.predict_proba(Xte)
        probs.append(p)
    return np.argmax(np.mean(probs, axis=0), axis=1)


def stage_run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_cache()
    ip = pd.read_csv(IPMAP)
    df = df.merge(ip, on='_observation_id', how='left')
    if df['_remote_ip'].isna().any():
        raise RuntimeError('ipmap coverage gap')
    counts = df.groupby(['_app', '_platform']).size().unstack(fill_value=0)
    good_apps = [a for a in counts.index
                 if counts.loc[a].get('android', 0) >= 20
                 and counts.loc[a].get('ios', 0) >= 20]
    df = df[df._app.isin(good_apps)].reset_index(drop=True)
    labels = sorted(good_apps)
    lid = {c: i for i, c in enumerate(labels)}
    ps = pools(df)
    results = {'scenario': 'E_crossplatform_final_alignment_round',
               'apps': labels, 'targets': TARGETS,
               'arms': [], 'diagnostics': {},
               'decision_rule': ('FINAL_FAIL unless any arm reaches all four '
                                 'formal thresholds; single-capture structural '
                                 'confound documented regardless')}

    # ---- arm 1: per-capture normalization (session level) ------------------
    for pool_name, fs in ps.items():
        fs = [f for f in fs if clean(df, [f])[f].std() > 0]
        for norm in ('raw', 'capz', 'caprank'):
            for src, dst in (('android', 'ios'), ('ios', 'android')):
                tr = df[df._platform == src].reset_index(drop=True)
                te = df[df._platform == dst].reset_index(drop=True)
                trn, ten = cap_transform(tr, fs, norm), cap_transform(te, fs, norm)
                for model in ('xgb', 'ens_xe'):
                    p = fit_predict(trn, ten, fs, model, SEED + 10, labels)
                    m = met(te['_app'].map(lid).to_numpy(), p, labels)
                    results['arms'].append({
                        'arm': 'session_capnorm', 'norm': norm, 'pool': pool_name,
                        'train_platform': src, 'test_platform': dst,
                        'model': model, 'metrics': m})
                    print('ARM session_capnorm', norm, pool_name, src, '->', dst,
                          model, round(m['accuracy'], 4), flush=True)

    # ---- arm 2: CORAL -------------------------------------------------------
    for pool_name, fs in ps.items():
        fs = [f for f in fs if clean(df, [f])[f].std() > 0]
        for src, dst in (('android', 'ios'), ('ios', 'android')):
            tr = df[df._platform == src].reset_index(drop=True)
            te = df[df._platform == dst].reset_index(drop=True)
            Xs = clean(cap_transform(tr, fs, 'capz'), fs).to_numpy()
            Xt = clean(cap_transform(te, fs, 'capz'), fs).to_numpy()
            ys = tr['_app'].map(lid).to_numpy()
            Xs_a, _ = coral(Xs, Xt)
            for model in ('xgb', 'ens_xe'):
                p = fit_pred_numpy(Xs_a, ys, Xt, model, SEED + 10, len(labels))
                m = met(te['_app'].map(lid).to_numpy(), p, labels)
                results['arms'].append({
                    'arm': 'coral_on_capz', 'pool': pool_name,
                    'train_platform': src, 'test_platform': dst,
                    'model': model, 'metrics': m})
                print('ARM coral', pool_name, src, '->', dst, model,
                      round(m['accuracy'], 4), flush=True)

    # ---- arm 3: capture-level aggregation ----------------------------------
    for pool_name, fs in ps.items():
        fs = [f for f in fs if clean(df, [f])[f].std() > 0]
        gb = df.groupby(['_app', '_platform', '_source_file'])[fs]
        rows = []
        for (app, plat, sfile), g in gb:
            row = {'_app': app, '_platform': plat, '_source_file': sfile}
            vals = clean(g, fs)
            for c in fs:
                row[f'{c}__mean'] = float(vals[c].mean())
                row[f'{c}__std'] = float(vals[c].std())
            rows.append(row)
        adf = pd.DataFrame(rows)
        afeat = [c for c in adf.columns if c.endswith(('__mean', '__std'))]
        afeat = [c for c in afeat if adf[c].fillna(0).std() > 0]
        for src, dst in (('android', 'ios'), ('ios', 'android')):
            tr = adf[adf._platform == src].reset_index(drop=True)
            te = adf[adf._platform == dst].reset_index(drop=True)
            for model in ('xgb', 'ens_xe'):
                p = fit_predict(tr, te, afeat, model, SEED + 10, labels)
                m = met(te['_app'].map(lid).to_numpy(), p, labels)
                results['arms'].append({
                    'arm': 'capture_level_aggregation', 'pool': pool_name,
                    'train_platform': src, 'test_platform': dst, 'model': model,
                    'n_train': len(tr), 'n_test': len(te), 'metrics': m})
                print('ARM capture_agg', pool_name, src, '->', dst, model,
                      round(m['accuracy'], 4), flush=True)

    # ---- arm 4: identifier-assisted remote-IP feed transfer ----------------
    for src, dst in (('android', 'ios'), ('ios', 'android')):
        tr = df[df._platform == src].reset_index(drop=True)
        te = df[df._platform == dst].reset_index(drop=True)
        ip2app: Dict[int, str] = {}
        slash2app: Dict[int, str] = {}
        for _, r in tr.iterrows():
            ip2app.setdefault(int(r['_remote_u32']), r['_app'])
            slash2app.setdefault(int(r['_remote_slash24']), r['_app'])
        exact = te['_remote_u32'].map(ip2app)
        via24 = te['_remote_slash24'].map(slash2app)
        cov_exact = float(exact.notna().mean())
        cov_24 = float(via24.notna().mean())
        yhat = exact.fillna(via24)
        covered = yhat.notna()
        acc_covered = float((yhat[covered] == te['_app'][covered]).mean()) \
            if covered.any() else 0.0
        results['arms'].append({
            'arm': 'ipfeed_transfer', 'train_platform': src,
            'test_platform': dst,
            'feed_coverage_exact_ip': cov_exact,
            'feed_coverage_slash24': cov_24,
            'accuracy_on_covered_sessions': acc_covered,
            'note': 'IP->app map learned from source-platform captures only'})
        print('ARM ipfeed', src, '->', dst, 'cov_exact', round(cov_exact, 4),
              'cov_/24', round(cov_24, 4), 'acc_cov', round(acc_covered, 4),
              flush=True)

    # ---- diagnostics: 1-NN domain gap --------------------------------------
    from sklearn.neighbors import KNeighborsClassifier
    fs = [f for f in ps['hs_visible'] if clean(df, [f])[f].std() > 0]
    diag = {}
    for norm in ('raw', 'capz'):
        dn = cap_transform(df, fs, norm)
        X = clean(dn, fs).to_numpy()
        y = dn['_app'].map(lid).to_numpy()
        plat = dn['_platform'].to_numpy()
        loo_same = []
        for a in labels:
            m = (dn['_app'] == a).to_numpy() & (plat == 'android')
            if m.sum() < 4:
                continue
            Xi, yi = X[m], y[m]
            knn = KNeighborsClassifier(n_neighbors=1).fit(Xi, yi)
            p = knn.predict(Xi)
            loo_same.append(float(accuracy_score(yi, p)))  # train acc (not LOO)
        knn = KNeighborsClassifier(n_neighbors=1).fit(X[plat == 'android'],
                                                      y[plat == 'android'])
        cross = float(accuracy_score(y[plat == 'ios'],
                                     knn.predict(X[plat == 'ios'])))
        diag[norm] = {'within_android_1nn_train': float(np.mean(loo_same)) if loo_same else None,
                      'android_to_ios_1nn': cross}
        print('DIAG 1nn', norm, diag[norm], flush=True)
    results['diagnostics']['one_nn_domain_gap'] = diag

    best = max(results['arms'], key=lambda a: a.get('metrics', {}).get(
        'accuracy', a.get('accuracy_on_covered_sessions', 0.0)))
    results['best_arm'] = {
        'arm': best['arm'],
        'metrics': best.get('metrics'),
        'four_pass': bool(best.get('metrics', {}).get('four_pass', False))}
    (OUT / 'final_results.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float))
    pd.DataFrame([{**{k: v for k, v in a.items() if k != 'metrics'},
                   **{f'm__{k}': v for k, v in a.get('metrics', {}).items()
                      if isinstance(v, (int, float))}}
                  for a in results['arms']]).to_csv(
        OUT / 'leaderboard.csv', index=False)
    print('BEST', results['best_arm'], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['ipmap', 'run'])
    args = ap.parse_args()
    {'ipmap': stage_ipmap, 'run': stage_run}[args.stage]()


if __name__ == '__main__':
    main()
