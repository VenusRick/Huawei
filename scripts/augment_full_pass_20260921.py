# -*- coding: utf-8 -*-
"""Train-side mid-session window augmentation test (dev/validation only).

For train_pool sessions longer than 128 packets, add ONE extra training
observation built from packets [64:128] of the same session (same label).
Validation/test are untouched; augmented rows get new observation ids and
never enter any split. Question: does the extra class signal lift the
weak classes' validation recall?
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/Huawei')
import importlib.util
spec = importlib.util.spec_from_file_location(
    'fp', '/workspace/Huawei/scripts/run_mixed_dataset_full_pass_20260921.py')
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)

from src.features.basic.feature_extractor import BasicFeatureExtractor
from src.features.advanced.advanced_extractor import AdvancedFeatureExtractor
from src.parser.pcap_reader import PCAPReader
from src.parser.session.session_manager import SessionManager

OUT = fp.OUT


def window_view(session, lo: int, hi: int):
    """Prefix-style view over packets[lo:hi] (same construction rules as the
    runtime prefix view: shallow copy, sliced packets, recomputed totals)."""
    s = copy.copy(session)
    s.packets = list(session.packets[lo:hi])
    s.total_fwd_packets = sum(1 for p in s.packets if p.direction == 1)
    s.total_bwd_packets = sum(1 for p in s.packets if p.direction == -1)
    s.total_fwd_bytes = sum(p.length for p in s.packets if p.direction == 1)
    s.total_bwd_bytes = sum(p.length for p in s.packets if p.direction == -1)
    if s.packets:
        last = s.packets[-1]
        s.end_time = last.timestamp
        s.last_activity_time = s.end_time
    return s


def build_augment(train: pd.DataFrame, lo: int = 64, hi: int = 128) -> pd.DataFrame:
    want = train[train['_full_packets'] > hi]
    by_file: Dict[str, set] = {}
    for _, r in want.iterrows():
        fname, idx = r['_observation_id'].rsplit(':', 1)
        by_file.setdefault(r['_source_file'], set()).add(int(idx))
    basic_ext = BasicFeatureExtractor()
    advanced_ext = AdvancedFeatureExtractor()
    rows: List[Dict[str, object]] = []
    for src, idxs in sorted(by_file.items()):
        sm = SessionManager()
        reader = PCAPReader(sm)
        for idx, session in enumerate(reader.read_pcap_generator(src, max_read_packets=None)):
            if idx not in idxs:
                continue
            meta = want[want['_observation_id'] == f'{Path(src).name}:{idx}'].iloc[0]
            view = window_view(session, lo, hi)
            row = {'_dataset': meta['_dataset'], '_class': meta['_class'],
                   '_label': meta['_label'], '_source_file': src,
                   '_observation_id': f'{Path(src).name}:w{lo}:{idx}',
                   '_full_packets': int(session.total_packets),
                   '_full_bytes': int(session.total_bytes)}
            row.update(basic_ext.extract_all(view))
            row.update(advanced_ext.extract_all(view))
            row.update(fp.sequence_features(view.packets))
            rows.append(row)
        print('AUG', Path(src).name, 'done', flush=True)
    aug = pd.DataFrame(rows)
    ids = set(train['_observation_id']) | set(aug['_observation_id'])
    if len(ids) != len(train) + len(aug):
        raise RuntimeError('augmentation id collision')
    return aug


def main() -> None:
    df = fp.load_split()
    train = df[df['_split'] == 'train_pool'].reset_index(drop=True)
    val = df[df['_split'] == 'validation'].reset_index(drop=True)
    labels = sorted(df['_label'].unique())
    fs_all = fp.common_shape_features(df.columns)
    seq = [f for f in fs_all if f.startswith('seq_')]

    aug_path = OUT / 'augment_windows.csv'
    if aug_path.exists():
        aug = pd.read_csv(aug_path)
    else:
        aug = build_augment(train)
        aug.to_csv(aug_path, index=False)
    print('augmented rows:', len(aug), 'by class:',
          aug.groupby('_class').size().to_dict(), flush=True)
    # sanity: no overlap with val/test ids
    split_ids = set(df['_observation_id'])
    assert not (set(aug['_observation_id']) & split_ids)

    tr_aug = pd.concat([train, aug.assign(_split='train_pool')],
                       ignore_index=True, sort=False)
    # union of columns; missing values are filled by clean() downstream
    for model in ('ens_xe', 'ens_all'):
        m0 = fp.evaluate(train, val, seq, model, 20260922 + 33, labels)
        m1 = fp.evaluate(tr_aug, val, seq, model, 20260922 + 33, labels)
        print(f'{model}: base acc={m0["accuracy"]:.4f} mR={m0["macro_recall"]:.4f} | '
              f'+win acc={m1["accuracy"]:.4f} mR={m1["macro_recall"]:.4f}', flush=True)
        weak = [(c, round(v['recall'], 3))
                for c, v in sorted(m1['per_class'].items(), key=lambda kv: kv[1]['recall'])[:6]]
        print('   weak after aug:', weak, flush=True)


if __name__ == '__main__':
    main()
