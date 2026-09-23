# -*- coding: utf-8 -*-
"""Regression tests for the final-experiment closure round (2026-09-22).

Covers the two reusable pure-ish pieces of the closure scripts:
1. mixed16 capture-scoped endpoint aggregation is label-free, deterministic
   and never mixes files (the grouping contract behind the one-shot test
   opening);
2. behavior capture-relative normalization transforms each capture with its
   OWN statistics only (splits/files never mix) and is invariant to row
   order within a capture.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path('/workspace/Huawei')


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


agg_mod = _load('closure_mixed16_agg',
                'scripts/run_closure_mixed16_endpoint_agg_20260922.py')
beh_mod = _load('closure_behavior_capdisjoint',
                'scripts/run_closure_behavior_capdisjoint_20260922.py')


def _frame():
    return pd.DataFrame({
        '_source_file': ['a.pcap', 'a.pcap', 'a.pcap', 'b.pcap', 'b.pcap'],
        '_remote_ip': ['1.2.3.4', '1.2.3.4', '9.9.9.9', '1.2.3.4', '1.2.3.4'],
        '_remote_slash24': [0x01020300, 0x01020300, 0x09090900,
                            0x01020300, 0x01020300],
    })


def test_endpoint_aggregation_pools_within_file_and_endpoint():
    proba = np.array([
        [0.8, 0.2], [0.4, 0.6], [0.9, 0.1], [0.3, 0.7], [0.3, 0.7]])
    out = agg_mod.aggregate(proba, _frame(), 'file_ip')
    # a.pcap/1.2.3.4 pooled together; different file never pooled
    assert np.allclose(out[0], out[1])
    assert not np.allclose(out[0], out[3])
    assert np.allclose(out[2], proba[2])          # singleton group untouched


def test_endpoint_aggregation_none_is_identity():
    proba = np.arange(10, dtype=float).reshape(5, 2)
    assert np.allclose(agg_mod.aggregate(proba, _frame(), 'none'), proba)


def test_endpoint_aggregation_is_label_free_and_deterministic():
    proba = np.random.RandomState(0).rand(5, 2)
    df = _frame()
    df['_label'] = ['x', 'x', 'x', 'y', 'y']
    a = agg_mod.aggregate(proba, df, 'file_slash24')
    df['_label'] = ['z', 'z', 'z', 'w', 'w']      # relabel: must not change
    b = agg_mod.aggregate(proba, df, 'file_slash24')
    assert np.allclose(a, b)


def _bframe():
    cols = ['f1', 'f2']
    data = {
        '_source_file': ['capA'] * 3 + ['capB'] * 3,
        '_split': ['train'] * 3 + ['validation'] * 3,
    }
    data['f1'] = [10.0, 20.0, 30.0, 0.0, 0.0, 0.0]
    data['f2'] = [1.0, 1.0, 1.0, 5.0, 10.0, 15.0]
    return pd.DataFrame(data), cols


def test_capz_uses_only_own_capture_stats():
    d, cols = _bframe()
    tr, va, _ = beh_mod.apply_norm(d[d._split == 'train'].reset_index(drop=True),
                                   d[d._split == 'validation'].reset_index(drop=True),
                                   d.iloc[0:0], cols, 'capz')
    # capB has constant f1 -> z-score falls back to 0; f2 standardized by capB
    assert np.allclose(va['f1'].to_numpy(), 0.0)
    assert abs(va['f2'].to_numpy()[0] + va['f2'].to_numpy()[2]) < 1e-9
    assert not np.allclose(tr['f1'].to_numpy(), 0.0)   # capA has variance
    # a capture with identical values in both frames normalizes identically
    d2 = d.copy()
    d2['f1'] = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    tr2, va2, _ = beh_mod.apply_norm(
        d2[d2._split == 'train'].reset_index(drop=True),
        d2[d2._split == 'validation'].reset_index(drop=True),
        d2.iloc[0:0], cols, 'capz')
    assert np.allclose(tr2['f1'].to_numpy(), va2['f1'].to_numpy())


def test_caprank_is_within_capture_pct_rank():
    d, cols = _bframe()
    tr, va, _ = beh_mod.apply_norm(d[d._split == 'train'].reset_index(drop=True),
                                   d[d._split == 'validation'].reset_index(drop=True),
                                   d.iloc[0:0], cols, 'caprank')
    assert np.allclose(sorted(tr['f1'].to_numpy()), [1 / 3, 2 / 3, 1.0])
    assert np.allclose(va['f2'].to_numpy(), [1 / 3, 2 / 3, 1.0])


def test_raw_norm_is_identity():
    d, cols = _bframe()
    tr, va, _ = beh_mod.apply_norm(d[d._split == 'train'].reset_index(drop=True),
                                   d[d._split == 'validation'].reset_index(drop=True),
                                   d.iloc[0:0], cols, 'raw')
    assert np.allclose(tr['f1'].to_numpy(), [10.0, 20.0, 30.0])


def test_slash24_key_never_crosses_files():
    df = _frame()
    key = agg_mod.group_key(df, 'file_slash24')
    assert len(set(key)) == 3   # a.pcap/24, a.pcap/other24, b.pcap/24
