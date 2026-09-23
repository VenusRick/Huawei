# -*- coding: utf-8 -*-
"""TRAIN_THREADS 回归：默认值须收敛在 1..8，环境变量显式覆盖须生效。

src.config 在 import 时一次性读取环境变量（src/config.py），同进程内
reload 会受模块缓存与其他测试导入顺序干扰，故统一在子进程中求值。
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROBE = "from src.config import TRAIN_THREADS; print(TRAIN_THREADS)"


def _train_threads(env_value: Optional[str] = None) -> int:
    """在干净子进程中求值 TRAIN_THREADS，隔离 import 缓存与本进程环境。"""
    env = {k: v for k, v in os.environ.items() if k != "TRAIN_THREADS"}
    if env_value is not None:
        env["TRAIN_THREADS"] = env_value
    out = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=ROOT, env=env, check=True,
        capture_output=True, text=True, timeout=120,
    )
    return int(out.stdout.strip())


def test_default_bounded_1_to_8():
    # 未显式设置时取 min(8, cpu_count)，防 OpenMP 超额订阅；且恒 >= 1
    assert 1 <= _train_threads() <= 8


@pytest.mark.parametrize("raw,expected", [("3", 3), ("16", 16)])
def test_env_override_honored(raw: str, expected: int):
    # 显式覆盖直接生效：不受默认 8 上限约束（显式值视为运维知情）
    assert _train_threads(raw) == expected


@pytest.mark.parametrize("raw", ["0", "-4"])
def test_env_override_clamped_to_1(raw: str):
    # 低位非法值由 max(1, ...) 兜底，避免 0/负数传给 n_jobs
    assert _train_threads(raw) == 1
