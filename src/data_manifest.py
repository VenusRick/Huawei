# -*- coding: utf-8 -*-
"""数据清单服务（统一契约层 P0）。

全工程唯一的数据入口契约：
    Manifest -> Observation -> FeatureRecord -> RuleBundle -> Prediction -> Evaluation

应用识别 / 精细行为识别 / 匿名工具识别共用本模块，区别只存在于
task config + manifest + feature spec + rule bundle，不建三套系统。

要点：
- 试次清单（trial manifest）是正式真值来源；all_data 文件名前缀只记为
  candidate_label，不算正式真值（建议.md 第四点）。
- 数据划分在本层写死：先按 trial/capture group 分组，再产生窗口；
  同一 capture_id / trial_id 永不跨集合；同设备一次连续采集不拆分；
  平台留出实验单独配置（防同试次连续窗口泄漏，建议.md 第五点）。
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field, fields as dc_fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 正式试次 manifest 必填字段（P1.5 schema，缺一即校验失败）
TRIAL_REQUIRED_FIELDS = (
    "capture_id", "trial_id", "platform", "device_id", "app",
    "behavior", "pcap", "label_source",
)

PLATFORM_CHOICES = ("android", "ios", "pc")
LABEL_SOURCE_CHOICES = ("ui_event", "automation", "manual_verified",
                         "dataset_official")   # 数据集官方分文件标注
SPLIT_CHOICES = ("train", "validation", "test")

# label_source 可信度排序（越靠后越可信，供下游筛选参考，不改变真值）
LABEL_SOURCE_TRUST = {"ui_event": 1, "automation": 2,
                       "dataset_official": 3, "manual_verified": 3}

# 任务类型 -> 试次里承载标签的字段（应用/行为/工具同一体系）
TASK_LABEL_FIELD = {
    "app": "app",
    "behavior": "behavior",
    "tool": "app",   # 匿名工具沿用 app 字段（Tor/Psiphon/Session 记在 app）
}


@dataclass
class TrialRecord:
    """一次受控行为试次的清单记录（正式真值单位）。"""
    capture_id: str
    trial_id: str
    platform: str                    # android | ios | pc
    device_id: str
    app: str
    behavior: str
    pcap: str                        # 相对 manifest 的路径或绝对路径
    label_source: str                # ui_event | automation | manual_verified
    app_version: str = ""
    behavior_start: Optional[float] = None   # 相对 pcap 首包的秒数
    behavior_end: Optional[float] = None
    network_env: str = ""
    terminal_ip: str = ""        # 可见终端IP（M2 行为窗口方向基准）
    split: Optional[str] = None      # train | validation | test（未分时为 None）
    candidate_label: str = ""        # 文件名前缀派生的候选标签，非真值

    def label(self, task: str) -> str:
        """按任务取标签字段（应用/行为/工具统一入口）。"""
        return getattr(self, TASK_LABEL_FIELD[task])

    def to_dict(self) -> Dict:
        return asdict(self)


class ManifestError(ValueError):
    """manifest 校验失败（正式输入错误必须显式失败，不得静默降级）。"""


def load_manifest(path: str) -> List[TrialRecord]:
    """读取 jsonl 试次清单；任一行缺必填字段/枚举非法即抛 ManifestError。"""
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"manifest 不存在: {path}")
    records: List[TrialRecord] = []
    with open(p, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ManifestError(f"第{lineno}行不是合法JSON: {e}") from e
            missing = [k for k in TRIAL_REQUIRED_FIELDS if not obj.get(k)]
            if missing:
                raise ManifestError(f"第{lineno}行缺必填字段: {missing}")
            if obj["platform"] not in PLATFORM_CHOICES:
                raise ManifestError(
                    f"第{lineno}行 platform 非法: {obj['platform']}"
                    f"（合法值 {list(PLATFORM_CHOICES)}）")
            if obj["label_source"] not in LABEL_SOURCE_CHOICES:
                raise ManifestError(
                    f"第{lineno}行 label_source 非法: {obj['label_source']}")
            if obj.get("split"):
                # 别名归一化：val/valid -> validation
                if obj["split"] in ("val", "valid"):
                    obj["split"] = "validation"
                if obj["split"] not in SPLIT_CHOICES:
                    raise ManifestError(f"第{lineno}行 split 非法: {obj['split']}")
            if (obj.get("behavior_start") is None) != (obj.get("behavior_end") is None):
                raise ManifestError(
                    f"第{lineno}行 behavior_start/behavior_end 必须同时给出")
            if (obj.get("behavior_start") is not None
                    and obj["behavior_end"] < obj["behavior_start"]):
                raise ManifestError(f"第{lineno}行 behavior_end 早于 behavior_start")
            known = {f.name for f in dc_fields(TrialRecord)}
            records.append(TrialRecord(**{k: v for k, v in obj.items() if k in known}))
    if not records:
        raise ManifestError(f"manifest 为空: {path}")
    return records


def validate_manifest(records: List[TrialRecord], manifest_dir: str = ".",
                      task: str = "behavior") -> Dict:
    """正式挖掘前的数据体检：pcap 存在性、标签完整性、采集组统计。

    返回统计 dict；发现阻断性问题时抛 ManifestError（对应 --mode validate-data）。
    """
    problems: List[str] = []
    base = Path(manifest_dir)
    for r in records:
        pcap_path = Path(r.pcap)
        if not pcap_path.is_absolute():
            pcap_path = base / r.pcap
        if not pcap_path.exists():
            problems.append(f"pcap 缺失: {r.trial_id} -> {pcap_path}")
    if task == "behavior":
        for r in records:
            if r.behavior_start is None:
                problems.append(
                    f"behavior 任务缺行为时间区间: {r.trial_id}"
                    "（无区间无法生成窗口真值）")
    if problems:
        raise ManifestError("; ".join(problems[:10]) +
                            (f"...共{len(problems)}项" if len(problems) > 10 else ""))
    # 采集组统计（按 capture_id；同组永不跨 split 由 split_by_group 保证）
    groups: Dict[str, set] = {}
    for r in records:
        groups.setdefault(r.capture_id, set()).add(r.split or "unassigned")
    return {
        "trials": len(records),
        "captures": len(groups),
        "platforms": sorted({r.platform for r in records}),
        "apps": sorted({r.app for r in records}),
        "behaviors": sorted({r.behavior for r in records}),
        "label_sources": sorted({r.label_source for r in records}),
    }


def build_label_vocab(records: List[TrialRecord], task: str) -> Dict:
    """构建标签词表（应用/行为/工具同一体系，排序稳定分配 ID）。

    返回 {"task", "label_map": {name: id}, "names": [...]}；
    split 各集合沿用全局 label_map，不重编号（M1 R27 语义）。
    """
    names = sorted({r.label(task) for r in records})
    if not names:
        raise ManifestError(f"task={task} 下无可用标签（清单为空？）")
    return {"task": task, "label_map": {n: i for i, n in enumerate(names)},
            "names": names}


def split_by_group(records: List[TrialRecord],
                   ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
                   seed: int = 42) -> List[TrialRecord]:
    """按采集组分 train/validation/test（写死的防泄漏规则）。

    - 分组单位是 capture_id（同设备一次连续采集不拆分）；
    - 同一 capture_id 的全部 trial 落进同一集合；
    - 平台留出（held-out platform）不在此做，由调用方按 platform 过滤后
      单独配置，避免与随机分组语义混淆。
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ManifestError(f"ratios 之和必须为 1: {ratios}")
    groups: Dict[str, List[TrialRecord]] = {}
    for r in records:
        groups.setdefault(r.capture_id, []).append(r)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    assign: Dict[str, str] = {}
    for i, k in enumerate(keys):
        if i < n_train:
            assign[k] = "train"
        elif i < n_train + n_val:
            assign[k] = "validation"
        else:
            assign[k] = "test"
    out = []
    for k in keys:
        for r in groups[k]:
            r.split = assign[k]
            out.append(r)
    return out


def write_manifest(records: List[TrialRecord], path: str) -> None:
    """写出 jsonl 清单（保持字段顺序，便于 diff 与人工修订）。"""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def label_window(record: "TrialRecord", window_start: float,
                window_end: float, min_overlap: float = 0.5) -> str:
    """行为时间区间 -> 窗口标签（评价配置的一部分，M2）。

    窗口与行为区间的重叠占窗口时长比例 >= min_overlap 记该行为，
    否则记 background（纯背景窗口）。区间缺失时一律 background。
    """
    if record.behavior_start is None or record.behavior_end is None:
        return "background"
    overlap = (min(window_end, record.behavior_end)
               - max(window_start, record.behavior_start))
    dur = window_end - window_start
    if dur <= 0:
        return "background"
    return record.behavior if overlap / dur >= min_overlap else "background"


# ---------------------------------------------------------------------------
# candidate_label 盘点（all_data 文件名前缀 -> 候选标签，非正式真值）
# ---------------------------------------------------------------------------
_PREFIX_RE = re.compile(r"^([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*?)-")


def discover_candidate_labels(data_dir: str,
                              patterns: Optional[Dict[str, str]] = None
                              ) -> List[Dict]:
    """扫描目录内 pcap 文件名前缀，产出 candidate_label 盘点表。

    patterns: 手工复核后的前缀->标签映射（优先）；缺省时退化为首个
    '字母数字连字符段' 前缀，且结果仅作 candidate，不进正式真值。
    注意 pcap 后缀只剥一次（历史坑：双重切除导致 .p cap 误判）。
    """
    base = Path(data_dir)
    out: List[Dict] = []
    for p in sorted(base.glob("*.pcap")) + sorted(base.glob("*.pcapng")):
        name = p.name
        if patterns:
            for prefix, label in sorted(patterns.items(),
                                        key=lambda kv: -len(kv[0])):
                if name.startswith(prefix):
                    out.append({"pcap": str(p), "candidate_label": label,
                                "source": "manual_pattern"})
                    break
            else:
                out.append({"pcap": str(p), "candidate_label": "",
                            "source": "unmatched"})
        else:
            m = _PREFIX_RE.match(name)
            label = m.group(1) if m else ""
            out.append({"pcap": str(p), "candidate_label": label,
                        "source": "filename_prefix"})
    return out
