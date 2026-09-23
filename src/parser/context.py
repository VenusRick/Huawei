# -*- coding: utf-8 -*-
"""终端行为上下文聚合（M2）：包 -> 终端时间窗 -> 连接/流段 -> 窗口观测。

只做组织，不做分类（建议.md 第二点）：
    packet 事件流
      -> 按 (capture_id, 可见终端) 维护有限时间窗（首版 5s 窗 / 1s 步长，入 bundle）
      -> 窗内每条连接按活动间隔切流段（方向相对选定终端，不跨连接混用方向）
      -> 固定槽位 max_segments=3（起始时间+首包序号稳定排序）
         overflow.aggregate_remaining=true：剩余流段进窗口级统计，不静默丢弃
      -> 窗口观测交给 WindowFeatureBuilder（FeatureRecord 的一部分）

红线：
- 各窗只使用本窗可见信息，不用未来包；窗口在配置截止时即可产出；
- 窗口总量统计使用全部纳入窗口的包，不因槽位展示漏计。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_CONTEXT_CONFIG = {
    "window_sec": 5.0,
    "step_sec": 1.0,
    "max_segments": 3,
    "ordering": "start_time_then_first_packet_index",
    "segment_gap_sec": 2.0,      # 窗内连接的活动间隔超过该值切新流段
    "overflow": {"aggregate_remaining": True},
}


@dataclass
class _SegmentState:
    """窗内一条连接上的连续活动簇（相对窗口坐标）。"""
    conn_key: str
    start_ts: float
    end_ts: float
    first_pkt_index: int          # 全捕获内首包序号（稳定排序次键）
    pkt_count: int = 0
    up_bytes: int = 0             # 相对选定终端的上行
    down_bytes: int = 0
    up_pkts: int = 0
    down_pkts: int = 0
    lengths: List[int] = field(default_factory=list)

    @property
    def direction_changes(self) -> int:
        return 0  # 由 builder 从包序重算（此处仅聚合量）


class BehaviorContextTracker:
    """逐包事件聚合器：feed_packet 即时更新，窗口过期即发射。"""

    def __init__(self, config: Optional[dict] = None,
                 terminal: Optional[str] = None,
                 capture_id: str = ""):
        cfg = dict(DEFAULT_CONTEXT_CONFIG)
        if config:
            for k, v in config.items():
                if isinstance(v, dict):
                    cfg.setdefault(k, {}).update(v)
                else:
                    cfg[k] = v
        self.cfg = cfg
        self.window_sec = float(cfg["window_sec"])
        self.step_sec = float(cfg["step_sec"])
        self.max_segments = int(cfg["max_segments"])
        self.segment_gap_sec = float(cfg["segment_gap_sec"])
        self.aggregate_overflow = bool(
            cfg.get("overflow", {}).get("aggregate_remaining", True))
        self.terminal = terminal
        self.capture_id = capture_id
        # 窗口发射回调：on_window(window_dict)
        self.on_window: Optional[Callable[[dict], None]] = None
        # 活动缓冲（ts, src_ip, dst_ip, canonical, length, pkt_index, up_flag）
        self._events: List[Tuple[float, str, str, str, int, int, int]] = []
        self._pkt_index = 0
        self._next_window_start = 0.0
        self._t0_seen = False
        self._windows_emitted = 0

    # ------------------------------------------------------------------
    def feed_packet(self, ts: float, src_ip: str, dst_ip: str,
                    canonical: str, length: int,
                    terminal: Optional[str] = None) -> None:
        """喂一个包；terminal 未显式给定时用首包 src（首观测方向语义）。"""
        if terminal is not None and self.terminal is None:
            self.terminal = terminal
        if not self._t0_seen:
            self._t0_seen = True
            self._origin_ts = ts
        rel = ts - self._origin_ts
        term = self.terminal or src_ip
        up = 1 if src_ip == term else 0
        self._events.append(
            (rel, src_ip, dst_ip, canonical, length, self._pkt_index, up))
        self._pkt_index += 1
        # 窗口发射：当前包时间已越过 [t0, t0+w) 的尾部 -> 该窗再无可能收包
        while rel >= self._next_window_start + self.window_sec:
            self._emit_one(self._next_window_start)
            self._next_window_start += self.step_sec

    def finalize(self) -> None:
        """截止冲刷：剩余未发射窗口按当前可见信息产出（不用未来包）。"""
        if not self._t0_seen:
            return
        rel = self._events[-1][0] if self._events else 0.0
        while self._next_window_start <= rel:
            self._emit_one(self._next_window_start)
            self._next_window_start += self.step_sec

    # ------------------------------------------------------------------
    def _emit_one(self, t0: float) -> None:
        win = self._build_window(t0)
        if win is not None and self.on_window is not None:
            self.on_window(win)
            self._windows_emitted += 1

    def _build_window(self, t0: float) -> Optional[dict]:
        t1 = t0 + self.window_sec
        evs = [e for e in self._events if t0 <= e[0] < t1]
        if not evs:
            return None
        truncated_prefix = t0 > 0.0   # 窗口起点晚于捕获首包：R30 截断前缀标记
        # 每条连接按活动间隔切流段
        segments: List[_SegmentState] = []
        last_by_conn: Dict[str, _SegmentState] = {}
        for (rel, src, dst, canonical, length, pidx, up) in evs:
            seg = last_by_conn.get(canonical)
            if seg is None or rel - seg.end_ts > self.segment_gap_sec:
                seg = _SegmentState(conn_key=canonical, start_ts=rel,
                                    end_ts=rel, first_pkt_index=pidx)
                segments.append(seg)
            seg.end_ts = rel
            seg.pkt_count += 1
            if up:
                seg.up_bytes += length
                seg.up_pkts += 1
            else:
                seg.down_bytes += length
                seg.down_pkts += 1
            seg.lengths.append(length)
            last_by_conn[canonical] = seg
        # 固定槽位：起始时间 + 首包序号稳定排序
        segments.sort(key=lambda s: (s.start_ts, s.first_pkt_index))
        k = self.max_segments
        slotted, overflow = segments[:k], segments[k:]
        conns = {s.conn_key for s in segments}
        return {
            "capture_id": self.capture_id,
            "terminal": self.terminal or evs[0][1],
            "window_start": t0,
            "window_end": t1,
            "truncated_prefix": truncated_prefix,
            "n_packets": len(evs),
            "n_connections": len(conns),
            "n_segments": len(segments),
            "n_overflow_segments": len(overflow),
            "segments": slotted,
            "overflow": overflow if self.aggregate_overflow else [],
            "events": evs,          # 供 builder 计算方向变化等窗级统计
        }


class WindowFeatureBuilder:
    """窗口观测 -> 特征 dict（window_* / seg{i}_* / rel_{i}{j}_* 命名）。

    全部为标量，进共享 FeatureSpec；缺槽位输出 None（有效性标记），
    缺失不零填（R22 语义），由训练侧显式处理。
    """

    PREFIX = "behavior"

    def build(self, win: dict) -> Dict[str, Optional[float]]:
        f: Dict[str, Optional[float]] = {}
        p = self.PREFIX
        # ---- 窗级（用全部包，不因槽位漏计） ----
        evs = win["events"]
        ups = sum(1 for e in evs if e[6])
        downs = len(evs) - ups
        up_bytes = sum(e[4] for e in evs if e[6])
        down_bytes = sum(e[4] for e in evs if not e[6])
        # 方向变化（按包序，全窗）
        changes = sum(1 for a, b in zip(evs, evs[1:]) if a[6] != b[6])
        span = max(e[0] for e in evs) - min(e[0] for e in evs) if evs else 0.0
        f[f"{p}_win_n_conn"] = float(win["n_connections"])
        f[f"{p}_win_n_seg"] = float(win["n_segments"])
        f[f"{p}_win_n_pkt"] = float(win["n_packets"])
        f[f"{p}_win_dir_changes"] = float(changes)
        f[f"{p}_win_active_span"] = span
        f[f"{p}_win_idle"] = max(
            0.0, win["window_end"] - win["window_start"] - span)
        f[f"{p}_win_up_ratio"] = (ups / len(evs)) if evs else None
        f[f"{p}_win_up_byte_ratio"] = (
            up_bytes / (up_bytes + down_bytes)
            if (up_bytes + down_bytes) > 0 else None)
        f[f"{p}_win_overflow_n_seg"] = float(win["n_overflow_segments"])
        if win["overflow"]:
            ob = sum(s.up_bytes + s.down_bytes for s in win["overflow"])
            tot = ob + sum(s.up_bytes + s.down_bytes for s in win["segments"])
            f[f"{p}_win_overflow_byte_ratio"] = ob / tot if tot > 0 else None
        else:
            f[f"{p}_win_overflow_byte_ratio"] = 0.0
        # ---- 段级槽位 ----
        for i in range(3):   # 契约固定三槽命名；max_segments 可配但特征名稳定
            name = f"{p}_seg{i}"
            if i < len(win["segments"]):
                s = win["segments"][i]
                total_b = s.up_bytes + s.down_bytes
                mean_len = sum(s.lengths) / len(s.lengths) if s.lengths else None
                var = (sum((x - (mean_len or 0)) ** 2 for x in s.lengths)
                       / len(s.lengths)) if s.lengths else None
                f[f"{name}_valid"] = 1.0
                f[f"{name}_duration"] = s.end_ts - s.start_ts
                f[f"{name}_pkt"] = float(s.pkt_count)
                f[f"{name}_up_ratio"] = (
                    s.up_pkts / s.pkt_count) if s.pkt_count else None
                f[f"{name}_up_byte_ratio"] = (
                    s.up_bytes / total_b) if total_b > 0 else None
                f[f"{name}_mean_len"] = mean_len
                f[f"{name}_std_len"] = var ** 0.5 if var is not None else None
            else:
                for suffix in ("valid", "duration", "pkt", "up_ratio",
                               "up_byte_ratio", "mean_len", "std_len"):
                    f[f"{name}_{suffix}"] = None
        # ---- 段间关系（绑定相邻槽位对，不拼不相关段） ----
        for i, j in ((0, 1), (1, 2)):
            name = f"{p}_rel{i}{j}"
            a = win["segments"][i] if i < len(win["segments"]) else None
            b = win["segments"][j] if j < len(win["segments"]) else None
            if a is None or b is None:
                for suffix in ("gap", "overlap", "overlap_ratio",
                               "cross_conn", "byte_ratio"):
                    f[f"{name}_{suffix}"] = None
                continue
            gap = b.start_ts - a.end_ts
            overlap = max(0.0, min(a.end_ts, b.end_ts) - max(a.start_ts, b.start_ts))
            denom = max(a.end_ts - a.start_ts, b.end_ts - b.start_ts, 1e-9)
            ba = a.up_bytes + a.down_bytes
            bb = b.up_bytes + b.down_bytes
            f[f"{name}_gap"] = gap
            f[f"{name}_overlap"] = overlap
            f[f"{name}_overlap_ratio"] = overlap / denom
            f[f"{name}_cross_conn"] = 0.0 if a.conn_key == b.conn_key else 1.0
            f[f"{name}_byte_ratio"] = (bb / ba) if ba > 0 else None
        return f


class TunnelWindowFeatureBuilder:
    """15s tunnel window -> 64 low-cost, identifier-free scalar features.

    The feature family deliberately emphasizes packet-size distribution and
    ACK/cell/MTU-like peaks because encrypted bulk transfer is the residual hard
    case in ISCXTor. It uses only window-visible sizes, timing and directions.
    """

    PREFIX = "tunnel"
    SIZE_BINS = (0, 64, 128, 256, 512, 768, 1024, 1280, 1601)

    @staticmethod
    def _stats(values: List[float]) -> List[float]:
        if not values:
            return [0.0] * 8
        a = sorted(float(x) for x in values)
        n = len(a)
        mean = sum(a) / n
        var = sum((x - mean) ** 2 for x in a) / n
        def q(frac: float) -> float:
            if n == 1:
                return a[0]
            pos = frac * (n - 1)
            lo = int(math.floor(pos)); hi = int(math.ceil(pos))
            if lo == hi:
                return a[lo]
            w = pos - lo
            return a[lo] * (1 - w) + a[hi] * w
        return [mean, var ** 0.5, a[0], a[-1], q(.5), q(.25), q(.75), q(.9)]

    @staticmethod
    def _entropy(values: List[int]) -> float:
        if not values:
            return 0.0
        c = Counter(values); n = float(len(values))
        return -sum((v / n) * math.log(v / n, 2) for v in c.values())

    def build(self, win: dict) -> Dict[str, float]:
        p = self.PREFIX
        evs = list(win.get("events") or [])
        lengths = [int(e[4]) for e in evs]
        rel = [float(e[0]) for e in evs]
        ups = [int(e[6]) for e in evs]
        up_lengths = [l for l, u in zip(lengths, ups) if u]
        down_lengths = [l for l, u in zip(lengths, ups) if not u]
        iats = [max(0.0, b - a) for a, b in zip(rel, rel[1:])]
        changes = sum(1 for a, b in zip(ups, ups[1:]) if a != b)
        up_b = sum(up_lengths); down_b = sum(down_lengths)
        total_b = up_b + down_b
        span = (max(rel) - min(rel)) if rel else 0.0
        win_dur = max(float(win["window_end"] - win["window_start"]), 1e-9)

        f: Dict[str, float] = {}
        # 14 general counters/ratios
        base = {
            "n_pkt": len(lengths), "n_conn": win.get("n_connections", 0),
            "n_seg": win.get("n_segments", 0), "active_span": span,
            "idle": max(0.0, win_dur - span), "total_bytes": total_b,
            "up_pkts": sum(ups), "down_pkts": len(ups) - sum(ups),
            "up_bytes": up_b, "down_bytes": down_b,
            "up_ratio": (sum(ups) / len(ups)) if ups else 0.0,
            "up_byte_ratio": (up_b / total_b) if total_b else 0.0,
            "dir_changes": changes,
            "dir_change_rate": changes / max(len(ups) - 1, 1),
        }
        for k, v in base.items(): f[f"{p}_{k}"] = float(v)

        # size 9 (8 stats + sum)
        for name, val in zip(("mean","std","min","max","median","q25","q75","q90"),
                             self._stats(lengths)):
            f[f"{p}_size_{name}"] = float(val)
        f[f"{p}_size_sum"] = float(sum(lengths))
        # iat 8
        for name, val in zip(("mean","std","min","max","median","q25","q75","q90"),
                             self._stats(iats)):
            f[f"{p}_iat_{name}"] = float(val)
        # directional size 5+5
        for prefix, vals in (("up_size", up_lengths), ("down_size", down_lengths)):
            ss = self._stats(vals)
            for name, val in zip(("mean","std","min","max","median"), ss[:5]):
                f[f"{p}_{prefix}_{name}"] = float(val)

        # 8-bin normalized packet-size histogram
        n = max(len(lengths), 1)
        for i, (lo, hi) in enumerate(zip(self.SIZE_BINS, self.SIZE_BINS[1:])):
            cnt = sum(1 for x in lengths if lo <= x < hi)
            f[f"{p}_size_hist_{i}"] = cnt / n

        # 8 shape/peak features
        mode = max(Counter(lengths).values()) if lengths else 0
        specials = {
            "size_eq40_ratio": sum(x == 40 for x in lengths) / n,
            "size_560_610_ratio": sum(560 <= x <= 610 for x in lengths) / n,
            "size_ge1450_ratio": sum(x >= 1450 for x in lengths) / n,
            "size_1300_1449_ratio": sum(1300 <= x < 1450 for x in lengths) / n,
            "size_le64_ratio": sum(x <= 64 for x in lengths) / n,
            "size_unique_ratio": len(set(lengths)) / n if lengths else 0.0,
            "size_mode_ratio": mode / n,
            "size_entropy": self._entropy(lengths),
        }
        for k, v in specials.items(): f[f"{p}_{k}"] = float(v)

        # 4 direction bigrams
        den = max(len(ups) - 1, 1)
        for name, a, b in (("uu",1,1),("ud",1,0),("du",0,1),("dd",0,0)):
            f[f"{p}_dir2_{name}"] = sum(
                1 for x, y in zip(ups, ups[1:]) if x == a and y == b) / den

        # 3 structural features -> exactly 64 total
        f[f"{p}_overflow_n_seg"] = float(win.get("n_overflow_segments", 0))
        ob = sum(s.up_bytes + s.down_bytes for s in (win.get("overflow") or []))
        f[f"{p}_overflow_byte_ratio"] = (ob / total_b) if total_b else 0.0
        f[f"{p}_truncated_prefix"] = 1.0 if win.get("truncated_prefix") else 0.0

        if len(f) != 64:
            raise AssertionError(f"tunnel feature contract must be 64 dims, got {len(f)}")
        return f


class BehaviorFlowWindowFeatureBuilder:
    """One long flow's fixed-time slice -> 64 behavior features.

    This mirrors the time-based flow unit that works on ISCXVPN: packet/payload
    sizes, IAT and direction statistics from one flow, rather than mixing every
    background/control connection in the capture into one terminal window.
    """

    PREFIX = "behavior_flow"

    @staticmethod
    def _stats(values: List[float]) -> List[float]:
        return TunnelWindowFeatureBuilder._stats(values)

    def build(self, packets: List) -> Dict[str, float]:
        p = self.PREFIX
        ps = sorted(packets, key=lambda x: float(getattr(x, "timestamp", 0.0)))
        lengths = [float(getattr(x, "length", 0) or 0) for x in ps]
        payload = [float(getattr(x, "payload_length", 0) or 0) for x in ps]
        dirs = [1 if int(getattr(x, "direction", 1) or 1) > 0 else -1 for x in ps]
        times = [float(getattr(x, "timestamp", 0.0)) for x in ps]
        iats = [0.0] + [max(0.0, b - a) for a, b in zip(times, times[1:])]
        headers = [max(0.0, a - b) for a, b in zip(lengths, payload)]
        fmask = [d > 0 for d in dirs]
        bmask = [not x for x in fmask]
        f_sizes = [v for v, m in zip(lengths, fmask) if m]
        b_sizes = [v for v, m in zip(lengths, bmask) if m]
        f_iat = [v for v, m in zip(iats, fmask) if m]
        b_iat = [v for v, m in zip(iats, bmask) if m]
        n = len(ps)
        duration = max(times[-1] - times[0], 1e-9) if n else 1e-9
        fc, bc = len(f_sizes), len(b_sizes)
        fb, bb = sum(f_sizes), sum(b_sizes)
        changes = sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)

        vals: List[float] = [
            float(n), duration, sum(lengths), sum(payload), float(fc), float(bc),
            fb, bb, n / duration, sum(lengths) / duration,
            sum(payload) / max(sum(lengths), 1.0), fc / max(bc, 1),
            fb / max(bb, 1.0), float(changes), changes / max(n - 1, 1),
        ]
        vals += self._stats(lengths)
        vals += self._stats(payload)
        vals += self._stats(iats)
        vals += self._stats(f_sizes)[:5]
        vals += self._stats(b_sizes)[:5]
        for arr in (f_iat, b_iat):
            q = self._stats(arr)
            vals += [q[0], q[1], q[3], q[4]]
        vals += self._stats(headers)[:3]
        den = max(n - 1, 1)
        for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            vals.append(sum(1 for x, y in zip(dirs, dirs[1:])
                            if x == a and y == b) / den)
        if len(vals) != 64:
            raise AssertionError(f"behavior flow-window contract must be 64 dims, got {len(vals)}")
        names = [
            "n_packets", "duration", "total_bytes", "total_payload_bytes",
            "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
            "packet_rate", "byte_rate", "payload_ratio", "fwd_bwd_packet_ratio",
            "fwd_bwd_byte_ratio", "direction_changes", "direction_change_rate",
        ]
        for pre in ("pkt", "payload", "iat"):
            names += [f"{pre}_{s}" for s in
                      ("mean","std","min","max","median","q25","q75","q90")]
        for pre in ("fwd_pkt", "bwd_pkt"):
            names += [f"{pre}_{s}" for s in ("mean","std","min","max","median")]
        for pre in ("fwd_iat", "bwd_iat"):
            names += [f"{pre}_{s}" for s in ("mean","std","max","median")]
        names += ["header_mean","header_std","header_min",
                  "dir2_ff","dir2_fb","dir2_bf","dir2_bb"]
        return {f"{p}_{name}": float(v) for name, v in zip(names, vals)}
