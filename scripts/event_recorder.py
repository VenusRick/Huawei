#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人工行为事件记录器（P1.5）：与抓包同时启动，按键记录动作起止时刻。

用法：
  tcpdump -i any -w cap01.pcap &          # 抓包
  python scripts/event_recorder.py --capture-id cap01 \
      --terminal-ip 192.168.1.5 --app wechat --out events_cap01.jsonl
交互：输入行为名回车=开始计时；直接回车=结束当前行为；输入 q 退出。
输出 jsonl：{trial_id, behavior, t_start, t_end, wall_start, wall_end}
t_* 为相对记录器启动的秒数；与 pcap 首包的换算见采集清单第四节。
"""
import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="人工行为事件记录器")
    ap.add_argument("--capture-id", required=True)
    ap.add_argument("--terminal-ip", required=True)
    ap.add_argument("--app", required=True)
    ap.add_argument("--platform", default="pc")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    print(f"记录器启动（capture={args.capture_id}）。"
          f"抓包请与本进程同时开始。")
    t0 = time.time()
    print(f"启动墙钟: {t0:.3f}（写入输出供时间校准）")
    n = 0
    current = None
    current_t = None
    while True:
        try:
            line = input("[行为名回车=开始 / 回车=结束 / q=退出] > ").strip()
        except EOFError:
            break
        now = time.time()
        if line == "q":
            break
        if line:
            if current is not None:
                print("  警告：上一行为未结束，先自动结束")
                _dump(out, args, current, current_t, now, n)
                n += 1
            current, current_t = line, now
            print(f"  开始: {line} (+{now - t0:.2f}s)")
        elif current is not None:
            _dump(out, args, current, current_t, now, n)
            print(f"  结束: {current} 历时 {now - current_t:.2f}s")
            n += 1
            current = None
        else:
            print("  （当前无进行中行为）")
    if current is not None:
        _dump(out, args, current, current_t, time.time(), n)
    meta = {"t0_wall": t0, "capture_id": args.capture_id,
            "terminal_ip": args.terminal_ip, "app": args.app}
    with open(out.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"已写入 {n} 条试次 -> {out}；元数据 -> {out.with_suffix('.meta.json')}")


def _dump(out, args, behavior, t_start, t_end, idx):
    rec = {
        "trial_id": f"{args.capture_id}_{behavior}_{idx:03d}",
        "behavior": behavior,
        "t_start": round(t_start, 3),
        "t_end": round(t_end, 3),
        "wall_start": round(t_start, 3),
        "wall_end": round(t_end, 3),
        "app": args.app, "platform": args.platform,
        "terminal_ip": args.terminal_ip,
    }
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
