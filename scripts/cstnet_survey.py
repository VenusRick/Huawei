# -*- coding: utf-8 -*-
"""CSTNET 候选类 qualified session 分级普查（扩类选类的数据依据）。

对 scripts/cstnet_catalog.py 发现的每个类做有界扫描：
- 分级读包上限 tiers（默认 50k -> 100k -> 200k -> 全文件）；
- 只有当 qualified 会话数不足 min_qualified 且仍有文件未读满时才升级，
  不无脑全量重跑；
- 每级记录真实读包量/是否读满/耗时，逐类落 JSON（survey/<cls>.json），
  含全部 qualified 会话元数据（observation_id 供后续固定观测选择）。

qualified 与 runtime 口径一致：>=3 包且 >=100 字节。
读上限截断的边界会话与后续正式提取同 cap 复现，口径一致。

用法::

    python3 scripts/cstnet_survey.py --data data/all_data \
        --out output/cstnet60_20260918_exp/survey \
        [--classes a,b,c] [--min-qualified 140] [--tiers 50000,100000,200000,0]
    （tiers 中 0 表示全文件）
"""
import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cstnet_catalog import discover_cstnet  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def scan_file(path: Path, cap: Optional[int]) -> Dict[str, object]:
    """单文件有界扫描：qualified 会话元数据 + 真实读量统计。

    会话枚举顺序与 runtime.extract_feature_records_with_stats 的
    read_pcap_generator 同构（同 cap 同序），session_index 可跨脚本对齐。
    """
    from src.parser.pcap_reader import PCAPReader
    from src.parser.session.session_manager import SessionManager
    from src.features.runtime import MIN_SESSION_PACKETS, MIN_SESSION_BYTES

    sm = SessionManager()
    reader = PCAPReader(sm)
    rec = {'file': path.name, 'sessions_total': 0, 'qualified': 0,
           'sessions': []}
    for idx, sess in enumerate(reader.read_pcap_generator(
            str(path), max_read_packets=cap)):
        rec['sessions_total'] += 1
        if (sess.total_packets < MIN_SESSION_PACKETS
                or sess.total_bytes < MIN_SESSION_BYTES):
            continue
        rec['qualified'] += 1
        sni = ''
        has_tls = False
        for p in sess.packets:
            if p.tls_info:
                has_tls = True
                if p.tls_info.get('ch_sni'):
                    sni = p.tls_info['ch_sni']
                    break
        rec['sessions'].append({
            'file': path.name,
            'session_index': idx,
            'observation_id': f'{path.name}:{idx}',
            'proto': sess.protocol.name,
            'src': f'{sess.src_ip}:{sess.src_port}',
            'dst': f'{sess.dst_ip}:{sess.dst_port}',
            'n_packets': int(sess.total_packets),
            'n_bytes': int(sess.total_bytes),
            'duration': round(sess.duration, 6),
            'has_syn': any(p.tcp_flags & 0x02 for p in sess.packets),
            'has_tls': has_tls,
            'sni': sni,
        })
    rec['packets_read'] = reader.packets_last_read
    rec['read_capped'] = reader.read_capped
    rec['parse_errors'] = reader.parse_errors
    return rec


def survey_class(cls: str, paths: List[Path],
                 tiers: List[Optional[int]],
                 min_qualified: int) -> Dict[str, object]:
    """分级普查一个类：不足 min_qualified 且有文件未读满才升级。"""
    trace = []
    result = None
    for tier_i, cap in enumerate(tiers):
        t0 = time.time()
        files = [scan_file(p, cap) for p in paths]
        n_qualified = sum(f['qualified'] for f in files)
        any_capped = any(f['read_capped'] for f in files)
        result = {
            'cls': cls, 'tier_index': tier_i,
            'per_file_cap': cap, 'n_files': len(paths),
            'n_qualified': n_qualified, 'read_capped_any': any_capped,
            'packets_read': sum(f['packets_read'] for f in files),
            'wall_sec': round(time.time() - t0, 2),
            'files': files,
        }
        trace.append({'tier': tier_i, 'cap': cap, 'qualified': n_qualified,
                      'packets_read': result['packets_read'],
                      'read_capped_any': any_capped,
                      'wall_sec': result['wall_sec']})
        print(f"  [{cls}] tier{tier_i} cap={cap} qualified={n_qualified} "
              f"pkts={result['packets_read']} capped={any_capped} "
              f"wall={result['wall_sec']}s", flush=True)
        if n_qualified >= min_qualified or not any_capped:
            break  # 足量，或已读满全部文件（再升级无意义）
    assert result is not None
    result['tier_trace'] = trace
    result['read_full'] = not result['read_capped_any']
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='CSTNET 分级普查')
    ap.add_argument('--data', default='data/all_data')
    ap.add_argument('--out', required=True)
    ap.add_argument('--classes', help='逗号分隔子集（默认全部）')
    ap.add_argument('--min-qualified', type=int, default=140)
    ap.add_argument('--tiers', default='50000,100000,200000,0',
                    help='逗号分隔分级读包上限，0=全文件')
    args = ap.parse_args(argv)

    tiers = [None if int(x) == 0 else int(x) for x in args.tiers.split(',')]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_cstnet(args.data)
    if args.classes:
        wanted = set(args.classes.split(','))
        missing = wanted - set(discovered)
        if missing:
            raise SystemExit(f'未知类: {sorted(missing)}')
        classes = OrderedDict((c, discovered[c]) for c in sorted(wanted))
    else:
        classes = discovered

    summary = []
    t_all = time.time()
    for i, (cls, paths) in enumerate(classes.items(), 1):
        out_path = out_dir / f'{cls}.json'
        if out_path.exists():  # 断点续跑：已普查类跳过
            prev = json.loads(out_path.read_text())
            summary.append({'cls': cls, 'n_qualified': prev['n_qualified'],
                            'tier_index': prev['tier_index'],
                            'read_full': prev['read_full'],
                            'n_files': prev['n_files']})
            continue
        print(f'({i}/{len(classes)}) {cls}', flush=True)
        res = survey_class(cls, paths, tiers, args.min_qualified)
        out_path.write_text(json.dumps(res, ensure_ascii=False),
                            encoding='utf-8')
        summary.append({'cls': cls, 'n_qualified': res['n_qualified'],
                        'tier_index': res['tier_index'],
                        'read_full': res['read_full'],
                        'n_files': res['n_files']})

    (out_dir / '_summary.json').write_text(
        json.dumps({'data_dir': args.data, 'tiers': tiers,
                    'min_qualified': args.min_qualified,
                    'n_classes': len(summary),
                    'wall_sec': round(time.time() - t_all, 1),
                    'classes': summary}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    ok = sum(1 for s in summary if s['n_qualified'] >= 100)
    print(f"survey done: {len(summary)} classes, "
          f"{ok} with >=100 qualified, wall={time.time() - t_all:.0f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
