# -*- coding: utf-8 -*-
"""CSTNET-TLS1.3 数据目录通用发现/审计（扩类索引的通用实现）。

背景（2026-09-18）：output/fast_dpi_20260918_03/s0_index.py 的 SITE_FILES
只硬编码六个网站类，扩到 60 类时索引路径全部落错。本模块按
``CSTNET-TLS1.3-<class>[_pNNN].pcap`` 文件名模式自动发现全部类与分片，
不再依赖硬编码清单；对原始 data/all_data 只读。

要点：
- 类名解析用非贪婪 ``.+?`` + 可选 ``_p\\d+`` 分片后缀，正确处理
  ``xiaomi.com_p002.pcap``（cls=xiaomi.com, part=2）与 ``51.la.pcap``
  （cls=51.la, 无分片）。贪婪正则会把分片误并入类名（审计时踩过）。
- 排序：base 文件在前，分片按编号升序，保证发现结果确定性。
- LEGACY_SIX 与 s0_index.SITE_FILES 一致，用于六类兼容性断言。

用法（审计 CLI）::

    python3 scripts/cstnet_catalog.py --data data/all_data [--json OUT.json]
"""
import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

# CSTNET-TLS1.3-<class>[_pNNN].pcap；cls 非贪婪，先剥离 _p 分片后缀
CSTNET_RE = re.compile(
    r'^CSTNET-TLS1\.3-(?P<cls>.+?)(?:_p(?P<part>\d+))?\.pcap$')

# 兼容锚点：与 output/fast_dpi_20260918_03/s0_index.py 的 SITE_FILES 逐字一致
LEGACY_SIX: Dict[str, List[str]] = {
    'acm.org': ['CSTNET-TLS1.3-acm.org.pcap', 'CSTNET-TLS1.3-acm.org_p002.pcap',
                'CSTNET-TLS1.3-acm.org_p003.pcap'],
    'huawei.com': ['CSTNET-TLS1.3-huawei.com.pcap',
                   'CSTNET-TLS1.3-huawei.com_p002.pcap',
                   'CSTNET-TLS1.3-huawei.com_p003.pcap',
                   'CSTNET-TLS1.3-huawei.com_p004.pcap'],
    'overleaf.com': ['CSTNET-TLS1.3-overleaf.com.pcap',
                     'CSTNET-TLS1.3-overleaf.com_p002.pcap',
                     'CSTNET-TLS1.3-overleaf.com_p003.pcap',
                     'CSTNET-TLS1.3-overleaf.com_p004.pcap',
                     'CSTNET-TLS1.3-overleaf.com_p005.pcap',
                     'CSTNET-TLS1.3-overleaf.com_p006.pcap'],
    'qq.com': ['CSTNET-TLS1.3-qq.com.pcap', 'CSTNET-TLS1.3-qq.com_p002.pcap'],
    'twimg.com': ['CSTNET-TLS1.3-twimg.com.pcap',
                  'CSTNET-TLS1.3-twimg.com_p002.pcap',
                  'CSTNET-TLS1.3-twimg.com_p003.pcap'],
    'vivo.com.cn': ['CSTNET-TLS1.3-vivo.com.cn.pcap',
                    'CSTNET-TLS1.3-vivo.com.cn_p002.pcap',
                    'CSTNET-TLS1.3-vivo.com.cn_p003.pcap'],
}

_PCAP_MAGICS = {
    b'\xd4\xc3\xb2\xa1': 'pcap-le',
    b'\xa1\xb2\xc3\xd4': 'pcap-be',
    b'\x4d\x3c\xb2\xa1': 'pcap-ns-le',
    b'\x0a\x0d\x0d\x0a': 'pcapng',
}


def parse_cstnet_name(name: str) -> Optional[Dict[str, object]]:
    """解析单个文件名；非 CSTNET 命名返回 None。

    返回 {'cls': 类名, 'part': 分片号或 None}。
    """
    m = CSTNET_RE.match(name)
    if not m:
        return None
    part = m.group('part')
    return {'cls': m.group('cls'),
            'part': int(part) if part is not None else None}


def part_sort_key(name: str):
    """排序键：base 文件在前，分片按编号升序。"""
    info = parse_cstnet_name(name)
    part = info['part'] if info else None
    return (0, 0) if part is None else (1, part)


def discover_cstnet(data_dir) -> 'OrderedDict[str, List[Path]]':
    """发现 data_dir 下全部 CSTNET-TLS1.3 类及其分片文件。

    返回按类名排序的有序字典：cls -> [base, ...分片按编号升序]。
    只读目录项，不读文件内容；符号链接按名字参与（链接本身即数据问题证据）。
    """
    data_dir = Path(data_dir)
    classes: Dict[str, List[Path]] = {}
    for p in data_dir.iterdir():
        info = parse_cstnet_name(p.name)
        if info is None:
            continue
        classes.setdefault(info['cls'], []).append(p)
    out: 'OrderedDict[str, List[Path]]' = OrderedDict()
    for cls in sorted(classes):
        out[cls] = sorted(classes[cls],
                          key=lambda p: part_sort_key(p.name))
    return out


def pcap_format(path: Path) -> str:
    """读文件头 4 字节判断格式；不可读/未知魔数如实返回。"""
    try:
        with open(path, 'rb') as f:
            head = f.read(4)
    except OSError as e:
        return f'unreadable:{type(e).__name__}'
    return _PCAP_MAGICS.get(head, f'unknown:{head.hex()}')


def audit_cstnet(data_dir) -> Dict[str, object]:
    """数据审计：逐类文件数/大小/格式/符号链接/命名异常。

    产出可直接落档的证据结构（Task A 要求）。
    """
    data_dir = Path(data_dir)
    classes = discover_cstnet(data_dir)
    per_class = OrderedDict()
    anomalies = []
    fmt_counter: Dict[str, int] = {}
    total_bytes = 0
    for cls, paths in classes.items():
        files = []
        for p in paths:
            fmt = pcap_format(p)
            fmt_counter[fmt] = fmt_counter.get(fmt, 0) + 1
            try:
                size = p.stat().st_size
                if size == 0:
                    anomalies.append(f'zero-size: {p.name}')
            except OSError as e:
                size = -1
                anomalies.append(f'stat-error: {p.name}: {e}')
            if p.is_symlink():
                anomalies.append(f'symlink: {p.name} -> {p.readlink()}')
            total_bytes += max(size, 0)
            files.append({'file': p.name, 'part': parse_cstnet_name(p.name)['part'],
                          'bytes': size, 'format': fmt})
        # 命名异常：有分片却缺 base
        parts = [f for f in files if f['part'] is not None]
        if parts and not any(f['part'] is None for f in files):
            anomalies.append(f'missing-base-file: {cls}')
        per_class[cls] = {
            'n_files': len(files),
            'bytes': sum(f['bytes'] for f in files),
            'files': files,
        }
    return {
        'data_dir': str(data_dir),
        'n_classes': len(classes),
        'n_files': sum(len(v) for v in classes.values()),
        'total_bytes': total_bytes,
        'formats': fmt_counter,
        'multi_file_classes': sorted(c for c, v in classes.items() if len(v) > 1),
        'anomalies': anomalies,
        'per_class': per_class,
    }


def legacy_six_matches(discovered: 'OrderedDict[str, List[Path]]') -> bool:
    """六类兼容性：发现结果须与旧 SITE_FILES 逐类逐文件一致。"""
    for cls, names in LEGACY_SIX.items():
        got = [p.name for p in discovered.get(cls, [])]
        if got != names:
            return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='CSTNET 通用发现/审计')
    ap.add_argument('--data', default='data/all_data')
    ap.add_argument('--json', help='审计结果 JSON 落档路径（可选）')
    args = ap.parse_args(argv)

    report = audit_cstnet(args.data)
    print(f"classes={report['n_classes']} files={report['n_files']} "
          f"total={report['total_bytes'] / 1e9:.2f}GB "
          f"formats={report['formats']}")
    print(f"multi-file classes: {len(report['multi_file_classes'])}")
    print(f"anomalies: {len(report['anomalies'])}")
    for a in report['anomalies'][:20]:
        print('  -', a)

    disc = discover_cstnet(args.data)
    compat = legacy_six_matches(disc)
    print(f"legacy-six compat: {'OK' if compat else 'MISMATCH'}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
        print('saved:', args.json)
    return 0 if compat else 1


if __name__ == '__main__':
    sys.exit(main())
