# -*- coding: utf-8 -*-
"""扩类索引通用实现（scripts/cstnet_catalog.py / cstnet_survey.py）回归。

背景：s0_index.py SITE_FILES 只硬编码六类，扩类时索引路径全错
（candidate80 有 80 类、index64_pool 只落了 6 个 JSON）。
本标尺钉死：类发现不再依赖硬编码清单，且对六类结果与旧清单逐字兼容。
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.cstnet_catalog import (  # noqa: E402
    CSTNET_RE,
    LEGACY_SIX,
    audit_cstnet,
    discover_cstnet,
    legacy_six_matches,
    parse_cstnet_name,
    part_sort_key,
)

REAL_DATA = ROOT / 'data' / 'all_data'
HAS_REAL = REAL_DATA.is_dir()


class TestParseName:
    def test_base_file(self):
        assert parse_cstnet_name('CSTNET-TLS1.3-acm.org.pcap') == {
            'cls': 'acm.org', 'part': None}

    def test_part_file(self):
        # 非贪婪 cls：_p002 必须剥成 part，不许并入类名（贪婪正则的坑）
        assert parse_cstnet_name('CSTNET-TLS1.3-xiaomi.com_p002.pcap') == {
            'cls': 'xiaomi.com', 'part': 2}

    def test_cls_with_digits_and_dots(self):
        assert parse_cstnet_name('CSTNET-TLS1.3-51.la.pcap') == {
            'cls': '51.la', 'part': None}
        assert parse_cstnet_name('CSTNET-TLS1.3-leetcode-cn.com_p003.pcap') == {
            'cls': 'leetcode-cn.com', 'part': 3}

    def test_part_numeric_not_hex(self):
        assert parse_cstnet_name('CSTNET-TLS1.3-qq.com_p010.pcap')['part'] == 10

    def test_rejects_non_cstnet(self):
        for name in ['USTC-TLS1.3-qq.com.pcap',
                     'CSTNET-TLS1.3-qq.com.pcapng',
                     'CSTNET-TLS1.3-qq.com.txt',
                     'CSTNET-TLS1.3.pcap']:
            assert parse_cstnet_name(name) is None, name

    def test_sort_key_base_first_then_part(self):
        names = ['CSTNET-TLS1.3-a.com_p002.pcap', 'CSTNET-TLS1.3-a.com.pcap',
                 'CSTNET-TLS1.3-a.com_p010.pcap', 'CSTNET-TLS1.3-a.com_p003.pcap']
        assert sorted(names, key=part_sort_key) == [
            'CSTNET-TLS1.3-a.com.pcap',
            'CSTNET-TLS1.3-a.com_p002.pcap',
            'CSTNET-TLS1.3-a.com_p003.pcap',
            'CSTNET-TLS1.3-a.com_p010.pcap']


class TestDiscover:
    def test_discover_and_order(self, tmp_path):
        for n in ['CSTNET-TLS1.3-b.com_p002.pcap', 'CSTNET-TLS1.3-b.com.pcap',
                  'CSTNET-TLS1.3-a.com.pcap', 'USTC-x.pcap', 'CSTNET-TLS1.3-c.cn.pcap']:
            (tmp_path / n).write_bytes(b'x')
        got = discover_cstnet(tmp_path)
        assert list(got) == ['a.com', 'b.com', 'c.cn']  # 类名排序
        assert [p.name for p in got['b.com']] == [
            'CSTNET-TLS1.3-b.com.pcap', 'CSTNET-TLS1.3-b.com_p002.pcap']

    def test_audit_flags(self, tmp_path):
        (tmp_path / 'CSTNET-TLS1.3-ok.com.pcap').write_bytes(b'\xd4\xc3\xb2\xa1rest')
        (tmp_path / 'CSTNET-TLS1.3-empty.com.pcap').write_bytes(b'')
        (tmp_path / 'CSTNET-TLS1.3-orphan.com_p002.pcap').write_bytes(b'\xd4\xc3\xb2\xa1')
        link = tmp_path / 'CSTNET-TLS1.3-link.com.pcap'
        link.symlink_to(tmp_path / 'CSTNET-TLS1.3-ok.com.pcap')
        rep = audit_cstnet(tmp_path)
        assert rep['n_classes'] == 4 and rep['n_files'] == 4
        # 空文件读不满魔数记 unknown；链接跟随目标读到同魔数
        assert rep['formats'] == {'pcap-le': 3, 'unknown:': 1}
        msgs = ' '.join(rep['anomalies'])
        assert 'zero-size' in msgs and 'missing-base-file: orphan.com' in msgs \
            and 'symlink' in msgs


class TestManifestSplit:
    """build_manifest60.assign_class_split 的划分纪律（合成类记录）。"""

    @staticmethod
    def _cls_rec(counts: dict):
        return {'files': [
            {'file': f, 'qualified': n,
             'sessions': [{'observation_id': f'{f}:{i}', 'file': f}
                          for i in range(n)]}
            for f, n in counts.items()]}

    def test_capture_group_three_big_files(self):
        from scripts.build_manifest60 import assign_class_split
        rec = self._cls_rec({'a.pcap': 200, 'b.pcap': 60, 'c.pcap': 50})
        mode, got, used = assign_class_split('x.com', rec, 100)
        assert mode == 'capture_group'
        assert len(got['train']) == 60 and len(got['validation']) == 20 \
            and len(got['test']) == 20
        # 三组文件互斥：测试采集组与训练完全隔离
        assert not set(used['train_files']) & set(used['test_files'])
        assert not set(used['validation_files']) & set(used['test_files'])

    def test_test_only_two_files(self):
        from scripts.build_manifest60 import assign_class_split
        rec = self._cls_rec({'big.pcap': 90, 'test.pcap': 40})
        mode, got, used = assign_class_split('x.com', rec, 100)
        assert mode == 'capture_group_test_only'
        assert used['test_files'] == ['test.pcap']
        assert len(got['train']) == 60 and len(got['validation']) == 20

    def test_session_level_when_no_isolated_test_possible(self):
        from scripts.build_manifest60 import assign_class_split
        # 单文件类；或两文件都当不了 train+val 池（70<80）：必须降级并明示
        for counts in ({'only.pcap': 200}, {'a.pcap': 70, 'b.pcap': 30}):
            rec = self._cls_rec(counts)
            mode, got, used = assign_class_split('x.com', rec, 100)
            assert mode == 'session_level', counts
            assert len(got['train']) == 60 and len(got['test']) == 20

    def test_deterministic_sampling(self):
        from scripts.build_manifest60 import assign_class_split
        rec = self._cls_rec({'a.pcap': 200, 'b.pcap': 60, 'c.pcap': 50})
        r1 = assign_class_split('x.com', rec, 100)
        r2 = assign_class_split('x.com', rec, 100)
        assert r1[1] == r2[1]  # 同输入两次划分完全一致（固定观测）


@pytest.mark.skipif(not HAS_REAL, reason='真实 all_data 不在本机')
class TestRealData:
    def test_legacy_six_compat(self):
        """六类发现结果必须与 s0_index.SITE_FILES 逐类逐文件一致。"""
        disc = discover_cstnet(REAL_DATA)
        assert legacy_six_matches(disc)
        for cls, names in LEGACY_SIX.items():
            assert [p.name for p in disc[cls]] == names

    def test_real_audit_shape(self):
        rep = audit_cstnet(REAL_DATA)
        assert rep['n_classes'] == 118 and rep['n_files'] == 148
        assert set(rep['formats']) == {'pcap-be'}  # 主机审计口径
        assert rep['anomalies'] == []

    def test_scan_file_observation_id_deterministic(self):
        """同 cap 两次扫描 observation_id 完全一致（固定观测选择的前提）。"""
        from scripts.cstnet_survey import scan_file
        path = REAL_DATA / 'CSTNET-TLS1.3-qq.com.pcap'
        if not path.exists():
            pytest.skip('qq.com 分片不存在')
        a = scan_file(path, 20000)
        b = scan_file(path, 20000)
        ids_a = [s['observation_id'] for s in a['sessions']]
        ids_b = [s['observation_id'] for s in b['sessions']]
        assert ids_a == ids_b and ids_a
        assert a['qualified'] == len(ids_a)
