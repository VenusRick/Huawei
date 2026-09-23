# -*- coding: utf-8 -*-
"""基于分级普查结果构建 60 类固定观测 manifest（选类 + 采样 + 划分）。

选类规则（数据驱动，不拍脑袋）：
1. 候选 = survey 中 qualified >= min_obs(默认100) 的类；
2. 多文件类优先（可按 capture 分组划分，跨采集组泛化更可信），
   单文件类按 qualified 降序补足到 60 类；
3. 若 100/类 不够 60 类，允许 qualified >= 60 的类以 60 观测入选并如实记录。

划分纪律（防同 PCAP 随机拆分虚高）：
- capture_group：>=3 个文件各有 >=25 qualified —— train/val/test 各绑不同文件，
  测试观测来自训练未见过的采集分片；
- capture_group_test_only：恰好 2 个文件各 >=40 —— train+val 用文件A（会话级拆），
  test 绑文件B（测试采集组仍与训练隔离）；
- session_level：其余（含全部单文件类）—— 同文件内会话级 60/20/20，
  必须明示为 session-level 规模实验，不冒充跨 capture 泛化。

所有采样用固定 seed 的确定性洗牌；每类记录 survey tier cap，
后续特征提取必须用同一 cap 才能对齐 observation_id。

用法::

    python3 scripts/build_manifest60.py \
        --survey output/cstnet60_20260918_exp/survey \
        --out output/cstnet60_20260918_exp
"""
import argparse
import json
import random
from collections import Counter, OrderedDict
from pathlib import Path

SEED = 20260918


def load_class(survey_dir: Path, cls: str):
    return json.loads((survey_dir / f'{cls}.json').read_text(encoding='utf-8'))


def shuffled_obs(qualified_ids, cls: str, fname: str):
    """同文件观测确定性洗牌（seed 绑定 类+文件，与上一轮方法一致）。"""
    obs = sorted(set(qualified_ids))
    rng = random.Random(f'{SEED}:{cls}:{fname}')
    rng.shuffle(obs)
    return obs


def take_round_robin(pools: dict, n: int):
    """pools: {file: [obs,...]} 已洗牌；跨文件 round-robin 取 n 条。"""
    selected = []
    pos = {f: 0 for f in pools}
    while len(selected) < n:
        progressed = False
        for f in sorted(pools):
            i = pos[f]
            if i < len(pools[f]):
                selected.append({'observation_id': pools[f][i], 'file': f})
                pos[f] += 1
                progressed = True
                if len(selected) >= n:
                    break
        if not progressed:
            break
    return selected


def assign_class_split(cls: str, class_rec: dict, target: int):
    """决定一个类的 split_mode 并直接选出各 split 的固定观测。

    返回 (split_mode, got, files_used)：
    - capture_group：>=3 文件各 >=20 qualified 且剩余池可覆盖 train 配额
      —— train/val/test 绑不同文件；
    - capture_group_test_only：文件A(>=train+val 配额) 承担 train/val 会话级拆，
      文件B(>=20) 独占 test（测试采集组与训练隔离）；
    - session_level：其余 —— 全池洗牌 round-robin 合并后顺序切 60/20/20。
    """
    files = class_rec['files']
    by_file = {f['file']: [s['observation_id'] for s in f['sessions']]
               for f in files}
    counts = {f: len(v) for f, v in by_file.items()}
    ranked = sorted(counts, key=lambda f: (-counts[f], f))
    n_tr, n_va, n_te = target * 3 // 5, target // 5, target // 5

    # capture_group：>=3 文件各 >= n_te(20)，且剩余文件池能覆盖 train 配额
    big = [f for f in ranked if counts[f] >= n_te]
    if len(big) >= 3:
        val_file, test_file = big[1], big[2]
        train_files = [f for f in ranked if f not in (val_file, test_file)]
        if sum(counts[f] for f in train_files) >= n_tr:
            got = {
                'train': take_round_robin(
                    {f: shuffled_obs(by_file[f], cls, f) for f in train_files}, n_tr),
                'validation': take_round_robin(
                    {val_file: shuffled_obs(by_file[val_file], cls, val_file)}, n_va),
                'test': take_round_robin(
                    {test_file: shuffled_obs(by_file[test_file], cls, test_file)}, n_te),
            }
            return 'capture_group', got, {'train_files': train_files,
                                          'validation_files': [val_file],
                                          'test_files': [test_file]}

    # test_only：最大文件A(>=train+val 配额) 承担 train/val，文件B(>=20) 独占 test
    file_a = ranked[0]
    others = [f for f in ranked[1:] if counts[f] >= n_te] if ranked else []
    if counts.get(file_a, 0) >= n_tr + n_va and others:
        file_b = others[0]
        pool_a = shuffled_obs(by_file[file_a], cls, file_a)
        got = {
            'train': [{'observation_id': o, 'file': file_a}
                      for o in pool_a[:n_tr]],
            'validation': [{'observation_id': o, 'file': file_a}
                           for o in pool_a[n_tr:n_tr + n_va]],
            'test': take_round_robin(
                {file_b: shuffled_obs(by_file[file_b], cls, file_b)}, n_te),
        }
        return 'capture_group_test_only', got, {
            'train_files': [file_a], 'validation_files': [file_a],
            'test_files': [file_b]}

    # session_level：全池洗牌后 round-robin 合并，再顺序切 60/20/20
    pools = {f: shuffled_obs(by_file[f], cls, f) for f in sorted(by_file)}
    merged = take_round_robin(pools, target)
    got = {
        'train': merged[:n_tr],
        'validation': merged[n_tr:n_tr + n_va],
        'test': merged[n_tr + n_va:n_tr + n_va + n_te],
    }
    return 'session_level', got, {
        'train_files': sorted(by_file), 'validation_files': sorted(by_file),
        'test_files': sorted(by_file)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='60类 manifest 构建')
    ap.add_argument('--survey', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-classes', type=int, default=60)
    ap.add_argument('--min-obs', type=int, default=100)
    args = ap.parse_args(argv)

    survey_dir = Path(args.survey)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((survey_dir / '_summary.json').read_text(encoding='utf-8'))

    recs = []
    for s in summary['classes']:
        recs.append((s['cls'], s['n_qualified'], s['n_files'], s['tier_index'],
                     s['read_full']))
    multi = sorted([r for r in recs if r[2] > 1 and r[1] >= args.min_obs],
                   key=lambda r: (-r[1], r[0]))
    single = sorted([r for r in recs if r[2] == 1 and r[1] >= args.min_obs],
                    key=lambda r: (-r[1], r[0]))
    chosen = multi + single
    # 100/类不够时允许 60/类 的类补足（如实记录）
    relaxed = []
    if len(chosen) < args.n_classes:
        relaxed = sorted([r for r in recs if r not in chosen and r[1] >= 60],
                         key=lambda r: (-r[1], r[0]))
        chosen += relaxed[:args.n_classes - len(chosen)]
    chosen = chosen[:args.n_classes]

    rows = []
    class_cfg = OrderedDict()
    for cls, n_qual, n_files, tier, read_full in chosen:
        target = 100 if n_qual >= 100 else 60
        crec = load_class(survey_dir, cls)
        mode, plan, files_used = assign_class_split(cls, crec, target)
        got = plan
        cnt = {}
        for split, items in got.items():
            cnt[split] = len(items)
            for r in items:
                rows.append({'observation_id': r['observation_id'],
                             'label': cls, 'file': r['file'], 'split': split,
                             'split_mode': mode})
        class_cfg[cls] = {
            'n_files': n_files, 'qualified_survey': n_qual,
            'survey_tier': tier, 'per_file_cap': crec['per_file_cap'],
            'read_full': read_full, 'target': target,
            'split_mode': mode, 'n_obs': cnt, **files_used,
        }
        assert cnt['train'] + cnt['validation'] + cnt['test'] == target, cls

    order = {'train': 0, 'validation': 1, 'test': 2}
    rows.sort(key=lambda x: (order[x['split']], x['label'], x['file'],
                             x['observation_id']))
    manifest = out_dir / 'manifest_60c.jsonl'
    with manifest.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    meta = {
        'seed': SEED, 'n_classes': len(class_cfg),
        'n_observations': len(rows),
        'split_mode_counts': dict(Counter(c['split_mode']
                                          for c in class_cfg.values())),
        'classes_relaxed_60': [c for c, v in class_cfg.items()
                               if v['target'] == 60],
        'tier_cap_distribution': dict(Counter(c['per_file_cap']
                                              for c in class_cfg.values())),
        'classes': class_cfg,
    }
    (out_dir / 'manifest_60c_class_config.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')

    print(f'classes={len(class_cfg)} observations={len(rows)}')
    print('split modes:', meta['split_mode_counts'])
    print('tier caps:', meta['tier_cap_distribution'])
    cnt = Counter((r['label'], r['split']) for r in rows)
    for cls in class_cfg:
        print(f'  {cls:22s} {class_cfg[cls]["split_mode"]:26s} '
              f'train={cnt[cls, "train"]:3d} val={cnt[cls, "validation"]:3d} '
              f'test={cnt[cls, "test"]:3d} qual={class_cfg[cls]["qualified_survey"]}')
    print('saved:', manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
