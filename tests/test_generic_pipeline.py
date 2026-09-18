"""
通用端到端流水线
支持两种模式：

模式1: 预划分目录（比赛/正式评测）
    python tests/test_generic_pipeline.py --train data/train --val data/val --test data/test

模式2: 单目录自动随机划分 6:2:2（开发/自测）
    python tests/test_generic_pipeline.py --data data/samples_demo

目录结构（每个子目录名 = 一个类别标签）：
    data/train/class_A/*.pcap
    data/val/class_A/*.pcap
    data/test/class_A/*.pcap
"""
import sys, os, time, argparse, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')  # 抑制joblib物理核心检测警告

# 修复Windows GBK编码导致tqdm.write()无法输出Unicode字符的问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings('ignore', message='Could not find the number of physical cores')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import f_oneway
from xgboost import XGBClassifier

from src.parser.pcap_reader import PCAPReader
from src.parser.session.session_manager import SessionManager
from src.features.basic.feature_extractor import BasicFeatureExtractor
from src.features.advanced.advanced_extractor import AdvancedFeatureExtractor
from src.features.selection.feature_selector import FeatureSelector
from src.engine.rule_compiler.optimized_generator import OptimizedRuleGenerator
from src.engine.matcher.optimized_engine import OptimizedDPIEngine


# =====================================================================
#  核心函数
# =====================================================================

def parse_pcap_files(data_dir: str, label_names: dict = None,
                     min_sessions: int = 3, max_sessions: int = 300,
                     max_file_mb: int = 500, max_packets: int = 500):
    """
    模式A: 文件名 = 类别名。
    每个 .pcap/.pcapng 文件是一个独立类别，类别名 = 文件名（去扩展名）。
    同目录下同名的7z会自动解压。

    Args:
        data_dir: 数据目录
        label_names: 已有的 {ID: 类别名} 映射
        min_sessions: 最少会话数，不足则跳过
        max_sessions: 每类最大会话数
        max_file_mb: 单文件大小上限MB
        max_packets: 每会话最大包数（用于截断）
    """
    from tqdm import tqdm

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"  ⚠ 目录不存在: {data_dir}")
        return [], [], label_names or {}

    if label_names is None:
        label_names = {}
    label_map = {v: k for k, v in label_names.items()}

    # 收集所有PCAP文件（含7z解压）
    pcap_files = []
    for f in sorted(data_path.glob('*.pcap')) + sorted(data_path.glob('*.pcapng')):
        size_mb = f.stat().st_size / (1024*1024)
        if size_mb > max_file_mb:
            print(f"  跳过 {f.name}（{size_mb:.0f}MB > {max_file_mb}MB）")
            continue
        pcap_files.append(f)

    # 解压7z文件
    for f_7z in sorted(data_path.glob('*.7z')):
        try:
            import py7zr
            extract_dir = data_path / f_7z.stem
            if not extract_dir.exists():
                print(f"  解压: {f_7z.name} ...", end=' ', flush=True)
                with py7zr.SevenZipFile(str(f_7z), mode='r') as z:
                    z.extractall(str(extract_dir))
                print("完成")
            for p in sorted(extract_dir.rglob('*.pcap')):
                size_mb = p.stat().st_size / (1024*1024)
                if size_mb <= max_file_mb:
                    pcap_files.append(p)
                else:
                    print(f"  跳过 {p.name}（{size_mb:.0f}MB > {max_file_mb}MB）")
        except ImportError:
            print(f"  ⚠ 需要 py7zr: pip install py7zr")
        except Exception as e:
            print(f"  ⚠ 解压 {f_7z.name} 失败: {e}")

    # 分配类别ID（文件名去扩展名 = 类别名）
    next_id = max(label_map.values(), default=-1) + 1
    file_groups = {}  # {class_name: [file_path, ...]}
    for f in pcap_files:
        name = f.stem
        file_groups.setdefault(name, []).append(f)
        if name not in label_map:
            label_map[name] = next_id
            next_id += 1

    # 解析每个类别
    sessions, labels = [], []
    class_tcp_stats = {}
    pbar = tqdm(sorted(file_groups.items()), desc='解析PCAP', unit='类别')
    for class_name, files in pbar:
        pbar.set_postfix_str(class_name)
        lid = label_map[class_name]
        class_sessions = []
        cls_retrans = 0
        cls_ooo = 0

        for f in files:
            try:
                sm = SessionManager(tcp_timeout=300, udp_timeout=60)
                reader = PCAPReader(sm)
                for s in reader.read_pcap(str(f)):
                    cls_retrans += s.num_retransmissions
                    cls_ooo += s.num_out_of_order
                    if s.total_packets >= min_sessions:
                        class_sessions.append(s)
                    if len(class_sessions) >= max_sessions:
                        break
            except Exception:
                pass
            if len(class_sessions) >= max_sessions:
                break

        class_sessions = class_sessions[:max_sessions]
        n = len(class_sessions)

        # 统计加密会话数（载荷是否可读：可打印字符占比 < 70% 则视为加密）
        cls_encrypted = 0
        for s in class_sessions:
            payload_bytes = b""
            for p in s.packets:
                if p.payload:
                    payload_bytes += p.payload[:512]  # 取前512字节采样
                    if len(payload_bytes) >= 1024:
                        break
            if len(payload_bytes) > 0:
                printable = sum(1 for b in payload_bytes if 0x20 <= b <= 0x7e)
                if printable / len(payload_bytes) < 0.70:
                    cls_encrypted += 1
        enc_pct = cls_encrypted / n * 100 if n > 0 else 0
        enc_tag = "加密" if n > 0 and cls_encrypted > 0 else ("明文" if n > 0 else "")

        if n >= min_sessions:
            sessions.extend(class_sessions)
            labels.extend([lid] * n)
            tqdm.write(f"  ✓ {class_name:<20} {n:>5} 个会话  {enc_tag}")
        else:
            tqdm.write(f"  ✗ {class_name:<20} {n:>5} 个会话（不足）")

        class_tcp_stats[class_name] = {
            'sessions': n,
            'total_pkts': sum(s.total_packets for s in class_sessions),
            'retrans': cls_retrans,
            'ooo': cls_ooo,
        }

    pbar.close()

    # 格式化输出: TCP重传/乱序诊断（按类别）
    total_pkts = sum(v['total_pkts'] for v in class_tcp_stats.values())
    total_retrans = sum(v['retrans'] for v in class_tcp_stats.values())
    total_ooo = sum(v['ooo'] for v in class_tcp_stats.values())

    if total_pkts > 0:
        wire_total = total_pkts + total_retrans
        sep = "  " + "-" * 72
        print(f"\n  -- 重传/乱序诊断 --")
        print(sep)
        print(f"  {'类别':<12} {'会话':>4} {'总包数':>7} {'重传包':>7} {'重传率':>5} {'乱序包':>7} {'乱序率':>5}")
        print(sep)
        for name in sorted(class_tcp_stats.keys()):
            v = class_tcp_stats[name]
            if v['sessions'] == 0:
                continue
            wire = v['total_pkts'] + v['retrans']
            pr = v['retrans'] / wire * 100 if wire > 0 else 0
            po = v['ooo'] / wire * 100 if wire > 0 else 0
            print(f"  {name:<14} {v['sessions']:>6} {wire:>10} {v['retrans']:>10} {pr:>7.1f}% {v['ooo']:>10} {po:>7.1f}%")
        print(sep)
        pr_all = total_retrans / wire_total * 100
        po_all = total_ooo / wire_total * 100
        print(f"  {'合计':<12} {len(sessions):>6} {wire_total:>10} {total_retrans:>10} {pr_all:>7.1f}% {total_ooo:>10} {po_all:>7.1f}%")
        print()

    # 重映射为连续ID
    # R27: 标签沿用全局label_map的ID，不按split内出现的类别重新编号
    label_names = {old_id: name for name, old_id in label_map.items()}

    return sessions, labels, label_names


def parse_pcap_dir(pcap_dir: str, label_names: dict = None,
                   min_sessions: int = 3, max_sessions: int = 300,
                   max_packets: int = 500):
    """
    模式B: 目录名 = 类别名。
    每个子目录是一个类别，目录下放多个 .pcap/.pcapng 文件。

    Args:
        pcap_dir: PCAP目录路径
        label_names: 已有的 {ID: 类别名} 映射
        min_sessions: 最少会话数
        max_sessions: 每类最大会话数
        max_packets: 每会话最大包数
    """
    pcap_path = Path(pcap_dir)
    if not pcap_path.exists():
        print(f"  ⚠ 目录不存在: {pcap_dir}")
        return [], [], label_names or {}

    if label_names is None:
        label_names = {}
    label_map = {v: k for k, v in label_names.items()}

    class_dirs = sorted([d for d in pcap_path.iterdir() if d.is_dir()])
    next_id = max(label_map.values(), default=-1) + 1
    for d in class_dirs:
        if d.name not in label_map:
            label_map[d.name] = next_id
            next_id += 1

    sessions, labels = [], []
    class_tcp_stats = {}
    for class_dir in class_dirs:
        name = class_dir.name
        lid = label_map[name]
        pcap_files = list(class_dir.glob("*.pcap")) + list(class_dir.glob("*.pcapng"))
        class_sessions = []
        cls_retrans = 0
        cls_ooo = 0
        for f in pcap_files:
            sm = SessionManager(tcp_timeout=600, udp_timeout=300)
            reader = PCAPReader(sm)
            for s in reader.read_pcap(str(f)):
                cls_retrans += s.num_retransmissions
                cls_ooo += s.num_out_of_order
                if s.total_packets >= min_sessions:
                    class_sessions.append(s)
                if len(class_sessions) >= max_sessions:
                    break
            if len(class_sessions) >= max_sessions:
                break

        class_sessions = class_sessions[:max_sessions]
        n = len(class_sessions)

        # 统计加密会话数（载荷是否可读：可打印字符占比 < 70% 则视为加密）
        cls_encrypted = 0
        for s in class_sessions:
            payload_bytes = b""
            for p in s.packets:
                if p.payload:
                    payload_bytes += p.payload[:512]
                    if len(payload_bytes) >= 1024:
                        break
            if len(payload_bytes) > 0:
                printable = sum(1 for b in payload_bytes if 0x20 <= b <= 0x7e)
                if printable / len(payload_bytes) < 0.70:
                    cls_encrypted += 1
        enc_pct = cls_encrypted / n * 100 if n > 0 else 0
        enc_tag = "加密" if n > 0 and cls_encrypted > 0 else ("明文" if n > 0 else "")

        if n >= min_sessions:
            sessions.extend(class_sessions)
            labels.extend([lid] * n)
            print(f"    {name}: {n} 个会话  {enc_tag}")
        else:
            print(f"    ⚠ {name}: {n} 个会话（不足），跳过")

        class_tcp_stats[name] = {
            'sessions': n,
            'total_pkts': sum(s.total_packets for s in class_sessions),
            'retrans': cls_retrans,
            'ooo': cls_ooo,
        }

    # 格式化输出: TCP重传/乱序诊断（按类别）
    total_pkts = sum(v['total_pkts'] for v in class_tcp_stats.values())
    total_retrans = sum(v['retrans'] for v in class_tcp_stats.values())
    total_ooo = sum(v['ooo'] for v in class_tcp_stats.values())

    if total_pkts > 0:
        wire_total = total_pkts + total_retrans
        sep = "  " + "-" * 72
        print(f"\n  -- 重传/乱序诊断 --")
        print(sep)
        print(f"  {'类别':<12} {'会话':>4} {'总包数':>7} {'重传包':>7} {'重传率':>5} {'乱序包':>7} {'乱序率':>5}")
        print(sep)
        for cname in sorted(class_tcp_stats.keys()):
            v = class_tcp_stats[cname]
            if v['sessions'] == 0:
                continue
            wire = v['total_pkts'] + v['retrans']
            pr = v['retrans'] / wire * 100 if wire > 0 else 0
            po = v['ooo'] / wire * 100 if wire > 0 else 0
            print(f"  {cname:<14} {v['sessions']:>6} {wire:>10} {v['retrans']:>10} {pr:>7.1f}% {v['ooo']:>10} {po:>7.1f}%")
        print(sep)
        pr_all = total_retrans / wire_total * 100
        po_all = total_ooo / wire_total * 100
        print(f"  {'合计':<12} {len(sessions):>6} {wire_total:>10} {total_retrans:>10} {pr_all:>7.1f}% {total_ooo:>10} {po_all:>7.1f}%")
        print()

    used_ids = sorted(set(labels))
    remap = {old: new for new, old in enumerate(used_ids)}
    labels = [remap[l] for l in labels]
    label_names = {}
    for name, old_id in label_map.items():
        if old_id in remap:
            label_names[remap[old_id]] = name

    return sessions, labels, label_names


def extract_features(sessions, labels, max_packets=500, save_dir=None):
    """薄包装：提取序列统一走 src.features.runtime 共享实现（防双实现漂移）。"""
    from src.features.runtime import extract_features_from_sessions
    feature_records = extract_features_from_sessions(sessions, max_packets)
    df = pd.DataFrame(feature_records)
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, 'all_features.csv'), index=False)
    return df, list(labels) if hasattr(labels, '__iter__') else labels


def select_features(df_train, y_train, feat_cols, max_features=40):
    """在训练集上做特征选择"""
    selector = FeatureSelector(
        correlation_threshold=0.92,
        max_features=max_features,
        min_features=5,
    )
    selector.fit(df_train[feat_cols], y_train, feat_cols)
    selected = selector.selected_features_
    report = selector.get_selection_report()
    return selected, report


def train_model(X_train, y_train, selected):
    """训练XGBoost分类器（带进度条）"""
    from tqdm import tqdm
    import xgboost as xgb

    class TQDMCallback(xgb.callback.TrainingCallback):
        def __init__(self, n_estimators):
            self.pbar = tqdm(total=n_estimators, desc='XGBoost训练', unit='轮')
        def after_iteration(self, model, epoch, evals_log):
            self.pbar.update(1)
            return False
        def after_training(self, model):
            self.pbar.close()
            return model

    n_estimators = 200
    clf = XGBClassifier(
        n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
        min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
        random_state=42, use_label_encoder=False,
        eval_metric='mlogloss', verbosity=0,
        callbacks=[TQDMCallback(n_estimators)],
    )
    clf.fit(X_train[selected].values, y_train.values)
    # 清除callback，避免序列化报错
    clf.set_params(callbacks=[])
    return clf


def evaluate(clf, X_test, y_test, selected, label_names, confidence_threshold):
    """在测试集上评估"""
    X = X_test[selected].values
    y_true = y_test.values

    proba = clf.predict_proba(X)
    y_pred = np.argmax(proba, axis=1)
    confidences = np.max(proba, axis=1)

    # 带置信度过滤
    y_pred_filtered = y_pred.copy()
    y_pred_filtered[confidences < confidence_threshold] = -1

    return y_true, y_pred, y_pred_filtered, confidences


def tune_confidence_threshold(clf, X_val, y_val, selected,
                               thresholds=None):
    """
    在验证集上搜索最优置信度阈值。

    遍历候选阈值，选择 Macro-F1 最高的阈值。
    当多个阈值 F1 相同时，优先选更高的阈值（更保守）。

    Returns:
        (best_threshold, best_f1, details_df)
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.96, 0.05)

    X = X_val[selected].values
    y_true = y_val.values
    proba = clf.predict_proba(X)
    y_pred = np.argmax(proba, axis=1)
    confidences = np.max(proba, axis=1)

    results = []
    for t in thresholds:
        y_filt = y_pred.copy()
        y_filt[confidences < t] = -1
        # R29: 拒识(-1)计入分母——被拒known按预测错误处理，防止高阈值下
        # "拒掉大半样本后剩余全对"被选为best（F1含拒识惩罚）
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_filt, average='macro')
        acc = float(np.mean(y_true == y_filt))
        coverage = float(np.mean(confidences >= t))
        if coverage == 0:
            results.append({'threshold': t, 'f1': 0, 'accuracy': 0,
                            'coverage': 0})
            continue

        results.append({'threshold': round(t, 2), 'f1': round(f1, 4),
                       'accuracy': round(acc, 4),
                       'coverage': round(coverage, 4)})

    details = pd.DataFrame(results)
    # 选 F1 最高的，相同时取阈值更高的
    best = details.loc[details['f1'].idxmax()]
    return best['threshold'], best['f1'], details


def print_report(y_true, y_pred, y_pred_filtered, confidences,
                 label_names, confidence_threshold):
    """打印完整指标报告（含召回率/精确率/假阳率独立统计）"""
    # 纯分类性能
    print(f"\n  ── 纯分类性能（不过滤）──")
    base_metrics = _print_table(y_true, y_pred, label_names)

    # 带置信度过滤
    print(f"\n  ── 带置信度过滤（阈值={confidence_threshold}）──")
    _print_table(y_true, y_pred_filtered, label_names, allow_unknown=True)

    # 统计
    unknown = np.sum(y_pred_filtered == -1)
    print(f"\n  平均置信度: {np.mean(confidences):.4f}")
    print(f"  未知样本: {unknown}/{len(y_true)} ({unknown/len(y_true)*100:.1f}%)")

    # 独立输出召回率/精确率/假阳率
    print(f"\n  ── 核心指标汇总（题目要求）──")
    total = len(y_true)
    macro_recall = base_metrics['macro_recall']
    macro_precision = base_metrics['macro_precision']
    accuracy = base_metrics['accuracy']
    macro_f1 = base_metrics['macro_f1']

    # 假阳率：将非A类错误判为A类的比例，对所有类取平均
    fpr_per_class = []
    for lid in sorted(label_names.keys()):
        fp = np.sum((y_true != lid) & (y_pred == lid))
        tn = np.sum((y_true != lid) & (y_pred != lid))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fpr_per_class.append(fpr)
    macro_fpr = np.mean(fpr_per_class)

    print(f"  召回率(Macro):    {macro_recall:.4f}  {'✅' if macro_recall >= 0.98 else '⚠️'} 目标≥0.98")
    print(f"  精确率(Macro):    {macro_precision:.4f}  {'✅' if macro_precision >= 0.95 else '⚠️'} 目标≥0.95")
    print(f"  准确率:           {accuracy:.4f}  {'✅' if accuracy >= 0.95 else '⚠️'} 目标≥0.95")
    print(f"  Macro F1:         {macro_f1:.4f}  {'✅' if macro_f1 >= 0.80 else '⚠️'} 目标≥0.80")
    print(f"  假阳率(Macro):    {macro_fpr:.4f}  {'✅' if macro_fpr <= 0.05 else '⚠️'} 目标≤0.05")

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_recall': macro_recall,
        'macro_precision': macro_precision,
        'macro_fpr': macro_fpr,
        'per_class': base_metrics['per_class'],
    }


def _print_table(y_true, y_pred, label_names, allow_unknown=False):
    active = sorted(label_names.keys())
    print(f"\n  {'类别':<25} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'-'*57}")

    precisions, recalls, f1s = [], [], []
    per_class = {}
    for lid in active:
        tp = np.sum((y_true == lid) & (y_pred == lid))
        fp = np.sum((y_true != lid) & (y_pred == lid))
        fn = np.sum((y_true == lid) & (y_pred != lid))
        support = int(np.sum(y_true == lid))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        per_class[label_names[lid]] = {
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1': round(float(f1), 4),
            'support': support,
        }
        print(f"  {label_names[lid]:<25} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {support:>8}")

    print(f"  {'-'*57}")
    print(f"  {'Macro Average':<25} {np.mean(precisions):>8.4f} {np.mean(recalls):>8.4f} {np.mean(f1s):>8.4f}")

    if allow_unknown:
        mask = y_pred != -1
        if mask.sum() > 0:
            print(f"  准确率(已识别): {np.mean(y_true[mask] == y_pred[mask]):.4f}")
    else:
        print(f"  准确率: {np.mean(y_true == y_pred):.4f}")

    return {
        'accuracy': float(np.mean(y_true == y_pred)),
        'macro_precision': float(np.mean(precisions)),
        'macro_recall': float(np.mean(recalls)),
        'macro_f1': float(np.mean(f1s)),
        'per_class': per_class,
    }


def save_all_features(selected, df_train, y_train, feat_cols, output_dir):
    """保存完整的特征目录：基础196维 + 高级123维 + 选中40维"""
    from src.features.basic.feature_extractor import BasicFeatureExtractor
    from src.features.advanced.advanced_extractor import AdvancedFeatureExtractor

    basic_ext = BasicFeatureExtractor()
    advanced_ext = AdvancedFeatureExtractor()

    basic_catalog = basic_ext.get_feature_catalog()
    advanced_catalog = advanced_ext.get_feature_catalog()

    # 构建完整特征目录
    all_features = {
        'summary': {
            'total_features': len(basic_catalog) + len(advanced_catalog),
            'basic_count': len(basic_catalog),
            'advanced_count': len(advanced_catalog),
            'selected_count': len(selected),
            'selected_ratio': f"{len(selected)/(len(basic_catalog)+len(advanced_catalog))*100:.1f}%",
        },
        'basic_features': {
            'total': len(basic_catalog),
            'categories': {},
            'features': {},
        },
        'advanced_features': {
            'total': len(advanced_catalog),
            'categories': {},
            'features': {},
        },
        'selected_features': selected,
    }

    # 按子类别分组基础特征
    for name, info in basic_catalog.items():
        subcat = info['subcategory']
        all_features['basic_features']['categories'].setdefault(subcat, []).append(name)
        feat_info = {'description': info['description'], 'subcategory': subcat}
        # 添加统计信息（如果在训练集中）
        if name in df_train.columns:
            col = df_train[name].dropna()
            if len(col) > 0 and np.issubdtype(col.dtype, np.number):
                feat_info['stats'] = {
                    'min': round(float(col.min()), 4),
                    'max': round(float(col.max()), 4),
                    'mean': round(float(col.mean()), 4),
                    'std': round(float(col.std()), 4),
                }
        all_features['basic_features']['features'][name] = feat_info

    # 按子类别分组高级特征
    for name, info in advanced_catalog.items():
        subcat = info['subcategory']
        all_features['advanced_features']['categories'].setdefault(subcat, []).append(name)
        feat_info = {'description': info['description'], 'subcategory': subcat}
        if name in df_train.columns:
            col = df_train[name].dropna()
            if len(col) > 0 and np.issubdtype(col.dtype, np.number):
                feat_info['stats'] = {
                    'min': round(float(col.min()), 4),
                    'max': round(float(col.max()), 4),
                    'mean': round(float(col.mean()), 4),
                    'std': round(float(col.std()), 4),
                }
        all_features['advanced_features']['features'][name] = feat_info

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    feat_path = str(output_path / 'all_features.json')
    with open(feat_path, 'w', encoding='utf-8') as f:
        json.dump(all_features, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print(f"  总特征: {all_features['summary']['total_features']} "
          f"(基础{all_features['summary']['basic_count']} + "
          f"高级{all_features['summary']['advanced_count']})")
    print(f"  选中: {all_features['summary']['selected_count']} "
          f"({all_features['summary']['selected_ratio']})")
    print(f"  基础特征类别:")
    for cat, feats in all_features['basic_features']['categories'].items():
        print(f"    {cat}: {len(feats)} 维")
    print(f"  高级特征类别:")
    for cat, feats in all_features['advanced_features']['categories'].items():
        print(f"    {cat}: {len(feats)} 维")
    print(f"  完整特征目录: {feat_path}")

    return feat_path


def save_outputs(clf, selected, label_names, confidence_threshold,
                 output_dir, df_train=None, y_train=None, feat_cols=None):
    """保存模型+规则(含特征定义)+DPI输出格式"""
    from src.engine.model_io import save_model_bundle

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 规则文件（含完整特征定义和可解释规则）──
    gen = OptimizedRuleGenerator(confidence_threshold=confidence_threshold)
    gen.classifier = clf
    gen.label_names = label_names

    # 生成特征定义
    if df_train is not None and y_train is not None and feat_cols is not None:
        gen.feature_importances_ = dict(zip(selected, clf.feature_importances_))
        gen.generate_feature_definitions(df_train[feat_cols], selected, y_train, label_names)

    # 生成可解释规则（决策树 + 统计）
    if df_train is not None and y_train is not None and feat_cols is not None:
        X_sel = df_train[selected].fillna(0)
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
        tree.fit(X_sel, y_train)
        tree_rules = gen._extract_tree_rules(X_sel, y_train, selected, label_names)
        stat_rules = gen._generate_statistical_rules(
            df_train[feat_cols], y_train, selected, label_names)
    else:
        tree_rules = []
        stat_rules = []

    gen.rules = [
        # 集成规则
        {'id': 'ENS_0001', 'name': 'XGBoost集成分类器', 'type': 'ensemble_classifier',
         'confidence_threshold': confidence_threshold, 'num_classes': len(label_names),
         'selected_features': selected},
    ] + tree_rules + stat_rules

    rule_path = str(output_path / 'rules.yaml')
    gen.export_yaml(rule_path, label_names)

    # ── 模型文件（pickle兼容 + XGBoost原生JSON）──
    model_path = str(output_path / 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'classifier': clf, 'selected_features': selected,
                     'label_names': label_names,
                     'confidence_threshold': confidence_threshold}, f)

    # ── 独立推理模型包（XGBoost JSON + 特征列表 + 引擎配置）──
    bundle_paths = save_model_bundle(
        clf, selected, label_names, confidence_threshold, output_dir)

    # ── 特征有效率报告 ──
    if df_train is not None and y_train is not None and feat_cols is not None:
        feat_effective = _compute_feature_effectiveness(
            df_train[feat_cols], y_train, selected)
        eff_path = str(output_path / 'feature_effectiveness.json')
        with open(eff_path, 'w', encoding='utf-8') as f:
            json.dump(feat_effective, f, indent=2, ensure_ascii=False)
        print(f"  特征有效率: {feat_effective['effective_rate']:.1%} "
              f"({feat_effective['effective_count']}/{feat_effective['total_selected']})")
        print(f"  有效特征报告: {eff_path}")

    print(f"  规则文件: {rule_path}")
    print(f"  模型文件: {model_path}")
    print(f"  模型JSON: {bundle_paths['model']}")
    print(f"  引擎配置: {bundle_paths['config']}")

    return rule_path, model_path


def save_dpi_output(clf, X_test, y_test, selected, label_names,
                    confidence_threshold, output_dir):
    """保存DPI标准识别结果输出"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X = X_test[selected].values
    y_true = y_test.values
    proba = clf.predict_proba(X)
    y_pred = np.argmax(proba, axis=1)
    confidences = np.max(proba, axis=1)

        # R23: 低于置信度阈值的样本预测置为unknown（拒识写入预测结果，而非仅标记）
    y_pred[confidences < confidence_threshold] = -1
# DPI标准输出格式
    dpi_results = []
    for i in range(len(y_true)):
        pred_label = label_names.get(int(y_pred[i]), 'unknown')
        true_label = label_names.get(int(y_true[i]), 'unknown')
        confidence = float(confidences[i])

        dpi_results.append({
            'session_id': i,
            'true_label': true_label,
            'predicted_app': pred_label.split('_')[0] if '_' in pred_label else pred_label,
            'predicted_behavior': '_'.join(pred_label.split('_')[1:]) if '_' in pred_label else 'general',
            'predicted_label': pred_label,
            'confidence': round(confidence, 6),
            'is_correct': bool(y_true[i] == y_pred[i]),
            'meets_threshold': bool(confidence >= confidence_threshold),
        })

    # 保存JSON
    dpi_path = str(output_path / 'dpi_results.json')
    with open(dpi_path, 'w', encoding='utf-8') as f:
        json.dump({
            'version': '1.0',
            'engine': 'OptimizedDPIEngine',
            'confidence_threshold': confidence_threshold,
            'total_sessions': len(dpi_results),
            'correct': sum(1 for r in dpi_results if r['is_correct']),
            'above_threshold': sum(1 for r in dpi_results if r['meets_threshold']),
            'results': dpi_results,
        }, f, indent=2, ensure_ascii=False)

    # 保存CSV
    csv_path = str(output_path / 'dpi_results.csv')
    pd.DataFrame(dpi_results).to_csv(csv_path, index=False)

    print(f"  DPI结果(JSON): {dpi_path}")
    print(f"  DPI结果(CSV):  {csv_path}")

    return dpi_path


def _compute_feature_effectiveness(df, y, selected):
    """
    计算特征维度输出有效率。

    定义：选中特征中，对分类有显著贡献的特征占比。
    判定标准：该特征在XGBoost中的重要性 > 平均重要性的一半，
              且该特征的类间区分度 > 0。
    """
    from scipy.stats import f_oneway

    effective = []
    ineffective = []
    details = {}

    for feat in selected:
        if feat not in df.columns:
            ineffective.append(feat)
            continue

        # ANOVA F检验：各类别均值是否有显著差异
        groups = [df.loc[y == lid, feat].dropna().values
                  for lid in sorted(y.unique()) if (y == lid).sum() > 1]
        groups = [g for g in groups if len(g) > 1]

        if len(groups) >= 2:
            try:
                f_stat, p_value = f_oneway(*groups)
                is_effective = p_value < 0.05  # 95%置信度
            except Exception:
                is_effective = False
                p_value = 1.0
        else:
            is_effective = False
            p_value = 1.0

        details[feat] = {
            'effective': bool(is_effective),
            'p_value': round(float(p_value), 6),
        }

        if is_effective:
            effective.append(feat)
        else:
            ineffective.append(feat)

    effective_rate = len(effective) / len(selected) if selected else 0

    return {
        'total_selected': len(selected),
        'effective_count': len(effective),
        'ineffective_count': len(ineffective),
        'effective_rate': round(effective_rate, 4),
        'effective_features': effective,
        'ineffective_features': ineffective,
        'details': details,
    }


# =====================================================================
#  主流程
# =====================================================================

def run(train_dir=None, val_dir=None, test_dir=None, data_dir=None,
        output_dir='output/results', confidence_threshold=0.65,
        max_features=60, max_sessions=300, max_file_mb=500, max_packets=500,
        no_eval=False, save_features=False):

    print("\n" + "★" * 60)
    print("  通用加密流量特征挖掘流水线")
    print("★" * 60 + "\n")

    t0 = time.time()

    # ────────────────────────────────────────
    #  Step 1: 解析数据
    # ────────────────────────────────────────
    print("=" * 60)
    print("Step 1: 解析PCAP")
    print("=" * 60)

    label_names = {}  # {id: name}

    if train_dir and test_dir:
        # ── 模式A: 预划分目录（目录名=类别名）──
        print(f"\n  模式: 预划分目录（目录名 = 类别名）")
        print(f"  训练集: {train_dir}")
        train_sessions, train_labels, label_names = parse_pcap_dir(
            train_dir, label_names, max_sessions=max_sessions)

        val_sessions, val_labels = [], []
        if val_dir:
            print(f"  验证集: {val_dir}")
            val_sessions, val_labels, label_names = parse_pcap_dir(
                val_dir, label_names, max_sessions=max_sessions)

        print(f"  测试集: {test_dir}")
        test_sessions, test_labels, label_names = parse_pcap_dir(
            test_dir, label_names, max_sessions=max_sessions)

        if not train_sessions:
            print("错误：训练集无有效数据"); return None

        print(f"\n  训练集: {len(train_sessions)} 个会话")
        if val_sessions:
            print(f"  验证集: {len(val_sessions)} 个会话")
        print(f"  测试集: {len(test_sessions)} 个会话")

    elif data_dir:
        # ── 模式B: 单目录（文件名=类别名，自动随机划分 6:2:2）──
        print(f"\n  模式: 单目录自动随机划分（文件名 = 类别名）")
        print(f"  数据目录: {data_dir}")
        all_sessions, all_labels, label_names = parse_pcap_files(
            data_dir, label_names,
            max_sessions=max_sessions, max_file_mb=max_file_mb)

        if not all_sessions:
            print("错误：无有效数据"); return None

        # 6:2:2 随机划分
        X_temp, test_sessions, y_temp, test_labels = train_test_split(
            all_sessions, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
        train_sessions, val_sessions, train_labels, val_labels = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

        print(f"\n  训练集: {len(train_sessions)} 个会话 (60%)")
        print(f"  验证集: {len(val_sessions)} 个会话 (20%)")
        print(f"  测试集: {len(test_sessions)} 个会话 (20%)")
    else:
        print("错误：必须指定 --train + --test 或 --data"); return None

    print(f"  类别: {len(label_names)} 个")
    for lid, name in sorted(label_names.items()):
        parts = [f"训={train_labels.count(lid)}"]
        if val_labels:
            parts.append(f"验={val_labels.count(lid)}")
        parts.append(f"测={test_labels.count(lid)}")
        print(f"    [{lid}] {name}: {', '.join(parts)}")

    # ────────────────────────────────────────
    #  Step 2: 特征提取
    # ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: 特征提取")
    print("=" * 60)

    feat_save_dir = str(Path(output_dir).parent / 'features') if save_features else None

    print("  训练集特征提取...")
    df_train, y_train, feat_cols = extract_features(
        train_sessions, train_labels, save_dir=feat_save_dir)
    print(f"    {df_train.shape[0]} x {len(feat_cols)} 特征")

    if test_sessions and not no_eval:
        print("  测试集特征提取...")
        df_test, y_test, _ = extract_features(test_sessions, test_labels)
        print(f"    {df_test.shape[0]} × {len(feat_cols)} 特征")
    else:
        df_test, y_test = None, None

    # ────────────────────────────────────────
    #  Step 3: 特征选择（仅在训练集上）
    # ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: 特征选择（训练集）")
    print("=" * 60)

    selected, report = select_features(df_train, y_train, feat_cols, max_features)
    print(f"  {len(feat_cols)} → {len(selected)} 维")
    print(f"  Top-5:")
    for f, imp in report['importance_ranking'][:5]:
        print(f"    {f}: {imp:.4f}")

    # ────────────────────────────────────────
    #  Step 4: 训练（仅用训练集）
    # ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: 训练XGBoost")
    print("=" * 60)

    clf = train_model(df_train, y_train, selected)

    # 交叉验证（重新训练一个无callback的分类器用于CV）
    from copy import deepcopy
    clf_cv = deepcopy(clf)
    cv_scores = cross_val_score(clf_cv, df_train[selected], y_train, cv=5, scoring='f1_macro')
    print(f"  训练集 5折CV Macro-F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ────────────────────────────────────────
    #  Step 5: 评估（验证集 + 测试集）
    # ────────────────────────────────────────
    metrics = None
    df_val, y_val = None, None

    if not no_eval:
        # 验证集评估 + 置信度阈值调优
        if val_sessions:
            print("\n" + "=" * 60)
            print("Step 5a: 验证集评估 + 置信度阈值调优")
            print("=" * 60)
            df_val, y_val, _ = extract_features(val_sessions, val_labels)
            print(f"    {df_val.shape[0]} x {len(feat_cols)} 特征")

            # 在验证集上搜索最优阈值
            best_t, best_f1, tune_details = tune_confidence_threshold(
                clf, df_val, y_val, selected)
            print(f"\n  阈值搜索结果:")
            for _, row in tune_details.iterrows():
                mark = " <-- best" if row['threshold'] == best_t else ""
                print(f"    threshold={row['threshold']:.2f}  "
                      f"F1={row['f1']:.4f}  acc={row['accuracy']:.4f}  "
                      f"coverage={row['coverage']:.4f}{mark}")
            print(f"\n  最优阈值: {best_t:.2f} (原默认: {confidence_threshold:.2f})")
            confidence_threshold = float(best_t)

            # 用最优阈值评估验证集
            y_true_v, y_pred_v, y_pred_vf, confs_v = evaluate(
                clf, df_val, y_val, selected, label_names, confidence_threshold)
            val_metrics = print_report(y_true_v, y_pred_v, y_pred_vf, confs_v,
                                      label_names, confidence_threshold)

        # 测试集评估
        if test_sessions:
            print("\n" + "=" * 60)
            print("Step 5b: 测试集评估")
            print("=" * 60)
            df_test, y_test, _ = extract_features(test_sessions, test_labels)
            print(f"    {df_test.shape[0]} × {len(feat_cols)} 特征")
            y_true, y_pred, y_pred_filt, confs = evaluate(
                clf, df_test, y_test, selected, label_names, confidence_threshold)
            metrics = print_report(y_true, y_pred, y_pred_filt, confs,
                                  label_names, confidence_threshold)

    # ────────────────────────────────────────
    #  Step 6: 保存（含特征定义、可解释规则、特征有效率）
    # ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: 保存模型、规则和特征报告")
    print("=" * 60)

    # ── 完整特征目录（基础196 + 高级123 + 选中40）──
    save_all_features(selected, df_train, y_train, feat_cols, output_dir)

    rule_path, model_path = save_outputs(
        clf, selected, label_names, confidence_threshold, output_dir,
        df_train=df_train, y_train=y_train, feat_cols=feat_cols)

    # ── DPI标准输出 ──
    if df_test is not None and y_test is not None and not no_eval:
        print("\n  生成DPI识别结果...")
        save_dpi_output(clf, df_test, y_test, selected, label_names,
                       confidence_threshold, output_dir)

    elapsed = time.time() - t0

    # ────────────────────────────────────────
    #  达标检查（对标题目全部指标）
    # ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  达标检查（总耗时 {elapsed:.1f}s）")
    print(f"{'='*60}")

    if metrics:
        checks = [
            ("召回率(Macro) ≥ 0.98", metrics['macro_recall'] >= 0.98,
             f"{metrics['macro_recall']:.4f}"),
            ("精确率(Macro) ≥ 0.95", metrics['macro_precision'] >= 0.95,
             f"{metrics['macro_precision']:.4f}"),
            ("准确率 ≥ 0.95", metrics['accuracy'] >= 0.95,
             f"{metrics['accuracy']:.4f}"),
            ("Macro F1 ≥ 0.80", metrics['macro_f1'] >= 0.80,
             f"{metrics['macro_f1']:.4f}"),
            ("假阳率(Macro) ≤ 0.05", metrics['macro_fpr'] <= 0.05,
             f"{metrics['macro_fpr']:.4f}"),
        ]
        for name, passed, detail in checks:
            s = "✅" if passed else "⚠️"
            print(f"  {s} {name} ({detail})")

    return metrics


# =====================================================================
#  CLI
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='通用加密流量特征挖掘流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 模式A: 预划分目录（目录名=类别名）
  python tests/test_generic_pipeline.py --train data/train --val data/val --test data/test

  # 模式B: 单目录（文件名=类别名，自动随机划分 6:2:2）
  python tests/test_generic_pipeline.py --data data/USTC-TFC2016-master
  python tests/test_generic_pipeline.py --data data/samples_demo

  # 只训练不评测
  python tests/test_generic_pipeline.py --data data/USTC-TFC2016-master --no-eval
        """
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--data', help='单目录模式（文件名=类别名，自动划分6:2:2）')
    mode.add_argument('--train', help='预划分模式：训练集目录（目录名=类别名）')

    parser.add_argument('--val', help='预划分模式：验证集目录')
    parser.add_argument('--test', help='预划分模式：测试集目录')
    parser.add_argument('--output', default='output/results', help='输出目录')
    parser.add_argument('--confidence', type=float, default=0.65, help='置信度阈值')
    parser.add_argument('--max-features', type=int, default=60, help='最大特征数')
    parser.add_argument('--max-sessions', type=int, default=300, help='每类最大会话数')
    parser.add_argument('--max-file-mb', type=int, default=500, help='单文件大小上限MB')
    parser.add_argument('--max-packets', type=int, default=500, help='每会话最大包数')
    parser.add_argument('--no-eval', action='store_true', help='只训练，不评测')
    parser.add_argument('--save-features', action='store_true',
                        help='保存提取的特征向量到 output/features/')

    args = parser.parse_args()

    run(
        train_dir=args.train,
        val_dir=args.val,
        test_dir=args.test,
        data_dir=args.data,
        output_dir=args.output,
        confidence_threshold=args.confidence,
        max_features=args.max_features,
        max_sessions=args.max_sessions,
        max_file_mb=args.max_file_mb,
        max_packets=args.max_packets,
        no_eval=args.no_eval,
        save_features=args.save_features,
    )
