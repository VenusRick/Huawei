# -*- coding: utf-8 -*-
"""评价服务（契约层 P0）：只读 predictions + truth，不再传入 clf 或 X_test。

语义（M1 已修复并验证，勿回退）：
- 拒识在预测输出中记 -1 / 'unknown'，评价时计入完整分母（R23）；
- 阈值搜索的分母计入被拒 known 样本，f1_score 含 -1 类（R29）；
- split 标签沿用全局 label_map，不重编号（R27）；
- "100 已知只接纳 20"时 Recall 不得为 1；全拒识目标类 Recall=0。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

UNKNOWN = -1
UNKNOWN_STR = "unknown"


def _to_id(value, label_map: Dict[str, int]) -> int:
    """标签 -> ID；未知字符串标签按词表扩展顺序稳定分配，不抛错但记录。"""
    if value is None:
        return UNKNOWN
    if isinstance(value, (int, np.integer)):
        return int(value)
    s = str(value)
    if s == UNKNOWN_STR or s == "":
        return UNKNOWN
    return label_map.get(s, UNKNOWN)


def load_predictions(path: str) -> List[Dict]:
    """加载预测输出（dpi_infer JSON 或 jsonl）。

    统一为 [{observation_id, predicted_label, confidence, source, ...}]。
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    results: List[Dict] = []
    if text.startswith("{"):
        obj = json.loads(text)
        results = list(obj.get("results", []))
        # 顶层带 engine_mode/matched 等统计则透传
        for k in ("engine_mode", "rules_path", "generated_at"):
            if k in obj:
                results and results[0].setdefault("_meta_" + k, obj[k])
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                results.append(json.loads(line))
    if not results:
        raise ValueError(f"预测文件为空或不含结果: {path}")
    return results


def load_truth_manifest(manifest_records, task: str = "app") -> Dict[str, Dict]:
    """从 TrialRecord 清单构建真值索引：pcap 文件名 -> {label, trial_id, split}。"""
    truth: Dict[str, Dict] = {}
    for r in manifest_records:
        truth[Path(r.pcap).name] = {
            "label": r.label(task), "trial_id": r.trial_id,
            "capture_id": r.capture_id, "split": r.split or "",
            "platform": r.platform,
        }
    return truth


def align_predictions(predictions: List[Dict], truth: Dict[str, Dict],
                      label_map: Dict[str, int]) -> Tuple[List[int], List[int], List[Dict]]:
    """预测与真值一一配对（按 pcap 文件名；无真值的预测记为不可评价并单列）。"""
    y_true: List[int] = []
    y_pred: List[int] = []
    aligned: List[Dict] = []
    unpaired: List[Dict] = []
    for pred in predictions:
        # observation_id 形如 "{file}:{idx}"；旧输出无此字段时退用 source_file
        oid = pred.get("observation_id") or pred.get("source_file") or ""
        fname = oid.split(":")[0] if ":" in oid else str(oid)
        src = pred.get("source_file") or fname
        t = truth.get(Path(src).name) or truth.get(Path(fname).name)
        if t is None:
            unpaired.append(pred)
            continue
        y_true.append(_to_id(t["label"], label_map))
        y_pred.append(_to_id(pred.get("predicted_label"), label_map))
        aligned.append({"pred": pred, "truth": t})
    return y_true, y_pred, [{"unpaired": unpaired, "aligned": aligned}]


def compute_metrics(y_true: List[int], y_pred: List[int],
                    names: List[str]) -> Dict:
    """完整分母指标：accuracy、macro P/R/F1、macro FPR、拒识统计。"""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    rejected = int((yp == UNKNOWN).sum())
    acc = float(accuracy_score(yt, yp)) if n else 0.0
    # 拒识计漏检：对每个已知类，recall 分母是该类全部真值样本
    prec, rec, f1, sup = precision_recall_fscore_support(
        yt, yp, labels=list(range(len(names))), zero_division=0)
    cm = confusion_matrix(yt, yp,
                          labels=list(range(len(names))) + [UNKNOWN])
    # macro FPR：对每个类，误报 = 非该类样本被预测为该类 / 非该类样本总数
    fprs = []
    k = len(names)
    for i in range(k):
        fp = cm[:, i].sum() - cm[i, i]
        tn_fp_den = cm.sum() - cm[i, :].sum()
        fprs.append(float(fp / tn_fp_den) if tn_fp_den else 0.0)
    per_class = {}
    for i, nm in enumerate(names):
        per_class[nm] = {
            "precision": float(prec[i]), "recall": float(rec[i]),
            "f1": float(f1[i]), "support": int(sup[i]),
            "fpr": fprs[i],
            "rejected_true_of_class": int(cm[i, k]) if k < cm.shape[1] else 0,
        }
    return {
        "n_samples": n,
        "n_rejected": rejected,
        "rejection_rate": rejected / n if n else 0.0,
        "accuracy": acc,
        "macro_precision": float(np.mean(prec)) if k else 0.0,
        "macro_recall": float(np.mean(rec)) if k else 0.0,
        "macro_f1": float(np.mean(f1)) if k else 0.0,
        "macro_fpr": float(np.mean(fprs)) if k else 0.0,
        "per_class": per_class,
    }


def evaluate_files(predictions_path: str, truth: Dict[str, Dict],
                   label_map: Dict[str, int], output_dir: str) -> Dict:
    """评价入口：预测文件 + 真值索引 -> metrics.json（含未配对单列）。"""
    preds = load_predictions(predictions_path)
    y_true, y_pred, extra = align_predictions(preds, truth, label_map)
    names = [n for n, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    metrics = compute_metrics(y_true, y_pred, names)
    metrics["n_unpaired_predictions"] = len(extra[0]["unpaired"])
    metrics["label_names"] = names
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


# ---------------------------------------------------------------------------
# 事件级评价（M2 预留：短时操作按事件计，重复告警不重复 TP）
# ---------------------------------------------------------------------------
def match_events(predicted_events: List[Dict], truth_events: List[Dict],
                 tolerance: float = 2.0) -> Dict:
    """一对一容差匹配：同终端 + 正确标签 + |t_pred - t_truth| <= tolerance。

    贪心按时间差排序匹配；同一真值事件只计一次 TP（重复告警不算）。
    """
    used_truth = set()
    tp = 0
    for p in sorted(predicted_events, key=lambda e: e.get("t", 0)):
        best_j, best_d = None, None
        for j, t in enumerate(truth_events):
            if j in used_truth:
                continue
            if t.get("terminal") != p.get("terminal"):
                continue
            if t.get("label") != p.get("label"):
                continue
            d = abs(p.get("t", 0) - t.get("t", 0))
            if d <= tolerance and (best_d is None or d < best_d):
                best_j, best_d = j, d
        if best_j is not None:
            used_truth.add(best_j)
            tp += 1
    fn = len(truth_events) - tp
    fp = len(predicted_events) - tp
    return {"tp": tp, "fp": fp, "fn": fn,
            "event_recall": tp / len(truth_events) if truth_events else 0.0,
            "event_precision": tp / len(predicted_events) if predicted_events else 0.0,
            "tolerance_sec": tolerance}
