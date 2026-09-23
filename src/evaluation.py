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
# 词表外数值ID哨兵（审计 2026-09-18 第二轮）：预测/真值出现词表外数字ID时
# 不再被 confusion_matrix(labels=...) 静默丢出分母，映射为独立列并单列计数。
OUT_OF_VOCAB = -2
_PREDICTION_KEYS = ("predicted_label", "prediction", "label", "result")


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
    """加载预测输出（dpi_infer JSON、JSON 数组或 jsonl）。

    统一为 [{observation_id, predicted_label, confidence, source, ...}]。
    格式判定：先尝试整体 JSON 解析；失败则按行解析 JSONL（首行以'{'
    开头的 JSONL 不再被误判为单 JSON——审计 2026-09-18）。
    单个 JSON 对象：带 results 包装按包装取；无包装但含预测键
    （predicted_label 等）按单条预测记录处理，不再"解析成功却返回空"
    （审计 2026-09-18 第二轮）。
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"预测文件为空: {path}")
    results: List[Dict] = []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            results = list(obj)
        elif isinstance(obj, dict):
            if isinstance(obj.get("results"), list):
                results = list(obj["results"])
                # 顶层带 engine_mode/matched 等统计则透传
                for k in ("engine_mode", "rules_path", "generated_at"):
                    if k in obj and results:
                        results[0].setdefault("_meta_" + k, obj[k])
            elif any(k in obj for k in _PREDICTION_KEYS):
                # 单行 JSONL 预测对象（无 results 包装）= 1 条预测
                results = [obj]
            else:
                raise ValueError(
                    f"预测JSON对象无 results 包装且不含预测键"
                    f"（{'/'.join(_PREDICTION_KEYS)}）: {path}")
    except json.JSONDecodeError:
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"预测文件第{lineno}行不是合法JSON: {e}") from e
    if not results:
        raise ValueError(f"预测文件为空或不含结果: {path}")
    return results


def load_truth_manifest(manifest_records, task: str = "app") -> Dict[str, Dict]:
    """从 TrialRecord 清单构建真值索引：pcap 文件名 -> {label, trial_id, split}。

    重名 basename 显式报错（评价按 basename 配对，静默覆盖=默默丢分母）。
    """
    truth: Dict[str, Dict] = {}
    seen: Dict[str, str] = {}
    for r in manifest_records:
        name = Path(r.pcap).name
        if name in truth:
            raise ValueError(
                f"真值清单 basename 重复: {name}"
                f"（{seen[name]} 与 {r.trial_id}；按文件名配对会静默覆盖，"
                f"请改用唯一文件名或按 trial_id 评价）")
        seen[name] = r.trial_id
        truth[name] = {
            "label": r.label(task), "trial_id": r.trial_id,
            "capture_id": r.capture_id, "split": r.split or "",
            "platform": r.platform,
            "record": r,
        }
    return truth


def align_predictions(predictions: List[Dict], truth: Dict[str, Dict],
                      label_map: Dict[str, int],
                      task: str = "app") -> Tuple[List[int], List[int], List[Dict]]:
    """预测与真值一一配对（按 pcap 文件名；无真值的预测记为不可评价并单列）。

    behavior 任务：预测是窗口（带 window_start/window_end），真值按
    label_window(试次区间, 窗口区间) 逐窗判定——不用整个 pcap 的行为
    标签覆盖背景窗（审计 2026-09-18）。
    重复 observation_id（同一观测被预测两次）只计首条，计数单列返回
    （审计 2026-09-18 第二轮：重复入分母=同一会话计两次成绩）。
    仅当预测行携带真实 observation_id 时才去重——无 observation_id 的
    旧输出按行保留（不把同文件多会话误当重复）。
    """
    from src.data_manifest import label_window
    y_true: List[int] = []
    y_pred: List[int] = []
    aligned: List[Dict] = []
    unpaired: List[Dict] = []
    seen_obs: Dict[str, int] = {}
    duplicates: List[Dict] = []
    for pred in predictions:
        # observation_id 形如 "{file}:{idx}"；旧输出无此字段时退用 source_file
        oid = pred.get("observation_id") or pred.get("source_file") or ""
        fname = oid.split(":")[0] if ":" in oid else str(oid)
        src = pred.get("source_file") or fname
        dedup_key = pred.get("observation_id")
        if dedup_key and dedup_key in seen_obs:
            seen_obs[dedup_key] += 1
            duplicates.append(pred)
            continue          # 同一观测的重复预测不重复入分母
        t = truth.get(Path(src).name) or truth.get(Path(fname).name)
        if t is None:
            unpaired.append(pred)
            continue
        if dedup_key:
            seen_obs[dedup_key] = 1
        if (task == "behavior"
                and pred.get("window_start") is not None
                and pred.get("window_end") is not None):
            true_label = label_window(t["record"],
                                      float(pred["window_start"]),
                                      float(pred["window_end"]))
        else:
            true_label = t["label"]
        y_true.append(_to_id(true_label, label_map))
        y_pred.append(_to_id(pred.get("predicted_label"), label_map))
        aligned.append({"pred": pred, "truth": t, "true_label": true_label})
    return y_true, y_pred, [{"unpaired": unpaired, "aligned": aligned,
                             "duplicates": duplicates}]


def compute_metrics(y_true: List[int], y_pred: List[int],
                    names: List[str],
                    label_ids: Optional[List[int]] = None) -> Dict:
    """完整分母指标：accuracy、macro P/R/F1、macro FPR、拒识统计。

    label_ids: names 对应的标签 ID（默认 0..k-1）。词表 ID 非连续时必须
    显式给出，否则 sklearn labels 与 names 错位（审计 2026-09-18）。
    词表外数字 ID（真值或预测）不再被 confusion_matrix(labels=...) 静默
    丢出分母：映射为独立 OOV 列/行进入矩阵，并在 out_of_vocab 单列计数
    （审计 2026-09-18 第二轮）。
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    ids = list(label_ids) if label_ids is not None else list(range(len(names)))
    rejected = int((yp == UNKNOWN).sum())
    acc = float(accuracy_score(yt, yp)) if n else 0.0
    # 词表外 ID 显式接管（不静默丢分母）：已知类 = ids ∪ {UNKNOWN}
    known = set(ids) | {UNKNOWN}
    extra = sorted((set(yt.tolist()) | set(yp.tolist())) - known)
    oov_pred_counts = {int(v): int((yp == v).sum()) for v in extra
                       if int((yp == v).sum())}
    oov_true_counts = {int(v): int((yt == v).sum()) for v in extra
                       if int((yt == v).sum())}
    if extra:
        # OOV 值归一到哨兵 -2 进矩阵（保留完整 n 行列，不与拒识混同）
        yt = np.where(np.isin(yt, extra), OUT_OF_VOCAB, yt)
        yp = np.where(np.isin(yp, extra), OUT_OF_VOCAB, yp)
        n_oov_pred = int(sum(oov_pred_counts.values()))
        n_oov_true = int(sum(oov_true_counts.values()))
    else:
        n_oov_pred = n_oov_true = 0
    # 拒识计漏检：对每个已知类，recall 分母是该类全部真值样本
    prec, rec, f1, sup = precision_recall_fscore_support(
        yt, yp, labels=ids, zero_division=0)
    cm_labels = ids + [UNKNOWN] + ([OUT_OF_VOCAB] if extra else [])
    cm = confusion_matrix(yt, yp, labels=cm_labels)
    # macro FPR：对每个类，误报 = 非该类样本被预测为该类 / 非该类样本总数
    fprs = []
    k = len(ids)
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
    out = {
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
    if extra:
        out["out_of_vocab"] = {
            "n_pred": n_oov_pred, "n_true": n_oov_true,
            "pred_id_counts": oov_pred_counts,
            "true_id_counts": oov_true_counts,
            "note": "词表外数字ID按独立列计入矩阵分母（不静默丢、"
                    "不与拒识unknown混同）；出现即说明预测/真值词表"
                    "与label_map不一致，应先对齐词表",
        }
    return out


def aggregate_windows_to_events(results: List[Dict],
                                step_gap: float = 1.5) -> List[Dict]:
    """行为窗口预测 -> 事件序列（同源文件同终端同标签的相邻窗口合并）。

    事件 t 取连续段首窗 window_start；供 match_events 一对一匹配。
    - 按终端分组：同一文件里不同终端的窗口不互相合并（审计
      2026-09-18 第二轮：按文件合并会把两终端的同标签近时窗错并）；
    - background 不是行为事件：显式背景类预测只进背景FPR口径，
      不进事件序列（否则被 match_events 记为行为FP）。
    """
    by_key: Dict[Tuple[str, str], List[Dict]] = {}
    for r in results:
        if r.get("predicted_label") in (None, UNKNOWN_STR, "", "background"):
            continue
        key = (r.get("source_file", ""), r.get("terminal", ""))
        by_key.setdefault(key, []).append(r)
    events: List[Dict] = []
    for (fname, terminal), rows in by_key.items():
        rows.sort(key=lambda r: r.get("window_start") or 0.0)
        cur = None
        for r in rows:
            if (cur is not None
                    and r.get("predicted_label") == cur["label"]
                    and (r.get("window_start", 0.0)
                         - cur["end"]) <= step_gap):
                cur["end"] = r.get("window_end", cur["end"])
            else:
                if cur is not None:
                    events.append(cur)
                cur = {"label": r["predicted_label"],
                       "t": r.get("window_start", 0.0),
                       "end": r.get("window_end", 0.0),
                       "terminal": terminal,
                       "source_file": fname}
        if cur is not None:
            events.append(cur)
    return events


def evaluate_files(predictions_path: str, truth: Dict[str, Dict],
                   label_map: Dict[str, int], output_dir: str,
                   task: str = "app") -> Dict:
    """评价入口：预测文件 + 真值索引 -> metrics.json（含未配对单列）。

    - 空 truth / 空预测：显式抛错（无真值即不可评价，不产出假指标）；
    - behavior：逐窗 label_window 真值 + 背景窗误报率 + 事件级一对一匹配
      （事件按 source_file 隔离，不跨试次配对）；
    - 标签 ID 与名称按 label_map 实际取值对齐（支持非连续 ID）；
    - truth 覆盖率单列：哪些真值文件没有任何预测（漏检文件不再隐形，
      不用"有预测的文件数"伪造分母——审计 2026-09-18 第二轮）；
    - 重复 observation_id 只计首条并计数（不重复入分母）。
    """
    if not truth:
        raise ValueError("真值索引为空：无真值即不可评价（不产出指标）")
    preds = load_predictions(predictions_path)
    y_true, y_pred, extra = align_predictions(preds, truth, label_map, task)
    ordered = sorted(label_map.items(), key=lambda kv: kv[1])
    names = [n for n, _ in ordered]
    label_ids = [i for _, i in ordered]
    # behavior：背景窗真值不在行为词表中，按 UNKNOWN(-1) 行进入混淆矩阵
    # （预测为任一已知类即计入该类 FP 分子；拒识(unknown)按正确处理），
    # 完整分母保留，不静默丢窗、不新增零支持类拉低 macro。
    metrics = compute_metrics(y_true, y_pred, names, label_ids=label_ids)
    metrics["n_unpaired_predictions"] = len(extra[0]["unpaired"])
    metrics["n_duplicate_observations"] = len(extra[0].get("duplicates", []))
    # truth 覆盖率：配对过的真值文件 vs 全部真值文件
    # 使用已配对真值对象，避免仅带 file:session 的预测被误报为漏文件。
    covered_truth = {id(a["truth"]) for a in extra[0]["aligned"]}
    covered = {name for name, t in truth.items() if id(t) in covered_truth}
    missing = [{"file": name, "label": t["label"], "trial_id": t["trial_id"]}
               for name, t in truth.items() if name not in covered]
    metrics["truth_coverage"] = {
        "n_truth_files": len(truth),
        "n_covered_files": len(truth) - len(missing),
        "missing_files": missing,
        "note": "missing=有真值但无任何预测的文件（会话全被过滤/漏检/"
                "漏跑），这些文件按无预测处理不产假分母",
    }
    incomplete_reasons = []
    if missing:
        incomplete_reasons.append("missing_truth_files")
    if metrics["n_unpaired_predictions"]:
        incomplete_reasons.append("unpaired_predictions")
    if metrics.get("out_of_vocab"):
        incomplete_reasons.append("out_of_vocab_labels")
    if not y_true:
        incomplete_reasons.append("no_aligned_observations")
    metrics["evaluation_scope"] = {
        "status": "incomplete" if incomplete_reasons else "covered_sources",
        "reasons": incomplete_reasons,
        "denominator": "aligned_unique_observations",
        "note": "覆盖全部文件不等于验证了全部会话/窗口分母；"
                "未提供独立观测清单时，不宣称缺失观测数为零",
    }
    metrics["label_names"] = names
    metrics["task"] = task
    if task == "behavior":
        aligned = extra[0]["aligned"]
        n_bg = sum(1 for a in aligned if a["true_label"] == "background")
        bg_fired = sum(
            1 for a in aligned
            if a["true_label"] == "background"
            and a["pred"].get("predicted_label") not in
            (None, UNKNOWN_STR, "", "background"))
        metrics["background"] = {
            "n_background_windows": n_bg,
            "background_false_alarms": bg_fired,
            "background_fpr": (bg_fired / n_bg) if n_bg else None,
            "note": "真值背景窗被预测为任一已知行为类即计背景误报；"
                    "拒识(unknown)与显式背景类命中(background)不计误报",
        }
        # 背景窗拒识(unknown)按正确处理的重命名口径（不改accuracy定义，
        # 单列供报告；accuracy仍按严格矩阵）
        bg_rejected = sum(
            1 for a in aligned
            if a["true_label"] == "background"
            and a["pred"].get("predicted_label") in (None, UNKNOWN_STR, ""))
        metrics["background"]["background_rejected_as_unknown"] = bg_rejected
        # 事件级一对一（窗口合并 -> 与试次行为区间容差匹配；
        # 真值事件带 source_file，与预测事件同源才可配对）
        truth_events = [
            {"terminal": getattr(t["record"], "terminal_ip", "") or "",
             "label": t["label"],
             "t": getattr(t["record"], "behavior_start", 0.0) or 0.0,
             "source_file": name}
            for name, t in truth.items()]
        metrics["event_matching"] = match_events(
            aggregate_windows_to_events(preds), truth_events)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def load_observation_list(path: str) -> List[Dict]:
    """加载预先固定的观测清单（JSONL：observation_id/label 必填）。

    清单即分母（2026-09-18 第三轮）：会话级批次评价不再以文件覆盖
    冒充完整观测分母；observation_id 重复显式报错（同观测两次入清单
    = 分母伪造）。
    """
    p = Path(path)
    obs: List[Dict] = []
    seen = set()
    for lineno, line in enumerate(
            p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        oid = rec.get("observation_id")
        label = rec.get("label")
        if not oid or label in (None, ""):
            raise ValueError(
                f"观测清单第{lineno}行缺 observation_id/label: {line[:80]}")
        if oid in seen:
            raise ValueError(f"观测清单 observation_id 重复: {oid}")
        seen.add(oid)
        obs.append(rec)
    if not obs:
        raise ValueError(f"观测清单为空: {path}")
    return obs


def evaluate_observations(predictions_path: str,
                          observations: List[Dict],
                          output_dir: str,
                          label_map: Optional[Dict[str, int]] = None
                          ) -> Dict:
    """观测级评价：预先固定的观测清单是唯一分母（会话级批次）。

    - 清单内 observation_id 无预测行 -> 按 missing_prediction 计错
      （y_pred=UNKNOWN 且单列计数，不静默剔除不冒充拒识）；
    - 预测文件中的清单外行计数单列（extra_predictions，不入分母）；
    - 同一 observation_id 多条预测取首条并计数（与文件级口径一致）；
    - 指标/词表/混淆矩阵与 compute_metrics 同源（拒识入分母、词表外
      单列、macro=类等权）。
    """
    if not observations:
        raise ValueError("观测清单为空：无清单即不可评价（不产出指标）")
    preds = load_predictions(predictions_path)
    by_oid: Dict[str, Dict] = {}
    n_dup_pred = 0
    n_extra = 0
    for pred in preds:
        oid = pred.get("observation_id")
        if oid is None:
            n_extra += 1            # 无observation_id的行不参与观测级配对
            continue
        if oid in by_oid:
            n_dup_pred += 1
            continue
        by_oid[oid] = pred
    listed_ids = {o["observation_id"] for o in observations}
    n_extra += sum(1 for oid in by_oid if oid not in listed_ids)

    if label_map is None:
        ordered = sorted({str(o["label"]) for o in observations})
        label_map = {n: i for i, n in enumerate(ordered)}
    y_true: List[int] = []
    y_pred: List[int] = []
    missing: List[Dict] = []
    for o in observations:
        y_true.append(_to_id(o["label"], label_map))
        pred = by_oid.get(o["observation_id"])
        if pred is None:
            y_pred.append(UNKNOWN)
            missing.append(o["observation_id"])
        else:
            y_pred.append(_to_id(pred.get("predicted_label"), label_map))
    ordered = sorted(label_map.items(), key=lambda kv: kv[1])
    names = [n for n, _ in ordered]
    metrics = compute_metrics(y_true, y_pred, names,
                              label_ids=[i for _, i in ordered])
    metrics["n_observations"] = len(observations)
    metrics["n_missing_predictions"] = len(missing)
    metrics["missing_prediction_ids"] = missing[:50]
    metrics["n_extra_predictions"] = n_extra
    metrics["n_duplicate_predictions"] = n_dup_pred
    metrics["evaluation_scope"] = {
        "status": ("incomplete" if missing or n_extra or n_dup_pred
                   else "fixed_observation_batch"),
        "denominator": "pre_fixed_observation_list",
        "note": "分母=预先固定的观测清单；missing=清单内无预测行(计错)；"
                "extra=预测文件中清单外行(不入分母)",
    }
    metrics["task"] = "app"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


# ---------------------------------------------------------------------------
# 事件级评价（M2 预留：短时操作按事件计，重复告警不重复 TP）
# ---------------------------------------------------------------------------
def _eligible_pairs(predicted_events: List[Dict],
                    truth_events: List[Dict],
                    tolerance: float) -> List[Tuple[int, int, float]]:
    """可配对(预测,真值)列表：同源 + 同终端 + 同标签 + 时间差<=容差。

    source_file 隔离：两侧都带 source_file 时要求同文件——不同试次
    （独立 pcap）的同终端同标签近时事件不得跨试次配对（审计
    2026-09-18 第二轮）。任一侧无 source_file（旧API）不按源过滤。
    """
    pairs = []
    for i, p in enumerate(predicted_events):
        for j, t in enumerate(truth_events):
            if t.get("terminal") != p.get("terminal"):
                continue
            if t.get("label") != p.get("label"):
                continue
            ps, ts = p.get("source_file"), t.get("source_file")
            if (ps is not None and ts is not None
                    and Path(str(ps)).name != Path(str(ts)).name):
                continue
            d = abs(p.get("t", 0) - t.get("t", 0))
            if d <= tolerance:
                pairs.append((i, j, d))
    return pairs


def match_events(predicted_events: List[Dict], truth_events: List[Dict],
                 tolerance: float = 2.0) -> Dict:
    """一对一容差匹配：同源 + 同终端 + 正确标签 + |t_pred - t_truth| <= tolerance。

    最大一对一配对（增广路算法，不依赖 scipy，不因环境改变评价语义）。
    贪心逐预测取最近可用真值不是最大配对——容差边界情形会少算 TP
    （审计 2026-09-18 第二轮）。同一真值事件只计一次 TP（重复告警不算）。
    """
    pairs = _eligible_pairs(predicted_events, truth_events, tolerance)
    tp = 0
    matching = "max_cardinality"
    # 二分图左右编号分开维护；非递归增广避免长事件序列触发递归上限。
    from collections import deque
    adjacent = [[] for _ in predicted_events]
    for i, j, _d in pairs:
        adjacent[i].append(j)
    left_match, right_match = {}, {}
    for root in range(len(predicted_events)):
        queue = deque([root])
        seen_left = {root}
        parent_right = {}
        free_right = None
        while queue and free_right is None:
            i = queue.popleft()
            for j in adjacent[i]:
                if j in parent_right:
                    continue
                parent_right[j] = i
                if j not in right_match:
                    free_right = j
                    break
                owner = right_match[j]
                if owner not in seen_left:
                    seen_left.add(owner)
                    queue.append(owner)
        if free_right is not None:
            j = free_right
            while j is not None:
                i = parent_right[j]
                old_j = left_match.get(i)
                left_match[i] = j
                right_match[j] = i
                j = old_j
    tp = len(left_match)
    fn = len(truth_events) - tp
    fp = len(predicted_events) - tp
    return {"tp": tp, "fp": fp, "fn": fn,
            "event_recall": tp / len(truth_events) if truth_events else 0.0,
            "event_precision": tp / len(predicted_events) if predicted_events else 0.0,
            "tolerance_sec": tolerance,
            "matching": matching}
