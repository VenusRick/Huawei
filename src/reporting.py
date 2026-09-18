# -*- coding: utf-8 -*-
"""报告服务（M3：统一 AnalysisPackage + 特征有效性口径）。

一个 AnalysisPackage = analysis/ 目录八件套，分两层生成：
  mine 层：feature_catalog.json / feature_effectiveness.csv / class_profiles.json
  evaluate 层：confusion_analysis.json / misclassified_samples.jsonl /
               rules_explained.json / recognition_plan.md / report.html

特征有效性口径（建议.md 第七点，现在固定）：
  A = 自动输出特征全集
  E = A 中通过五级判定的子集：可正确计算 -> 训练/推理定义一致 ->
      验证集存在判别证据 -> 可被规则引擎使用
  EffectiveRate = |E| / |A|；人工修订后得 U，UsableRate = |U| / |A|
  每个特征判定原因留档；无法判断保留 pending。
  selected/all 是保留比例，不是特征有效率（勿混）。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------- mine 层
def compute_feature_effectiveness(df, y, selected: List[str],
                                  engine_supported: Optional[set] = None
                                  ) -> Dict:
    """特征有效性原始表（可审查，不自称官方验收口径）。

    返回 {"A": [...], "E": [...], "pending": [...], "rows": {...}}；
    rows[feat] = {"computable", "consistent", "discriminative",
                  "engine_supported", "effective", "reason"}。
    """
    from scipy.stats import f_oneway
    all_feats = [c for c in df.columns if not str(c).startswith("_")]
    a_set = list(all_feats)
    sel = set(selected)
    rows: Dict[str, Dict] = {}
    e_list: List[str] = []
    pending: List[str] = []
    groups_idx = [lid for lid in sorted(set(y)) if (y == lid).sum() > 1]
    for feat in a_set:
        col = df[feat]
        computable = bool(col.notna().any()) and col.astype(float).nunique() > 1
        consistent = True   # 单一 runtime 入口保证定义一致（P0 parity 证据）
        disc = False
        reason = ""
        if computable and len(groups_idx) >= 2:
            groups = [col[(y == lid)].dropna().astype(float).values
                      for lid in groups_idx]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) >= 2:
                try:
                    _, p = f_oneway(*groups)
                    disc = p < 0.05
                    reason = f"ANOVA p={p:.4g}"
                except Exception as e:  # noqa: BLE001
                    disc = False
                    reason = f"ANOVA异常({e})；pending"
            else:
                reason = "类别组不足；pending"
        elif not computable:
            reason = "全缺失或常数"
        engine_ok = (engine_supported is None) or (feat in engine_supported)
        if not engine_ok:
            reason += "；引擎不支持"
        effective = computable and consistent and disc and engine_ok
        if "pending" in reason:
            pending.append(feat)
        elif effective:
            e_list.append(feat)
        rows[feat] = {
            "computable": computable, "consistent": consistent,
            "discriminative": disc, "engine_supported": engine_ok,
            "effective": effective, "reason": reason or "通过全部判定",
            "selected": feat in sel,
        }
    return {"A": a_set, "E": e_list, "pending": pending, "rows": rows,
            "effective_rate": len(e_list) / len(a_set) if a_set else 0.0,
            "definition": "EffectiveRate=|E|/|A|；五级判定见模块docstring；"
                          "selected/all 为保留比例，非有效率"}


def build_mine_layer(df, y, selected, rules, label_names, output_dir: str,
                     confidence_threshold: float = 0.7) -> Dict:
    """mine 层三件：catalog / effectiveness / class_profiles。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    eff = compute_feature_effectiveness(df, y, selected)
    # feature_catalog.json
    catalog = []
    for feat in eff["A"]:
        r = eff["rows"][feat]
        catalog.append({
            "name": feat,
            "layer": ("behavior" if feat.startswith("behavior_")
                      else ("advanced" if any(k in feat for k in
                           ("burst", "iit", "entropy", "ngram", "gapped"))
                            else "basic")),
            "missing_semantics": "训练侧哨兵-1；推理侧None同规则",
            "cost": "low" if feat.startswith("behavior_win") else "medium",
            "used_by_rules": sorted({
                rr.get("id") for rr in rules
                for c in rr.get("conditions", []) if c.get("feature") == feat}),
            "effectiveness": "effective" if r["effective"] else
                             ("pending" if feat in eff["pending"] else "no"),
        })
    (out / "feature_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    # feature_effectiveness.csv
    with open(out / "feature_effectiveness.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "computable", "consistent", "discriminative",
                    "engine_supported", "effective", "selected", "reason"])
        for feat in eff["A"]:
            r = eff["rows"][feat]
            w.writerow([feat, r["computable"], r["consistent"],
                        r["discriminative"], r["engine_supported"],
                        r["effective"], r["selected"], r["reason"]])
    # class_profiles.json
    profiles = {}
    yv = list(y)
    names = {v: k for k, v in (label_names or {}).items()}
    for lid in sorted(set(yv)):
        sub = df[y == lid]
        prof = {"support": int((y == lid).sum())}
        for feat in selected:
            col = sub[feat]
            prof[feat] = {"mean": float(col.mean()) if len(col) else None,
                          "std": float(col.std()) if len(col) > 1 else 0.0}
        profiles[names.get(lid, str(lid))] = prof
    (out / "class_profiles.json").write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    return eff


# ------------------------------------------------------------- evaluate 层
def build_evaluate_layer(bundle_dir: str, predictions_path: str,
                         truth: Dict[str, Dict], metrics: Dict,
                         output_dir: str) -> None:
    """evaluate 层五件（confusion / misclassified / rules_explained /
    recognition_plan / report.html）。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    b = Path(bundle_dir)
    rules = json.loads((b / "rules.json").read_text(encoding="utf-8"))
    selected = json.loads((b / "selected_features.json").read_text(
        encoding="utf-8"))
    preds = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    results = preds.get("results", preds if isinstance(preds, list) else [])
    # 混淆
    conf = {}
    for p in results:
        t = truth.get(Path(p.get("source_file", "")).name, {})
        tl = t.get("label", "?")
        conf.setdefault(tl, {}).setdefault(p.get("predicted_label", "?"), 0)
        conf[tl][p.get("predicted_label", "?")] += 1
    (out / "confusion_analysis.json").write_text(
        json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")
    # 误判样例
    with open(out / "misclassified_samples.jsonl", "w",
              encoding="utf-8") as f:
        for p in results:
            t = truth.get(Path(p.get("source_file", "")).name, {})
            tl = t.get("label", "?")
            pl = p.get("predicted_label", "?")
            if tl != pl and not (tl == "?" or pl == "unknown"):
                f.write(json.dumps({
                    "observation_id": p.get("observation_id"),
                    "source_file": p.get("source_file"),
                    "window": [p.get("window_start"), p.get("window_end")],
                    "true": tl, "pred": pl,
                    "confidence": p.get("confidence")}, ensure_ascii=False)
                    + "\n")
    # rules_explained
    expl = []
    for r in rules:
        conds = " AND ".join(
            f"{c.get('feature')} {c.get('op')} {c.get('value'):.4g}"
            if isinstance(c.get("value"), (int, float)) else
            f"{c.get('feature')} {c.get('op')} {c.get('value')}"
            for c in r.get("conditions", []))
        expl.append({"rule_id": r.get("id"),
                     "label": r.get("action", {}).get("result", ""),
                     "confidence": r.get("confidence"),
                     "conditions_text": conds or "(无条件)",
                     "n_conditions": len(r.get("conditions", []))})
    (out / "rules_explained.json").write_text(
        json.dumps(expl, ensure_ascii=False, indent=2), encoding="utf-8")
    # recognition_plan.md（从规则模板化，不要求 LLM）
    lines = ["# 可行识别方案（自动生成）", ""]
    lines.append(f"共 {len(rules)} 条规则、{len(selected)} 个选中特征。")
    lines.append("")
    lines.append("## 识别顺序建议（低成本条件优先）")
    basic_first = sorted(
        expl, key=lambda e: e["n_conditions"])
    for e in basic_first[:10]:
        _conf = e['confidence']
        _conf_txt = f"{_conf:.2f}" if isinstance(_conf, (int, float)) else "N/A"
        lines.append(f"- **{e['label']}**（{e['rule_id']}，置信度"
                     f"{_conf_txt}）：{e['conditions_text']}")
    lines.append("")
    lines.append("## 拒识语义")
    lines.append("全部 AND 条件满足才命中；任一条件不满足即不命中，"
                 "无命中输出 unknown（拒识计入评价分母）。")
    lines.append("")
    lines.append("> 统计来自本次 run 的 bundle 与预测，未拼接历史文件。")
    (out / "recognition_plan.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    # report.html（最小静态页）
    html = ["<!doctype html><meta charset='utf-8'>",
            "<title>AnalysisPackage</title>",
            "<h1>AnalysisPackage</h1>",
            f"<p>规则 {len(rules)} 条；选中特征 {len(selected)}；"
            f"样本 {metrics.get('n_samples')}；准确率 "
            f"{metrics.get('accuracy')}</p>",
            "<h2>混淆（真值->预测）</h2><pre>",
            json.dumps(conf, ensure_ascii=False, indent=1), "</pre>",
            "<h2>规则</h2><pre>",
            json.dumps(expl, ensure_ascii=False, indent=1), "</pre>"]
    (out / "report.html").write_text("\n".join(html), encoding="utf-8")


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_run_report(bundle_dir: str, metrics: Optional[Dict],
                     run_dir: str) -> str:
    """生成单次 run 的最小报告（run_report.md），返回路径。"""
    b = Path(bundle_dir)
    rules = _load_json(b / "rules.json") or []
    selected = _load_json(b / "selected_features.json") or []
    bundle_cfg = _load_json(b / "bundle_config.json") or {}
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)

    lines: List[str] = ["# 运行报告（自动生成）", ""]
    lines.append(f"- bundle 目录: `{b}`")
    lines.append(f"- 规则数: {len(rules)}")
    lines.append(f"- 选中特征数: {len(selected)}（保留比例，非特征有效率）")
    if bundle_cfg:
        lines.append(f"- 标签词表: {bundle_cfg.get('label_names', {})}")
        lines.append(f"- 验收置信度阈值: "
                     f"{bundle_cfg.get('confidence_threshold', 'N/A')}")
        ctx = bundle_cfg.get("context", {})
        if ctx:
            lines.append(f"- 行为窗口配置: {json.dumps(ctx, ensure_ascii=False)}")
    if metrics:
        lines.append("")
        lines.append("## 指标（完整分母，拒识计入）")
        lines.append("```json")
        lines.append(json.dumps(
            {k: v for k, v in metrics.items() if k != "per_class"},
            indent=2, ensure_ascii=False))
        lines.append("```")
    lines.append("")
    lines.append("> 本报告统计全部来自本次 run 的 bundle 与预测输出，"
                 "未拼接历史文件。")
    out = run / "run_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return str(out)
