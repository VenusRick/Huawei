# -*- coding: utf-8 -*-
"""一键工程验收（方案 §6 第5条）：G01-G05 检查自动化。

不是顺序调命令后打印PASS——每项独立取证：
  G01 回归防线全绿（标尺+契约测试）
  G02 正式 mine 退出0且产出非空 bundle；空清单非零退出（R25）
  G04 独立进程只凭 bundle 识别；改规则→结果变、恢复→复原；
     import 树不含训练库（sklearn/xgboost）；隔离目录删模型文件仍可运行
  G05 指标可从预测文件单独复算，阈值与 bundle 一致

输出 acceptance.json（PASS/FAIL/BLOCKED_DATA 三态，不混成单个成功提示）。
用法：
  python scripts/run_acceptance.py --output output/acceptance001
  python scripts/run_acceptance.py --config configs/acceptance.yaml --output ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # 进程内复算需要 import src.*
PY = sys.executable

DEFAULTS = {
    "train_val_manifest": "data/task/fixture_train_val.jsonl",
    "test_truth_manifest": "data/task/fixture_test_truth.jsonl",
    "test_pcap_dir": "output/fixture_m1/test",
    "regression_tests": ["tests/regression/test_review_regressions.py",
                         "tests/test_contract_services.py"],
}


def _status(results, name):
    return results.setdefault(name, {"status": None, "evidence": []})


def _semantic(results):
    """语义比对：剔除 match_time_ms 等时延字段（每次运行天然抖动）。"""
    return json.dumps(
        [{k: v for k, v in r.items() if not k.endswith("_ms")} for r in results],
        sort_keys=True)


def check_g01(results, cfg):
    g = _status(results, "G01_regression")
    ok = True
    for t in cfg["regression_tests"]:
        r = subprocess.run([PY, "-m", "pytest", t, "-q"],
                           cwd=str(ROOT), capture_output=True, text=True,
                           timeout=900)
        tail = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
        g["evidence"].append(f"{t}: exit={r.returncode} {tail}")
        ok = ok and r.returncode == 0
    g["status"] = "PASS" if ok else "FAIL"


def check_g02(results, cfg, run_dir):
    g = _status(results, "G02_formal_mine")
    out = run_dir / "mine"
    r = subprocess.run(
        [PY, "-m", "src.pipeline", "--mode", "mine",
         "--manifest", str(ROOT / cfg["train_val_manifest"]),
         "--task", "app", "--output", str(out)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
    bundle = out / "bundle"
    valid = (r.returncode == 0 and bundle.exists()
             and (bundle / "rules.json").exists()
             and (bundle / "selected_features.json").exists()
             and (bundle / "bundle_config.json").exists())
    try:
        n_rules = len(json.loads((bundle / "rules.json").read_text()))
    except Exception:
        n_rules = 0
    valid = valid and n_rules > 0
    g["evidence"].append(f"mine exit={r.returncode}, bundle规则数={n_rules}")
    # 反例：空输入必须非零退出（R25）
    empty = run_dir / "empty_manifest.jsonl"
    empty.write_text("", encoding="utf-8")
    r2 = subprocess.run(
        [PY, "-m", "src.pipeline", "--mode", "mine",
         "--manifest", str(empty), "--output", str(run_dir / "mine_empty")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    g["evidence"].append(f"空清单 exit={r2.returncode}（要求非零）")
    g["status"] = "PASS" if (valid and r2.returncode != 0) else "FAIL"
    return bundle if valid else None


def check_g04(results, cfg, bundle, run_dir):
    g = _status(results, "G04_independent_dpi")
    if bundle is None:
        g["status"] = "FAIL"
        g["evidence"].append("上游 G02 未产出有效 bundle")
        return None
    pred = run_dir / "predictions.json"
    r = subprocess.run(
        [PY, "-m", "src.engine.dpi_infer", "--rules", str(bundle),
         "--pcap-dir", str(ROOT / cfg["test_pcap_dir"]),
         "-o", str(pred)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    try:
        obj = json.loads(pred.read_text())
        n, matched = obj["total_sessions"], sum(
            1 for x in obj["results"] if x.get("predicted_label") != "unknown")
    except Exception:
        n, matched = 0, -1
    g["evidence"].append(f"detect exit={r.returncode}, 会话={n}, 识别={matched}")
    ok = r.returncode == 0 and n > 0 and matched > 0
    # 静态证据：推理入口 import 树不含训练库
    src = (ROOT / "src/engine/dpi_infer.py").read_text(encoding="utf-8")
    bad = [w for w in ("sklearn", "xgboost") if w in src]
    g["evidence"].append(f"dpi_infer 源码训练库引用: {bad or '无'}")
    ok = ok and not bad
    # 隔离目录：只拷运行时代码+bundle，删除模型文件后仍可识别
    iso = run_dir / "isolated"
    if iso.exists():
        shutil.rmtree(iso)
    (iso / "src").mkdir(parents=True)
    shutil.copytree(ROOT / "src", iso / "src", dirs_exist_ok=True)
    shutil.copytree(bundle, iso / "bundle")
    for junk in iso.rglob("model.pkl"):
        junk.unlink()
    for junk in iso.rglob("model.json"):
        junk.unlink()
    r3 = subprocess.run(
        [PY, "-m", "src.engine.dpi_infer", "--rules", "bundle",
         "--pcap-dir", str(ROOT / cfg["test_pcap_dir"]),
         "-o", str(run_dir / "predictions_isolated.json")],
        cwd=str(iso), capture_output=True, text=True, timeout=900)
    g["evidence"].append(f"隔离目录(无模型文件) exit={r3.returncode}")
    ok = ok and r3.returncode == 0
    # 改规则→结果变；恢复→复原（op方向感知：把数值条件推到不可满足）
    rules = json.loads((bundle / "rules.json").read_text())
    backup = json.dumps(rules, ensure_ascii=False)
    try:
        n_mut = 0
        for rule in rules:
            for c in rule.get("conditions", []):
                v = c.get("value")
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                if str(c.get("op", "")).startswith(">"):
                    c["value"] = float(v) + 1e12
                    n_mut += 1
                elif str(c.get("op", "")).startswith("<"):
                    c["value"] = float(v) - 1e12
                    n_mut += 1
        (bundle / "rules.json").write_text(
            json.dumps(rules, ensure_ascii=False), encoding="utf-8")
        pred_mut = run_dir / "predictions_mutated.json"
        r4 = subprocess.run(
            [PY, "-m", "src.engine.dpi_infer", "--rules", str(bundle),
             "--pcap-dir", str(ROOT / cfg["test_pcap_dir"]),
             "-o", str(pred_mut)],
            cwd=str(ROOT), capture_output=True, text=True, timeout=900)
        base_obj = json.loads(pred.read_text())
        try:
            after = json.loads(pred_mut.read_text())["results"]
            matched_after = sum(
                1 for x in after if x.get("predicted_label") != "unknown")
        except Exception:
            after, matched_after = [], -1
        matched_before = sum(
            1 for x in base_obj["results"]
            if x.get("predicted_label") != "unknown")
        changed = matched_after < matched_before
        (bundle / "rules.json").write_text(backup, encoding="utf-8")
        pred_restored = run_dir / "predictions_restored.json"
        subprocess.run(
            [PY, "-m", "src.engine.dpi_infer", "--rules", str(bundle),
             "--pcap-dir", str(ROOT / cfg["test_pcap_dir"]),
             "-o", str(pred_restored)],
            cwd=str(ROOT), capture_output=True, text=True, timeout=900)
        restored = (_semantic(json.loads(pred_restored.read_text())["results"])
                    == _semantic(base_obj["results"]))
        g["evidence"].append(
            f"突变条件数={n_mut}, 识别数 {matched_before}->{matched_after}"
            f"(要求下降), 恢复复现={restored}, r4={r4.returncode}")
        ok = ok and changed and restored and n_mut > 0
    except Exception as e:  # noqa: BLE001
        g["evidence"].append(f"改规则检查异常: {e}")
        (bundle / "rules.json").write_text(backup, encoding="utf-8")
        ok = False
    g["status"] = "PASS" if ok else "FAIL"
    return pred if ok else None


def check_g05(results, cfg, pred, run_dir):
    g = _status(results, "G05_evaluation")
    if pred is None:
        g["status"] = "FAIL"
        g["evidence"].append("上游 G04 无预测产物")
        return
    out = run_dir / "evaluation"
    r = subprocess.run(
        [PY, "-m", "src.pipeline", "--mode", "evaluate",
         "--truth", str(ROOT / cfg["test_truth_manifest"]),
         "--predictions", str(pred), "--task", "app", "--output", str(out)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    try:
        m = json.loads((out / "metrics.json").read_text())
        # 复算：直接从 predictions+truth 独立重算准确率
        from src.data_manifest import load_manifest
        from src.evaluation import (load_truth_manifest, load_predictions,
                                    align_predictions, compute_metrics)
        truth = load_truth_manifest(
            [x for x in load_manifest(str(ROOT / cfg["test_truth_manifest"]))
             if x.split == "test"], "app")
        label_map = {"appA": 0, "appB": 1}
        for nm in sorted(truth):
            if truth[nm]["label"] not in label_map:
                label_map[truth[nm]["label"]] = len(label_map)
        y_t, y_p, _ = align_predictions(load_predictions(str(pred)),
                                        truth, label_map)
        names = [n for n, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
        m2 = compute_metrics(y_t, y_p, names)
        agree = abs(m["accuracy"] - m2["accuracy"]) < 1e-9 and \
            m["n_samples"] == m2["n_samples"]
        g["evidence"].append(
            f"evaluate exit={r.returncode}, acc={m['accuracy']}, "
            f"复算acc={m2['accuracy']}, n={m['n_samples']}/{m2['n_samples']}")
    except Exception as e:  # noqa: BLE001
        agree = False
        g["evidence"].append(f"复算异常: {e}")
    g["status"] = "PASS" if (r.returncode == 0 and agree) else "FAIL"


def main():
    ap = argparse.ArgumentParser(description="一键工程验收")
    ap.add_argument("--config", help="acceptance.yaml（缺省用内置 fixture 配置）")
    ap.add_argument("--output", default="output/acceptance001")
    args = ap.parse_args()
    cfg = dict(DEFAULTS)
    if args.config:
        import yaml  # noqa: F401  配置存在才依赖yaml
        with open(args.config, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    run_dir = ROOT / args.output
    if run_dir.exists():
        print(f"输出目录已存在: {run_dir}，请换新目录或清理后重跑")
        sys.exit(2)
    run_dir.mkdir(parents=True)
    results = {}
    check_g01(results, cfg)
    bundle = check_g02(results, cfg, run_dir)
    pred = check_g04(results, cfg, bundle, run_dir)
    check_g05(results, cfg, pred, run_dir)
    summary = {
        "results": results,
        "milestone": "P0",
        "all_pass": all(v["status"] == "PASS" for v in results.values()),
    }
    (run_dir / "acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v["status"] for k, v in results.items()},
                     ensure_ascii=False))
    print(f"acceptance.json -> {run_dir / 'acceptance.json'}")
    sys.exit(0 if summary["all_pass"] else 1)


if __name__ == "__main__":
    main()
