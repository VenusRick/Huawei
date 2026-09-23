# -*- coding: utf-8 -*-
"""Competition-oriented Behavior15 acceptance with session-disjoint split.

This is intentionally less strict than file-disjoint validation: sessions from
the same controlled capture may appear in different splits, but windows from
the same reconstructed 5-tuple session never cross train/validation/test.
The split is deterministic and frozen before test replay.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.engine.matcher.optimized_engine import OptimizedDPIEngine
from src.engine.model_io import save_rule_bundle
from src.engine.rule_compiler.optimized_generator import OptimizedRuleGenerator
from src.profile_specs import get_profile_spec


ROOT = Path("/workspace/Huawei")
SRC = ROOT / "output/final_profile_runtime_20260919/behavior"
OUT = ROOT / "output/protocol_and_system_acceptance_20260920/behavior_relaxed"
SEED = 20260920
LABELS = ["chat", "audio", "video"]
LID = {x: i for i, x in enumerate(LABELS)}
PAT = re.compile(r":bw(\d+):(\d+)$")


def metric(y: np.ndarray, p: np.ndarray) -> Dict[str, object]:
    ids = list(range(3))
    P, R, F1, s = precision_recall_fscore_support(y, p, labels=ids, zero_division=0)
    cm = confusion_matrix(y, p, labels=ids)
    fprs = []
    for i in ids:
        fp = cm[:, i].sum() - cm[i, i]
        neg = cm.sum() - cm[i, :].sum()
        fprs.append(fp / neg if neg else 0.0)
    r = {
        "accuracy": float(accuracy_score(y, p)),
        "macro_precision": float(P.mean()),
        "macro_recall": float(R.mean()),
        "macro_f1": float(F1.mean()),
        "max_class_fpr": float(max(fprs)),
        "errors": int((p != y).sum()),
        "per_class": {LABELS[i]: {"P": float(P[i]), "R": float(R[i]), "F1": float(F1[i]),
                                        "FPR": float(fprs[i]), "N": int(s[i])} for i in ids},
        "cm": cm.tolist(),
    }
    r["target_pass"] = bool(r["accuracy"] >= .95 and r["macro_precision"] >= .95
                            and r["macro_recall"] >= .98 and r["max_class_fpr"] <= .05)
    return r


def key(r: Dict[str, object]) -> Tuple:
    return (1 if r["target_pass"] else 0, r["macro_recall"], r["accuracy"],
            r["macro_precision"], -r["max_class_fpr"])


def parse_ids(df: pd.DataFrame) -> pd.DataFrame:
    sids = []
    wids = []
    for oid in df["_observation_id"].astype(str):
        m = PAT.search(oid)
        if not m:
            raise ValueError(f"bad behavior observation id: {oid}")
        sids.append(int(m.group(1)))
        wids.append(int(m.group(2)))
    df = df.copy()
    df["_session_index"] = sids
    df["_window_index"] = wids
    df["_session_group"] = df["_source_file"].astype(str) + "::" + df["_session_index"].astype(str)
    return df


def make_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    import random
    assignment: Dict[str, str] = {}
    stats = {}
    for label in LABELS:
        groups = sorted(df.loc[df["_label_name"] == label, "_session_group"].unique())
        rng = random.Random(f"{SEED}:{label}:session-disjoint")
        rng.shuffle(groups)
        n = len(groups)
        n_test = max(1, round(n * .20))
        n_val = max(1, round(n * .20))
        test = set(groups[:n_test])
        val = set(groups[n_test:n_test + n_val])
        train = set(groups[n_test + n_val:])
        for g in train:
            assignment[g] = "train"
        for g in val:
            assignment[g] = "validation"
        for g in test:
            assignment[g] = "test"
        stats[label] = {"sessions": n, "train": len(train), "validation": len(val), "test": len(test)}
    out = df.copy()
    out["_relaxed_split"] = out["_session_group"].map(assignment)
    if out["_relaxed_split"].isna().any():
        raise RuntimeError("unassigned behavior session")
    # Guard: no reconstructed session may cross splits.
    cross = out.groupby("_session_group")["_relaxed_split"].nunique()
    if int((cross > 1).sum()) != 0:
        raise RuntimeError("session leakage across relaxed splits")
    manifest = {
        "seed": SEED,
        "split_unit": "reconstructed-session",
        "relaxation": "same PCAP/capture may appear across splits; same session never crosses splits",
        "label_stats": stats,
        "rows": {sp: int((out["_relaxed_split"] == sp).sum()) for sp in ["train", "validation", "test"]},
        "sessions": {sp: int(out.loc[out["_relaxed_split"] == sp, "_session_group"].nunique())
                     for sp in ["train", "validation", "test"]},
    }
    return out, manifest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = parse_ids(pd.read_csv(SRC / "mine_flow/raw_features.csv"))
    df, split_info = make_split(df)
    split_csv = OUT / "relaxed_split_rows.csv"
    df[["_observation_id", "_source_file", "_session_index", "_window_index", "_label_name",
        "_session_group", "_relaxed_split"]].to_csv(split_csv, index=False)
    split_info["split_csv_sha256"] = hashlib.sha256(split_csv.read_bytes()).hexdigest()
    (OUT / "split_manifest.json").write_text(json.dumps(split_info, indent=2, ensure_ascii=False))

    tr = df[df["_relaxed_split"] == "train"].reset_index(drop=True)
    va = df[df["_relaxed_split"] == "validation"].reset_index(drop=True)
    te = df[df["_relaxed_split"] == "test"].reset_index(drop=True)
    features = [c for c in df.columns if not c.startswith("_")]
    Xall = tr[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    ytr = tr["_label_id"].astype(int).to_numpy()
    ranker = XGBClassifier(n_estimators=180, max_depth=5, learning_rate=.05, min_child_weight=1,
                           subsample=.9, colsample_bytree=.9, reg_lambda=1.5,
                           random_state=SEED, n_jobs=8, eval_metric="mlogloss", tree_method="hist")
    ranker.fit(Xall, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    order = np.argsort(ranker.feature_importances_)[::-1]
    rank = [features[i] for i in order]
    pd.DataFrame({"feature": rank,
                  "importance": [float(ranker.feature_importances_[i]) for i in order]}).to_csv(
                      OUT / "train_feature_ranking.csv", index=False)

    yv = va["_label_id"].astype(int).to_numpy()
    upper_rows = []
    candidates = []
    best = None
    for k in [16, 26, 32, 48, 64]:
        fs = rank[:min(k, len(rank))]
        up = XGBClassifier(n_estimators=220, max_depth=6, learning_rate=.04, min_child_weight=1,
                           subsample=.9, colsample_bytree=.9, reg_lambda=1.5,
                           random_state=42, n_jobs=8, eval_metric="mlogloss", tree_method="hist")
        up.fit(tr[fs].fillna(0), ytr, sample_weight=compute_sample_weight("balanced", ytr))
        um = metric(yv, up.predict(va[fs].fillna(0)))
        upper_rows.append({"k": len(fs), **{z: v for z, v in um.items() if z not in ("per_class", "cm")}})
        for depth in [6, 10, 14]:
            for leaf in [1, 2]:
                gen = OptimizedRuleGenerator(confidence_threshold=0.0, min_confidence=0.0,
                                             tree_max_depth=depth, tree_min_samples_leaf=leaf,
                                             tree_class_weight="balanced")
                rules = gen.fit_and_generate(tr[fs], tr["_label_id"], fs,
                                             {0: "chat", 1: "audio", 2: "video"},
                                             validation=(va[fs], va["_label_id"]),
                                             groups=tr["_session_group"])
                deploy = [r for r in rules if r.get("type") != "ensemble_classifier"]
                for th in [0, .5, .7, .8, .9, .95]:
                    eng = OptimizedDPIEngine()
                    eng.rules = deploy; eng.selected_features = fs
                    eng.label_names = {0: "chat", 1: "audio", 2: "video"}
                    eng.confidence_threshold = th; eng.classifier = None
                    pred = []
                    for _, row in va.iterrows():
                        m = eng.match({f: row[f] for f in fs})
                        pred.append(LID.get(m[0].result, -1) if m else -1)
                    mm = metric(yv, np.asarray(pred))
                    rr = {"k": len(fs), "depth": depth, "leaf": leaf, "threshold": th,
                          "n_rules": len(deploy), **{z: v for z, v in mm.items() if z not in ("per_class", "cm")}}
                    candidates.append(rr)
                    if best is None or key(mm) > key(best[0]):
                        best = (mm, fs, depth, leaf, th, rules, deploy)
    pd.DataFrame(upper_rows).to_csv(OUT / "offline_upper_sweep.csv", index=False)
    pd.DataFrame(candidates).to_csv(OUT / "validation_rule_sweep.csv", index=False)
    vm, fs, depth, leaf, th, rules, deploy = best

    bundle = OUT / "bundle"
    shutil.rmtree(bundle, ignore_errors=True)
    spec = get_profile_spec("behavior15", "behavior")
    spec["required_features"] = fs
    save_rule_bundle(rules, fs, {0: "chat", 1: "audio", 2: "video"}, th, str(bundle),
                     bundle_meta={"task": "behavior", "feature_profile": "behavior15",
                                  "profile_spec": spec, "context": spec["context"],
                                  "split_unit": "reconstructed-session",
                                  "same_capture_cross_split": True,
                                  "max_read_packets_per_file": 50000})
    freeze = {
        "seed": SEED, "split_manifest_sha256": split_info["split_csv_sha256"],
        "selected_features": fs, "tree_max_depth": depth, "tree_min_samples_leaf": leaf,
        "confidence_threshold": th, "validation": vm,
        "test_policy": "test session groups not used for ranking/rule/threshold selection",
    }
    freeze_path = OUT / "frozen_config.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False))
    freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()

    # First test access after freeze. Direct rules over frozen test feature rows.
    eng = OptimizedDPIEngine(); eng.load_rule_bundle(str(bundle))
    yt = te["_label_id"].astype(int).to_numpy(); direct_pred = []; matcher_ms = []
    direct_by_id = {}
    for _, row in te.iterrows():
        t0 = time.perf_counter(); m = eng.match({f: row[f] for f in fs}); matcher_ms.append((time.perf_counter()-t0)*1000)
        lab = m[0].result if m else "unknown"
        direct_pred.append(LID.get(lab, -1)); direct_by_id[str(row["_observation_id"])] = lab
    direct_test = metric(yt, np.asarray(direct_pred))

    # Independent rules-only PCAP replay, then select only frozen test sessions.
    all_files = sorted(set(Path(x).resolve() for x in df["_source_file"].astype(str)))
    pcapdir = OUT / "pcaps"; shutil.rmtree(pcapdir, ignore_errors=True); pcapdir.mkdir()
    for p in all_files:
        dst = pcapdir / p.name
        if not dst.exists():
            dst.symlink_to(p)
    guard = OUT / "import_guard"; guard.mkdir(exist_ok=True)
    (guard / "sitecustomize.py").write_text(
        "import builtins\n_o=builtins.__import__\ndef g(n,*a,**k):\n"
        "    if n.split('.')[0] in {'sklearn','xgboost','scipy'}: raise ImportError('training lib blocked: '+n)\n"
        "    return _o(n,*a,**k)\nbuiltins.__import__=g\n")
    outjson = OUT / "independent_all_windows.json"
    env = os.environ.copy(); env["PYTHONPATH"] = str(guard) + os.pathsep + str(ROOT)
    t0 = time.perf_counter(); before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    proc = subprocess.run([sys.executable, "-m", "src.engine.dpi_infer", "--rules", str(bundle),
                           "--pcap-dir", str(pcapdir), "-o", str(outjson)],
                          capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0; rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    (OUT / "independent_cli.log").write_text(proc.stdout + "\nSTDERR\n" + proc.stderr)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    rows = json.loads(outjson.read_text())["results"]
    test_group_to_label = {g: lab for g, lab in zip(te["_session_group"], te["_label_name"])}
    selected = []
    mismatches = []
    for r in rows:
        m = PAT.search(str(r["observation_id"]))
        if not m:
            continue
        group = str((pcapdir / r["source_file"]).resolve()) + "::" + str(int(m.group(1)))
        # source raw_features use original absolute file path; symlink resolves to it.
        if group not in test_group_to_label:
            continue
        selected.append(r)
        if r["observation_id"] in direct_by_id and direct_by_id[r["observation_id"]] != r["predicted_label"]:
            mismatches.append((r["observation_id"], direct_by_id[r["observation_id"]], r["predicted_label"]))
    if len(selected) != len(te):
        raise RuntimeError(f"independent test rows mismatch: {len(selected)} != {len(te)}")
    y_ind = np.asarray([LID[test_group_to_label[str((pcapdir / r["source_file"]).resolve()) + "::" + str(int(PAT.search(str(r["observation_id"])).group(1)))]] for r in selected])
    p_ind = np.asarray([LID.get(r["predicted_label"], -1) for r in selected])
    independent_test = metric(y_ind, p_ind)
    json.dump(independent_test, open(OUT / "metrics.json", "w"), indent=2)
    json.dump({"consistent": not mismatches, "compared": len(selected), "mismatches": mismatches[:50]},
              open(OUT / "parity.json", "w"), indent=2)
    performance = {
        "independent_wall_sec": wall, "peak_rss_mib": rss / 1024.0,
        "test_windows": len(selected), "windows_per_sec": len(selected) / max(wall, 1e-9),
        "input_bytes": sum(p.stat().st_size for p in all_files),
        "mib_per_sec": (sum(p.stat().st_size for p in all_files)/(1024.0*1024.0))/max(wall,1e-9),
        "matcher_ms_mean": float(np.mean(matcher_ms)), "matcher_ms_p95": float(np.quantile(matcher_ms,.95)),
        "feature_count": len(fs), "rule_count": len(deploy),
    }
    json.dump(performance, open(OUT / "performance.json", "w"), indent=2)
    acceptance = {
        "scope": "competition-oriented session-disjoint 60/20/20 split; same PCAP may cross splits",
        "strict_file_disjoint_reference": json.load(open(SRC / "acceptance.json"))["independent_rule_validation"],
        "validation": vm, "test_direct": direct_test, "test_independent": independent_test,
        "parity_consistent": not mismatches, "formal_test_pass": independent_test["target_pass"],
        "freeze_sha256": freeze_sha,
    }
    json.dump(acceptance, open(OUT / "acceptance.json", "w"), indent=2)
    print("SPLIT", split_info)
    print("VALIDATION", vm)
    print("TEST", independent_test)
    print("PARITY", not mismatches, "compared", len(selected))
    print("PERF", performance)


if __name__ == "__main__":
    main()
