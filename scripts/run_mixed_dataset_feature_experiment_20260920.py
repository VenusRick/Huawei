# -*- coding: utf-8 -*-
"""Small multi-dataset feature-selection diagnostic.

Goal: test whether mixing heterogeneous datasets pushes selection toward
cross-dataset traffic-shape features and whether that costs within-dataset or
pooled classification accuracy.

This is an exploratory diagnostic, not a competition acceptance benchmark.
All datasets are transformed by the same current runtime with max_packets=64.
"""
from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.features.runtime import extract_feature_records_with_stats


ROOT = Path("/workspace/Huawei")
OUT = ROOT / "output/mixed_dataset_feature_experiment_20260920"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260920
TARGET_PER_CLASS = 80


def dataset_files() -> Dict[str, Dict[str, List[Path]]]:
    ar = Path("/workspace/cz-华为杯/data/all_data")
    datasets: Dict[str, Dict[str, List[Path]]] = {
        "USTC": {
            c: [ar / f"USTC-TFC2016-{c}.pcap"]
            for c in ["FTP", "Gmail", "MySQL", "WorldOfWarcraft"]
        },
        "CSTNET": {
            c: sorted(ar.glob(f"CSTNET-TLS1.3-{c}*.pcap"))
            for c in ["acm.org", "huawei.com", "overleaf.com", "vivo.com.cn"]
        },
    }

    cr = Path("/workspace/datasets/CrossPlatform/china/android")
    cclasses = ["bubei.tingshu", "com.aikan", "com.autonavi.minimap", "com.baidu.BaiduMap"]
    datasets["CrossPlatform"] = {c: [cr / f"{c}.pcap"] for c in cclasses}

    vr = Path("/workspace/datasets/VisQUIC/data/VisQUIC")
    vclasses = ["cloudflare.com", "discord.com", "google.com", "cdnetworks.com"]
    datasets["VisQUIC"] = {}
    for c in vclasses:
        datasets["VisQUIC"][c] = sorted((vr / c).rglob("*quic_anonymized_filtered.pcap"))
    return datasets


IDENTIFIER_EXACT = {
    "src_port", "dst_port", "is_well_known_src_port", "is_well_known_dst_port",
    "tls_ja3_hash_prefix", "tls_ja4_hash_prefix",
}
TCP_FLAG_FEATURES = {
    "syn_flag_count", "ack_flag_count", "fin_flag_count", "rst_flag_count", "psh_flag_count",
    "urg_flag_count", "ece_flag_count", "cwr_flag_count",
    "fwd_syn_flag_count", "fwd_ack_flag_count", "fwd_fin_flag_count", "fwd_rst_flag_count",
    "fwd_psh_flag_count", "fwd_urg_flag_count", "fwd_ece_flag_count", "fwd_cwr_flag_count",
    "bwd_syn_flag_count", "bwd_ack_flag_count", "bwd_fin_flag_count", "bwd_rst_flag_count",
    "bwd_psh_flag_count", "bwd_urg_flag_count", "bwd_ece_flag_count", "bwd_cwr_flag_count",
}


def feature_spaces(all_features: Iterable[str]) -> Dict[str, List[str]]:
    feats = sorted(all_features)
    broad = [f for f in feats if f not in IDENTIFIER_EXACT]
    shape = []
    for f in broad:
        if f.startswith(("tls_", "quic_", "protocol_", "tcp_window_")):
            continue
        if f in TCP_FLAG_FEATURES or f in {"syn_ack_rtt", "num_retransmissions", "retransmission_ratio"}:
            continue
        shape.append(f)
    return {"broad_no_identifier": broad, "common_shape": shape}


def extract_dataset_cache() -> pd.DataFrame:
    cache = OUT / "all_datasets_prefix64.csv"
    if cache.exists():
        return pd.read_csv(cache)
    configs = dataset_files()
    rows: List[Dict[str, object]] = []
    for dname, classes in configs.items():
        for cname, files in classes.items():
            if not files:
                raise RuntimeError(f"no files for {dname}:{cname}")
            rng = random.Random(f"{SEED}:{dname}:{cname}:files")
            files = list(files)
            rng.shuffle(files)
            collected: List[Dict[str, object]] = []
            for fi, p in enumerate(files):
                if len(collected) >= TARGET_PER_CLASS:
                    break
                # Large single-capture datasets only need a bounded prefix; VisQUIC
                # files are already small and normally contain 1-2 sessions.
                if dname == "CSTNET":
                    max_read = 50000
                elif dname in {"USTC", "CrossPlatform"}:
                    max_read = 12000
                else:
                    max_read = 50000
                recs, stats = extract_feature_records_with_stats(
                    str(p), max_packets=64, max_read_packets=max_read)
                rr = list(recs)
                random.Random(f"{SEED}:{p}").shuffle(rr)
                need = TARGET_PER_CLASS - len(collected)
                for r in rr[:need]:
                    row = {
                        "_dataset": dname,
                        "_class": cname,
                        "_label": f"{dname}::{cname}",
                        "_source_file": str(p),
                        "_observation_id": r.observation.observation_id,
                    }
                    row.update({k: v for k, v in r.features.items()})
                    collected.append(row)
                print("EXTRACT", dname, cname, p.name, "records", len(rr),
                      "collected", len(collected), "packets", stats.packets_read,
                      flush=True)
            if len(collected) < TARGET_PER_CLASS:
                raise RuntimeError(f"insufficient records {dname}:{cname} {len(collected)}<{TARGET_PER_CLASS}")
            rows.extend(collected[:TARGET_PER_CLASS])
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def _xgb(seed: int, n_estimators: int = 180) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators, max_depth=5, learning_rate=.05,
        min_child_weight=1, subsample=.9, colsample_bytree=.9,
        reg_lambda=1.5, random_state=seed, n_jobs=8,
        eval_metric="mlogloss", tree_method="hist",
    )


def fused_scores(X: pd.DataFrame, y: pd.Series, features: List[str], seed: int) -> Dict[str, float]:
    usable = [f for f in features if f in X.columns and pd.to_numeric(X[f], errors="coerce").fillna(0).std() > 1e-10]
    Xv = X[usable].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    classes = sorted(pd.Series(y).astype(str).unique())
    lid = {c: i for i, c in enumerate(classes)}
    yn = pd.Series(y).astype(str).map(lid).to_numpy()
    model = _xgb(seed, 140)
    model.fit(Xv, yn, sample_weight=compute_sample_weight("balanced", yn))
    xgb = np.asarray(model.feature_importances_, float)
    try:
        mi = np.asarray(mutual_info_classif(Xv, yn, random_state=seed), float)
    except Exception:
        mi = np.zeros(len(usable), float)
    def norm(a):
        if len(a) == 0:
            return a
        lo, hi = float(np.min(a)), float(np.max(a))
        return (a - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(a)
    score = .6 * norm(xgb) + .4 * norm(mi)
    return {f: float(s) for f, s in zip(usable, score)}


def rank_scores(scores: Dict[str, float]) -> List[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def split_dataset(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=.40, random_state=seed, stratify=df["_class"])
    va, te = train_test_split(tmp, test_size=.50, random_state=seed, stratify=df.iloc[tmp]["_class"])
    return df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def eval_subset(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame,
                features: List[str], seed: int, target: str = "_class") -> Dict[str, float]:
    fs = [f for f in features if f in tr.columns]
    train = pd.concat([tr, va], ignore_index=True)
    classes = sorted(train[target].unique())
    lid = {c: i for i, c in enumerate(classes)}
    Xtr = train[fs].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    Xte = te[fs].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    ytr = train[target].map(lid).to_numpy()
    yte = te[target].map(lid).to_numpy()
    model = _xgb(seed, 220)
    model.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    p = model.predict(Xte)
    return {"accuracy": float(accuracy_score(yte, p)),
            "macro_f1": float(f1_score(yte, p, average="macro", zero_division=0)),
            "n_test": int(len(yte)), "n_features": len(fs)}


def individual_and_common(df: pd.DataFrame, space_name: str, features: List[str]) -> Dict[str, object]:
    per_dataset = {}
    indiv_scores = {}
    split_cache = {}
    for i, d in enumerate(sorted(df._dataset.unique())):
        dd = df[df._dataset == d].reset_index(drop=True)
        tr, va, te = split_dataset(dd, SEED + i)
        split_cache[d] = (tr, va, te)
        scores = fused_scores(tr, tr["_class"], features, SEED + i)
        indiv_scores[d] = scores
        top64 = rank_scores(scores)[:64]
        per_dataset[d] = {
            "specific_top64": top64,
            "specific_metrics": eval_subset(tr, va, te, top64, SEED + 100 + i),
        }

    # Consensus score: mean normalized project-selector score; absent/constant=0.
    consensus_scores = {}
    for f in features:
        consensus_scores[f] = float(np.mean([indiv_scores[d].get(f, 0.0) for d in indiv_scores]))
    consensus_top64 = rank_scores(consensus_scores)[:64]

    # Pooled 16-class selector on all dataset-specific training folds.
    pooled_train = pd.concat([split_cache[d][0] for d in sorted(split_cache)], ignore_index=True)
    pooled_scores = fused_scores(pooled_train, pooled_train["_label"], features, SEED + 999)
    pooled_top64 = rank_scores(pooled_scores)[:64]

    for i, d in enumerate(sorted(per_dataset)):
        tr, va, te = split_cache[d]
        per_dataset[d]["consensus_metrics"] = eval_subset(tr, va, te, consensus_top64, SEED + 200 + i)
        per_dataset[d]["pooled_top64_metrics"] = eval_subset(tr, va, te, pooled_top64, SEED + 300 + i)
        per_dataset[d]["delta_consensus_acc"] = (
            per_dataset[d]["consensus_metrics"]["accuracy"] - per_dataset[d]["specific_metrics"]["accuracy"])
        per_dataset[d]["delta_pooled_acc"] = (
            per_dataset[d]["pooled_top64_metrics"]["accuracy"] - per_dataset[d]["specific_metrics"]["accuracy"])

    top_sets = {d: set(per_dataset[d]["specific_top64"]) for d in per_dataset}
    support = Counter()
    for d, s in top_sets.items():
        for f in s:
            support[f] += 1
    overlap = {}
    ds = sorted(top_sets)
    for a in ds:
        for b in ds:
            inter = len(top_sets[a] & top_sets[b]); union = len(top_sets[a] | top_sets[b])
            overlap[f"{a}|{b}"] = {"intersection": inter, "jaccard": inter / union if union else 0.0}
    pooled_support = Counter(support[f] for f in pooled_top64)
    consensus_support = Counter(support[f] for f in consensus_top64)
    return {
        "feature_space": space_name,
        "per_dataset": per_dataset,
        "consensus_top64": consensus_top64,
        "pooled_top64": pooled_top64,
        "top64_support_counts_all_features": dict(Counter(support.values())),
        "pooled_top64_support_distribution": dict(pooled_support),
        "consensus_top64_support_distribution": dict(consensus_support),
        "top64_overlap": overlap,
        "features_in_all_4_specific_top64": sorted([f for f, n in support.items() if n == 4]),
        "features_in_3plus_specific_top64": sorted([f for f, n in support.items() if n >= 3]),
    }


def pooled_mix_experiment(df: pd.DataFrame, space_name: str, features: List[str]) -> Dict[str, object]:
    datasets = sorted(df._dataset.unique())
    scenarios = {
        "fixed_per_class_40": lambda k: 40,
        "fixed_total_about_320": lambda k: max(20, int(80 / k)),
    }
    rows = []
    for scenario, count_fun in scenarios.items():
        for k in range(1, len(datasets) + 1):
            n_per_class = count_fun(k)
            for combo in combinations(datasets, k):
                for rep in range(3):
                    sample_parts = []
                    for d in combo:
                        for c, g in df[df._dataset == d].groupby("_class"):
                            n = min(n_per_class, len(g))
                            sample_parts.append(g.sample(n=n, random_state=SEED + rep * 100 + k * 10 + sum(map(ord, d+c))))
                    sub = pd.concat(sample_parts, ignore_index=True)
                    # Stratified train/val/test on union dataset::class label.
                    idx = np.arange(len(sub))
                    tr_idx, tmp_idx = train_test_split(idx, test_size=.40, random_state=SEED+rep,
                                                       stratify=sub["_label"])
                    va_idx, te_idx = train_test_split(tmp_idx, test_size=.50, random_state=SEED+rep,
                                                       stratify=sub.iloc[tmp_idx]["_label"])
                    tr = sub.iloc[tr_idx].reset_index(drop=True)
                    va = sub.iloc[va_idx].reset_index(drop=True)
                    te = sub.iloc[te_idx].reset_index(drop=True)
                    scores = fused_scores(tr, tr["_label"], features, SEED + rep + k * 100)
                    top64 = rank_scores(scores)[:64]
                    m = eval_subset(tr, va, te, top64, SEED + 500 + rep + k * 10, target="_label")
                    rows.append({
                        "feature_space": space_name, "scenario": scenario, "n_datasets": k,
                        "datasets": "+".join(combo), "rep": rep, "n_per_class": n_per_class,
                        "n_classes": int(sub["_label"].nunique()), "n_total": len(sub),
                        "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                        "top64": top64,
                    })
    tab = pd.DataFrame([{k: v for k, v in r.items() if k != "top64"} for r in rows])
    tab.to_csv(OUT / f"pooled_mix_{space_name}.csv", index=False)
    summary = (tab.groupby(["scenario", "n_datasets"])
               .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                    macro_f1_mean=("macro_f1", "mean"), n_runs=("accuracy", "size"),
                    n_classes_mean=("n_classes", "mean"), n_total_mean=("n_total", "mean"))
               .reset_index())
    summary.to_csv(OUT / f"pooled_mix_summary_{space_name}.csv", index=False)
    return {"rows": rows, "summary": summary.to_dict(orient="records")}


def main() -> None:
    t0 = time.time()
    df = extract_dataset_cache()
    meta = {
        "datasets": sorted(df._dataset.unique()),
        "classes": {d: sorted(df.loc[df._dataset == d, "_class"].unique()) for d in sorted(df._dataset.unique())},
        "samples_per_dataset_class": {
            d: {c: int(n) for c, n in g.groupby("_class").size().items()}
            for d, g in df.groupby("_dataset")
        },
        "runtime_prefix_packets": 64,
        "target_per_class": TARGET_PER_CLASS,
    }
    all_features = [c for c in df.columns if not c.startswith("_")]
    spaces = feature_spaces(all_features)
    meta["feature_space_sizes"] = {k: len(v) for k, v in spaces.items()}
    json.dump(meta, open(OUT / "dataset_summary.json", "w"), indent=2, default=lambda x: str(x))
    full = {"meta": meta, "spaces": {}}
    for sname, feats in spaces.items():
        print("\n=== SPACE", sname, len(feats), "===", flush=True)
        indiv = individual_and_common(df, sname, feats)
        pooled = pooled_mix_experiment(df, sname, feats)
        full["spaces"][sname] = {"individual_common": indiv, "pooled_mix": pooled}
        json.dump(full["spaces"][sname], open(OUT / f"results_{sname}.json", "w"), indent=2)

    full["wall_sec"] = time.time() - t0
    json.dump(full, open(OUT / "results.json", "w"), indent=2)

    # Human-readable summary from the more conservative common_shape space.
    s = full["spaces"]["common_shape"]
    lines = ["# 多数据集混合特征选择诊断（2026-09-20）", "",
             "数据集：USTC / CSTNET / PARROT / VisQUIC，每个数据集4类，每类80个会话/flow样本；统一当前runtime前64包。", "",
             "特征空间 common_shape 排除端口、TLS/QUIC/protocol显式字段、TCP flags/window等明显协议/数据集标识，主要保留包长、IAT、方向、burst、payload/header统计等跨协议形状特征。", "",
             "## 单数据集Top64 vs 多数据集共同Top64", "",
             "| Dataset | Specific Top64 Acc | Consensus Top64 Acc | Delta | Pooled-16class Top64 Acc | Delta |", "|---|---:|---:|---:|---:|---:|"]
    for d, v in sorted(s["individual_common"]["per_dataset"].items()):
        a=v["specific_metrics"]["accuracy"];c=v["consensus_metrics"]["accuracy"];p=v["pooled_top64_metrics"]["accuracy"]
        lines.append(f"| {d} | {a:.3f} | {c:.3f} | {c-a:+.3f} | {p:.3f} | {p-a:+.3f} |")
    lines += ["", "共同Top64支持度：", ""]
    lines.append(f"- 四个数据集各自Top64的全四集交集：{len(s['individual_common']['features_in_all_4_specific_top64'])}维。")
    lines.append(f"- 至少出现在3/4个数据集Top64中的特征：{len(s['individual_common']['features_in_3plus_specific_top64'])}维。")
    lines.append(f"- Consensus Top64支持度分布：{s['individual_common']['consensus_top64_support_distribution']}。")
    lines.append(f"- Pooled 16-class Top64支持度分布：{s['individual_common']['pooled_top64_support_distribution']}。")
    lines += ["", "## 混合闭集准确率", ""]
    ss=pd.DataFrame(s["pooled_mix"]["summary"])
    lines.append(ss.to_markdown(index=False))
    lines += ["", "## 初步结论", "",
              "该实验用于观察趋势，不作为正式赛题精度。是否支持‘混合数据集会迫使选择更共享的特征并降低精度’应同时看：(1) common Top64对各数据集的同split精度损失；(2) 混合数据集数增加时固定预算准确率变化；(3) Top64跨数据集支持度。"]
    (OUT / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print((OUT/"summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
