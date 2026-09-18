# -*- coding: utf-8 -*-
"""契约层四服务（data_manifest/runtime/evaluation/reporting）守门测试。

P0 红线用例：
- 数据划分永不跨 capture/trial（建议.md 第五点反例）；
- 评价拒识语义："100 已知只接纳 20"时 Recall=0.2 而非 1；全拒识 Recall=0；
- runtime 与 dpi_infer 旧提取路径逐字一致（parity 前哨）。
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_manifest import (ManifestError, TrialRecord, build_label_vocab,
                               discover_candidate_labels, load_manifest,
                               split_by_group, validate_manifest,
                               write_manifest)
from src.evaluation import (UNKNOWN, compute_metrics, evaluate_files,
                            match_events)
from src.reporting import write_run_report

FIXTURE = Path(__file__).resolve().parents[1] / "output" / "fixture_m1"


# ---------------------------------------------------------------- manifest
def _trial(**kw):
    base = dict(capture_id="c1", trial_id="t1", platform="pc",
                device_id="d1", app="appA", behavior="text",
                pcap="x.pcap", label_source="manual_verified",
                behavior_start=1.0, behavior_end=9.0)
    base.update(kw)
    return TrialRecord(**base)


def test_load_manifest_rejects_missing_fields(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"capture_id": "c1"}) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(str(p))


def test_load_manifest_rejects_bad_platform(tmp_path):
    obj = _trial().to_dict(); obj["platform"] = "web"
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(obj) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(str(p))


def test_load_manifest_rejects_half_time_window(tmp_path):
    obj = _trial().to_dict(); obj["behavior_end"] = None
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(obj) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(str(p))


def test_manifest_roundtrip(tmp_path):
    recs = [_trial(trial_id="t1"), _trial(trial_id="t2", app="appB")]
    p = tmp_path / "m.jsonl"
    write_manifest(recs, str(p))
    back = load_manifest(str(p))
    assert [r.app for r in back] == ["appA", "appB"]


def test_split_never_breaks_capture_group():
    """同一 capture_id 的多个 trial 必须落进同一 split（防窗口泄漏）。"""
    recs = []
    for c in range(12):
        for t in range(3):
            recs.append(_trial(capture_id=f"cap{c:02d}",
                               trial_id=f"cap{c:02d}_t{t}"))
    out = split_by_group(recs, seed=7)
    where = {}
    for r in out:
        where.setdefault(r.capture_id, set()).add(r.split)
    assert all(len(v) == 1 for v in where.values()), "capture 组被拆裂"
    splits = {r.split for r in out}
    assert splits <= {"train", "validation", "test"}


def test_label_vocab_stable_ids():
    recs = [_trial(app="appB"), _trial(app="appA"), _trial(app="appA")]
    v1 = build_label_vocab(recs, "app")
    v2 = build_label_vocab(recs, "app")
    assert v1["label_map"] == {"appA": 0, "appB": 1} == v2["label_map"]


def test_validate_manifest_flags_missing_pcap(tmp_path):
    recs = [_trial(pcap=str(tmp_path / "nope.pcap"))]
    with pytest.raises(ManifestError):
        validate_manifest(recs, manifest_dir=str(tmp_path))


def test_validate_manifest_requires_time_window_for_behavior(tmp_path):
    pcap = tmp_path / "ok.pcap"; pcap.write_bytes(b"x")
    recs = [_trial(pcap=str(pcap), behavior_start=None, behavior_end=None)]
    with pytest.raises(ManifestError):
        validate_manifest(recs, task="behavior")


def test_candidate_label_discovery(tmp_path):
    (tmp_path / "CIC-IoT-2022-a.pcap").write_bytes(b"x")
    (tmp_path / "PARROT-2025-b.pcap").write_bytes(b"x")
    rows = discover_candidate_labels(str(tmp_path))
    assert len(rows) == 2
    assert all(r["candidate_label"] for r in rows)
    assert all(r["source"] == "filename_prefix" for r in rows)


# ---------------------------------------------------------------- evaluation
def _metrics_for(known_n, accepted_correct, n_classes=2):
    """known_n 个已知真值，其中 accepted_correct 个被正确接纳，其余拒识。"""
    names = [f"c{i}" for i in range(n_classes)]
    y_true = [0] * known_n
    y_pred = [0] * accepted_correct + [UNKNOWN] * (known_n - accepted_correct)
    return compute_metrics(y_true, y_pred, names)


def test_partial_acceptance_recall_is_not_one():
    """100 已知只正确接纳 20：Recall 必须是 0.2，不得报 1。"""
    m = _metrics_for(known_n=100, accepted_correct=20)
    assert m["per_class"]["c0"]["recall"] == pytest.approx(0.2)
    assert m["accuracy"] == pytest.approx(0.2)
    assert m["n_rejected"] == 80


def test_all_rejected_recall_zero():
    m = _metrics_for(known_n=10, accepted_correct=0)
    assert m["per_class"]["c0"]["recall"] == 0.0
    assert m["macro_f1"] == 0.0


def test_metrics_full_denominator():
    m = _metrics_for(known_n=10, accepted_correct=10)
    assert m["n_samples"] == 10
    assert m["per_class"]["c0"]["rejected_true_of_class"] == 0


def test_evaluate_files_writes_metrics(tmp_path):
    pred = {"results": [
        {"observation_id": "a.pcap:0", "predicted_label": "appA",
         "confidence": 1.0},
        {"observation_id": "a.pcap:1", "predicted_label": "unknown",
         "confidence": 0.1},
        {"observation_id": "zzz.pcap:0", "predicted_label": "appA",
         "confidence": 1.0},
    ]}
    pp = tmp_path / "pred.json"
    pp.write_text(json.dumps(pred), encoding="utf-8")
    truth = {"a.pcap": {"label": "appA", "trial_id": "t1",
                        "capture_id": "c1", "split": "test", "platform": "pc"}}
    m = evaluate_files(str(pp), truth, {"appA": 0}, str(tmp_path / "ev"))
    assert m["n_samples"] == 2
    assert m["n_unpaired_predictions"] == 1
    assert (tmp_path / "ev" / "metrics.json").exists()


def test_match_events_one_to_one():
    truth = [{"terminal": "10.0.0.1", "label": "text", "t": 5.0}]
    preds = [{"terminal": "10.0.0.1", "label": "text", "t": 5.5},
             {"terminal": "10.0.0.1", "label": "text", "t": 6.0}]
    r = match_events(preds, truth, tolerance=2.0)
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 0
    assert r["event_recall"] == 1.0


# ---------------------------------------------------------------- runtime
@pytest.mark.skipif(not FIXTURE.exists(), reason="M1 fixture 未生成")
def test_runtime_parity_with_dpi_infer_path():
    """runtime 提取与 dpi_infer 原路径逐特征一致（parity 前哨）。"""
    from src.features.runtime import extract_feature_records
    from src.engine.dpi_infer import extract_features_from_pcap
    pcap = sorted((FIXTURE / "test" / "appA").glob("*.pcap"))[0]
    recs = extract_feature_records(str(pcap))
    old = extract_features_from_pcap(str(pcap))
    assert len(recs) == len(old)
    for r, o in zip(recs, old):
        assert set(r.features) == set(o)
        for k in r.features:
            assert r.features[k] == o[k], f"特征失配 {k}"


@pytest.mark.skipif(not FIXTURE.exists(), reason="M1 fixture 未生成")
def test_runtime_schema_stable_across_pcaps():
    from src.features.runtime import extract_feature_records, feature_names
    pcaps = sorted((FIXTURE / "test").rglob("*.pcap"))[:3]
    schemas = [set(feature_names(extract_feature_records(str(p))))
               for p in pcaps]
    assert all(s == schemas[0] for s in schemas), "schema 不稳定（R16 回退）"


# ---------------------------------------------------------------- reporting
def test_write_run_report(tmp_path):
    bundle = tmp_path / "bundle"; bundle.mkdir()
    (bundle / "rules.json").write_text(
        json.dumps([{"label": "appA", "conditions": []}]), encoding="utf-8")
    (bundle / "selected_features.json").write_text(
        json.dumps(["f1", "f2"]), encoding="utf-8")
    (bundle / "bundle_config.json").write_text(
        json.dumps({"label_names": {"appA": 0}, "confidence_threshold": 0.9}),
        encoding="utf-8")
    metrics = compute_metrics([0, 0], [0, 1], ["appA", "appB"])
    out = write_run_report(str(bundle), metrics, str(tmp_path / "run"))
    text = Path(out).read_text(encoding="utf-8")
    assert "规则数: 1" in text
    assert "保留比例，非特征有效率" in text
