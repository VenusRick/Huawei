"""主机验收：实际规则依赖、评价覆盖和最大事件匹配。"""
import builtins
import json
import random

from src import evaluation
from src.features.operators import required_features_of_rules, resolve_required_families


def test_unused_selected_feature_does_not_activate_expensive_family():
    rules = [{"type": "decision_tree", "conditions": [
        {"feature": "total_packets", "op": ">", "value": 2}]}]
    required = required_features_of_rules(
        rules, ["total_packets", "iat_hurst_exponent"])
    assert required == ["total_packets"]
    assert resolve_required_families(required) == {"basic"}


def test_ensemble_placeholder_is_not_a_runtime_dependency():
    rules = [{"type": "ensemble_classifier", "conditions": [
        {"feature": "iat_hurst_exponent"}]}]
    assert required_features_of_rules(rules, ["iat_hurst_exponent"]) == []


def test_observation_only_coverage_and_missing_source_status(tmp_path):
    predictions = tmp_path / "pred.json"
    predictions.write_text(json.dumps([
        {"observation_id": "a.pcap:0", "predicted_label": "a"}]))
    truth = {"a.pcap": {"label": "a", "trial_id": "a"}}
    out = evaluation.evaluate_files(str(predictions), truth, {"a": 0},
                                    str(tmp_path / "one"))
    assert out["truth_coverage"]["n_covered_files"] == 1
    assert out["evaluation_scope"]["status"] == "covered_sources"
    truth["b.pcap"] = {"label": "a", "trial_id": "b"}
    out = evaluation.evaluate_files(str(predictions), truth, {"a": 0},
                                    str(tmp_path / "two"))
    assert out["evaluation_scope"]["status"] == "incomplete"
    assert out["evaluation_scope"]["reasons"] == ["missing_truth_files"]
    assert out["n_samples"] == 1  # 不用文件数伪造会话分母


def test_maximum_event_matching_without_optional_optimizer(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("scipy.optimize"):
            raise ImportError("test: optimizer unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    predicted = [{"label": "a", "terminal": "x", "t": t}
                 for t in (0.9, 1.05)]
    truth = [{"label": "a", "terminal": "x", "t": t}
             for t in (0.0, 1.0)]
    result = evaluation.match_events(predicted, truth, tolerance=1.0)
    assert result["tp"] == 2
    assert result["matching"] == "max_cardinality"


def test_maximum_matching_agrees_with_exhaustive_small_graphs(monkeypatch):
    rng = random.Random(42)
    for _ in range(30):
        n, m = rng.randint(1, 4), rng.randint(1, 4)
        edges = {(i, j) for i in range(n) for j in range(m)
                 if rng.random() < 0.5}

        def exhaustive(i, used):
            if i == n:
                return 0
            return max([exhaustive(i + 1, used)] + [
                1 + exhaustive(i + 1, used | {j}) for j in range(m)
                if (i, j) in edges and j not in used])

        monkeypatch.setattr(evaluation, "_eligible_pairs",
                            lambda *args: [(i, j, 0.0) for i, j in edges])
        result = evaluation.match_events([{}] * n, [{}] * m)
        assert result["tp"] == exhaustive(0, set())
