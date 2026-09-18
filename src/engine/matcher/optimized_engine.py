"""
优化版DPI引擎
支持XGBoost集成分类器 + 决策树规则 + 统计规则的混合匹配。

关键改进：
1. 支持加载XGBoost模型进行直接预测
2. 软匹配规则引擎（加权评分，非全或无）
3. 置信度校准
"""
import yaml
import time
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class MatchResult:
    """匹配结果"""
    rule_id: str
    rule_name: str
    app: str
    behavior: str
    result: str
    confidence: float
    source: str
    match_time_ms: float
    match_score: float = 1.0   # 软匹配得分 (0~1)


class OptimizedDPIEngine:
    """优化版DPI规则识别引擎"""

    def __init__(self, soft_match_threshold: float = 0.75):
        self.classifier = None
        self.selected_features: List[str] = []
        self.label_names: Dict[int, str] = {}
        self.confidence_threshold: float = 0.70
        self.rules: List[Dict] = []
        self.soft_match_threshold = soft_match_threshold  # 软匹配最低得分

        # 统计
        self.total_matches = 0
        self.total_predictions = 0
        self.avg_match_time_ms = 0.0

    def load_rules(self, rule_file_path: str):
        """加载规则文件"""
        path = Path(rule_file_path)
        if not path.exists():
            raise FileNotFoundError(f"规则文件不存在: {rule_file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            rule_data = yaml.safe_load(f)

        self.rules = rule_data.get('rules', [])

        # 加载分类器配置
        classifier_config = rule_data.get('classifier', {})
        self.confidence_threshold = classifier_config.get('confidence_threshold', 0.70)
        self.label_names = classifier_config.get('label_names', {})
        # label_names 的 key 是字符串，转为 int
        self.label_names = {int(k): v for k, v in self.label_names.items()}

        print(f"[DPI Engine v2] 加载了 {len(self.rules)} 条规则, "
              f"{len(self.label_names)} 个类别, "
              f"置信度阈值={self.confidence_threshold}")

    def load_classifier(self, classifier, selected_features: List[str]):
        """直接加载训练好的分类器（用于pipeline集成）"""
        self.classifier = classifier
        self.selected_features = selected_features

    def load_rule_bundle(self, bundle_dir: str):
        """从规则包目录加载（独立规则推理；不加载任何预测器/训练库）"""
        from src.engine.model_io import load_rule_bundle
        rules, features, label_names, config = load_rule_bundle(bundle_dir)
        self.rules = rules
        self.selected_features = features
        self.label_names = label_names
        self.confidence_threshold = config.get('confidence_threshold', 0.65)
        self.classifier = None  # 规则模式：禁用预测器，仅严格规则匹配
        print(f"[DPI Engine] 从 {bundle_dir} 加载规则包: {len(rules)} 条规则, "
              f"{len(features)} 特征, 阈值={self.confidence_threshold}")
        return self

    def load_model_dir(self, model_dir: str):
        """
        从目录加载完整模型包（独立推理用）。

        目录内容：
        - model.json           ← XGBoost原生JSON模型
        - selected_features.json ← 选中特征列表
        - engine_config.json   ← 引擎配置（阈值、类别映射等）
        """
        from src.engine.model_io import load_model_bundle
        clf, features, label_names, config = load_model_bundle(model_dir)
        self.classifier = clf
        self.selected_features = features
        self.label_names = label_names
        self.confidence_threshold = config.get('confidence_threshold', 0.65)
        self.soft_match_threshold = config.get('soft_match_threshold', 0.75)
        print(f"[DPI Engine] 从 {model_dir} 加载模型: "
              f"{len(features)} 特征, {len(label_names)} 类别, "
              f"阈值={self.confidence_threshold}")

    def match(self, features: Dict[str, float]) -> List[MatchResult]:
        """
        使用混合策略匹配

        优先级：
        1. XGBoost集成分类器（如果有）
        2. 决策树规则
        3. 统计规则
        """
        start_time = time.perf_counter()
        results = []
        self.total_predictions += 1

        # 方式1：XGBoost集成分类器
        if self.classifier is not None and self.selected_features:
            x = np.array([features.get(f, 0.0)
                         for f in self.selected_features]).reshape(1, -1)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            proba = self.classifier.predict_proba(x)[0]
            pred_class = np.argmax(proba)
            confidence = float(proba[pred_class])

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if confidence >= self.confidence_threshold:
                label_name = self.label_names.get(pred_class, f'class_{pred_class}')
                results.append(MatchResult(
                    rule_id='ENS_0001',
                    rule_name='XGBoost集成分类器',
                    app=self._extract_app(label_name),
                    behavior=self._extract_behavior(label_name),
                    result=label_name,
                    confidence=confidence,
                    source='xgboost_ensemble',
                    match_time_ms=elapsed_ms,
                ))

        # 方式2：决策树规则 + 统计规则（备选）
        if not results:
            for rule in self.rules:
                if rule.get('type') == 'ensemble_classifier':
                    continue
                if rule.get('type') in ('decision_tree', 'statistical'):
                    match_result = self._evaluate_rule(rule, features)
                    if match_result:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        match_result.match_time_ms = elapsed_ms
                        results.append(match_result)
                        break  # 取第一个匹配的规则

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.total_matches += len(results)
        self.avg_match_time_ms = (
            (self.avg_match_time_ms * (self.total_predictions - 1) + elapsed_ms)
            / self.total_predictions
        )

        return results

    def _evaluate_rule(self, rule: Dict,
                       features: Dict[str, float]) -> Optional[MatchResult]:
        """评估单条规则（软匹配：按条件命中比例评分）"""
        conditions = rule.get('conditions', [])
        if not conditions:
            return None

        matched = 0
        total = len(conditions)

        for cond in conditions:
            feat_val = features.get(cond['feature'])
            op = cond.get('op', 'range')
            # R22: 规则所需特征缺失（除not_exists外）→ 整条规则不匹配，不静默跳过
            if feat_val is None:
                if op == 'not_exists':
                    matched += 1
                    continue
                return None
            try:
                if op == 'range':
                    if cond['min'] <= float(feat_val) <= cond['max']:
                        matched += 1
                elif op == '<=':
                    if float(feat_val) <= float(cond['value']):
                        matched += 1
                elif op == '>':
                    if float(feat_val) > float(cond['value']):
                        matched += 1
                elif op == '==':
                    if str(feat_val) == str(cond['value']):
                        matched += 1
                elif op == 'contains':
                    if isinstance(cond['value'], list):
                        if any(v in str(feat_val) for v in cond['value']):
                            matched += 1
                    elif str(cond['value']) in str(feat_val):
                        matched += 1
                elif op == 'not_exists':
                    # R19: 0是合法观测值不算缺失；缺失已在前置分支处理
                    if feat_val == '':
                        matched += 1
            except (ValueError, TypeError):
                pass

        score = matched / total
        # R18: 严格AND——全部条件命中(score==1.0)才匹配；soft_match_threshold仅作legacy对照保留
        if score >= 1.0:
            action = rule.get('action', {})
            base_confidence = rule.get('confidence', 0.0)
            # 软匹配置信度 = 基础置信度 × 匹配得分
            adjusted_confidence = base_confidence * score
            return MatchResult(
                rule_id=rule.get('id', ''),
                rule_name=rule.get('name', ''),
                app=rule.get('app', ''),
                behavior=rule.get('behavior', ''),
                result=action.get('result', rule.get('name', '')),
                confidence=adjusted_confidence,
                source=action.get('source', 'unknown'),
                match_time_ms=0.0,
                match_score=score,
            )

        return None

    def _extract_app(self, label: str) -> str:
        parts = label.split('_')
        return parts[0] if parts else label

    def _extract_behavior(self, label: str) -> str:
        parts = label.split('_')
        return '_'.join(parts[1:]) if len(parts) > 1 else 'general'

    def get_statistics(self) -> Dict:
        return {
            'total_rules': len(self.rules),
            'total_predictions': self.total_predictions,
            'total_matches': self.total_matches,
            'avg_match_time_ms': round(self.avg_match_time_ms, 4),
            'match_rate': (self.total_matches / max(self.total_predictions, 1)),
        }
