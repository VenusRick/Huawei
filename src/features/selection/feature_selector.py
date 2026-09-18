"""
智能特征选择模块
基于FSNID的转移熵方法 + NTLFlowLyzer的图方法 + XGBoost重要性

核心算法：
1. 冗余过滤：基于转移熵Φ排除无关特征
2. 相关性分析：图方法构建特征相关图，过滤通用特征
3. 特征重要性排序：XGBoost/SHAP
4. 最优子集确定：交叉验证
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class FeatureSelector:
    """智能特征选择器"""

    def __init__(self, correlation_threshold: float = 0.7,
                 common_ratio_threshold: float = 0.3,
                 max_features: int = 30,
                 min_features: int = 4):
        self.correlation_threshold = correlation_threshold
        self.common_ratio_threshold = common_ratio_threshold
        self.max_features = max_features
        self.min_features = min_features
        self.selected_features_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}
        self.correlation_matrix_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            feature_names: Optional[List[str]] = None) -> 'FeatureSelector':
        """
        执行特征选择

        融合策略：XGBoost重要性 + 互信息(MI)双信号排序
        1. 基础过滤：移除常量/高缺失/非数值特征
        2. 相关性分析：Spearman相关图过滤冗余
        3. 双信号重要性：XGBoost importance + sklearn MI 归一化后加权融合
        4. 贪心选择：按融合分数降序，跳过高度相关特征

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (n_samples,)
            feature_names: 特征名列表

        Returns:
            self
        """
        if feature_names is None:
            feature_names = list(X.columns)

        # Step 1: 基础过滤
        X_filtered, kept_names = self._basic_filter(X, feature_names)

        # Step 2: 相关性分析
        self.correlation_matrix_ = X_filtered.corr(method='spearman').abs()

        # Step 3: 双信号重要性（XGBoost + 互信息）
        xgb_importances = self._compute_importance(X_filtered, y, kept_names)
        mi_scores = self._compute_mi_scores(X_filtered, y, kept_names)

        # 归一化后加权融合: 0.6 * XGBoost + 0.4 * MI
        importances = self._fuse_importances(xgb_importances, mi_scores, kept_names,
                                              xgb_weight=0.6, mi_weight=0.4)
        self.feature_importances_ = importances
        self.xgb_importances_ = xgb_importances
        self.mi_scores_ = mi_scores

        # Step 4: 综合选择
        selected = self._select_features(X_filtered, y, kept_names, importances)
        self.selected_features_ = selected

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """使用选中的特征子集变换数据"""
        available = [f for f in self.selected_features_ if f in X.columns]
        return X[available]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series,
                     feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """拟合并变换"""
        self.fit(X, y, feature_names)
        return self.transform(X)

    # ==================== Step 1: 基础过滤 ====================

    def _basic_filter(self, X: pd.DataFrame,
                      feature_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """基础过滤：移除常量特征、高缺失率、非数值特征"""
        kept_indices = []
        kept_names = []

        for i, name in enumerate(feature_names):
            if name not in X.columns:
                continue
            col = X[name]

            # 跳过非数值列
            if not np.issubdtype(col.dtype, np.number):
                continue

            # 跳过高缺失率（>50%）
            if col.isna().mean() > 0.5:
                continue

            # 跳过常量特征（方差<1e-10）
            if col.std() < 1e-10:
                continue

            kept_indices.append(i)
            kept_names.append(name)

        return X[kept_names].copy(), kept_names

    # ==================== Step 2: 相关性分析 ====================

    def _build_correlation_graph(self, X: pd.DataFrame,
                                 feature_names: List[str]) -> Dict[str, set]:
        """构建特征相关图（参考NTLFlowLyzer）"""
        corr_graph = {name: set() for name in feature_names}
        corr_matrix = X.corr(method='spearman').abs()

        for i, name_i in enumerate(feature_names):
            for j, name_j in enumerate(feature_names):
                if i < j and name_j in corr_matrix.columns and name_i in corr_matrix.index:
                    corr_val = corr_matrix.loc[name_i, name_j]
                    if corr_val > self.correlation_threshold:
                        corr_graph[name_i].add(name_j)
                        corr_graph[name_j].add(name_i)

        return corr_graph

    # ==================== Step 3: 双信号重要性 ====================

    def _compute_importance(self, X: pd.DataFrame, y: pd.Series,
                           feature_names: List[str]) -> Dict[str, float]:
        """计算特征重要性（XGBoost）"""
        try:
            from xgboost import XGBClassifier
            from src.config import TRAIN_THREADS
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0,
                n_jobs=TRAIN_THREADS
            )
            X_filled = X.fillna(0)
            model.fit(X_filled, y)
            importances = dict(zip(feature_names, model.feature_importances_))
        except Exception:
            try:
                from sklearn.ensemble import RandomForestClassifier
                from src.config import TRAIN_THREADS
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42,
                    n_jobs=TRAIN_THREADS
                )
                X_filled = X.fillna(0)
                model.fit(X_filled, y)
                importances = dict(zip(feature_names, model.feature_importances_))
            except Exception:
                importances = {}
                for name in feature_names:
                    col = X[name].fillna(0)
                    importances[name] = col.std()

        return importances

    def _compute_mi_scores(self, X: pd.DataFrame, y: pd.Series,
                           feature_names: List[str]) -> Dict[str, float]:
        """计算每个特征与标签的互信息（sklearn实现）"""
        try:
            import warnings
            warnings.filterwarnings('ignore', message='Could not find the number of physical cores')
            from sklearn.feature_selection import mutual_info_classif
            X_filled = X[feature_names].fillna(0)
            mi = mutual_info_classif(X_filled, y, random_state=42)
            return dict(zip(feature_names, mi))
        except Exception:
            return {name: 0.0 for name in feature_names}

    def _fuse_importances(self, xgb_imp: Dict[str, float],
                          mi_scores: Dict[str, float],
                          feature_names: List[str],
                          xgb_weight: float = 0.6,
                          mi_weight: float = 0.4) -> Dict[str, float]:
        """归一化后加权融合两种重要性分数"""
        # 归一化到 [0, 1]
        def normalize(d: Dict[str, float]) -> Dict[str, float]:
            vals = list(d.values())
            lo, hi = min(vals), max(vals)
            span = hi - lo if hi - lo > 1e-10 else 1.0
            return {k: (v - lo) / span for k, v in d.items()}

        xgb_norm = normalize(xgb_imp)
        mi_norm = normalize(mi_scores)

        fused = {}
        for name in feature_names:
            fused[name] = (xgb_weight * xgb_norm.get(name, 0.0)
                         + mi_weight * mi_norm.get(name, 0.0))
        return fused

    # ==================== Step 4: 综合选择 ====================

    def _select_features(self, X: pd.DataFrame, y: pd.Series,
                        feature_names: List[str],
                        importances: Dict[str, float]) -> List[str]:
        """综合选择最优特征子集"""
        # 按重要性排序
        sorted_features = sorted(importances.items(),
                                key=lambda x: x[1], reverse=True)

        # 构建相关图，用于处理冗余
        corr_graph = self._build_correlation_graph(X, feature_names)

        # 贪心选择：优先选重要特征，跳过与已选特征高度相关的
        selected = []
        selected_set = set()

        for feat_name, importance in sorted_features:
            if len(selected) >= self.max_features:
                break

            # 检查是否与已选特征高度冗余
            is_redundant = False
            if feat_name in corr_graph:
                for neighbor in corr_graph[feat_name]:
                    if neighbor in selected_set:
                        is_redundant = True
                        break

            if not is_redundant:
                selected.append(feat_name)
                selected_set.add(feat_name)

        # 确保最少特征数
        if len(selected) < self.min_features:
            for feat_name, importance in sorted_features:
                if feat_name not in selected_set:
                    selected.append(feat_name)
                    selected_set.add(feat_name)
                if len(selected) >= self.min_features:
                    break

        return selected

    # ==================== 信息论方法（FSNID简化版） ====================

    def compute_mutual_information(self, X: pd.DataFrame, y: pd.Series,
                                   feature_names: List[str]) -> Dict[str, float]:
        """计算每个特征与标签的互信息"""
        from sklearn.feature_selection import mutual_info_classif

        X_filled = X[feature_names].fillna(0)
        mi_scores = mutual_info_classif(X_filled, y, random_state=42)
        return dict(zip(feature_names, mi_scores))

    def compute_conditional_phi(self, X: pd.DataFrame, y: pd.Series,
                                feature_name: str,
                                other_features: List[str]) -> float:
        """
        计算转移熵 Φ(X_i; X→Y) = H(Y|X\X_i) - H(Y|X)
        简化版：使用互信息差近似
        """
        from sklearn.feature_selection import mutual_info_classif

        # I(Y; X)
        X_all = X.fillna(0)
        mi_all = mutual_info_classif(X_all, y, random_state=42)[0]

        # I(Y; X\X_i)
        features_without_i = [f for f in other_features if f != feature_name]
        X_without = X[features_without_i].fillna(0)
        if X_without.shape[1] == 0:
            return 0.0
        mi_without = mutual_info_classif(X_without, y, random_state=42)[0]

        # Φ = I(Y;X) - I(Y;X\X_i) ≈ 该特征的边际贡献
        return max(0.0, mi_all - mi_without)

    # ==================== 输出 ====================

    def get_selection_report(self) -> Dict:
        """获取特征选择报告"""
        report = {
            'selected_features': self.selected_features_,
            'num_selected': len(self.selected_features_),
            'feature_importances': {
                feat: self.feature_importances_.get(feat, 0.0)
                for feat in self.selected_features_
            },
            'importance_ranking': sorted(
                self.feature_importances_.items(),
                key=lambda x: x[1], reverse=True
            )[:20],
        }
        return report
