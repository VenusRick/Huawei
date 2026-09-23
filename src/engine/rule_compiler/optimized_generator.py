"""
优化版规则生成器
用 XGBoost 集成分类器 + 置信度阈值 + 规则提取替代纯决策树。

关键改进：
1. 集成分类器：XGBoost 替代单棵决策树
2. 置信度阈值：低置信度结果标记为"未知"，降低误报
3. 规则提取：从XGBoost中提取可解释规则
4. 统计规则增强：更精细的特征范围计算
"""
import numpy as np
import pandas as pd
import yaml
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class OptimizedRuleGenerator:
    """优化版规则生成器"""

    def __init__(self, confidence_threshold: float = 0.70,
                 min_support: float = 0.1,
                 min_confidence: float = 0.4,
                 tree_max_depth: Optional[int] = None,
                 tree_min_samples_leaf: Optional[int] = None,
                 tree_class_weight: Optional[str] = None):
        self.confidence_threshold = confidence_threshold
        self.min_support = min_support
        self.min_confidence = min_confidence
        # 浅树参数：None = 按样本量/类数自适应（小样本放宽叶子下限，
        # 防止 6类×41样本 因 min_samples_leaf=5 无法为小类产出纯叶规则）
        self.tree_max_depth = tree_max_depth
        self.tree_min_samples_leaf = tree_min_samples_leaf
        # 类均衡浅树（2026-09-18 第三轮）：'balanced' 时 sklearn 以类权重
        # 拟合分裂；导出规则的 confidence 一律用"未加权训练回放的真实
        # precision"，加权纯度只作 leaf_purity 元数据，不冒充 precision
        self.tree_class_weight = tree_class_weight
        self.rules: List[Dict] = []
        self.feature_definitions: Dict[str, Dict] = {}
        self.classifier = None
        self.label_names: Dict[int, str] = {}

    def _adaptive_tree_params(self, n_samples: int, n_classes: int) -> Tuple[int, int]:
        """自适应浅树参数（2026-09-18 第二轮；固定 depth5/leaf5 在小样本
        多类下欠拟合：41样本6类只导出1条树规则）。

        leaf = max(1, min(5, n // (2k)))——每类期望样本的一半为叶下限；
        depth 默认 5，仅当 leaf 放宽后仍不足以给每类一个纯叶时加深。
        显式构造参数优先（受控实验用；fit 只用 train）。
        """
        k = max(1, n_classes)
        auto_leaf = max(1, min(5, n_samples // (2 * k)))
        leaf = (self.tree_min_samples_leaf if self.tree_min_samples_leaf
                is not None else auto_leaf)
        depth = (self.tree_max_depth if self.tree_max_depth is not None
                 else 5)
        return int(depth), int(leaf)

    def fit_and_generate(self, X: pd.DataFrame, y: pd.Series,
                         selected_features: List[str],
                         label_names: Dict[int, str],
                         validation: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                         groups: Optional[pd.Series] = None) -> List[Dict]:
        """
        训练集成分类器并生成规则

        Args:
            X: 特征矩阵（仅训练集）
            y: 标签（仅训练集）
            selected_features: 选中的特征列表
            label_names: 标签名映射
            validation: 可选 (X_val, y_val)——显式验证集，只用于一次性离线
                对照评估，不参与拟合（防泄漏；替代旧的5折CV）
            groups: 训练样本的采集组（capture_id）；无验证集且组数>=2时
                按组做 GroupKFold 对照，组数不足则明确跳过CV

        Returns:
            规则列表
        """
        from xgboost import XGBClassifier
        from sklearn.tree import DecisionTreeClassifier

        self.label_names = label_names
        X_sel = X[selected_features].fillna(0)

        # 1. 训练XGBoost集成分类器
        from src.config import TRAIN_THREADS
        self.classifier = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0,
            n_jobs=TRAIN_THREADS,
        )
        self.classifier.fit(X_sel, y)

        # 离线对照评估（XGBoost 成绩只作特征/配置对照，不是规则DPI成绩）：
        # - 有显式验证集 -> 验证集一次评估（不参与拟合）
        # - 无验证集但采集组>=2 -> GroupKFold（同采集组不跨折）
        # - 组数不足 -> 明确跳过（旧版固定5折CV按会话泄漏且小样本崩溃，已废除）
        if validation is not None:
            from sklearn.metrics import f1_score
            X_val, y_val = validation
            pred = self.classifier.predict(
                X_val[selected_features].fillna(0))
            print(f"  [规则生成] XGBoost 验证集 Macro-F1: "
                  f"{f1_score(y_val, pred, average='macro', zero_division=0):.4f}"
                  f"（离线对照，非规则DPI成绩；验证集不参与拟合）")
        elif groups is not None:
            from sklearn.metrics import f1_score
            from sklearn.model_selection import GroupKFold, cross_val_predict
            g = pd.Series(list(groups)).reset_index(drop=True)
            n_groups = int(g.nunique())
            n_splits = min(5, n_groups)
            # 组-类完全重合守卫：每个采集组只含一个类时，任何组折的训练集
            # 都是单类，CV不可拟也不可信（常见于 fixture：组=类）
            group_class_overlap = all(
                y[g == gv].nunique() > 1 for gv in g.unique())
            if n_splits >= 2 and group_class_overlap:
                try:
                    gkf = GroupKFold(n_splits=n_splits)
                    pred = cross_val_predict(
                        self.classifier, X_sel, y, groups=g, cv=gkf)
                    print(f"  [规则生成] XGBoost GroupKFold({n_splits}折,"
                          f"{n_groups}采集组) Macro-F1: "
                          f"{f1_score(y, pred, average='macro', zero_division=0):.4f}"
                          f"（离线对照，非规则DPI成绩）")
                except Exception as _e:  # noqa: BLE001
                    print(f"  [规则生成] CV对照执行失败已跳过: {_e}")
            else:
                why = (f"采集组仅{n_groups}组" if n_splits < 2
                       else "采集组与类别完全重合（组折训练集单类）")
                print(f"  [规则生成] 跳过CV对照：{why}"
                      f"（不按会话随机折防泄漏）")
        else:
            print("  [规则生成] 跳过CV对照：未提供验证集/采集组"
                  "（不按会话随机折防泄漏）")

        # 2. 提取特征重要性
        importances = dict(zip(selected_features, self.classifier.feature_importances_))
        self.feature_importances_ = importances

        # 3. 从XGBoost中提取置信度规则
        rules = []

        # 方式A：直接使用XGBoost作为分类器（输出置信度）
        rules.append(self._create_ensemble_rule(selected_features, importances))

        # 方式B：从XGBoost训练的浅层决策树中提取可解释规则
        # （树条件只用训练内完全present的特征——训练fillna(0)造出的阈值
        #   与推理缺失即不匹配的语义不一致，2026-09-18 第二轮修复）
        tree_rules = self._extract_tree_rules(X, y, selected_features, label_names)
        rules.extend(tree_rules)

        # 方式C：统计阈值规则（每个类别的特征范围；
        #   confidence=train precision，零覆盖死规则不部署）
        stat_rules = self._generate_statistical_rules(X, y, selected_features, label_names)
        rules.extend(stat_rules)

        self.rules = rules
        return rules

    def predict_with_rules(self, features: Dict[str, float],
                           selected_features: List[str]) -> Tuple[int, float, str]:
        """
        使用训练好的分类器进行预测

        Args:
            features: 特征字典
            selected_features: 选中的特征列表

        Returns:
            (预测标签ID, 置信度, 规则来源)
        """
        # 构建特征向量
        x = np.array([features.get(f, 0.0) for f in selected_features]).reshape(1, -1)

        # XGBoost预测概率
        proba = self.classifier.predict_proba(x)[0]
        pred_class = np.argmax(proba)
        confidence = proba[pred_class]

        # 置信度阈值检查
        if confidence < self.confidence_threshold:
            return -1, confidence, "low_confidence"

        return int(pred_class), confidence, "xgboost_ensemble"

    def generate_feature_definitions(self, X: pd.DataFrame,
                                      selected_features: List[str],
                                      y: pd.Series,
                                      label_names: Dict[int, str]):
        """
        生成特征定义（含统计信息和区分度），用于rules.yaml的features字段。

        每个特征记录：
        - type: 数据类型
        - description: 特征描述
        - stats: 全局统计 (min/max/mean/std)
        - per_class_stats: 每个类别的统计
        - importance: XGBoost特征重要性
        - discriminative_power: 区分度评分
        """
        self.feature_definitions = {}

        for feat in selected_features:
            if feat not in X.columns:
                continue

            col = X[feat].dropna()
            if len(col) == 0:
                continue

            # 全局统计
            stats = {
                'min': round(float(col.min()), 6),
                'max': round(float(col.max()), 6),
                'mean': round(float(col.mean()), 6),
                'std': round(float(col.std()), 6),
                'median': round(float(col.median()), 6),
            }

            # 每个类别的统计
            per_class = {}
            for lid, lname in label_names.items():
                class_mask = y == lid
                if class_mask.sum() > 0:
                    class_vals = X.loc[class_mask, feat].dropna()
                    if len(class_vals) > 0:
                        per_class[lname] = {
                            'mean': round(float(class_vals.mean()), 6),
                            'std': round(float(class_vals.std()), 6),
                            'q05': round(float(class_vals.quantile(0.05)), 6),
                            'q95': round(float(class_vals.quantile(0.95)), 6),
                        }

            # 区分度：用类间方差 / 类内方差衡量
            disc_power = 0.0
            class_means = []
            for lid in label_names.keys():
                class_mask = y == lid
                if class_mask.sum() > 0:
                    class_means.append(X.loc[class_mask, feat].mean())
            if len(class_means) > 1 and col.std() > 0:
                between_var = np.var(class_means)
                disc_power = round(float(between_var / (col.std() ** 2 + 1e-10)), 6)

            self.feature_definitions[feat] = {
                'type': 'float',
                'description': self._feature_description(feat),
                'stats': stats,
                'per_class_stats': per_class,
                'importance': round(float(
                    self.feature_importances_.get(feat, 0)), 6),
                'discriminative_power': disc_power,
            }

    def export_yaml(self, output_path: str, label_names: Dict[int, str]):
        """导出规则文件为YAML格式"""
        # 注册numpy类型的YAML序列化
        import yaml
        def _numpy_representer(dumper, data):
            if hasattr(data, 'item'):
                return dumper.represent_scalar('tag:yaml.org,2002:float', str(data.item()))
            return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))
        yaml.add_representer(np.float64, _numpy_representer)
        yaml.add_representer(np.float32, _numpy_representer)
        yaml.add_representer(np.int64, _numpy_representer)
        yaml.add_representer(np.int32, _numpy_representer)

        rule_file = {
            'version': '2.0',
            'generated_at': datetime.now().isoformat(),
            'description': '优化版DPI规则（XGBoost集成 + 置信度阈值）',
            'engine': 'optimized_v2',

            'classifier': {
                'type': 'xgboost_ensemble',
                'confidence_threshold': float(self.confidence_threshold),
                'num_classes': int(len(label_names)),
                'label_names': {int(k): str(v) for k, v in label_names.items()},
            },

            'features': {},
            'rules': [],
            'metadata': {
                'total_rules': len(self.rules),
                'label_names': label_names,
            }
        }

        # 特征定义
        for name, defn in self.feature_definitions.items():
            rule_file['features'][name] = defn

        # 规则（确保所有值为Python原生类型）
        for rule in self.rules:
            rule_file['rules'].append(self._sanitize_for_yaml(rule))

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(rule_file, f, allow_unicode=True,
                      default_flow_style=False, sort_keys=False)

        return output_path

    # ==================== 规则生成方法 ====================

    def _create_ensemble_rule(self, features: List[str],
                              importances: Dict[str, float]) -> Dict:
        """创建集成分类器规则"""
        return {
            'id': 'ENS_0001',
            'name': 'XGBoost集成分类器',
            'type': 'ensemble_classifier',
            'confidence_threshold': self.confidence_threshold,
            'num_features': len(features),
            'top_features': [[k, float(v)] for k, v in
                            sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]],
            'action': {
                'result': 'classify_by_ensemble',
                'confidence': 'from_predict_proba',
                'source': 'xgboost',
            }
        }

    def _extract_tree_rules(self, X: pd.DataFrame, y: pd.Series,
                            features: List[str],
                            label_names: Dict[int, str]) -> List[Dict]:
        """从浅层决策树提取可解释规则。

        - 参数自适应小样本（_adaptive_tree_params）；
        - tree_class_weight='balanced' 时以类权重拟合分裂（第三轮候选）；
        - 导出规则的 confidence 一律为"未加权训练回放的真实precision"：
          加权树的叶纯度（sklearn加权value）只作 leaf_purity 元数据，
          不得冒充 precision（计划 S2.3）；
        - 条件特征只在"训练行完全present（无NaN）"的特征里选：训练侧
          fillna(0) 会造出"缺失=0"的阈值，而推理侧特征缺失=规则不匹配
          （R22 严格语义），两侧语义必须一致——缺特征的特征不进规则，
          而不是靠0填对齐（2026-09-18 第二轮）。
        """
        from sklearn.tree import DecisionTreeClassifier

        usable = [f for f in features
                  if f in X.columns and not X[f].isna().any()]
        if not usable:
            return []
        depth, leaf = self._adaptive_tree_params(len(X), y.nunique())
        tree = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=leaf, random_state=42,
            class_weight=self.tree_class_weight
        )
        tree.fit(X[usable].fillna(0), y)

        tree_model = tree.tree_
        rules = []
        # R31: 传入树自身classes_，叶标签按真实类别值解析（支持字符串/非连续ID）
        self._traverse_tree(tree_model, 0, usable, label_names, rules, [], tree.classes_)
        # 真实未加权回放统计：confidence/train_precision/coverage（加权树
        # 的叶纯度不可冒充 precision）；零命中死规则剔除（与STAT一致）
        name_to_id = {v: k for k, v in label_names.items()}
        kept = []
        for r in rules:
            covered = pd.Series(True, index=X.index)
            for c in r['conditions']:
                col = X[c['feature']]
                if c['op'] == '<=':
                    covered &= (col <= c['value'])
                else:
                    covered &= (col > c['value'])
            n_fire = int(covered.sum())
            if n_fire == 0:
                continue
            lid = name_to_id.get(r['action']['result'])
            precision = float((y[covered] == lid).mean()) if lid is not None else 0.0
            coverage = float(covered[y == lid].mean()) if lid is not None else 0.0
            r['leaf_purity'] = r.pop('confidence')
            r['confidence'] = precision
            r['train_precision'] = round(precision, 4)
            r['train_coverage'] = round(coverage, 4)
            r['n_train_fires'] = n_fire
            r['action']['confidence'] = precision
            kept.append(r)
        return kept

    def _traverse_tree(self, tree, node_id, features, label_names, rules, conditions,
                        classes=None):
        """遍历决策树"""
        if tree.children_left[node_id] == tree.children_right[node_id]:
            values = tree.value[node_id][0]
            idx = int(np.argmax(values))
            predicted = classes[idx] if classes is not None else idx
            confidence = float(values[idx] / values.sum()) if values.sum() > 0 else 0.0

            # R21: class 0与其它类别同等导出；R20: 阈值用配置而非硬编码0.7
            if confidence >= self.confidence_threshold:
                label = label_names.get(predicted, str(predicted))
                rules.append({
                    'id': f'DT_{len(rules)+1:04d}',
                    'name': f'{label}_决策树规则',
                    'type': 'decision_tree',
                    'app': self._extract_app_name(label),
                    'behavior': self._extract_behavior_name(label),
                    'priority': 100,
                    'confidence': float(confidence),
                    'conditions': conditions.copy(),
                    'action': {'result': label, 'confidence': float(confidence), 'source': 'decision_tree'}
                })
            return

        feat_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feat_name = features[feat_idx]

        conditions.append({'feature': feat_name, 'op': '<=', 'value': float(threshold)})
        self._traverse_tree(tree, tree.children_left[node_id], features, label_names, rules, conditions, classes)
        conditions.pop()

        conditions.append({'feature': feat_name, 'op': '>', 'value': float(threshold)})
        self._traverse_tree(tree, tree.children_right[node_id], features, label_names, rules, conditions, classes)
        conditions.pop()

    def _generate_statistical_rules(self, X: pd.DataFrame, y: pd.Series,
                                     features: List[str],
                                     label_names: Dict[int, str]) -> List[Dict]:
        """基于统计的特征范围规则。

        语义修复（2026-09-18 第二轮）：
        - confidence = train precision（该规则在训练集上命中时的类别纯度），
          不再是 class coverage（召回）——引擎/报告消费的是"命中可信度"，
          召回语义既误导阈值判断也压低真实可信规则；
        - train_coverage 与 train_precision 一并写入规则元数据；
        - 零覆盖规则（在训练集上一次都不命中）不再部署（死规则只会
          在未知数据上盲发预测）；
        - 条件特征只在训练行完全present的特征里选（缺特征不靠0填造
          范围，与推理缺失即不匹配一致）。
        """
        rules = []
        usable = [f for f in features
                  if f in X.columns and not X[f].isna().any()]
        if not usable:
            return rules

        for label_id, label_name in label_names.items():
            mask = y == label_id
            if mask.sum() < 3:
                continue

            X_class = X[usable][mask]
            X_other = X[usable][~mask]

            conditions = []
            for feat in usable:
                class_vals = X_class[feat]
                other_vals = X_other[feat]

                if class_vals.std() < 1e-10:
                    continue

                # 使用5%-95%范围
                q05 = float(class_vals.quantile(0.05))
                q95 = float(class_vals.quantile(0.95))

                if q95 > q05:
                    # 计算区分度
                    in_range_other = ((other_vals >= q05) & (other_vals <= q95)).mean()
                    if in_range_other < 0.5:
                        conditions.append({
                            'feature': feat,
                            'op': 'range',
                            'min': round(q05, 4),
                            'max': round(q95, 4),
                        })

                if len(conditions) >= 8:
                    break

            if len(conditions) >= 2:
                # 训练集回放：coverage（类内命中比例）与 precision（命中纯度）
                covered = pd.Series(True, index=X.index)
                for cond in conditions:
                    covered &= (X[cond['feature']] >= cond['min'])
                    covered &= (X[cond['feature']] <= cond['max'])
                n_fire = int(covered.sum())
                coverage = float(covered[mask].mean()) if mask.sum() else 0.0
                precision = (float((y[covered] == label_id).mean())
                             if n_fire else 0.0)
                if n_fire == 0:
                    continue          # 零覆盖死规则不部署
                if precision < self.min_confidence:
                    continue          # 命中纯度低于配置下限不部署

                rules.append({
                    'id': f'STAT_{len(rules)+1:04d}',
                    'name': f'{label_name}_统计规则',
                    'type': 'statistical',
                    'app': self._extract_app_name(label_name),
                    'behavior': self._extract_behavior_name(label_name),
                    'priority': 200,
                    'confidence': float(precision),
                    'train_coverage': round(coverage, 4),
                    'train_precision': round(precision, 4),
                    'conditions': conditions,
                    'action': {'result': label_name, 'confidence': float(precision), 'source': 'statistical'}
                })

        return rules

    # ==================== 辅助函数 ====================

    FEATURE_DESCRIPTIONS = {
        # 会话信息
        'duration': '流持续时间(秒)',
        'protocol_tcp': '是否TCP协议',
        'protocol_udp': '是否UDP协议',
        'src_port': '源端口号',
        'dst_port': '目的端口号',
        # TLS指纹
        'tls_version': 'TLS版本号',
        'tls_has_sni': '是否包含SNI(域名)',
        'tls_sni_length': 'SNI域名长度',
        'tls_num_cipher_suites': '支持的密码套件数量',
        'tls_num_extensions': 'TLS扩展数量',
        'tls_has_tls13_ciphers': '是否支持TLS1.3密码套件',
        'tls_ja3_hash_prefix': 'JA3指纹哈希前缀(数值化)',
        # 流统计
        'total_packets': '总包数',
        'total_bytes': '总字节数',
        'total_fwd_packets': '前向包数',
        'total_bwd_packets': '后向包数',
        'total_fwd_bytes': '前向字节数',
        'total_bwd_bytes': '后向字节数',
        'fwd_bwd_byte_ratio': '前向/后向字节比率',
        'fwd_bwd_packet_ratio': '前向/后向包比率',
        'down_up_byte_ratio': '下行/上行字节比率',
        'byte_rate': '字节速率(bytes/s)',
        'packet_rate': '包速率(pkts/s)',
        'fwd_byte_rate': '前向字节速率',
        'bwd_byte_rate': '后向字节速率',
        # 包大小
        'pkt_size_mean': '包大小均值',
        'pkt_size_std': '包大小标准差',
        'pkt_size_max': '包大小最大值',
        'pkt_size_min': '包大小最小值',
        'fwd_pkt_size_mean': '前向包大小均值',
        'bwd_pkt_size_std': '后向包大小标准差',
        'fwd_pkt_size_max': '前向包大小最大值',
        'bwd_pkt_size_mean': '后向包大小均值',
        # IAT
        'iat_mean': '包间到达时间均值',
        'iat_std': '包间到达时间标准差',
        'iat_max': '包间到达时间最大值',
        'fwd_iat_mean': '前向IAT均值',
        'bwd_iat_mean': '后向IAT均值',
        # 标志
        'syn_flag_count': 'SYN标志计数',
        'ack_flag_count': 'ACK标志计数',
        'psh_flag_count': 'PSH标志计数',
        'rst_flag_count': 'RST标志计数',
        'fin_flag_count': 'FIN标志计数',
        # 行为
        'num_bursts': '突发传输次数',
        'pkt_size_entropy': '包大小信息熵',
        'payload_byte_entropy': '载荷字节信息熵',
        'num_retransmissions': '重传包数',
        'retransmission_ratio': '重传比率',
        # 高级
        'size_iqr': '包大小四分位距',
        'size_cv': '包大小变异系数',
        'iat_spectral_entropy': 'IAT频谱熵',
        'iat_hurst_exponent': 'IAT Hurst指数',
    }

    def _feature_description(self, name: str) -> str:
        """返回特征的可读描述"""
        if name in self.FEATURE_DESCRIPTIONS:
            return self.FEATURE_DESCRIPTIONS[name]
        # 自动生成描述
        name_lower = name.lower()
        if 'fwd' in name_lower:
            return f'前向{name.replace("fwd_", "").replace("_", " ")}'
        elif 'bwd' in name_lower:
            return f'后向{name.replace("bwd_", "").replace("_", " ")}'
        elif 'flag' in name_lower:
            return f'TCP标志-{name.replace("_flag_count", "").upper()}计数'
        else:
            return name.replace('_', ' ')

    def _extract_app_name(self, label: str) -> str:
        parts = label.split('_')
        return parts[0] if parts else label

    def _extract_behavior_name(self, label: str) -> str:
        parts = label.split('_')
        return '_'.join(parts[1:]) if len(parts) > 1 else 'general'

    @staticmethod
    def _sanitize_for_yaml(obj):
        """递归将numpy类型转为Python原生类型"""
        if isinstance(obj, dict):
            return {str(k): OptimizedRuleGenerator._sanitize_for_yaml(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [OptimizedRuleGenerator._sanitize_for_yaml(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
