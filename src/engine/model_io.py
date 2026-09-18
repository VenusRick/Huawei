"""
模型序列化/反序列化模块
支持XGBoost原生JSON格式（跨版本兼容）替代pickle。

独立推理所需的文件：
- model.json           ← XGBoost原生格式
- selected_features.json ← 选中特征列表
- engine_config.json   ← 引擎配置（置信度阈值、类别映射等）
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def save_model_bundle(clf, selected_features: List[str],
                      label_names: Dict[int, str],
                      confidence_threshold: float,
                      output_dir: str,
                      soft_match_threshold: float = 0.75) -> Dict[str, str]:
    """
    导出推理所需的完整模型包。

    Args:
        clf: 训练好的XGBoost分类器
        selected_features: 选中的特征名列表
        label_names: {类别ID: 类别名} 映射
        confidence_threshold: 置信度阈值
        output_dir: 输出目录
        soft_match_threshold: 软匹配阈值

    Returns:
        导出的文件路径字典
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. XGBoost原生JSON模型
    model_path = output_path / 'model.json'
    clf.save_model(str(model_path))
    paths['model'] = str(model_path)

    # 2. 选中特征列表
    feat_path = output_path / 'selected_features.json'
    with open(feat_path, 'w', encoding='utf-8') as f:
        json.dump(selected_features, f, indent=2, ensure_ascii=False)
    paths['features'] = str(feat_path)

    # 3. 引擎配置
    config = {
        'confidence_threshold': float(confidence_threshold),
        'soft_match_threshold': float(soft_match_threshold),
        'label_names': {int(k): str(v) for k, v in label_names.items()},
        'num_classes': len(label_names),
        'num_features': len(selected_features),
        'model_format': 'xgboost_json',
    }
    config_path = output_path / 'engine_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    paths['config'] = str(config_path)

    return paths


def load_model_bundle(model_dir: str) -> Tuple:
    """
    加载模型包，返回引擎初始化所需的全部数据。

    Args:
        model_dir: 模型目录路径

    Returns:
        (classifier, selected_features, label_names, config)

    Raises:
        FileNotFoundError: 缺少必要文件
    """
    from xgboost import XGBClassifier

    model_path = Path(model_dir)

    # 检查必要文件
    model_file = model_path / 'model.json'
    feat_file = model_path / 'selected_features.json'
    config_file = model_path / 'engine_config.json'

    for f in [model_file, feat_file, config_file]:
        if not f.exists():
            raise FileNotFoundError(f"模型包缺少文件: {f}")

    # 加载XGBoost模型
    clf = XGBClassifier()
    clf.load_model(str(model_file))

    # 加载特征列表
    with open(feat_file, 'r', encoding='utf-8') as f:
        selected_features = json.load(f)

    # 加载配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 解析类别映射
    label_names = {int(k): v for k, v in config.get('label_names', {}).items()}

    return clf, selected_features, label_names, config

def save_rule_bundle(rules: list, selected_features: List[str],
                     label_names: Dict, confidence_threshold: float,
                     output_dir: str, bundle_meta: Optional[Dict] = None) -> Dict[str, str]:
    """导出规则包（独立规则推理用；与旧模型包分开，不含任何预测器）。
    文件：rules.json / selected_features.json / bundle_config.json
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    rules_path = output_path / 'rules.json'
    with open(rules_path, 'w', encoding='utf-8') as f:
        json.dump(rules, f, indent=2, ensure_ascii=False, default=float)
    paths['rules'] = str(rules_path)
    feat_path = output_path / 'selected_features.json'
    with open(feat_path, 'w', encoding='utf-8') as f:
        json.dump(selected_features, f, indent=2, ensure_ascii=False)
    paths['features'] = str(feat_path)
    config = {
        'bundle_type': 'rules',
        'confidence_threshold': float(confidence_threshold),
        'label_names': {str(k): str(v) for k, v in label_names.items()},
        'num_rules': len(rules),
        'num_features': len(selected_features),
    }
    if bundle_meta:
        config.update(bundle_meta)
    config_path = output_path / 'bundle_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    paths['config'] = str(config_path)
    return paths


def load_rule_bundle(bundle_dir: str) -> Tuple:
    """加载规则包，返回 (rules, selected_features, label_names, config)。
    不导入任何训练/预测库（xgboost/sklearn）。"""
    p = Path(bundle_dir)
    files = {
        'rules': p / 'rules.json',
        'features': p / 'selected_features.json',
        'config': p / 'bundle_config.json',
    }
    for key, f in files.items():
        if not f.exists():
            raise FileNotFoundError(f"规则包缺少文件: {f}")
    with open(files['rules'], 'r', encoding='utf-8') as f:
        rules = json.load(f)
    with open(files['features'], 'r', encoding='utf-8') as f:
        selected_features = json.load(f)
    with open(files['config'], 'r', encoding='utf-8') as f:
        config = json.load(f)
    label_names = config.get('label_names', {})
    return rules, selected_features, label_names, config
