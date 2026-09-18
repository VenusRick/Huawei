"""
全局配置文件 - 智能化加密流量特征挖掘工具
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import os
import yaml

# 训练侧线程上限：XGBoost/sklearn 默认吃满全部核，在多核共享机上造成
# OpenMP 超额订阅（64核曾见193线程互踩、wall-time 数倍劣化）。
# 数据规模（千级会话）下 8 线程已够；TRAIN_THREADS 环境变量可覆盖。
TRAIN_THREADS = max(1, int(os.environ.get('TRAIN_THREADS', min(8, os.cpu_count() or 1))))


@dataclass
class ParserConfig:
    """流量解析配置"""
    max_packet_size: int = 65535
    tcp_timeout: int = 300        # TCP会话超时(秒)
    udp_timeout: int = 60         # UDP会话超时(秒)
    tcp_establish_timeout: int = 5 # TCP建连超时(秒)
    max_sessions: int = 1000000   # 最大并发会话数
    handle_retransmission: bool = True
    handle_out_of_order: bool = True


@dataclass
class FeatureConfig:
    """特征提取配置"""
    # 时间窗口参数 (参考PacketPrint)
    burst_window_sec: float = 1.0       # 突发级窗口
    behavior_window_sec: float = 5.0    # 行为级窗口
    sliding_step_sec: float = 1.0       # 滑动步长

    # 流统计
    max_flow_duration: float = 600.0    # 最大流持续时间(秒)
    payload_byte_count: int = 8         # 前N字节载荷分布

    # IAT统计
    iat_features_enabled: bool = True

    # 高级特征
    bow_vocabulary_size: int = 256       # 词袋词汇量
    ngram_size: int = 3                  # n-gram大小
    entropy_window_count: int = 10       # 熵窗口数


@dataclass
class SelectionConfig:
    """特征选择配置"""
    correlation_threshold: float = 0.7   # 相关性阈值
    common_feature_ratio: float = 0.3    # 跨活动通用特征比例阈值
    phi_significance: float = 0.95       # FSNID统计显著性水平
    max_features: int = 30               # 最大特征数
    min_features: int = 4                # 最小特征数


@dataclass
class RuleConfig:
    """规则生成配置"""
    min_support: float = 0.1             # FP-Growth最小支持度
    min_confidence: float = 0.4          # 最小置信度
    confidence_threshold: float = 0.95   # 规则置信度阈值
    output_format: str = "yaml"          # 规则输出格式


@dataclass
class DPIConfig:
    """DPI引擎配置"""
    max_rules: int = 10000
    cache_size: int = 100000
    batch_size: int = 1000
    worker_threads: int = 4


@dataclass
class AppConfig:
    """应用全局配置"""
    parser: ParserConfig = field(default_factory=ParserConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    rule: RuleConfig = field(default_factory=RuleConfig)
    dpi: DPIConfig = field(default_factory=DPIConfig)

    # 路径配置（自动定位项目根目录，不依赖绝对路径）
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    output_dir: Path = Path(__file__).resolve().parent.parent / "output"

    # 识别场景
    scenarios: List[str] = field(default_factory=lambda: [
        "social_post",         # 社交应用发帖
        "im_voice_call",       # IM语音通话
        "im_video_call",       # IM视频通话
        "im_text_message",     # IM文字消息
        "anonymous_tor",       # Tor匿名工具
        "anonymous_psiphon",   # Psiphon匿名工具
        "anonymous_session",   # Session匿名工具
    ])

    @classmethod
    def from_yaml(cls, path: str) -> 'AppConfig':
        """从YAML加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        cfg = cls()
        if data:
            for section, values in data.items():
                if hasattr(cfg, section) and isinstance(values, dict):
                    sub = getattr(cfg, section)
                    for k, v in values.items():
                        if hasattr(sub, k):
                            setattr(sub, k, v)
        return cfg

    def to_yaml(self, path: str):
        """保存配置到YAML"""
        from dataclasses import asdict
        d = asdict(self)
        # R26: 路径字段为运行时派生值，序列化为纯字符串；from_yaml按类型过滤不恢复，保持自动定位语义
        for k in ('project_root', 'data_dir', 'output_dir'):
            if k in d:
                d[k] = str(d[k])
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(d, f, allow_unicode=True, default_flow_style=False)


# 全局默认配置
DEFAULT_CONFIG = AppConfig()
