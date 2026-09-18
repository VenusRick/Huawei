"""
高级特征维度生成模块
基于基础特征，提取或生成更高维度的流量刻画信息。

技术来源：
- 多时间尺度结构模式：PacketPrint的H-BoW思想
- 特征交互与协同特征：FSNID的转移熵方法
- 统计分布特征：NTLFlowLyzer的KDE方法
- 时序依赖特征：FSNID的LSTM方法
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
from src.parser.session.session_manager import FlowSession, PacketInfo


class AdvancedFeatureExtractor:
    """高级特征提取器"""

    def __init__(self, ngram_size: int = 3, bow_vocab_size: int = 256,
                 num_quantiles: int = 10):
        self.ngram_size = ngram_size
        self.bow_vocab_size = bow_vocab_size
        self.num_quantiles = num_quantiles

    def extract_all(self, session: FlowSession) -> Dict[str, float]:
        """提取全部高级特征"""
        features = {}

        # 1. 多时间尺度结构模式
        features.update(self._extract_multi_scale_features(session))

        # 2. 方向序列模式
        features.update(self._extract_directional_patterns(session))

        # 3. 包大小分布特征
        features.update(self._extract_distribution_features(session))

        # 4. 时序复杂度特征
        features.update(self._extract_complexity_features(session))

        # 5. 交叉特征
        features.update(self._extract_cross_features(session))

        return features

    # ==================== 1. 多时间尺度结构模式 ====================

    def _extract_multi_scale_features(self, session: FlowSession) -> Dict[str, float]:
        """
        多时间尺度结构模式（借鉴PacketPrint H-BoW）
        在三个时间尺度上提取模式：包级(100ms)、突发级(1s)、行为级(5s)
        """
        features = {}
        packets = session.packets
        if len(packets) < 2:
            for prefix in ['packet_bow', 'burst_bow', 'behavior_bow']:
                for i in range(min(self.bow_vocab_size, 32)):
                    features[f'{prefix}_{i}'] = 0.0
            return features

        # 包级"词汇"：将(方向, 大小区间)编码为词汇
        packet_words = self._encode_packet_words(packets)

        # 突发级"词汇"：将包级词汇按1s窗聚合（保留窗口时间，避免长度错位）
        burst_items = self._aggregate_to_bursts(packet_words, packets, window_sec=1.0)

        # 行为级"词汇"：将突发级词汇按5s窗再聚合（基于burst窗口时间而非包索引）
        behavior_words = self._aggregate_bursts_to_behavior(burst_items, 1.0, 5.0)

        # 计算词袋向量（简化版：前32个词汇的频率）
        vocab_size = min(self.bow_vocab_size, 32)
        for prefix, words in [('packet_bow', packet_words),
                              ('burst_bow', [w for w, _ in burst_items]),
                              ('behavior_bow', behavior_words)]:
            bow = self._compute_bow_vector(words, vocab_size)
            for i, val in enumerate(bow):
                features[f'{prefix}_{i}'] = float(val)

        return features

    def _encode_packet_words(self, packets: List[PacketInfo]) -> List[int]:
        """将每个包编码为'词汇'：方向(2bit) + 大小区间(6bit)"""
        if not packets:
            return []

        sizes = [p.length for p in packets]
        max_size = max(sizes) if sizes else 1500

        # 分为4个大小区间
        bins = [0, max_size * 0.25, max_size * 0.5, max_size * 0.75, max_size + 1]

        words = []
        for p in packets:
            direction_bit = 0 if p.direction == 1 else 2
            size_bin = 0
            for i in range(len(bins) - 1):
                if p.length < bins[i + 1]:
                    size_bin = i
                    break
            word = direction_bit * 4 + size_bin  # 0-7
            words.append(word)

        return words

    def _aggregate_to_bursts(self, words: List[int], packets: List[PacketInfo],
                             window_sec: float) -> List[Tuple[int, int]]:
        """按时间窗聚合词汇，返回[(burst词汇, 窗口序号)]——窗口序号保留时间信息"""
        if not words or len(packets) < 2:
            return []
        start_time = packets[0].timestamp
        burst_items: List[Tuple[int, int]] = []
        current_window = None
        window_words: List[int] = []
        for word, pkt in zip(words, packets):
            window_idx = int((pkt.timestamp - start_time) / window_sec)
            if window_idx != current_window:
                if window_words:
                    burst_items.append((self._hash_words_to_word(window_words), current_window))
                window_words = []
                current_window = window_idx
            window_words.append(word)
        if window_words:
            burst_items.append((self._hash_words_to_word(window_words), current_window))
        return burst_items

    def _aggregate_bursts_to_behavior(self, burst_items: List[Tuple[int, int]],
                                      burst_window_sec: float,
                                      behavior_window_sec: float) -> List[int]:
        """将burst词汇按更大时间窗聚合为behavior词汇（时间基准=burst窗口起点）"""
        if not burst_items or burst_window_sec <= 0:
            return []
        scale = max(int(behavior_window_sec / burst_window_sec), 1)
        groups = {}
        for word, window_idx in burst_items:
            groups.setdefault(window_idx // scale, []).append(word)
        return [self._hash_words_to_word(ws) for _, ws in sorted(groups.items())]

    def _hash_words_to_word(self, words: List[int]) -> int:
        """将一组词汇哈希为一个词汇"""
        # 简单哈希：词袋签名
        counter = Counter(words)
        signature = 0
        for w, count in counter.items():
            signature = (signature * 31 + w * count) % self.bow_vocab_size
        return signature

    def _compute_bow_vector(self, words: List[int], vocab_size: int) -> np.ndarray:
        """计算词袋向量"""
        bow = np.zeros(vocab_size)
        if not words:
            return bow
        for w in words:
            idx = w % vocab_size
            bow[idx] += 1
        # 归一化
        total = bow.sum()
        if total > 0:
            bow /= total
        return bow

    # ==================== 2. 方向序列模式 ====================

    def _extract_directional_patterns(self, session: FlowSession) -> Dict[str, float]:
        """方向序列模式特征"""
        features = {}
        directions = [p.direction for p in session.packets]

        if len(directions) < 2:
            features['direction_changes'] = 0.0
            features['longest_same_dir_run'] = 0.0
            features['direction_change_rate'] = 0.0
            # n-gram特征（2bit 2-gram只有4种组合，与长流路径键集一致）
            for i in range(4):
                features[f'dir_2gram_{i}'] = 0.0
            return features

        # 方向变化次数
        changes = sum(1 for i in range(len(directions)-1)
                     if directions[i] != directions[i+1])
        features['direction_changes'] = float(changes)
        features['direction_change_rate'] = changes / max(len(directions) - 1, 1)

        # 最长同向连续包数
        max_run = 1
        current_run = 1
        for i in range(1, len(directions)):
            if directions[i] == directions[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        features['longest_same_dir_run'] = float(max_run)

        # 2-gram方向模式 (2bit → 4种: ++, +-, -+, --)
        dir_map = {1: 0, -1: 1}  # 前向=0, 后向=1
        gram2 = np.zeros(4)
        for i in range(len(directions) - 1):
            idx = dir_map.get(directions[i], 0) * 2 + dir_map.get(directions[i+1], 0)
            gram2[idx] += 1
        total_grams = gram2.sum()
        if total_grams > 0:
            gram2 /= total_grams
        for i in range(4):
            features[f'dir_2gram_{i}'] = float(gram2[i])

        return features

    # ==================== 3. 包大小分布特征 ====================

    def _extract_distribution_features(self, session: FlowSession) -> Dict[str, float]:
        """包大小分布特征"""
        features = {}
        sizes = np.array([p.length for p in session.packets]) if session.packets else np.array([])

        if len(sizes) == 0:
            for i in range(self.num_quantiles):
                features[f'size_quantile_{i}'] = 0.0
            features['size_iqr'] = 0.0
            features['size_range'] = 0.0
            features['size_cv'] = 0.0
            return features

        # 分位数
        quantiles = np.percentile(sizes,
                                   np.linspace(0, 100, self.num_quantiles + 2)[1:-1])
        for i, q in enumerate(quantiles):
            features[f'size_quantile_{i}'] = float(q)

        # IQR
        q75 = np.percentile(sizes, 75)
        q25 = np.percentile(sizes, 25)
        features['size_iqr'] = float(q75 - q25)
        features['size_range'] = float(sizes.max() - sizes.min())

        # 变异系数
        mean = np.mean(sizes)
        if mean > 0:
            features['size_cv'] = float(np.std(sizes) / mean)
        else:
            features['size_cv'] = 0.0

        return features

    # ==================== 4. 时序复杂度特征 ====================

    def _extract_complexity_features(self, session: FlowSession) -> Dict[str, float]:
        """时序复杂度特征"""
        features = {}

        if len(session.packets) < 3:
            features['iat_spectral_entropy'] = 0.0
            features['iat_approximate_entropy'] = 0.0
            features['iat_hurst_exponent'] = 0.5
            return features

        timestamps = [p.timestamp for p in session.packets]
        iats = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

        iat_arr = np.array(iats)

        # IAT频谱熵
        if len(iat_arr) > 4:
            fft_vals = np.abs(np.fft.fft(iat_arr - np.mean(iat_arr)))
            fft_power = fft_vals ** 2
            total_power = fft_power.sum()
            if total_power > 0:
                probs = fft_power / total_power
                probs = probs[probs > 0]
                features['iat_spectral_entropy'] = float(-np.sum(probs * np.log2(probs)))
            else:
                features['iat_spectral_entropy'] = 0.0
        else:
            features['iat_spectral_entropy'] = 0.0

        # 简化近似熵
        features['iat_approximate_entropy'] = float(self._approximate_entropy(iat_arr))

        # Hurst指数（简化R/S方法）
        features['iat_hurst_exponent'] = float(self._hurst_exponent(iat_arr))

        return features

    # ==================== 5. 交叉特征 ====================

    def _extract_cross_features(self, session: FlowSession) -> Dict[str, float]:
        """特征交叉（组合不同维度的特征）"""
        features = {}

        # 基于会话数据计算交叉特征
        fwd_sizes = [p.length for p in session.packets if p.direction == 1]
        bwd_sizes = [p.length for p in session.packets if p.direction == -1]
        timestamps = [p.timestamp for p in session.packets]

        # 平均包大小 × 平均IAT
        if len(timestamps) >= 2:
            iats = [timestamps[i+1] - timestamps[i]
                    for i in range(len(timestamps)-1)]
            avg_iat = np.mean(iats)
            avg_size = np.mean([p.length for p in session.packets])
            features['avg_size_x_avg_iat'] = float(avg_size * avg_iat)
        else:
            features['avg_size_x_avg_iat'] = 0.0

        # 上下行比率 × 持续时间
        if session.total_fwd_bytes > 0 and session.duration > 0:
            features['updown_ratio_x_duration'] = float(
                (session.total_bwd_bytes / session.total_fwd_bytes) * session.duration)
        else:
            features['updown_ratio_x_duration'] = 0.0

        # 包速率 × 平均包大小（近似吞吐量特征）
        if session.duration > 0:
            features['pkt_rate_x_avg_size'] = float(
                (session.total_packets / session.duration) *
                (session.total_bytes / max(session.total_packets, 1)))
        else:
            features['pkt_rate_x_avg_size'] = 0.0

        # 前向突发度 × 后向突发度
        fwd_burst = np.std(fwd_sizes) if len(fwd_sizes) > 1 else 0
        bwd_burst = np.std(bwd_sizes) if len(bwd_sizes) > 1 else 0
        features['fwd_burst_x_bwd_burst'] = float(fwd_burst * bwd_burst)

        return features

    # ==================== 辅助函数 ====================

    def _approximate_entropy(self, arr: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """计算简化近似熵"""
        if len(arr) < m + 1:
            return 0.0

        std = np.std(arr)
        if std == 0:
            return 0.0
        r_scaled = r * std

        def count_matches(m_val):
            count = 0
            for i in range(len(arr) - m_val + 1):
                for j in range(i + 1, len(arr) - m_val + 1):
                    if np.max(np.abs(arr[i:i+m_val] - arr[j:j+m_val])) <= r_scaled:
                        count += 1
            return count

        cm = count_matches(m)
        cm1 = count_matches(m + 1)

        n = len(arr) - m + 1
        if n <= 0 or cm == 0 or cm1 == 0:
            return 0.0

        phi_m = cm / (n * (n - 1) / 2) if n > 1 else 0
        phi_m1 = cm1 / ((n - 1) * (n - 2) / 2) if n > 2 else 0

        if phi_m > 0 and phi_m1 > 0:
            return np.log(phi_m) - np.log(phi_m1)
        return 0.0

    def _hurst_exponent(self, arr: np.ndarray) -> float:
        """计算Hurst指数（简化R/S方法）"""
        if len(arr) < 10:
            return 0.5

        max_lag = min(len(arr) // 2, 50)
        if max_lag < 4:
            return 0.5

        lags = range(2, max_lag)
        tau = []
        for lag in lags:
            # 简化：使用自相关延迟
            if len(arr) > lag:
                diff = arr[lag:] - arr[:-lag]
                tau.append(np.std(diff) if len(diff) > 0 else 0)
            else:
                tau.append(0)

        lags_arr = np.array(list(lags), dtype=np.float64)
        tau_arr = np.array(tau)

        # 过滤无效值
        valid = (tau_arr > 0) & (lags_arr > 0)
        if valid.sum() < 2:
            return 0.5

        try:
            log_lags = np.log(lags_arr[valid])
            log_tau = np.log(tau_arr[valid])
            # 线性回归
            slope = np.polyfit(log_lags, log_tau, 1)[0]
            return max(0.0, min(1.0, slope))
        except:
            return 0.5

    def get_feature_catalog(self) -> Dict[str, Dict]:
        """
        返回高级特征的完整目录，按类别分组。
        格式: {feature_name: {category, subcategory, description}}
        """
        catalog = {}

        # 1. 多时间尺度词袋 (96维)
        for level in ['packet', 'burst', 'behavior']:
            for i in range(32):
                name = f'{level}_bow_{i}'
                catalog[name] = {
                    'category': 'advanced',
                    'subcategory': 'multi_scale_bow',
                    'description': f'{level}级词袋第{i}维频率',
                }

        # 2. 方向序列模式 (7维)
        for name, desc in [
            ('direction_changes', '方向变化次数'),
            ('longest_same_dir_run', '最长同向连续包数'),
            ('direction_change_rate', '方向变化率'),
            ('dir_2gram_0', '2-gram: 前→前 频率'),
            ('dir_2gram_1', '2-gram: 前→后 频率'),
            ('dir_2gram_2', '2-gram: 后→前 频率'),
            ('dir_2gram_3', '2-gram: 后→后 频率'),
        ]:
            catalog[name] = {'category': 'advanced', 'subcategory': 'direction_patterns', 'description': desc}

        # 3. 分布特征 (12维)
        for i in range(10):
            name = f'size_quantile_{i}'
            catalog[name] = {'category': 'advanced', 'subcategory': 'distribution',
                           'description': f'包大小第{(i+1)*10}%分位数'}
        for name, desc in [
            ('size_iqr', '包大小四分位距(Q3-Q1)'),
            ('size_range', '包大小极差(max-min)'),
            ('size_cv', '包大小变异系数(std/mean)'),
        ]:
            catalog[name] = {'category': 'advanced', 'subcategory': 'distribution', 'description': desc}

        # 4. 复杂度特征 (3维)
        for name, desc in [
            ('iat_spectral_entropy', 'IAT序列频谱熵'),
            ('iat_approximate_entropy', 'IAT序列近似熵'),
            ('iat_hurst_exponent', 'IAT序列Hurst指数'),
        ]:
            catalog[name] = {'category': 'advanced', 'subcategory': 'complexity', 'description': desc}

        # 5. 交叉特征 (4维)
        for name, desc in [
            ('avg_size_x_avg_iat', '平均包大小×平均IAT'),
            ('updown_ratio_x_duration', '上下行比率×持续时间'),
            ('pkt_rate_x_avg_size', '包速率×平均包大小(近似吞吐量)'),
            ('fwd_burst_x_bwd_burst', '前向突发度×后向突发度'),
        ]:
            catalog[name] = {'category': 'advanced', 'subcategory': 'cross', 'description': desc}

        return catalog
