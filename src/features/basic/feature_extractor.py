"""
基础特征维度提取模块
从会话(FlowSession)中提取6大类特征，共80-120维。

特征体系（参考NTLFlowLyzer 114维 + PacketPrint + FSNID）：
1. 协议指纹特征（Protocol Fingerprint）    — TLS/QUIC握手元数据
2. 流统计特征（Flow Statistics）           — 包数、字节数、方向性统计
3. 时序行为特征（Timing Behavior）         — IAT、活动/空闲、突发检测
4. TCP标志特征（Flag-based）               — TCP标志计数
5. 协议结构特征（Protocol Structure）      — 头部字节、段大小
6. 连接行为特征（Connection Behavior）     — 连接模式、批量传输
"""
import numpy as np
from typing import List, Dict, Optional
from collections import Counter
from src.parser.session.session_manager import FlowSession, PacketInfo, Protocol


class BasicFeatureExtractor:
    """基础特征提取器"""

    def extract_all(self, session: FlowSession) -> Dict[str, float]:
        """提取全部基础特征"""
        features = {}

        # 会话基本信息
        features.update(self._extract_session_info(session))

        # 1. 协议指纹特征
        features.update(self._extract_protocol_features(session))

        # 2. 流统计特征
        features.update(self._extract_flow_stats(session))

        # 3. 时序行为特征
        features.update(self._extract_timing_features(session))

        # 4. TCP标志特征
        features.update(self._extract_flag_features(session))

        # 5. 协议结构特征
        features.update(self._extract_structure_features(session))

        # 6. 连接行为特征
        features.update(self._extract_behavior_features(session))

        return features

    # ==================== 会话基本信息 ====================

    def _extract_session_info(self, session: FlowSession) -> Dict[str, float]:
        """会话基本信息"""
        return {
            'duration': session.duration,
            'protocol_tcp': float(session.protocol == Protocol.TCP),
            'protocol_udp': float(session.protocol == Protocol.UDP),
            'src_port': float(session.src_port),
            'dst_port': float(session.dst_port),
            'is_well_known_src_port': float(session.src_port < 1024),
            'is_well_known_dst_port': float(session.dst_port < 1024),
        }

    # ==================== 1. 协议指纹特征 ====================

    def _extract_protocol_features(self, session: FlowSession) -> Dict[str, float]:
        """协议指纹特征"""
        features = {}

        # TLS信息（从第一个包中提取）
        tls_info = None
        for pkt in session.packets:
            if pkt.tls_info:
                tls_info = pkt.tls_info
                break

        if tls_info:
            features['tls_version'] = float(tls_info.get('tls_version', 0))
            features['tls_has_sni'] = float(bool(tls_info.get('ch_sni', '')))
            features['tls_sni_length'] = float(tls_info.get('ch_sni_length', 0))
            features['tls_num_cipher_suites'] = float(
                tls_info.get('ch_num_cipher_suites', 0))
            features['tls_num_extensions'] = float(
                tls_info.get('ch_num_extensions', 0))
            features['tls_num_supported_versions'] = float(
                tls_info.get('ch_num_supported_versions', 0))
            features['tls_num_supported_groups'] = float(
                tls_info.get('ch_num_supported_groups', 0))
            features['tls_num_signature_algorithms'] = float(
                tls_info.get('ch_num_signature_algorithms', 0))
            features['tls_has_h2'] = float(tls_info.get('has_h2', 0))
            features['tls_has_http11'] = float(tls_info.get('has_http11', 0))
            features['tls_has_tls13_ciphers'] = float(
                tls_info.get('has_tls13_ciphers', 0))
            features['tls_num_certificates'] = float(
                tls_info.get('num_certificates', 0))
            features['tls_handshake_bytes'] = float(
                tls_info.get('total_handshake_bytes', 0))
            features['tls_has_encrypted_extensions'] = float(
                tls_info.get('has_encrypted_extensions', 0))

            # JA3/JA4哈希作为数值特征（取前8位hex转int）
            ja3 = tls_info.get('ja3_hash', '')
            features['tls_ja3_hash_prefix'] = float(int(ja3[:8], 16)) / 0xFFFFFFFF if len(ja3) >= 8 else 0.0
            ja4 = tls_info.get('ja4_hash', '')
            features['tls_ja4_hash_prefix'] = float(int(ja4[:8], 16)) / 0xFFFFFFFF if len(ja4) >= 8 else 0.0

            # QUIC特征
            quic_version = tls_info.get('quic_version', 0)
            features['quic_is_present'] = float(quic_version != 0)
            features['quic_version'] = float(quic_version)
            features['quic_is_v1'] = float(tls_info.get('quic_is_v1', 0))
            features['quic_has_token'] = float(tls_info.get('quic_has_token', 0))
            features['quic_token_length'] = float(tls_info.get('quic_token_length', 0))
        else:
            # 无TLS/QUIC信息的默认值
            for key in ['tls_version', 'tls_has_sni', 'tls_sni_length',
                       'tls_num_cipher_suites', 'tls_num_extensions',
                       'tls_num_supported_versions', 'tls_num_supported_groups',
                       'tls_num_signature_algorithms', 'tls_has_h2',
                       'tls_has_http11', 'tls_has_tls13_ciphers',
                       'tls_num_certificates', 'tls_handshake_bytes',
                       'tls_has_encrypted_extensions', 'tls_ja3_hash_prefix',
                       'tls_ja4_hash_prefix', 'quic_is_present',
                       'quic_version', 'quic_is_v1', 'quic_has_token',
                       'quic_token_length']:
                features[key] = 0.0

        return features

    # ==================== 2. 流统计特征 ====================

    def _extract_flow_stats(self, session: FlowSession) -> Dict[str, float]:
        """流统计特征（参考NTLFlowLyzer F25-F34）"""
        features = {}

        # 总体统计
        features['total_fwd_packets'] = float(session.total_fwd_packets)
        features['total_bwd_packets'] = float(session.total_bwd_packets)
        features['total_fwd_bytes'] = float(session.total_fwd_bytes)
        features['total_bwd_bytes'] = float(session.total_bwd_bytes)
        features['total_packets'] = float(session.total_packets)
        features['total_bytes'] = float(session.total_bytes)

        # 比率特征
        if session.total_packets > 0:
            features['fwd_bwd_packet_ratio'] = (
                session.total_fwd_packets / max(session.total_bwd_packets, 1))
            features['fwd_packet_rate'] = (
                session.total_fwd_packets / max(session.duration, 0.001))
            features['bwd_packet_rate'] = (
                session.total_bwd_packets / max(session.duration, 0.001))
            features['packet_rate'] = (
                session.total_packets / max(session.duration, 0.001))
        else:
            features['fwd_bwd_packet_ratio'] = 0.0
            features['fwd_packet_rate'] = 0.0
            features['bwd_packet_rate'] = 0.0
            features['packet_rate'] = 0.0

        if session.total_bytes > 0:
            features['fwd_bwd_byte_ratio'] = (
                session.total_fwd_bytes / max(session.total_bwd_bytes, 1))
            features['byte_rate'] = (
                session.total_bytes / max(session.duration, 0.001))
            features['fwd_byte_rate'] = (
                session.total_fwd_bytes / max(session.duration, 0.001))
            features['bwd_byte_rate'] = (
                session.total_bwd_bytes / max(session.duration, 0.001))
            features['down_up_byte_ratio'] = (
                session.total_bwd_bytes / max(session.total_fwd_bytes, 1))
        else:
            features['fwd_bwd_byte_ratio'] = 0.0
            features['byte_rate'] = 0.0
            features['fwd_byte_rate'] = 0.0
            features['bwd_byte_rate'] = 0.0
            features['down_up_byte_ratio'] = 0.0

        # 包大小统计
        fwd_sizes = [p.length for p in session.packets if p.direction == 1]
        bwd_sizes = [p.length for p in session.packets if p.direction == -1]
        all_sizes = [p.length for p in session.packets]

        features.update(self._array_stats(fwd_sizes, 'fwd_pkt_size'))
        features.update(self._array_stats(bwd_sizes, 'bwd_pkt_size'))
        features.update(self._array_stats(all_sizes, 'pkt_size'))

        # 载荷大小统计
        fwd_payload_sizes = [p.payload_length for p in session.packets if p.direction == 1]
        bwd_payload_sizes = [p.payload_length for p in session.packets if p.direction == -1]
        all_payload_sizes = [p.payload_length for p in session.packets]

        features.update(self._array_stats(fwd_payload_sizes, 'fwd_payload_size'))
        features.update(self._array_stats(bwd_payload_sizes, 'bwd_payload_size'))
        features.update(self._array_stats(all_payload_sizes, 'payload_size'))

        return features

    # ==================== 3. 时序行为特征 ====================

    def _extract_timing_features(self, session: FlowSession) -> Dict[str, float]:
        """时序行为特征（参考NTLFlowLyzer F1-F24 + PacketPrint时间尺度）"""
        features = {}

        # R16: 不再提前返回——各统计段对空序列已有零值兜底，保证特征键集与包数无关
        if len(session.packets) < 2:
            for prefix in ['iat', 'fwd_iat', 'bwd_iat', 'active', 'idle']:
                features.update(self._empty_stats(prefix))

        timestamps = [p.timestamp for p in session.packets]
        fwd_ts = [p.timestamp for p in session.packets if p.direction == 1]
        bwd_ts = [p.timestamp for p in session.packets if p.direction == -1]

        # 包间到达时间(IAT)
        iats = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        fwd_iats = [fwd_ts[i+1] - fwd_ts[i] for i in range(len(fwd_ts)-1)] if len(fwd_ts) > 1 else []
        bwd_iats = [bwd_ts[i+1] - bwd_ts[i] for i in range(len(bwd_ts)-1)] if len(bwd_ts) > 1 else []

        features.update(self._array_stats(iats, 'iat'))
        features.update(self._array_stats(fwd_iats, 'fwd_iat'))
        features.update(self._array_stats(bwd_iats, 'bwd_iat'))

        # 活动/空闲时间段
        active_periods, idle_periods = self._compute_active_idle(timestamps, iats)
        features.update(self._array_stats(active_periods, 'active'))
        features.update(self._array_stats(idle_periods, 'idle'))

        # 突发(Burst)特征
        burst_durations, burst_packet_counts = self._detect_bursts(session.packets)
        features.update(self._array_stats(burst_durations, 'burst_duration'))
        features.update(self._array_stats(burst_packet_counts, 'burst_pkt_count'))
        features['num_bursts'] = float(len(burst_durations))

        # 前N秒包数时间序列（参考PacketPrint的多时间尺度）
        for window_sec in [0.1, 0.5, 1.0, 3.0, 5.0]:
            start_ts = timestamps[0]
            count = sum(1 for t in timestamps if t - start_ts <= window_sec)
            features[f'pkts_first_{window_sec}s'] = float(count)

        return features

    # ==================== 4. TCP标志特征 ====================

    def _extract_flag_features(self, session: FlowSession) -> Dict[str, float]:
        """TCP标志特征（参考NTLFlowLyzer F71-F94）"""
        features = {}

        flag_names = ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'ECE', 'CWR']
        flag_masks = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]

        for name, mask in zip(flag_names, flag_masks):
            # 总计
            count = sum(1 for p in session.packets if p.tcp_flags & mask)
            features[f'{name.lower()}_flag_count'] = float(count)
            # 前向
            fwd_count = sum(1 for p in session.packets
                          if p.direction == 1 and p.tcp_flags & mask)
            features[f'fwd_{name.lower()}_flag_count'] = float(fwd_count)
            # 后向
            bwd_count = sum(1 for p in session.packets
                          if p.direction == -1 and p.tcp_flags & mask)
            features[f'bwd_{name.lower()}_flag_count'] = float(bwd_count)

        # SYN-ACK往返时间
        syn_time = None
        syn_ack_time = None
        for p in session.packets:
            if p.tcp_flags & 0x02 and not (p.tcp_flags & 0x10):  # SYN
                syn_time = p.timestamp
            elif p.tcp_flags & 0x02 and p.tcp_flags & 0x10:  # SYN-ACK
                syn_ack_time = p.timestamp
                break

        if syn_time is not None and syn_ack_time is not None:
            features['syn_ack_rtt'] = syn_ack_time - syn_time
        else:
            features['syn_ack_rtt'] = 0.0

        # 重传统计
        features['num_retransmissions'] = float(session.num_retransmissions)
        features['retransmission_ratio'] = (
            session.num_retransmissions / max(session.total_packets, 1))

        # 窗口大小统计
        windows = [p.tcp_window for p in session.packets if p.tcp_window > 0]
        features.update(self._array_stats(windows, 'tcp_window'))

        return features

    # ==================== 5. 协议结构特征 ====================

    def _extract_structure_features(self, session: FlowSession) -> Dict[str, float]:
        """协议结构特征（参考NTLFlowLyzer F35-F52）"""
        features = {}

        # 头部字节估算
        # TCP头部: 20-60字节, IP头部: 20-60字节
        # 以太网: 14字节
        header_sizes = []
        fwd_header_sizes = []
        bwd_header_sizes = []

        for p in session.packets:
            # 简单估算：总长度 - 载荷长度
            hdr = max(p.length - p.payload_length, 0)
            header_sizes.append(hdr)
            if p.direction == 1:
                fwd_header_sizes.append(hdr)
            else:
                bwd_header_sizes.append(hdr)

        features.update(self._array_stats(header_sizes, 'header_bytes'))
        features.update(self._array_stats(fwd_header_sizes, 'fwd_header_bytes'))
        features.update(self._array_stats(bwd_header_sizes, 'bwd_header_bytes'))

        # 平均段大小
        if session.total_packets > 0:
            features['avg_segment_size'] = (
                session.total_bytes / session.total_packets)
        else:
            features['avg_segment_size'] = 0.0

        if session.total_fwd_packets > 0:
            features['fwd_avg_segment_size'] = (
                session.total_fwd_bytes / session.total_fwd_packets)
        else:
            features['fwd_avg_segment_size'] = 0.0

        if session.total_bwd_packets > 0:
            features['bwd_avg_segment_size'] = (
                session.total_bwd_bytes / session.total_bwd_packets)
        else:
            features['bwd_avg_segment_size'] = 0.0

        # 载荷占比
        total_payload = sum(p.payload_length for p in session.packets)
        if session.total_bytes > 0:
            features['payload_ratio'] = total_payload / session.total_bytes
        else:
            features['payload_ratio'] = 0.0

        return features

    # ==================== 6. 连接行为特征 ====================

    def _extract_behavior_features(self, session: FlowSession) -> Dict[str, float]:
        """连接行为特征（参考NTLFlowLyzer F95-F114）"""
        features = {}

        # 批量传输检测
        bulk_threshold = 4  # 连续4个同向包视为批量
        fwd_bulk_count, fwd_bulk_bytes, fwd_bulk_pkt_count = \
            self._detect_bulk_transfers(session.packets, direction=1, threshold=bulk_threshold)
        bwd_bulk_count, bwd_bulk_bytes, bwd_bulk_pkt_count = \
            self._detect_bulk_transfers(session.packets, direction=-1, threshold=bulk_threshold)

        features['fwd_bulk_count'] = float(fwd_bulk_count)
        features['fwd_bulk_total_bytes'] = float(fwd_bulk_bytes)
        features['fwd_bulk_total_packets'] = float(fwd_bulk_pkt_count)
        features['bwd_bulk_count'] = float(bwd_bulk_count)
        features['bwd_bulk_total_bytes'] = float(bwd_bulk_bytes)
        features['bwd_bulk_total_packets'] = float(bwd_bulk_pkt_count)

        # 批量统计
        if fwd_bulk_count > 0:
            features['fwd_avg_bulk_size'] = fwd_bulk_bytes / fwd_bulk_count
            features['fwd_avg_bulk_rate'] = (
                fwd_bulk_bytes / max(session.duration, 0.001))
        else:
            features['fwd_avg_bulk_size'] = 0.0
            features['fwd_avg_bulk_rate'] = 0.0

        if bwd_bulk_count > 0:
            features['bwd_avg_bulk_size'] = bwd_bulk_bytes / bwd_bulk_count
            features['bwd_avg_bulk_rate'] = (
                bwd_bulk_bytes / max(session.duration, 0.001))
        else:
            features['bwd_avg_bulk_size'] = 0.0
            features['bwd_avg_bulk_rate'] = 0.0

        # 包大小分布特征
        all_sizes = [p.length for p in session.packets]
        if all_sizes:
            arr = np.array(all_sizes)
            features['pkt_size_entropy'] = float(self._compute_entropy(arr))
            features['pkt_size_skewness'] = float(self._compute_skewness(arr))
            features['pkt_size_kurtosis'] = float(self._compute_kurtosis(arr))
        else:
            features['pkt_size_entropy'] = 0.0
            features['pkt_size_skewness'] = 0.0
            features['pkt_size_kurtosis'] = 0.0

        # 载荷字节分布熵（前128字节）
        byte_counts = np.zeros(256)
        for p in session.packets[:100]:
            for b in p.payload[:128]:
                byte_counts[b] += 1
        if byte_counts.sum() > 0:
            # byte_counts已是频数分布：直接按频数算熵（均匀256字节=8bit）
            probs = byte_counts[byte_counts > 0] / byte_counts.sum()
            features['payload_byte_entropy'] = float(-np.sum(probs * np.log2(probs)))
        else:
            features['payload_byte_entropy'] = 0.0

        return features

    # ==================== 工具函数 ====================

    def _array_stats(self, values: list, prefix: str) -> Dict[str, float]:
        """计算数组的统计特征"""
        if not values:
            return self._empty_stats(prefix)

        arr = np.array(values, dtype=np.float64)
        return {
            f'{prefix}_max': float(np.max(arr)),
            f'{prefix}_min': float(np.min(arr)),
            f'{prefix}_mean': float(np.mean(arr)),
            f'{prefix}_std': float(np.std(arr)),
            f'{prefix}_sum': float(np.sum(arr)),
            f'{prefix}_median': float(np.median(arr)),
        }

    def _empty_stats(self, prefix: str) -> Dict[str, float]:
        """返回空统计特征"""
        return {
            f'{prefix}_max': 0.0,
            f'{prefix}_min': 0.0,
            f'{prefix}_mean': 0.0,
            f'{prefix}_std': 0.0,
            f'{prefix}_sum': 0.0,
            f'{prefix}_median': 0.0,
        }

    def _compute_active_idle(self, timestamps: list, iats: list,
                             idle_threshold: float = 1.0):
        """计算活动/空闲时间段"""
        active_periods = []
        idle_periods = []
        current_start = timestamps[0]
        current_is_idle = False

        for i, iat in enumerate(iats):
            if iat > idle_threshold:
                # 空闲
                if not current_is_idle:
                    active_periods.append(timestamps[i] - current_start)
                    current_start = timestamps[i]
                    current_is_idle = True
                idle_periods.append(iat)
            else:
                if current_is_idle:
                    current_start = timestamps[i]
                    current_is_idle = False

        # 最后一段
        if not current_is_idle:
            active_periods.append(timestamps[-1] - current_start)

        return active_periods, idle_periods

    def _detect_bursts(self, packets: List[PacketInfo],
                       threshold: float = 0.5):
        """检测突发传输"""
        if len(packets) < 2:
            return [], []

        burst_durations = []
        burst_pkt_counts = []
        current_burst_start = packets[0].timestamp
        current_burst_count = 1
        current_direction = packets[0].direction

        for i in range(1, len(packets)):
            iat = packets[i].timestamp - packets[i-1].timestamp
            if iat > threshold or packets[i].direction != current_direction:
                # 突发结束
                if current_burst_count >= 3:
                    burst_durations.append(
                        packets[i-1].timestamp - current_burst_start)
                    burst_pkt_counts.append(current_burst_count)
                # 开始新突发
                current_burst_start = packets[i].timestamp
                current_burst_count = 1
                current_direction = packets[i].direction
            else:
                current_burst_count += 1

        # 最后一个突发
        if current_burst_count >= 3:
            burst_durations.append(
                packets[-1].timestamp - current_burst_start)
            burst_pkt_counts.append(current_burst_count)

        return burst_durations, burst_pkt_counts

    def _detect_bulk_transfers(self, packets: List[PacketInfo],
                                direction: int, threshold: int = 4):
        """检测批量传输"""
        bulk_count = 0
        bulk_bytes = 0
        bulk_pkt_count = 0

        current_streak = 0
        current_bytes = 0

        for p in packets:
            if p.direction == direction:
                current_streak += 1
                current_bytes += p.length
            else:
                if current_streak >= threshold:
                    bulk_count += 1
                    bulk_bytes += current_bytes
                    bulk_pkt_count += current_streak
                current_streak = 0
                current_bytes = 0

        # 最后一段
        if current_streak >= threshold:
            bulk_count += 1
            bulk_bytes += current_bytes
            bulk_pkt_count += current_streak

        return bulk_count, bulk_bytes, bulk_pkt_count

    def _compute_entropy(self, arr: np.ndarray) -> float:
        """计算信息熵"""
        if len(arr) == 0:
            return 0.0
        # 用于连续值的分箱
        if arr.max() > 255:
            hist, _ = np.histogram(arr, bins=50)
        else:
            hist = np.bincount(arr.astype(int), minlength=256)
        total = hist.sum()
        if total == 0:
            return 0.0
        probs = hist[hist > 0] / total
        return -np.sum(probs * np.log2(probs))

    def _compute_skewness(self, arr: np.ndarray) -> float:
        """计算偏度"""
        if len(arr) < 3:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))

    def _compute_kurtosis(self, arr: np.ndarray) -> float:
        """计算峰度"""
        if len(arr) < 4:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 4) - 3)

    def get_feature_catalog(self) -> Dict[str, Dict]:
        """
        返回基础特征的完整目录，按类别分组。
        格式: {feature_name: {category, subcategory, description}}
        """
        # 显式枚举所有196个基础特征（与 extract_all 输出一致）
        STAT_SUFFIXES = ['_max', '_min', '_mean', '_std', '_sum', '_median']
        catalog = {}

        def _add(name, subcat):
            catalog[name] = {
                'category': 'basic',
                'subcategory': subcat,
                'description': self._feature_desc(name),
            }

        # 1. 会话信息 (7维)
        for f in ['duration', 'protocol_tcp', 'protocol_udp', 'src_port',
                   'dst_port', 'is_well_known_src_port', 'is_well_known_dst_port']:
            _add(f, 'session_info')

        # 2. 协议指纹 (21维)
        for f in ['tls_version', 'tls_has_sni', 'tls_sni_length',
                   'tls_num_cipher_suites', 'tls_num_extensions',
                   'tls_num_supported_versions', 'tls_num_supported_groups',
                   'tls_num_signature_algorithms', 'tls_has_h2', 'tls_has_http11',
                   'tls_has_tls13_ciphers', 'tls_num_certificates',
                   'tls_handshake_bytes', 'tls_has_encrypted_extensions',
                   'tls_ja3_hash_prefix', 'tls_ja4_hash_prefix',
                   'quic_is_present', 'quic_version', 'quic_is_v1',
                   'quic_has_token', 'quic_token_length']:
            _add(f, 'protocol_fingerprint')

        # 3. 流统计 (54维: 10比率 + 6×3包大小 + 6×3载荷大小)
        for f in ['total_fwd_packets', 'total_bwd_packets', 'total_fwd_bytes',
                   'total_bwd_bytes', 'total_packets', 'total_bytes',
                   'fwd_bwd_packet_ratio', 'fwd_packet_rate', 'bwd_packet_rate',
                   'packet_rate', 'fwd_bwd_byte_ratio', 'byte_rate',
                   'fwd_byte_rate', 'bwd_byte_rate', 'down_up_byte_ratio']:
            _add(f, 'flow_stats')
        for prefix in ['fwd_pkt_size', 'bwd_pkt_size', 'pkt_size',
                       'fwd_payload_size', 'bwd_payload_size', 'payload_size']:
            for sfx in STAT_SUFFIXES:
                _add(f'{prefix}{sfx}', 'flow_stats')

        # 4. 时序行为 (30+13+5=48维)
        for prefix in ['iat', 'fwd_iat', 'bwd_iat', 'active', 'idle']:
            for sfx in STAT_SUFFIXES:
                _add(f'{prefix}{sfx}', 'timing_behavior')
        for prefix in ['burst_duration', 'burst_pkt_count']:
            for sfx in STAT_SUFFIXES:
                _add(f'{prefix}{sfx}', 'timing_behavior')
        _add('num_bursts', 'timing_behavior')
        for w in [0.1, 0.5, 1.0, 3.0, 5.0]:
            _add(f'pkts_first_{w}s', 'timing_behavior')

        # 5. TCP标志 (33维: 8标志×3方向 + syn_ack_rtt + 重传2 + 窗口6)
        for flag in ['fin', 'syn', 'rst', 'psh', 'ack', 'urg', 'ece', 'cwr']:
            _add(f'{flag}_flag_count', 'tcp_flags')
            _add(f'fwd_{flag}_flag_count', 'tcp_flags')
            _add(f'bwd_{flag}_flag_count', 'tcp_flags')
        _add('syn_ack_rtt', 'tcp_flags')
        _add('num_retransmissions', 'tcp_flags')
        _add('retransmission_ratio', 'tcp_flags')
        for sfx in STAT_SUFFIXES:
            _add(f'tcp_window{sfx}', 'tcp_flags')

        # 6. 协议结构 (22维)
        for prefix in ['header_bytes', 'fwd_header_bytes', 'bwd_header_bytes']:
            for sfx in STAT_SUFFIXES:
                _add(f'{prefix}{sfx}', 'protocol_structure')
        for f in ['avg_segment_size', 'fwd_avg_segment_size',
                   'bwd_avg_segment_size', 'payload_ratio']:
            _add(f, 'protocol_structure')

        # 7. 连接行为 (16维)
        for f in ['fwd_bulk_count', 'fwd_bulk_total_bytes', 'fwd_bulk_total_packets',
                   'bwd_bulk_count', 'bwd_bulk_total_bytes', 'bwd_bulk_total_packets',
                   'fwd_avg_bulk_size', 'fwd_avg_bulk_rate',
                   'bwd_avg_bulk_size', 'bwd_avg_bulk_rate',
                   'pkt_size_entropy', 'pkt_size_skewness', 'pkt_size_kurtosis',
                   'payload_byte_entropy']:
            _add(f, 'connection_behavior')

        return catalog

    @staticmethod
    def _categorize_feature(name: str) -> str:
        """根据特征名判断所属子类别"""
        if name in ('duration', 'protocol_tcp', 'protocol_udp', 'src_port',
                     'dst_port', 'is_well_known_src_port', 'is_well_known_dst_port'):
            return 'session_info'
        if name.startswith('tls_') or name.startswith('quic_') or name.startswith('ja'):
            return 'protocol_fingerprint'
        if any(name.startswith(p) for p in ['total_', 'fwd_bwd_', 'down_up_',
               'fwd_packet_', 'bwd_packet_', 'packet_rate', 'byte_rate',
               'fwd_byte_', 'bwd_byte_', 'fwd_pkt_size_', 'bwd_pkt_size_',
               'pkt_size_', 'fwd_payload_', 'bwd_payload_', 'payload_size_']):
            return 'flow_stats'
        if any(name.startswith(p) for p in ['iat_', 'fwd_iat_', 'bwd_iat_',
               'active_', 'idle_', 'burst_', 'num_bursts', 'pkts_first_']):
            return 'timing_behavior'
        if 'flag_count' in name or name in ('syn_ack_rtt', 'num_retransmissions',
                'retransmission_ratio') or name.startswith('tcp_window_'):
            return 'tcp_flags'
        if any(name.startswith(p) for p in ['header_bytes_', 'fwd_header_',
               'bwd_header_', 'avg_segment_', 'fwd_avg_segment_',
               'bwd_avg_segment_', 'payload_ratio']):
            return 'protocol_structure'
        if any(name.startswith(p) for p in ['fwd_bulk_', 'bwd_bulk_',
               'fwd_avg_bulk_', 'bwd_avg_bulk_', 'pkt_size_entropy',
               'payload_byte_entropy', 'pkt_size_skewness', 'pkt_size_kurtosis']):
            return 'connection_behavior'
        return 'other'

    @staticmethod
    def _feature_desc(name: str) -> str:
        """特征的简短中文描述"""
        DESC = {
            'duration': '流持续时间(秒)', 'protocol_tcp': '是否TCP协议',
            'protocol_udp': '是否UDP协议', 'src_port': '源端口号',
            'dst_port': '目的端口号',
            'is_well_known_src_port': '源端口是否<1024',
            'is_well_known_dst_port': '目的端口是否<1024',
            'tls_version': 'TLS版本号', 'tls_has_sni': '是否包含SNI域名',
            'tls_sni_length': 'SNI域名长度', 'tls_num_cipher_suites': '密码套件数量',
            'tls_num_extensions': 'TLS扩展数量',
            'tls_num_supported_versions': '支持的TLS版本数',
            'tls_num_supported_groups': '支持的椭圆曲线数',
            'tls_num_signature_algorithms': '签名算法数',
            'tls_has_h2': 'ALPN是否含h2', 'tls_has_http11': 'ALPN是否含http/1.1',
            'tls_has_tls13_ciphers': '是否含TLS1.3密码套件',
            'tls_num_certificates': '证书数量', 'tls_handshake_bytes': '握手消息总字节数',
            'tls_has_encrypted_extensions': '是否有EncryptedExtensions',
            'tls_ja3_hash_prefix': 'JA3指纹哈希前缀(数值化)',
            'tls_ja4_hash_prefix': 'JA4指纹哈希前缀(数值化)',
            'total_fwd_packets': '前向包数', 'total_bwd_packets': '后向包数',
            'total_fwd_bytes': '前向字节数', 'total_bwd_bytes': '后向字节数',
            'total_packets': '总包数', 'total_bytes': '总字节数',
            'fwd_bwd_packet_ratio': '前向/后向包比率',
            'fwd_bwd_byte_ratio': '前向/后向字节比率',
            'down_up_byte_ratio': '下行/上行字节比率',
            'byte_rate': '字节速率(bytes/s)', 'packet_rate': '包速率(pkts/s)',
            'duration': '流持续时间', 'iat_mean': '包间到达时间均值',
            'iat_std': '包间到达时间标准差', 'pkt_size_mean': '包大小均值',
            'pkt_size_entropy': '包大小信息熵', 'payload_byte_entropy': '载荷字节信息熵',
            'num_bursts': '突发传输次数', 'num_retransmissions': '重传包数',
            'retransmission_ratio': '重传比率', 'syn_ack_rtt': 'SYN-ACK往返时间',
            'payload_ratio': '载荷占比',
        }
        return DESC.get(name, name.replace('_', ' '))
