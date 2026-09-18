# 智能化加密流量特征挖掘工具与DPI识别引擎详细实施方案

> 用途：交给编程Agent按阶段实施。
>
> 最终目标：交付两个可独立运行、以规则包衔接的程序：
>
> 1. `traffic-miner`：从标注PCAP中解析流量、挖掘特征、训练模型、生成和校验DPI规则。
> 2. `traffic-dpi`：从PCAP或实时网卡读取流量，加载规则包，输出应用、精细行为和匿名工具识别结果。

---

## 1. 任务定位

本项目不是单纯训练一个加密流量分类模型。完整链路必须覆盖：

```text
PCAP与标签
  → 协议解析与会话重建
  → 基础特征和高维特征挖掘
  → 可用特征筛选
  → 可解释规则生成
  → 规则校准和人工修正
  → DPI在线匹配
  → 应用、行为、匿名工具输出
  → 误报漏报反馈
```

赛题要求的六项核心能力分别落到以下模块：

| 赛题要求 | 工程模块 | 验收产物 |
|---|---|---|
| TLS 1.3、QUIC、TCP/UDP私有协议解析与会话重建 | `libtraffic_core` | 流索引、协议字段、乱序/重传统计、单元测试 |
| 基础特征提取 | `feature_runtime` | 特征注册表、Parquet特征文件 |
| 高级维度生成 | `teacher_models`、`sequence_mining` | 高维表示、序列模式、困难样本 |
| 智能总结可用特征和识别方案 | `feature_miner`、`rule_generator` | 特征报告、候选规则、规则有效性报告 |
| 可解释DPI规则体系 | `rule_schema`、`rule_compiler` | YAML规则、规则包、规则说明书 |
| DPI高效运行并输出识别结果 | `traffic-dpi` | PCAP回放结果、实时结果、性能报告 |

文档批注对应的技术决策：

1. **基础特征**指端口、IP、域名、DNS、TLS/QUIC握手、统计量和行为字段。
2. **高维特征**指机器学习或深度模型从包序列、Burst和多流关系中提取的抽象表示。
3. **两个系统必须解耦**。特征挖掘工具不能成为DPI运行时依赖。
4. **面向大规模实用场景**。在线路径只保留可增量计算、可索引和可早退的规则。
5. **小样本能力由预训练表征与规则迁移承担**，不要求把大模型放进在线数据面。
6. **域名、DNS、握手指纹可用，但只作为辅助证据**。主体仍应覆盖包序列、Burst、频率和多流行为。
7. **离线与在线边界固定**：复杂挖掘离线完成；DPI负责PCAP回放和实时识别。

---

## 2. 最终验收目标

### 2.1 赛题指标

最终测试报告至少给出：

- 特征维度自动输出有效率不低于80%。
- 人工修正后可用率不低于90%。
- 召回率不低于98%。
- 准确率不低于95%。
- 精确率不低于95%。
- 假阳率不高于5%。
- Android、iOS、PC分别报告结果。

### 2.2 内部工程验收线

以下为项目内部建议验收线，不替代赛题指标：

| 项目 | 验收线 |
|---|---|
| 离线与在线特征一致性 | 同一PCAP中离线工具与DPI输出完全一致；浮点特征误差不超过`1e-6` |
| Python与C++规则一致性 | 所有黄金样本判定结果一致 |
| 规则相对显式模型损失 | 宏平均F1下降不超过3个百分点 |
| 规则相对高维教师模型损失 | 首版不超过8个百分点，最终不超过5个百分点 |
| 规则覆盖率 | 已知类验证集不低于95% |
| 早期识别 | 至少80%的可识别流在前32包内输出，至少95%在前64包或3秒内输出 |
| PCAP回放性能 | 代表性PCAP至少达到5倍实时回放 |
| 活跃流内存 | 中位数不超过2KB，P95不超过4KB；复杂多流关联状态单独统计 |
| 稳定性 | 相同配置、相同种子、相同数据得到相同划分和可复现结果 |

性能目标必须在指定服务器上实测，不得用理论值代替。

---

## 3. 总体技术路线

### 3.1 双系统架构

```text
┌──────────────────────────────────────────────────────────────┐
│                    智能化特征挖掘工具                        │
│                                                              │
│  数据审计 → 统一解析 → 特征仓库 → 模型训练 → 特征分析       │
│                                  ↓                           │
│                   序列模式/规则提取 → 规则校准               │
│                                  ↓                           │
│                         YAML规则包                            │
└──────────────────────────────┬───────────────────────────────┘
                               │ 规则编译
                               ▼
                     二进制规则包 `.rpk`
                               │
┌──────────────────────────────┴───────────────────────────────┐
│                       DPI识别引擎                             │
│                                                              │
│  PCAP/网卡 → 包解析 → 流状态 → 增量特征 → 分层规则匹配      │
│                                      ↓                       │
│                   应用/行为/匿名工具/置信度/证据             │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 核心设计原则

1. **规则优先设计**：先定义DPI能计算的特征和规则，再训练模型。
2. **共享特征内核**：离线工具和在线DPI复用同一套C++特征代码。
3. **模型作为教师**：模型用于发现复杂规律，不默认部署到DPI主路径。
4. **任务分头输出**：应用、行为、匿名工具分别建模和判定，避免组合标签爆炸。
5. **公共规则与平台适配规则并存**：不强求Android、iOS、PC使用完全相同的阈值。
6. **先PCAP后实时**：先完成可复现的离线回放，再接libpcap，最后接DPDK。
7. **先单流后多流**：先完成流级识别，再增加行为窗口和多连接关联。
8. **域名/IP等易变字段分层管理**：提供增强配置，但主结果必须报告去除易变字段后的表现。

---

## 4. 明确范围

### 4.1 必须完成

- PCAP/PCAPNG读取。
- IPv4/IPv6、TCP、UDP解析。
- IP分片重组。
- TCP双向流识别、乱序和重传处理。
- TLS 1.2/1.3握手元数据解析。
- QUIC长头、版本、CID、Initial等元数据解析。
- DNS请求与后续连接关联。
- 包、流、Burst、序列和行为窗口特征。
- 应用识别、行为识别、匿名工具识别三类任务。
- 显式模型、高维教师模型和规则模型。
- 自动特征报告、候选规则、人工修正规则。
- 规则编译、加载、索引、匹配、冲突消解。
- PCAP回放和实时网卡两种DPI输入。
- 跨平台测试矩阵。
- 单元测试、集成测试、性能测试、测试报告。
- 完整源代码、说明书、部署手册、示例规则和标注数据清单。

### 4.2 首版不做

- 不解密TLS应用载荷。
- 不把Transformer或其他大模型放进DPI在线主路径。
- 不先做图形化大平台；首版以CLI和静态HTML报告为主。
- 不先做FPGA、SmartNIC或P4。待C++规则语义稳定后再扩展。
- 不使用随机包级划分作为主结果。
- 不把预处理后的PKL/CSV当作DPI规则挖掘主数据源。

---

## 5. 数据集使用计划

### 5.1 第一批审计对象

先扫描全部数据目录，不预设目录内容。优先检查：

```text
CrossPlatform
CrossPlatform-Android
ISCXVPN2016
ISCXTor2016
CipherSpectrum
CSTNET-TLS1.3
USTC-TFC2016-pcap
USTC-TFC2016
NetBench
TrafficGPT-data
```

推荐分工仅作为初始假设：

| 数据集 | 主要用途 | 是否要求原始PCAP |
|---|---|---|
| CrossPlatform、CrossPlatform-Android | 移动应用识别、Android/iOS适应性 | 是 |
| ISCXVPN2016 | Chat、VoIP、Streaming、File Transfer等行为；VPN识别 | 是 |
| ISCXTor2016 | Tor/非Tor和对应业务类别 | 是 |
| CipherSpectrum、CSTNET-TLS1.3 | TLS 1.3鲁棒性与现代握手特征 | 是 |
| USTC-TFC2016-pcap | 管线回归、正常/恶意应用流量 | 是 |
| NetBench、TrafficGPT-data | 模型对照或预训练辅助 | 否，不作为规则主数据 |

其余IDS、IoT和内存恶意软件数据仅用于扩展，不阻塞主链路。

### 5.2 数据审计必须输出

执行：

```bash
trafficctl data scan \
  --root "$DATA_ROOT" \
  --out artifacts/data_audit
```

输出：

```text
artifacts/data_audit/
├── datasets_detected.yaml
├── files.parquet
├── pcap_quality.parquet
├── label_candidates.csv
├── duplicate_groups.csv
├── task_coverage.csv
├── platform_coverage.csv
└── audit_report.md
```

每个文件至少记录：

- 相对路径、文件类型、文件大小、SHA-256。
- PCAP/PCAPNG格式、链路层类型、包数、时长、首尾时间。
- IPv4/IPv6、TCP/UDP、TLS、QUIC、DNS占比。
- 损坏包、截断包、时间戳倒序、空PCAP。
- 标签来源：目录名、文件名、CSV、JSON、README或未知。
- 平台来源：Android、iOS、PC或未知。
- 应用、行为、匿名工具标签候选。
- 相同文件哈希和近重复流指纹。

### 5.3 数据集注册表

人工核对后生成：

```yaml
name: iscxvpn2016
root: /data/datasets/ISCXVPN2016
format: pcap
platform: pc
capture_scope: mixed
label_source:
  type: path_regex
  pattern: '(?P<app>[^/]+)-(?P<behavior>[^/]+)'
client_direction:
  method: tcp_syn_or_first_packet
split_group:
  - capture_file
  - source_host
  - collection_day
tasks:
  application: true
  behavior: true
  anonymity: true
notes: ''
```

所有数据集必须通过JSON Schema或Pydantic校验。未确定的字段必须显式写`unknown`，不能静默推断。

### 5.4 标签体系

统一使用三层标签，不把三者拼接成一个大类别：

```text
application_label: whatsapp / facebook / skype / ...
behavior_label: text_chat / voice_call / video_call / streaming / file_transfer / ...
anonymity_label: direct / vpn / tor / psiphon / session / ...
```

补充元数据：

```text
platform: android / ios / pc / unknown
app_version: string / unknown
protocol_family: tls_tcp / quic / udp_private / tcp_private / unknown
dataset_name
capture_id
device_id
user_id
collection_day
```

### 5.5 数据划分

主评测禁止随机包级或随机流级混洗。按以下优先级分组：

1. 设备或用户。
2. 采集日期。
3. PCAP文件。
4. 应用版本。
5. 网络环境。

默认划分：

```text
train 60%
validation 20%
test 20%
```

若样本较少，使用分组五折交叉验证。所有划分保存到`split_manifest.parquet`，后续实验不得重新随机。

必须检查：

- 同一五元组不跨集合。
- 同一PCAP切出的相邻流不跨集合。
- 完全重复和近重复流不跨集合。
- 同一用户动作产生的多条流不跨集合。
- 规则阈值只用训练集和验证集确定。

---

## 6. 仓库结构

```text
intelligent-traffic-dpi/
├── README.md
├── STATUS.md
├── DECISIONS.md
├── CHANGELOG.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── pyproject.toml
├── uv.lock
├── CMakeLists.txt
├── CMakePresets.json
├── docker/
│   ├── Dockerfile.dev
│   ├── Dockerfile.runtime
│   └── compose.yaml
├── configs/
│   ├── datasets/
│   ├── features/
│   ├── models/
│   ├── rules/
│   └── runtime/
├── schemas/
│   ├── dataset.schema.json
│   ├── feature.schema.json
│   ├── rule.schema.json
│   └── result.schema.json
├── common/
│   ├── feature_registry.yaml
│   ├── label_registry.yaml
│   ├── generated/
│   └── scripts/generate_schema_code.py
├── cpp/
│   ├── libtraffic_core/
│   │   ├── include/
│   │   ├── src/
│   │   └── tests/
│   ├── rule_compiler/
│   ├── traffic_dpi/
│   ├── bindings/
│   └── fuzz/
├── python/
│   └── traffic_miner/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── mining/
│       ├── rules/
│       ├── evaluation/
│       ├── reporting/
│       └── cli/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── golden_pcaps/
│   ├── golden_rules/
│   └── performance/
├── scripts/
│   ├── bootstrap.sh
│   ├── build.sh
│   ├── run_demo.sh
│   ├── acceptance.sh
│   └── package_release.sh
├── docs/
│   ├── architecture.md
│   ├── data_spec.md
│   ├── protocol_parsing.md
│   ├── feature_spec.md
│   ├── rule_spec.md
│   ├── dpi_engine.md
│   ├── experiment_protocol.md
│   └── deployment.md
├── artifacts/
└── dist/
```

`artifacts/`、模型权重和大数据不得提交到Git。仅提交配置、清单、哈希和小型黄金PCAP。

---

## 7. 开发环境与依赖

### 7.1 语言和构建

- C++17。
- CMake + Ninja。
- Python 3.11或以上，项目内锁定具体版本。
- PyTorch用于教师模型。
- Parquet作为主特征存储格式。
- YAML作为可读规则源格式。
- MessagePack作为DPI加载的编译规则包格式，扩展名`.rpk`。

### 7.2 网络解析组件

建议组合：

- PcapPlusPlus：PCAP读取、协议分层、TCP重组、重传和乱序处理。
- OpenSSL：TLS字段辅助、QUIC Initial所需HKDF和AEAD。
- TShark：只用于离线差分验证，不作为DPI运行时依赖。
- nDPI：只作为第三方DPI基线或可选适配器，不作为本项目规则引擎核心。
- DPDK：最终性能阶段接入，不阻塞PCAP和libpcap版本。

第三方依赖必须固定提交或发布版本，并生成`THIRD_PARTY_NOTICES.md`。nDPI为LGPLv3，若链接或分发必须单独核对交付方式；默认采用独立可选适配器，核心引擎不依赖它。

### 7.3 一键环境

Agent必须提供：

```bash
./scripts/bootstrap.sh
./scripts/build.sh
pytest -q
ctest --test-dir build --output-on-failure
```

GPU不是数据解析必需条件。资源分工：

- CPU：PCAP扫描、流重建、特征提取、规则评测、DPI性能测试。
- GPU：包序列教师模型、自监督预训练和超参数搜索。
- 多GPU：只用于数据并行训练，不改变模型接口。

---

## 8. 共享数据模型

### 8.1 PacketRecord

```text
capture_id
frame_no
ts_ns
link_type
ip_version
src_ip
dst_ip
src_port
dst_port
l4_proto
wire_len
ip_len
transport_payload_len
direction
tcp_seq
tcp_ack
tcp_flags
is_retransmission
is_out_of_order
is_fragment
tls_record_type
quic_header_type
quic_version
```

### 8.2 FlowRecord

双向流使用规范化五元组和实例序号标识：

```text
flow_id = hash(capture_id, canonical_5tuple, flow_instance)
```

字段：

```text
flow_id
capture_id
initiator_ip
responder_ip
initiator_port
responder_port
transport
start_ts_ns
end_ts_ns
packet_count
close_reason
platform
application_label
behavior_label
anonymity_label
split
```

方向判断：

1. TCP优先使用首个SYN发起方。
2. 无SYN时使用数据集声明的客户端IP。
3. UDP/QUIC优先使用数据集客户端IP或QUIC Initial发起方。
4. 无元数据时以首包源端作为发起方，并将`direction_confidence=low`。

默认流超时：

```text
TCP idle timeout: 120 s
UDP/QUIC idle timeout: 30 s
hard lifetime: 1800 s
```

所有值可在数据集配置中覆盖。

### 8.3 BurstRecord

Burst定义：

> 连续的非空载荷包具有相同方向，并且相邻包间隔不超过`burst_gap_ms`。方向变化或间隔超阈值时结束。

默认：

```text
burst_gap_ms = 1000
ignore_ack_only = true
```

字段：

```text
flow_id
burst_index
direction
start_ts_ns
end_ts_ns
packet_count
payload_bytes
wire_bytes
max_packet_len
mean_iat
```

### 8.4 ActivityWindow

用于同一用户动作触发多条流的情况：

```text
activity_id
endpoint_key
window_start
window_end
flow_ids
flow_count
domain_set
remote_prefix_set
protocol_mix
application_label
behavior_label
platform
```

初始分组规则：

- 同一客户端端点。
- 相邻流启动时间不超过5秒。
- 共享域名、DNS答案、证书或远端网段时提高关联得分。
- 公开数据没有动作时间戳时，仅把ActivityWindow作为辅助实验，不替代流级主结果。

---

## 9. 共享特征内核

### 9.1 关键要求

离线工具和DPI引擎不得各写一套特征逻辑。所有在线可用特征由`libtraffic_core`计算，Python通过`pybind11`调用同一代码。

```text
libtraffic_core
├── packet decoder
├── flow manager
├── protocol parsers
├── feature states
├── burst state
├── sequence state
└── activity state
```

Python绑定接口采用批式迭代，避免一次把完整PCAP加载进内存：

```python
for batch in traffic_core.extract_pcap(path, config):
    writer.write(batch)
```

### 9.2 特征注册表

`common/feature_registry.yaml`是唯一事实来源。每个特征至少包含：

```yaml
id: 10017
name: flow.fwd_packet_count
dtype: int64
level: flow
online: true
cost_tier: 1
update_event: packet
window: full_flow
missing_value: 0
version: 1
description: 发起方到响应方的数据包数量
```

代码生成器生成：

- Python枚举和Pydantic模型。
- C++枚举和查找表。
- `docs/feature_spec.md`。
- 规则Schema中的合法特征列表。

禁止手工在Python或C++中新增未注册特征。

### 9.3 首版特征集合

#### A. 基础网络特征

- IP版本。
- 传输层协议。
- 源/目的端口及端口类别。
- TCP标志统计。
- TCP窗口、MSS、选项顺序。
- IPv4 TTL或IPv6 Hop Limit统计。
- 分片、重传、乱序计数。

原始IP只进入“增强规则配置”，不进入稳健主配置。

#### B. TLS特征

- TLS记录版本和握手版本。
- ClientHello/ServerHello是否存在。
- 密码套件数量、顺序哈希和集合哈希。
- 扩展数量、顺序哈希和集合哈希。
- 支持组、签名算法、EC点格式。
- ALPN值或集合。
- SNI是否存在、域名后缀和长度。
- 会话恢复、PSK、Early Data等标志。
- 握手包长度和方向序列。
- JA3/JA3S类指纹作为可插拔特征。

#### C. QUIC特征

- QUIC版本。
- Long Header类型。
- DCID/SCID长度。
- Token长度。
- Initial包长度、数量和方向。
- Retry、Version Negotiation等标志。
- QUIC Initial解密成功时提取TLS ClientHello中的SNI、ALPN和扩展。
- 根据CID关联连接迁移。

#### D. DNS关联特征

- 查询类型、回答数量和TTL统计。
- 查询名后缀。
- DNS响应到连接建立的时间差。
- 目的IP是否来自近期DNS答案。
- 一个行为窗口内域名数量和重复访问频率。

#### E. 流统计特征

- 双向包数和字节数。
- 持续时间。
- 上下行包数比和字节比。
- 包长均值、标准差、最小、最大、P25、P50、P75、P90、P95。
- IAT均值、标准差和分位数。
- 空载荷包比例。
- 前8、16、32、64包的累计统计。
- 首包、首个响应、首个大包、首个应用数据的时间差。

#### F. Burst特征

- Burst数量。
- 双向Burst数量和字节。
- Burst包数、字节和持续时间分布。
- Burst方向交替次数。
- 前8、16个Burst的方向、包数和字节序列。
- 长空闲间隔数量。
- 周期性发送程度。

#### G. 包序列特征

保存前128个有效包：

```text
signed_payload_length
signed_wire_length
log_binned_iat
tcp_flag_token
tls_record_token
quic_packet_token
```

在线DPI默认只保留前64包。128包用于离线教师模型和消融实验。

#### H. ActivityWindow特征

- 时间窗口内流数量。
- TCP、UDP、QUIC占比。
- 并发流峰值。
- 启动顺序和间隔。
- 域名、证书、远端IP前缀数量。
- 总上行、下行字节。
- 小控制流与大媒体流的组合。
- 周期性心跳和媒体包同时出现的模式。

### 9.4 特征成本等级

| 等级 | 定义 | 示例 |
|---|---|---|
| 0 | 单包直接读取 | 协议、端口、长度 |
| 1 | 常数状态计数 | 包数、字节、均值 |
| 2 | 前N包或Burst状态 | 序列、分位数近似 |
| 3 | 协议握手和重组 | TLS、QUIC Initial、DNS关联 |
| 4 | 多流关联 | ActivityWindow |

规则生成时把成本作为惩罚项。高成本规则只有在明显提高指标时才能保留。

---

## 10. 协议解析实施细节

### 10.1 TCP

必须完成：

- 双向流表。
- SYN/FIN/RST状态。
- 序列号排序。
- 重传识别。
- 乱序缓存。
- IP分片重组。
- 缺失段标记。
- TCP重组数据只用于握手或协议字段，不参与明文内容分析。

统计策略：

- `raw_*`特征保留重传包。
- `unique_*`特征去除重传包。
- 包序列主版本默认去除纯ACK和重传；另保存是否发生重传。

### 10.2 TLS 1.3

实现最小握手解析器：

```text
TLS record
  → Handshake message
  → ClientHello / ServerHello
  → ciphers / extensions / SNI / ALPN / supported_groups / signatures
```

需要处理：

- 一个握手消息跨多个TCP段。
- 一个TLS记录含多个握手消息。
- GREASE值归一化。
- 扩展顺序和集合分别编码。
- 加密握手后的应用数据只统计长度、方向和时间。

### 10.3 QUIC

分两级完成。

#### 第一级：元数据解析

- 识别UDP上的QUIC Long Header。
- 解析版本、类型、DCID、SCID、Token、Payload Length。
- 识别Version Negotiation、Retry、Initial、Handshake。
- 维护CID到连接的映射。

#### 第二级：Initial解析

- 支持标准QUIC v1和v2 Initial密钥派生。
- 用OpenSSL完成HKDF、Header Protection和AEAD。
- 重组CRYPTO帧。
- 从ClientHello提取SNI、ALPN和TLS扩展。
- 不解密1-RTT应用数据。

对不支持的版本必须返回：

```text
quic_detected = true
quic_version = value
quic_initial_decrypted = false
```

不得因解密失败丢弃该流。

### 10.4 私有TCP/UDP协议

不假设明确应用层格式。保留：

- 包长方向序列。
- IAT。
- Burst。
- 端口变化。
- 多流行为。
- 首包字节长度和有限的不可逆字节统计；不得把原始载荷写入最终规则。

### 10.5 解析差分验证

建立`tests/golden_pcaps/`。对每个黄金PCAP：

1. 用本项目解析器输出字段。
2. 用TShark导出对应字段。
3. 比较TCP流、TLS握手、QUIC头、DNS记录。
4. 允许差异必须写入`known_differences.yaml`。

至少覆盖：

- 正常TCP握手。
- 无SYN抓包。
- 重传。
- 乱序。
- IP分片。
- TLS 1.2。
- TLS 1.3。
- QUIC v1 Initial。
- QUIC v2 Initial。
- DNS后连接。
- 截断PCAP。

---

## 11. 智能化特征挖掘工具

### 11.1 CLI

```bash
traffic-miner data scan ...
traffic-miner data register ...
traffic-miner extract ...
traffic-miner train ...
traffic-miner analyze-features ...
traffic-miner generate-rules ...
traffic-miner validate-rules ...
traffic-miner review-server ...
traffic-miner export ...
```

### 11.2 特征提取

```bash
traffic-miner extract \
  --dataset configs/datasets/iscxvpn2016.yaml \
  --feature-profile configs/features/core.yaml \
  --workers 32 \
  --out artifacts/features/iscxvpn2016
```

输出：

```text
metadata.json
packet_index.parquet         # 可选，默认不长期保存完整包级表
flow_index.parquet
flow_features.parquet
sequence_features.parquet
burst_features.parquet
activity_features.parquet
extract_errors.jsonl
extract_report.md
```

提取任务必须可断点续跑。每个PCAP生成独立完成标记和内容哈希，输入或配置不变时跳过。

### 11.3 三类任务

#### 应用识别

```text
输入：握手、流统计、Burst、包序列
输出：application_label
```

#### 精细行为识别

```text
输入：流统计、Burst、包序列、ActivityWindow
输出：behavior_label
```

行为任务优先在ISCXVPN2016等有明确服务标签的数据上验证，再用官方或受控采集数据细化到同应用动作。

#### 匿名工具识别

```text
输入：握手、流统计、包序列、多流关系
输出：anonymity_label
```

首版至少覆盖`direct / vpn / tor`。Psiphon和Session在官方数据或受控采集数据到位后补充。

### 11.4 特征配置分层

至少维护两套配置：

```text
core_robust:
  不使用原始IP、固定端口、完整域名等易变字段

enhanced_operational:
  允许域名后缀、DNS、端口、IP前缀、握手指纹等辅助字段
```

主报告必须同时给出两套结果。增强配置用于实际覆盖，稳健配置用于证明方法不是只记住基础设施标识。

---

## 12. 模型体系

### 12.1 模型层次

```text
Level 0：规则/统计基线
Level 1：显式可解释模型
Level 2：包序列高维教师模型
Level 3：小样本适配模型
```

### 12.2 Level 0：基础基线

至少实现：

- 端口规则。
- 域名/DNS规则。
- TLS握手指纹规则。
- 最近邻或原型距离基线。
- Logistic Regression。

作用：确定简单特征能达到的下限，并发现数据泄漏。

### 12.3 Level 1：显式模型

首选LightGBM或等价梯度提升树，输入在线可计算显式特征。

推荐初始参数范围：

```yaml
num_leaves: [8, 16, 31]
max_depth: [3, 4, 6, 8]
learning_rate: [0.03, 0.05, 0.1]
n_estimators: [200, 500, 1000]
min_child_samples: [20, 50, 100]
feature_fraction: [0.7, 0.9, 1.0]
```

目标：

- 获得可解释强基线。
- 提取特征重要性。
- 抽取树路径规则。
- 分析困难类别。

### 12.4 Level 2：包序列教师模型

首版实现轻量Transformer，不先复现大型基础模型。

建议结构：

```text
输入：前128包
每包token：signed_length + iat_bin + flag/protocol token
Embedding
4层Transformer Encoder
统计特征MLP
拼接
应用/行为/匿名工具三个分类头
```

初始规模：

```yaml
d_model: 128
n_heads: 4
n_layers: 4
ffn_dim: 512
dropout: 0.1
max_packets: 128
```

训练要求：

- 支持单GPU和DDP。
- 混合精度。
- 梯度裁剪。
- 固定随机种子。
- 保存最佳验证集权重、训练日志和完整配置。

### 12.5 自监督预训练

当无标签PCAP充足时增加：

1. Masked Packet Modeling：预测被遮蔽的长度桶、方向和IAT桶。
2. 双视图对比：随机丢包、轻微IAT扰动、裁剪前缀，不改变标签。
3. Burst顺序预测：判断Burst顺序是否被打乱。

预训练只在训练集合和额外无标签数据上进行，不得看到测试标签。

### 12.6 小样本适配

每类设置：

```text
5-shot
10-shot
20-shot
50-shot
full-data
```

比较：

- 从头训练线性头。
- 冻结教师编码器训练线性头。
- 类原型最近邻。
- 显式特征树模型。
- 最终规则。

小样本规则生成流程：

```text
少量正样本
  → 冻结编码器得到表征
  → 找到最接近的困难负类
  → 对正类和困难负类做显式特征差异分析
  → 生成候选规则
  → 在留出样本上校准
```

---

## 13. 特征分析与智能总结

### 13.1 分析方法

每个任务至少计算：

- 单变量互信息或单变量AUC。
- 树模型增益重要性。
- 分组Permutation Importance。
- 树SHAP。
- 特征组消融。
- 不同随机种子和数据划分下的稳定性。
- 跨平台分布偏移。
- 教师模型与显式模型预测差异。
- 类别对之间的差异特征。

### 13.2 特征评分

所有子项归一化到`[0,1]`：

```text
feature_score =
  0.30 * predictive_gain
+ 0.20 * split_stability
+ 0.15 * cross_platform_stability
+ 0.15 * rule_precision
+ 0.10 * rule_coverage
+ 0.10 * teacher_alignment
- online_cost_penalty
- volatility_penalty
```

说明：

- `predictive_gain`：加入该特征后的验证集提升。
- `split_stability`：多次分组划分的重要性一致性。
- `cross_platform_stability`：平台间条件分布和贡献一致性。
- `rule_precision`：由该特征参与的规则精确率。
- `teacher_alignment`：该特征对教师模型困难样本的解释程度。
- `volatility_penalty`：IP、完整域名、固定端口等易变字段惩罚。

权重写入配置，允许实验调整，但每次调整必须保存版本。

### 13.3 自动有效特征判定

候选特征进入“自动有效”集合需满足：

- 在线可计算。
- 至少在两个独立划分上提升指标。
- 重要性方向或贡献稳定。
- 覆盖不少于目标类验证样本的10%，或属于高精度小覆盖规则。
- DPI成本在预算内。
- 不是纯数据集文件名、采集时间或固定主机泄漏。

工具输出：

```text
feature_candidates.csv
feature_validated.csv
feature_rejected.csv
feature_report.html
```

### 13.4 人工修正闭环

人工界面必须支持：

- 查看特征定义。
- 查看正负样本分布。
- 查看涉及该特征的规则。
- 查看命中和误判PCAP片段。
- 接受、拒绝、修改阈值、标记易变。
- 记录审阅人、时间和理由。

自动有效率：

```text
人工确认有效的自动候选数 / 自动候选总数
```

人工修正后可用率：

```text
最终可用特征数 / 人工审阅后的保留特征总数
```

必须生成可审计记录，不能只给一个人工填写的百分比。

---

## 14. 序列模式挖掘

高维教师模型不能直接转成DPI规则，因此增加显式序列模式挖掘。

### 14.1 离散化

包token：

```text
方向：U / D
长度桶：0, 1-63, 64-127, 128-255, 256-511, 512-1023, 1024-1499, 1500+
IAT桶：<1ms, 1-10ms, 10-100ms, 100ms-1s, >1s
协议token：TLS-HS, TLS-APP, QUIC-I, QUIC-H, QUIC-1RTT, OTHER
```

例如：

```text
U:256-511:<10ms:TLS-HS
D:1024-1499:<100ms:TLS-HS
```

### 14.2 候选模式

挖掘：

- 连续子序列。
- 允许跳过少量包的稀疏子序列。
- Burst方向模式。
- 周期性IAT模式。
- 前N包长度区间模式。

每个模式计算：

```text
support_positive
support_negative
precision
recall
lift
platform_stability
```

### 14.3 转换为在线自动机

最终模式编译为有限状态机：

```text
state 0 --token A--> state 1
state 1 --token B within 3 packets--> state 2
state 2 --token C before 2s--> accept
```

禁止在线使用任意复杂正则表达式。所有模式必须可编译成有界状态机。

---

## 15. DPI规则DSL

### 15.1 规则层次

支持四类规则：

1. 字段规则。
2. 数值统计规则。
3. 序列状态规则。
4. 多流行为规则。

### 15.2 YAML示例

```yaml
schema_version: 1
rule_id: behavior.voip.001
rule_version: 3
task: behavior
label: voice_call
platforms: [android, ios, pc, unknown]
priority: 80
cost_tier: 2
scope:
  transports: [udp]
  protocols: [quic, udp_private]
trigger:
  evaluate_at_packets: [8, 16, 32, 64]
  timeout_ms: 3000
conditions:
  all:
    - feature: flow.duration_ms
      op: gte
      value: 1500
    - feature: flow.byte_ratio_fwd_bwd
      op: between
      value: [0.65, 1.45]
    - feature: burst.count
      op: gte
      value: 6
    - sequence:
        source: packet.signed_length_bin
        pattern: [U_SMALL, D_SMALL, U_SMALL, D_SMALL]
        max_skip: 2
        within_packets: 24
score:
  base: 0.72
  calibration_id: behavior_voip_v3
output:
  confidence: calibrated
  evidence: true
```

### 15.3 支持的操作符

标量：

```text
eq, ne, lt, lte, gt, gte, between, in, not_in, exists
```

字符串和集合：

```text
suffix_in, prefix_in, set_contains, set_intersects
```

序列：

```text
sequence_exact
sequence_with_skip
sequence_prefix
periodic_pattern
```

时间和多流：

```text
count_within
rate_within
transition_within
concurrent_count
```

在线引擎不支持通用脚本、不支持动态代码、不支持无界循环。

### 15.4 规则冲突消解

匹配顺序：

1. 任务独立：应用、行为、匿名工具分别判定。
2. 高优先级优先。
3. 同优先级比较校准分数。
4. 分数接近时选择条件更具体的规则。
5. 仍冲突则输出候选列表和`ambiguous=true`。
6. 未达到任务阈值时输出`unknown`或`unresolved`。

### 15.5 规则版本

规则包必须包含：

```text
schema_version
feature_registry_version
rule_pack_version
build_timestamp
training_data_manifest_hash
validation_report_hash
rules_checksum
```

DPI启动时校验特征注册表版本和规则包版本，不兼容时拒绝加载。

---

## 16. 自动规则生成

### 16.1 树路径规则

流程：

1. 训练限制深度的显式树模型。
2. 提取高纯度叶节点路径。
3. 把路径转成AND条件。
4. 在验证集重新计算精确率、召回率和覆盖率。
5. 删除无贡献条件。
6. 合并相邻区间和相似规则。
7. 用测试前固定阈值生成正式候选。

候选门槛初始值：

```text
precision >= 0.95
support_positive >= 10
coverage >= 0.02
```

小样本类别可降低支持数，但必须单独标记。

### 16.2 握手和字段规则

- 类别条件频繁项集。
- TLS扩展组合。
- ALPN、SNI后缀和DNS组合。
- QUIC版本/CID/Initial特征组合。
- 端口只作为组合条件，不单独定义强规则，除非验证集证明稳定。

### 16.3 序列规则

- 从判别性子序列中选择高lift模式。
- 转换成有限状态机。
- 用验证集选择长度桶容差、最大跳过包数和截止包数。

### 16.4 多流行为规则

- 对ActivityWindow显式特征训练浅树。
- 规则表达并发流数量、协议组合、流启动顺序和字节结构。
- 多流规则必须设置窗口上限和状态预算。

### 16.5 规则压缩

目标：在满足指标的前提下减少规则数和运行成本。

采用：

- 重复规则删除。
- 子集规则合并。
- 相邻阈值合并。
- 贪心集合覆盖。
- 冲突规则裁剪。
- 高成本条件替换为低成本近似条件。

每次压缩后必须重新运行完整验证集。

### 16.6 规则校准

规则原始分数用验证集校准。每个任务和类别保存：

```text
accept_threshold
reject_threshold
class_prior
calibration_curve
```

优先选择满足假阳率约束的阈值，再最大化召回率。

---

## 17. 规则编译器

### 17.1 输入输出

```bash
rule-compiler \
  --input rules/source/behavior.yaml \
  --schema schemas/rule.schema.json \
  --feature-registry common/feature_registry.yaml \
  --output rules/compiled/behavior.rpk
```

### 17.2 编译步骤

1. YAML语法检查。
2. JSON Schema检查。
3. 特征ID和类型检查。
4. 平台、任务、标签合法性检查。
5. 序列规则转状态机。
6. 域名后缀转反向Trie。
7. 精确集合转哈希表。
8. 数值区间排序和合并。
9. 规则按协议、触发阶段、成本等级建立索引。
10. 输出MessagePack规则包和人类可读清单。

### 17.3 编译期拒绝条件

- 引用不存在特征。
- 在线规则引用`online=false`特征。
- 无界时间窗口。
- 无界序列跳转。
- 条件类型与特征类型不匹配。
- 重复`rule_id + rule_version`。
- 规则包与特征注册表版本不一致。

---

## 18. DPI识别引擎

### 18.1 可执行程序

```bash
traffic-dpi --pcap input.pcap --rules rules/compiled/all.rpk --out result.jsonl
traffic-dpi --interface eth0 --rules rules/compiled/all.rpk --out result.jsonl
traffic-dpi --dpdk ...
```

### 18.2 模块

```text
capture_source
  ├── PcapFileSource
  ├── LibpcapLiveSource
  └── DpdkSource
packet_decoder
flow_sharder
flow_manager
protocol_parser
feature_runtime
rule_index
rule_vm
activity_correlator
result_emitter
metrics_server
```

### 18.3 FlowContext

每条活跃流保存：

```text
canonical tuple
start/last timestamps
protocol state
incremental counters
first-N packet ring buffer
burst state
TLS/QUIC handshake state
candidate rule bitmap
matched rules
current task scores
```

不得保存完整应用载荷。

### 18.4 分层匹配

```text
L0 首包：协议、端口、IP版本
L1 握手：TLS/QUIC/DNS
L2 前8/16/32/64包：统计和序列
L3 流结束或超时：完整流/Burst
L4 行为窗口：多流关联
```

每层只评估可能命中的规则桶。达到高置信阈值后可早退，但必须满足：

- 当前规则允许早退。
- 不存在未评估的更高优先级冲突规则。
- 任务置信度超过`early_accept_threshold`。

### 18.5 规则索引

- `transport + protocol + task`一级索引。
- 精确字段用哈希索引。
- 域名后缀用反向Trie。
- 数值区间用排序区间表。
- 序列规则用共享前缀状态机。
- 多流规则按端点和时间窗口索引。

### 18.6 多线程

PCAP和实时版本采用：

```text
capture thread
  → bounded queue
  → N flow workers, hash(flow_id)分片
  → result queue
  → writer
```

要求同一流始终进入同一worker，保持包顺序。多流关联可按客户端端点路由到固定分片，或由独立关联线程接收流摘要。

### 18.7 流表管理

- LRU和超时轮。
- 最大活跃流数可配置。
- 达到内存上限时优先驱逐长期无活动且低价值流。
- 被驱逐流输出`close_reason=evicted`。
- 所有驱逐和丢包计数进入运行指标。

### 18.8 识别结果

```json
{
  "flow_id": "...",
  "capture_id": "...",
  "timestamp_ns": 0,
  "platform_hint": "android",
  "application": {
    "label": "whatsapp",
    "confidence": 0.97,
    "status": "accepted"
  },
  "behavior": {
    "label": "voice_call",
    "confidence": 0.94,
    "status": "accepted"
  },
  "anonymity": {
    "label": "direct",
    "confidence": 0.99,
    "status": "accepted"
  },
  "decision_packet": 28,
  "decision_latency_ms": 731,
  "rule_ids": ["app.whatsapp.013", "behavior.voip.001"],
  "evidence": {
    "alpn": "h3",
    "burst_pattern": "bp_03"
  }
}
```

支持流结束后的最终结果和早期结果更新。若结果更新，使用相同`flow_id`和递增`result_version`。

---

## 19. 跨平台方案

### 19.1 规则结构

```text
common_rules.rpk
android_adapter.rpk
ios_adapter.rpk
pc_adapter.rpk
```

公共规则负责平台共享规律。平台适配规则只补偿：

- 协议栈差异。
- 默认网络行为差异。
- App实现差异。
- 阈值偏移。

### 19.2 训练流程

1. 在多平台训练集上训练公共显式模型。
2. 生成公共规则。
3. 分平台分析公共规则误差。
4. 对剩余错误训练浅层残差模型。
5. 只把稳定残差转成平台适配规则。

### 19.3 评测矩阵

| 训练 | 测试 | 目的 |
|---|---|---|
| Android | Android | 同平台上限 |
| iOS | iOS | 同平台上限 |
| PC | PC | 同平台上限 |
| Android | iOS | 跨平台迁移 |
| iOS | Android | 跨平台迁移 |
| Android+iOS | 分平台测试 | 公共模型 |
| 全平台 | 分平台测试 | 最终公共规则 |
| 公共规则 | 分平台测试 | 公共覆盖率 |
| 公共+适配 | 分平台测试 | 最终性能 |

平台未知时：

- 先运行公共规则。
- 置信度不足时运行所有适配包。
- 选择证据最充分的结果，并输出`platform_hint`和依据。

---

## 20. 实验设计

### 20.1 主实验

每个任务报告：

```text
基础规则
Logistic Regression
显式树模型
序列教师模型
自动生成规则
人工修正规则
```

### 20.2 消融实验

至少包含：

- 去掉域名/DNS/IP。
- 去掉TLS/QUIC握手。
- 去掉流统计。
- 去掉Burst。
- 去掉包序列。
- 去掉ActivityWindow。
- 只用前8/16/32/64包。
- 公共规则与平台适配规则。
- 去除重传与保留重传。
- 不同Burst定义。

### 20.3 鲁棒性实验

- 时间划分。
- 跨PCAP划分。
- 跨平台。
- 跨协议版本。
- 样本量变化。
- 包丢失模拟：1%、5%、10%。
- 时间抖动。
- 前缀截断。
- 域名或IP特征缺失。

扰动只用于评测或训练增强，不能改变测试标签。

### 20.4 指标

分类指标：

- Accuracy。
- Macro/Micro Precision。
- Macro/Micro Recall。
- Macro/Micro F1。
- 每类Precision、Recall、FPR。
- 混淆矩阵。

规则指标：

- 规则数量。
- 平均条件数。
- 规则覆盖率。
- 冲突率。
- 早退比例。
- 平均触发包数。
- 特征成本分布。
- 自动有效率和人工修正可用率。

性能指标：

- Gbps和Mpps。
- CPU占用和每包周期。
- 内存和每流状态。
- P50/P95/P99判定时延。
- 丢包数、驱逐流数和队列积压。
- 规则数从100、1K、10K扩展时的性能。

### 20.5 赛题指标核对

每次正式评测生成`competition_metrics.json`：

```json
{
  "accuracy": 0.0,
  "macro_precision": 0.0,
  "macro_recall": 0.0,
  "false_positive_rate": 0.0,
  "auto_feature_valid_rate": 0.0,
  "post_review_feature_usable_rate": 0.0
}
```

脚本必须明确是否达标，不得只输出图表。

---

## 21. 性能优化路线

### 21.1 先正确后优化

优化顺序固定：

1. 单线程PCAP正确性。
2. 多线程PCAP回放。
3. libpcap实时抓包。
4. 规则索引和早退。
5. 内存压缩。
6. DPDK。
7. SmartNIC/P4可选扩展。

### 21.2 具体优化

- 按规则桶缩小候选集合。
- 先计算低成本特征。
- 仅对候选规则需要的特征启用状态。
- 第一阶段确定不可能命中的规则后清除候选位图。
- 固定长度数组代替频繁堆分配。
- 对流状态使用对象池。
- 按worker分片，减少锁。
- 域名后缀Trie和序列自动机共享内存。
- 结果批量写盘。
- 性能测试关闭调试日志。

### 21.3 DPDK阶段

DPDK只替换`capture_source`和包分发层，不修改规则语义。要求：

- 选择稳定LTS版本并固定。
- 提供HugePage、NIC绑定和队列配置脚本。
- 用RSS保持流亲和性。
- 测试代表性包长分布和最小包场景。
- 输出单核、多核扩展曲线。

---

## 22. 测试体系

### 22.1 单元测试

C++：

- 五元组规范化。
- 方向判断。
- TCP重传和乱序。
- Burst切分。
- TLS字段解析。
- QUIC Long Header和Initial。
- 特征增量更新。
- 规则操作符。
- 序列自动机。
- 规则冲突消解。

Python：

- 数据集配置校验。
- 划分无泄漏。
- 特征分析。
- 模型加载与推理。
- 树路径转规则。
- 规则评测。
- 报告生成。

### 22.2 黄金PCAP集成测试

每个黄金PCAP绑定：

```text
expected_flows.json
expected_features.json
expected_results.json
```

CI运行：

```bash
traffic-dpi --pcap tests/golden_pcaps/x.pcap --rules tests/golden_rules/x.rpk
```

输出必须与黄金结果一致。

### 22.3 离线/在线一致性测试

对同一PCAP：

```text
traffic-miner extract
traffic-dpi --dump-features
```

比较每条流所有在线特征。任何新特征合并前必须通过此测试。

### 22.4 模糊测试和Sanitizer

至少为以下入口建立fuzz target：

- TLS握手解析。
- QUIC头和CRYPTO帧解析。
- YAML/MessagePack规则加载。
- 序列自动机。

CI定期运行：

- AddressSanitizer。
- UndefinedBehaviorSanitizer。
- ThreadSanitizer单独任务。

### 22.5 性能回归

保存固定小型性能PCAP。每次发布比较：

- 吞吐下降超过10%则阻止发布。
- P95内存增加超过15%则阻止发布。
- 规则加载时间增加超过20%则记录原因。

---

## 23. 可复现与实验管理

每次实验保存：

```text
config.yaml
git_commit.txt
environment.json
dataset_manifest_hash.txt
split_manifest.parquet
metrics.json
per_class_metrics.csv
predictions.parquet
confusion_matrix.csv
feature_importance.csv
rule_pack.yaml
logs/
```

实验目录命名：

```text
artifacts/experiments/{task}/{dataset}/{timestamp}_{short_commit}/
```

禁止覆盖已有实验目录。

服务器资源通过配置声明：

```yaml
compute:
  cpu_workers: 32
  gpu_ids: [0,1,2,3]
  precision: bf16
  distributed: ddp
  seed: 20260830
```

---

## 24. 分阶段执行计划

## 阶段0：仓库和环境

### 任务

- 创建仓库结构。
- 配置Python和C++构建。
- 接入PcapPlusPlus、OpenSSL、pybind11、测试框架。
- 建立基础CI。
- 创建`STATUS.md`和`DECISIONS.md`。

### 验收

```bash
./scripts/bootstrap.sh
./scripts/build.sh
pytest -q
ctest --test-dir build --output-on-failure
```

全部通过。

---

## 阶段1：数据审计

### 任务

- 扫描全部数据目录。
- 输出PCAP质量和标签候选。
- 建立第一批数据集注册表。
- 生成固定划分。
- 完成重复和近重复检查。

### 验收

- 至少确认一个应用识别数据集、一个行为数据集、一个匿名流量数据集和一个TLS 1.3数据集。
- 每个数据集有明确标签来源、平台来源和划分方式。
- 不得开始模型训练前跳过此阶段。

---

## 阶段2：统一解析与特征MVP

### 任务

- PCAP读取。
- 双向流表。
- TCP重传/乱序。
- TLS基础握手。
- QUIC元数据第一级。
- 流统计、前32包、基础Burst。
- Python绑定和Parquet输出。

### 验收

- 黄金PCAP测试通过。
- 与TShark的关键字段一致。
- 同一PCAP重复提取结果一致。
- 可对一个完整数据集产出`flow_features.parquet`。

---

## 阶段3：首个端到端闭环

### 任务

选择ISCXVPN2016或实际审计后最合适的数据集，完成：

```text
PCAP
→ 特征
→ 显式树模型
→ 树路径规则
→ YAML
→ 规则编译
→ C++ DPI回放
→ 指标报告
```

### 验收

- `traffic-miner`和`traffic-dpi`均能独立运行。
- Python规则评估与C++结果一致。
- 输出至少一个行为或应用识别规则包。
- 建立`run_demo.sh`。

这是项目第一条必须打通的垂直链路。

---

## 阶段4：完整特征与三任务

### 任务

- 扩充TLS、QUIC Initial、DNS、Burst和前64/128包。
- 分别建立应用、行为和匿名工具任务。
- 增加稳健配置与增强配置。
- 完成主要基线。

### 验收

- 三任务均有独立训练、验证、测试结果。
- 每个任务至少有一套可加载规则包。
- 输出特征组消融报告。

---

## 阶段5：高维教师与特征智能分析

### 任务

- 实现包序列Transformer。
- 可选自监督预训练。
- 建立教师模型与显式模型差异分析。
- 实现特征综合评分。
- 实现序列模式挖掘。

### 验收

- 教师模型训练可复现。
- 输出困难样本列表和新增候选特征。
- 至少一组由教师差异分析发现的新规则在验证集上有效。

---

## 阶段6：规则工程化和人工审阅

### 任务

- 完整规则DSL。
- 规则压缩、冲突分析和校准。
- 静态HTML或轻量本地Web审阅界面。
- 审阅记录和可用率统计。

### 验收

- 自动特征有效率和人工修正可用率可计算。
- 规则包包含完整版本和哈希。
- 规则数量、覆盖率和冲突率可报告。

---

## 阶段7：跨平台和多流行为

### 任务

- 公共规则与平台适配规则。
- ActivityWindow。
- 跨平台训练和测试矩阵。
- Android、iOS、PC分别报告。

### 验收

- 证明公共规则覆盖率。
- 证明平台适配规则带来的增益。
- 多流行为模块关闭时不影响单流主链路。

---

## 阶段8：实时与性能

### 任务

- libpcap实时输入。
- 多线程流分片。
- 规则索引和早退。
- 内存上限和流驱逐。
- DPDK后端。

### 验收

- PCAP回放达到内部性能线。
- 实时抓包稳定运行。
- 给出吞吐、时延、内存和规则扩展曲线。
- DPDK后端与PCAP后端结果一致。

---

## 阶段9：最终交付

### 任务

- 清理代码和配置。
- 固定依赖。
- 打包规则、模型、示例数据和脚本。
- 完成技术说明书、部署手册、测试报告、数据说明。
- 生成软件物料清单和第三方许可说明。

### 验收

在全新环境运行：

```bash
./scripts/bootstrap.sh
./scripts/build.sh
./scripts/run_demo.sh
./scripts/acceptance.sh
```

能生成完整结果和报告。

---

## 25. 编程Agent执行规范

### 25.1 每个阶段的固定流程

1. 阅读`STATUS.md`、`DECISIONS.md`和本方案。
2. 检查已有实现，不重复创建功能相同的模块。
3. 先写或更新测试。
4. 完成最小可运行实现。
5. 运行单元测试和对应集成测试。
6. 运行真实数据命令并保存输出。
7. 更新文档、状态和决策记录。
8. 提交原子化Commit。

### 25.2 Agent不得做的事

- 不修改原始数据集。
- 不伪造实验结果、吞吐或指标。
- 不用随机划分掩盖数据泄漏。
- 不在Python和C++重复实现同一在线特征。
- 不把文件名、目录名或标签字段误作为模型输入。
- 不把测试集用于阈值选择。
- 不为了通过测试删掉失败样本。
- 不在未通过阶段验收时直接开发DPDK或UI。
- 不提交密钥、大型PCAP或模型权重到Git。

### 25.3 状态记录

`STATUS.md`格式：

```markdown
## 当前阶段
阶段2：统一解析与特征MVP

## 已完成
- ...

## 正在处理
- ...

## 阻塞项
- ...

## 最近一次可复现实验
- 命令：...
- 输出：...
- Git commit：...
```

`DECISIONS.md`记录技术决策、备选方案和原因，不记录泛泛讨论。

### 25.4 Commit建议

```text
feat(parser): add TCP reassembly state
feat(feature): add first-32-packet statistics
feat(rule): compile numeric predicates
test(quic): add v1 initial golden vector
perf(engine): shard flow table by worker
fix(split): prevent capture leakage across sets
```

---

## 26. 第一轮具体执行清单

编程Agent收到本文件后，先完成以下顺序，不先训练复杂模型。

### Step 1：初始化

```bash
mkdir intelligent-traffic-dpi
cd intelligent-traffic-dpi
git init
```

建立第6节目录，创建基础构建脚本和测试框架。

### Step 2：扫描数据

```bash
export DATA_ROOT=/path/to/datasets
trafficctl data scan --root "$DATA_ROOT" --out artifacts/data_audit
```

人工确认第一批主数据集和标签映射。

### Step 3：实现最小C++解析器

先支持：

- PCAP。
- IPv4/IPv6。
- TCP/UDP。
- 双向五元组。
- 包数、字节数、持续时间、前32个方向化包长。

### Step 4：Python绑定和Parquet

完成：

```bash
traffic-miner extract --dataset ... --out artifacts/features/demo
```

### Step 5：行为或应用基线

在第一批可用数据上训练Logistic Regression和浅树。

### Step 6：树规则导出

导出至少一套YAML规则，并用Python评估。

### Step 7：最小DPI引擎

加载规则，回放同一PCAP，输出JSONL。

### Step 8：一致性

确认：

- Python特征与C++ DPI特征一致。
- Python规则结果与C++结果一致。

只有Step 1至Step 8全部通过，才进入TLS/QUIC、教师模型和DPDK阶段。

---

## 27. CLI最终形态

### 27.1 数据审计

```bash
traffic-miner data scan \
  --root /data/datasets \
  --out artifacts/data_audit
```

### 27.2 特征提取

```bash
traffic-miner extract \
  --dataset configs/datasets/iscxvpn2016.yaml \
  --features configs/features/core_robust.yaml \
  --workers 32 \
  --out artifacts/features/iscxvpn2016
```

### 27.3 训练

```bash
traffic-miner train \
  --task behavior \
  --data artifacts/features/iscxvpn2016 \
  --model configs/models/lightgbm.yaml \
  --out artifacts/experiments/behavior/iscxvpn_lgbm
```

### 27.4 特征分析

```bash
traffic-miner analyze-features \
  --experiment artifacts/experiments/behavior/iscxvpn_lgbm \
  --out artifacts/analysis/behavior_iscxvpn
```

### 27.5 规则生成

```bash
traffic-miner generate-rules \
  --experiment artifacts/experiments/behavior/iscxvpn_lgbm \
  --feature-analysis artifacts/analysis/behavior_iscxvpn \
  --out rules/source/behavior.yaml
```

### 27.6 规则验证

```bash
traffic-miner validate-rules \
  --rules rules/source/behavior.yaml \
  --data artifacts/features/iscxvpn2016 \
  --split test \
  --out artifacts/rule_validation/behavior
```

### 27.7 编译

```bash
rule-compiler \
  --input rules/source/behavior.yaml \
  --output rules/compiled/behavior.rpk
```

### 27.8 DPI回放

```bash
traffic-dpi \
  --pcap /data/test.pcap \
  --rules rules/compiled/behavior.rpk \
  --out artifacts/dpi_results/test.jsonl \
  --metrics artifacts/dpi_results/test_metrics.json
```

---

## 28. 最终交付目录

```text
dist/
├── source/
│   └── intelligent-traffic-dpi.tar.gz
├── bin/
│   ├── traffic-miner
│   ├── traffic-dpi
│   └── rule-compiler
├── rules/
│   ├── source/
│   └── compiled/
├── models/
│   ├── explicit/
│   └── teacher/
├── configs/
├── examples/
│   ├── sample.pcap
│   ├── sample_labels.csv
│   └── expected_result.jsonl
├── datasets/
│   ├── dataset_manifest.csv
│   ├── label_schema.md
│   └── collection_and_annotation.md
├── docs/
│   ├── 系统架构设计说明书.md
│   ├── 协议解析与会话重建说明书.md
│   ├── 特征定义与智能挖掘说明书.md
│   ├── DPI规则语言与引擎说明书.md
│   ├── 测试报告.md
│   ├── 部署与使用手册.md
│   └── 第三方依赖与许可说明.md
└── scripts/
    ├── install.sh
    ├── run_demo.sh
    └── acceptance.sh
```

---

## 29. 最终测试报告结构

```text
1. 测试环境
2. 数据集和标签
3. 数据划分与泄漏检查
4. 协议解析正确性
5. 应用识别结果
6. 精细行为识别结果
7. 匿名工具识别结果
8. Android/iOS/PC跨平台结果
9. TLS 1.3和QUIC结果
10. 小样本结果
11. 特征有效性与人工审阅
12. 模型到规则的性能变化
13. 消融和鲁棒性
14. DPI吞吐、时延和内存
15. 规则规模扩展
16. 已知限制与后续扩展
17. 复现命令和产物哈希
```

---

## 30. 完成定义

项目只有同时满足以下条件才算完成：

- 有真实PCAP输入，不只处理CSV或PKL。
- 能完成TCP/UDP双向流和TLS/QUIC基础解析。
- 能提取基础、统计、Burst、序列和至少一类多流特征。
- 有可复现的应用、行为和匿名工具实验。
- 有显式模型和高维教师模型。
- 能自动生成可解释规则。
- 规则经过独立验证和人工审阅。
- DPI引擎不依赖训练框架即可运行。
- Python与C++特征、规则结果一致。
- 能读取PCAP和实时网卡。
- Android、iOS、PC分别有结果或明确的数据补齐计划。
- 有性能实测，不用推测值。
- 所有源代码、配置、规则、说明书和测试脚本可交付。
- 在新环境中能用一组命令复现演示结果。

---

## 31. 决策优先级

开发中出现冲突时，按以下顺序取舍：

```text
可验证正确性
> 离线/在线一致性
> 规则可解释和可部署
> 测试集泛化
> 识别指标
> 性能
> 功能丰富度
> 界面美观
```

第一阶段不追求一次覆盖全部赛题场景。先把一个真实数据集上的完整闭环做通，再按应用识别、行为识别、匿名工具、跨平台、实时性能的顺序扩展。最终系统结构从第一天起保持可扩展，不在后期重写规则语义和特征接口。
