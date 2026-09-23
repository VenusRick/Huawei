# CLAUDE.md — 项目指令（Claude Code 自动加载）

---

## 会话启动协议（每次必须执行）

1. 读取 `docs/进度.md`，了解当前完成状态和下一步工作
2. 检查 `data/task/` 清单与其中 PCAP 路径；原始 `data/all_data/` 保持只读
3. 如有未完成任务，直接继续；如无，等待用户指示

---

## 会话结束协议（每次必须执行）

所有实质性工作完成后，**自动更新 `docs/进度.md`**，无需用户提醒：

1. 「最后更新」日期改为当天
2. 各模块状态表中标记本次完成的模块为 ✅
3. 「已完成模块详情」中补充新模块信息（文件路径、功能说明、测试状态）
4. 「下一步工作计划」中删除已完成项，新增发现的待办
5. 「已生成的产出物」中补充新产出物
6. 如有新的技术决策，追加到「关键设计决策记录」

---

## 项目上下文

- **竞赛：** "华为杯"第五届中国研究生网络安全创新大赛 · 揭榜挑战赛 · 题目4
- **题目：** 基于智能化辅助分析对应用及精细化行为进行特征挖掘识别
- **截止：** 2026-10-22
- **进度：** `docs/进度.md`
- **论文总结：** `docs/资料/论文综合总结报告.md`
- **测试：** `tests/test_all.py`

---

## 系统架构

```
src/
├── config.py                              # 全局配置（TRAIN_THREADS 线程上限）
├── data_manifest.py                       # 试次清单契约（正式真值/分组划分）
├── pipeline.py                            # CLI 主入口（mine/detect/evaluate/validate-data/revise-rules）
├── parser/
│   ├── tls/tls_parser.py                  # TLS 1.3 (JA3/JA4/SNI/ALPN)
│   ├── quic/quic_parser.py                # QUIC
│   ├── session/session_manager.py         # 会话重建 (TCP状态机)
│   └── pcap_reader.py                     # PCAP/PCAPNG读取器（支持真读包上限）
├── features/
│   ├── runtime.py                         # ★ 共享提取入口（训练/推理/CLI 单一实现，
│   │                                      #   过滤<3包/<100B+前缀截断+ExtractStats计数）
│   ├── operators.py                       # 算子族调度（basic+adv_五族，按需计算）
│   ├── basic/feature_extractor.py         # 基础特征
│   ├── advanced/advanced_extractor.py     # 高级特征（5族方法）
│   └── selection/feature_selector.py      # 特征选择（XGBoost+互信息融合）
├── evaluation.py                          # 评价服务（完整分母/行为逐窗真值/背景FPR）
├── reporting.py                           # AnalysisPackage（证据语义：无证据=pending）
└── engine/
    ├── model_io.py                        # 模型/规则包序列化
    ├── dpi_infer.py                       # 独立推理CLI（--rules 规则包 / --model 对照）
    ├── rule_compiler/optimized_generator.py  # 规则生成（train拟合+validation对照）
    └── matcher/optimized_engine.py           # DPI引擎（严格AND，LEGAL_OPS全集）
```

### 关键纪律（2026-09-18 审计后，勿回退）

- **标签**：只来自清单记录字段或 basename 精确匹配；匹配不到显式报错，
  绝不子串猜测、绝不回落类0。
- **拟合隔离**：FeatureSelector/规则生成只在 train split 拟合；validation
  只用于对照与证据（按 capture_id 分组）；无 validation 明确记录，不随机混拆。
- **单一提取实现**：mine/detect/dpi_infer 一律走 `src/features/runtime.py`，
  不得复制过滤/截断/提取逻辑；前缀截断不得使用未来包时长/握手。
- **评价**：重名真值报错、非连续标签ID显式对齐、拒识入分母、行为按窗口
  区间判真值并单列背景FPR。
- **报告证据**：consistent/engine_supported 无实测证据即 pending；
  ANOVA 证据集必须标注来源（validation/训练集）。

---

---

## 代码规范

- Python 3.9+，UTF-8 编码
- 使用 type hints
- 每个模块需有文档字符串
- 新增功能须有对应测试；回归守门用例放 `tests/regression/`，
  脚本级冒烟在 `tests/test_all.py`（断言制，不返回布尔值）

---

## 常用命令

### 一、正式全链（清单驱动，主入口）

```bash
# 数据体检（清单校验+pcap存在性+采集组统计）
python3 -m src.pipeline --mode validate-data --manifest data/task/xxx_train_val.jsonl --task app

# 特征挖掘（train-only 拟合；validation 仅对照与证据）
TRAIN_THREADS=2 python3 -m src.pipeline --mode mine \
    --manifest data/task/xxx_train_val.jsonl --task app \
    --output output/<run>/mine \
    [--max-read-packets 2000] [--max-packets 500]   # 真读包上限/会话截断

# 独立规则检测（只加载规则包，不载 sklearn/xgboost）
python3 -m src.engine.dpi_infer --rules output/<run>/mine/bundle \
    --pcap-dir data/test_dir -o output/<run>/predictions.json

# 评价（完整分母；behavior 任务逐窗真值+背景FPR）
python3 -m src.pipeline --mode evaluate \
    --truth data/task/xxx_test_truth.jsonl \
    --predictions output/<run>/predictions.json \
    --task app --rules output/<run>/mine/bundle \
    --output output/<run>/evaluation

# 人工修规则（校验→验证集回放diff→新版本，不覆盖原bundle）
python3 -m src.pipeline --mode revise-rules --rules <bundle> --edits <json> \
    --manifest <清单> --output output/<run>/revised
```

**mine 输出（`output/<run>/mine/`）：**
- `bundle/` — 规则包三件套（rules.json + selected_features.json + bundle_config.json，独立推理的加载单位）
- `rules.yaml` / `rules_summary.json` — 人读规则
- `raw_features.csv` — 全部特征行（含 `_split`/`_capture_id`/`_label_id` 元列）
- `validation_report.json` — 验证集回放成绩（无 validation 则明确 `available:false`）
- `parity_report.json` — 训练侧 vs 独立推理入口实测 parity 探针
- `extraction_stats.json` — 读包/过滤/截断/提取失败计数
- `analysis/` — AnalysisPackage mine 层（catalog/effectiveness/profiles，证据语义）

### 二、独立推理（脱离训练流水线）

正式模式只需规则包目录 `--rules`（三件套）：

```bash
# 从PCAP/目录推理（解析→共享runtime特征→纯规则匹配）
python3 -m src.engine.dpi_infer --rules output/<run>/mine/bundle --pcap test.pcap -o result.json
python3 -m src.engine.dpi_infer --rules output/<run>/mine/bundle --pcap-dir data/test/

# behavior bundle 自动走窗口模式（可加 --terminal 指定可见终端）
python3 -m src.engine.dpi_infer --rules <behavior bundle> --pcap x.pcap --terminal 10.0.0.5

# 按需计算（只调规则所需算子族，调用计数留证）
python3 -m src.engine.dpi_infer --rules <bundle> --pcap x.pcap --on-demand

# [对照模式] 旧 XGBoost 模型包（离线对照，非正式DPI链路）
python3 -m src.engine.dpi_infer --model output/results/ --pcap test.pcap
```

### 三、旧调试流水线（legacy，文件名=类别名）

```bash
python3 tests/test_generic_pipeline.py --data data/USTC-TFC2016-master --max-packets 500
```

输出在 `output/results/`（历史正式产物，测试不再写入该目录）。

### 四、测试

```bash
python3 tests/test_all.py                      # 脚本级冒烟（6项，断言制）

# 全量回归（basetemp 放项目 output 下，避开 /tmp/pytest-of-root 所有权）
TRAIN_THREADS=2 python3 -m pytest tests/ -q \
    --basetemp=output/chain_audit_20260918/tests/pytest_tmp
```

### 五、在Python代码中使用DPI引擎

```python
from src.engine.matcher.optimized_engine import OptimizedDPIEngine

# 正式：规则包（不加载任何训练库）
engine = OptimizedDPIEngine()
engine.load_rule_bundle('output/<run>/mine/bundle/')

# 共享特征提取（训练/推理/CLI 单一实现，勿另写提取逻辑）
from src.features.runtime import extract_feature_records_with_stats
records, stats = extract_feature_records_with_stats('x.pcap', max_read_packets=2000)

results = engine.match(records[0].features)
for r in results:
    print(f"{r.result} (置信度: {r.confidence:.2f}, 来源: {r.source})")
```

### 六、PCAP/PCAPNG格式支持

代码通过magic number自动识别PCAP和PCAPNG格式，无需手动指定：

- PCAP: magic `0xa1b2c3d4` / `0xd4c3b2a1`（增量读，支持 `--max-read-packets` 真读包上限）
- PCAPNG: magic `0x0a0d0d0a`（支持SHB/IDB/EPB/SPB块，多接口，微秒/纳秒精度；
  按块增量读，支持真读包上限；会话生成器由正式 runtime 消费）

### 2026-09-18 第二轮主机复验
- 纯规则按需计算只以实际条件引用为依赖；selected_features 是候选/编辑目录，不能触发未引用特征计算。
- 统计规则 confidence 是训练精确率，不是训练覆盖率；两者分别留档。DT priority=100 先于 STAT=200，命中还须通过 bundle 阈值。
- 评价新增 evaluation_scope；漏真值文件、未配对预测或词表异常标 incomplete，不把文件覆盖当完整观测分母证明。
- 事件匹配始终用最大一对一增广路算法，不因 scipy.optimize 缺失退回贪心。
- 主机全套复验145 passed；最终证据见 output/chain_improve_20260918_02/host_verify/。
