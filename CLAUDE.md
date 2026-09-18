# CLAUDE.md — 项目指令（Claude Code 自动加载）

---

## 会话启动协议（每次必须执行）

1. 读取 `docs/进度.md`，了解当前完成状态和下一步工作
2. 检查 `data/train/` 和 `data/test/` 目录是否有新的 PCAP 数据
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
├── config.py                              # 全局配置
├── parser/
│   ├── tls/tls_parser.py                  # TLS 1.3 (JA3/JA4/SNI/ALPN)
│   ├── quic/quic_parser.py                # QUIC
│   ├── session/session_manager.py         # 会话重建 (TCP状态机)
│   └── pcap_reader.py                     # PCAP/PCAPNG读取器
├── features/
│   ├── basic/feature_extractor.py         # 基础特征 196维
│   ├── advanced/advanced_extractor.py     # 高级特征 123维
│   └── selection/feature_selector.py      # 特征选择（XGBoost+互信息融合）
└── engine/
    ├── model_io.py                        # 模型序列化（XGBoost JSON）
    ├── dpi_infer.py                       # 独立推理CLI
    ├── rule_compiler/optimized_generator.py  # 规则生成（XGBoost集成）
    └── matcher/optimized_engine.py           # DPI引擎（软匹配）
```

---

## 代码规范

- Python 3.9+，UTF-8 编码
- 使用 type hints
- 每个模块需有文档字符串
- 新增模块必须在 `tests/test_all.py` 中添加对应测试

---

## 常用命令

### 一、训练流水线（端到端）

```bash
# 模式A: 预划分目录（目录名=类别名）
python tests/test_generic_pipeline.py --train data/demo/train --val data/demo/val --test data/demo/test

# 模式B: 单目录（文件名=类别名，自动随机划分6:2:2）
python tests/test_generic_pipeline.py --data data/USTC-TFC2016-master --max-file-mb 99999 --max-packets 500

# 只训练不评测
python tests/test_generic_pipeline.py --data data/USTC-TFC2016-master --no-eval

# 训练时同时保存特征向量（供独立推理复用）
python tests/test_generic_pipeline.py --data data/USTC-TFC2016-master --max-file-mb 99999 --save-features
```

**训练流水线输出文件（`output/results/`）：**
- `model.json` — XGBoost原生JSON模型（跨版本兼容，独立推理用）
- `model.pkl` — pickle格式模型（保留兼容）
- `engine_config.json` — 引擎配置（置信度阈值+类别映射）
- `selected_features.json` — 选中60个特征列表
- `rules.yaml` — 规则+特征定义
- `all_features.json` — 完整特征目录（基础196+高级123）
- `feature_effectiveness.json` — 特征有效率报告
- `dpi_results.json` / `dpi_results.csv` — DPI识别结果

**使用 `--save-features` 时额外输出（`output/features/`）：**
- `features.json` — 所有会话的特征向量（供独立推理复用，不需要重新解析PCAP）

### 二、独立推理（脱离训练流水线）

独立推理只需3个文件：`model.json` + `selected_features.json` + `engine_config.json`

```bash
# 从PCAP文件推理（自动解析→特征提取→分类）
python -m src.engine.dpi_infer --model output/results/ --pcap data/test.pcap

# 批量推理目录下所有PCAP/PCAPNG文件
python -m src.engine.dpi_infer --model output/results/ --pcap-dir data/test/

# 从特征向量JSON推理（训练时用 --save-features 保存的特征）
python -m src.engine.dpi_infer --model output/results/ --features output/features/features.json

# 输出结果到文件
python -m src.engine.dpi_infer --model output/results/ --pcap test.pcap -o result.json

# 覆盖置信度阈值
python -m src.engine.dpi_infer --model output/results/ --pcap test.pcap --confidence 0.8
```

### 三、单元测试

```bash
python tests/test_all.py
```

### 四、在Python代码中使用DPI引擎

```python
from src.engine.matcher.optimized_engine import OptimizedDPIEngine

# 加载模型包（独立推理）
engine = OptimizedDPIEngine()
engine.load_model_dir('output/results/')

# 输入特征向量，返回识别结果
features = {'duration': 1.23, 'total_packets': 150, 'tls_has_sni': 1, ...}
results = engine.match(features)
for r in results:
    print(f"{r.result} (置信度: {r.confidence:.2f}, 来源: {r.source})")
```

### 五、PCAP/PCAPNG格式支持

代码通过magic number自动识别PCAP和PCAPNG格式，无需手动指定：

- PCAP: magic `0xa1b2c3d4` / `0xd4c3b2a1`
- PCAPNG: magic `0x0a0d0d0a`（支持SHB/IDB/EPB/SPB块，多接口，微秒/纳秒精度）
