# CHANGELOG

## 2026-09-17（P0-P3）
- P0：四服务契约层（data_manifest/features-runtime/evaluation/reporting）、
  validate-data/evaluate 模式、mine 接 manifest 产 bundle、run_acceptance
  G01-G05 全 PASS、GREASE 精确集、W4 树一致性用例
- P1(M2)：context.py 行为窗口（流段/K=3/overflow/无未来包）、
  packet_listener 接线、runtime 行为入口、mine behavior 分支、
  dpi_infer --terminal 窗口模式；行为窗 32/32
- P1.5：采集套件（任务清单 + event_recorder.py）
- P2(M3)：AnalysisPackage 八件套、特征有效率 E/A 口径、revise-rules
- P3(M4)：CSTNET 六类真实数据全链（G09）、交付件五件套
- §5.1：TCP+TLS 联合用例（完整/分段/缺口不拼造/四元组独立）
- 回归终态：标尺41+契约17+行为10+test_all6 = 74 用例全绿

## 2026-09-16~17（M1 及整改）
- 检视报告 31 项中 30 项修复；M1 三阶段全链贯通；parity 319 维零失配
