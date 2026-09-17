# Choices13k pilot 实验结果

本次实验用于检验不同提示词是否能让 DeepSeek-V4-Flash 的二元风险选择更接近 Choices13k 的人类选择分布。实验与 CDA 市场模拟隔离，不会修改市场 agent。

## 数据与固定设计

- 数据来自官方 `jcpeterson/choices13k` 仓库，固定 commit：`821ae7e88386b508ebb46fae76fac63cb62ec876`。
- 下载并校验了 14,568 条 CSV 行；行号是样本身份，没有按重复的内部 Problem ID 去重。
- 使用随机种子 `20260916` 分层抽取 300 条：200 条 calibration、100 条 held-out。
- P0–P3 在 calibration 上各生成 10 次回答（每个 item 5 次 A-first、5 次 B-first），按 MAE 选择提示词；held-out 比较 P0 与冻结后的最佳提示词。

## API 执行

运行使用当前配置的 `aigc_relay`（模型 `DeepSeek-V4-Flash`）。共完成 10,000 个有效响应；发生 2 次失败并自动重试，重试后全部成功。输入 token 1,275,540，输出 token 69,954；请求和响应日志均经过敏感信息审计，没有保存 API key。

## 结果

Calibration 选择了 **P1**：P1 的 MAE 为 0.2703，优于 P0 的 0.2833。

Held-out 结果如下：

| 提示词 | MAE | RMSE | mean JS | Pearson | Spearman | 无效率 |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.2851 | 0.3494 | 0.1177 | 0.2267 | 0.1959 | 0% |
| P1 | 0.2702 | 0.3383 | 0.1163 | 0.4132 | 0.3655 | 0% |

P1−P0 的配对 MAE 差为 **−0.0149**，bootstrap 95% CI 为 **[−0.0613, 0.0290]**，区间跨过 0，因此当前结果只能说明 P1 有改善方向，不能宣称已经显著提升“人类相似度”。P1 的首选项比例为 41.6%，退化为单一选择的 item 比例为 24%。

### Subgroup MAE

以下分析全部来自既有 held-out 结果，没有新增 API 调用。负的 ΔMAE 表示 P1 优于 P0。

| Subgroup | P0 MAE | P1 MAE | ΔMAE |
|---|---:|---:|---:|
| Loss | 0.2638 | 0.2572 | −0.0066 |
| No loss | 0.3264 | 0.2954 | −0.0310 |
| Ambiguous | 0.2854 | 0.2433 | −0.0421 |
| Non-ambiguous | 0.2850 | 0.2765 | −0.0085 |
| Feedback | 0.2872 | 0.2644 | −0.0228 |
| No feedback | 0.2750 | 0.2985 | +0.0235 |

P1 的最大方向性改善出现在 ambiguous 与 no-loss 子组；在 no-feedback 子组反而更差。不过 no-feedback 只有 17 个 held-out items，这些结果是诊断性分析，不能当作新的 prompt-selection 依据。

### Risk 与 expected-value diagnostics

| Prompt | LLM 选择更高风险选项 | Human 选择更高风险选项 | LLM 选择更高 EV 选项 | Human 选择更高 EV 选项 |
|---|---:|---:|---:|---:|
| P0 | 38.28% | 48.09% | 56.19% | 64.74% |
| P1 | 33.33% | 48.09% | 59.07% | 64.74% |

在本批题目中，两个 prompt 都比真人更少选择高风险选项，也更少选择高 EV 选项。P1 进一步降低了高风险选择率，同时让高 EV 选择率更接近真人，因此误差变化不能简化成单一的“风险规避”或“EV maximization”解释。这些指标只用于诊断。

### Position bias

定义 `PositionBias = P(select displayed first) − 0.5`，并定义 `BPositionEffect = P(select original B | B first) − P(select original B | B second)`。置信区间使用 2,000 次 item-level bootstrap。

| Prompt | First-option rate | PositionBias (95% CI) | BPositionEffect (95% CI) |
|---|---:|---:|---:|
| P0 | 36.30% | −0.137 [−0.178, −0.095] | −0.274 [−0.356, −0.190] |
| P1 | 41.60% | −0.084 [−0.123, −0.042] | −0.168 [−0.246, −0.084] |

两个 prompt 都存在明显的 displayed-second 偏好；P1 减弱了这种位置效应，但没有消除它。

### Degenerate items

Degenerate item 指 10 次回答全部选择同一原始选项，即 `p_llm(B) ∈ {0,1}`。整体 P0 为 18%，P1 为 24%。

| Subgroup | P0 | P1 |
|---|---:|---:|
| Loss | 15.15% | 24.24% |
| No loss | 23.53% | 23.53% |
| Ambiguous | 21.05% | 15.79% |
| Non-ambiguous | 17.28% | 25.93% |
| Feedback | 16.87% | 22.89% |
| No feedback | 23.53% | 29.41% |

P1 的整体退化比例更高，主要来自 non-ambiguous 与 no-feedback 子组；这也是 confirmatory experiment 必须继续报告的采样诊断。

## 结果文件

- 汇总报告：`outputs/human_benchmark/choices13k_deepseek_pilot/evaluation/report.md`
- 完整指标与 bootstrap 区间：`outputs/human_benchmark/choices13k_deepseek_pilot/evaluation/summary.json`
- API 用量（不含 key）：`outputs/human_benchmark/choices13k_deepseek_pilot/api_usage.json`
- 可复现性审计：`outputs/human_benchmark/choices13k_deepseek_pilot/evaluation/audit.json`

审计确认 calibration 为 8,000 条、held-out 为 2,000 条，提示词集合、重复次数、A/B 展示顺序、响应状态和数据哈希均符合预注册配置。由于这是公开 benchmark，可能存在训练数据污染；Choices13k 的 feedback 行也缺少参与者级真实反馈轨迹，因此本实验只支持“聚合二元风险选择”的结论，不能直接外推到金融交易、herding 或 strategy switching。
