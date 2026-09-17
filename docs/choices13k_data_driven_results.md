# Choices13k data-driven calibration 实验结果

本实验检验来自真实人类选择数据的 few-shot demonstrations，能否比 vanilla
提示词或冻结的 persona 提示词更准确地复现 Choices13k 的聚合选择分布。实验与
CDA 市场隔离，未修改既有 pilot、confirmatory 或市场实验。

## 数据与冻结流程

排除 pilot 的 300 道题和 confirmatory 的 1,000 道题后，官方冻结数据中只有
1,331 条未使用的 `Feedback=False` 记录，无法按建议的 2,000/500/1,000 规模
切分。因此，本实验保持 no-feedback 主体并冻结为：

- calibration pool：600 道；
- development validation：300 道；
- final test：400 道；
- 另留 31 道未使用。

三组按 `has_loss`、`ambiguity` 和 EV-gap tertile 分层，构造 split 时未使用
human B-rate、human sample size 或 target error。三组互不重叠，也不与历史
1,300 道题重叠。冻结 SHA256 分别为：

| Split | SHA256 |
|---|---|
| Calibration | `15e9304cfc7bd9292c22c3c747873bfca0aa03417c606bd5247dfc391489fb35` |
| Development | `09a96481d4bfd58acf3d04555dbf6b1f99ea21fac76e9433f22a0a0b97514acb` |
| Final test | `9854bb96695335761e3ec40ac146fe2544bedd8918d3fbf8d79eaa767734abd9` |

模型为 `aigc_relay / DeepSeek-V4-Flash`，temperature 0.7，thinking 关闭，
`max_tokens=32`。每个 item-method 运行 10 次，严格保持 5 次 A-first 和 5 次
B-first。每个 demonstration 把 calibration item 的人类 B-rate 确定性地转换为
10 个个体 A/B 选择；提示中没有 B-rate、EV、variance、风险标签、分组标签或
解释。检索只使用目标题在作答前可见的题目特征。

协议与 split 在提交 `f1561a3` 冻结。开发集完成后，入选方法文件在提交
`0d7b792` 单独冻结，之后才允许运行 final test。

## Development selection

开发阶段完成 300 道题 × 8 个方法 × 10 次重复，共 24,000 个有效响应。
B0 与 B1 用于对照，只有六个 B2 候选有资格入选。预注册主指标为 item-level
MAE，结果如下：

| Method | MAE | RMSE | Pearson | Spearman |
|---|---:|---:|---:|---:|
| B0 | 0.3021 | 0.3643 | 0.2326 | 0.2118 |
| B1 | 0.3104 | 0.3667 | 0.3718 | 0.3717 |
| B2 Random K=4 | 0.2617 | 0.3238 | 0.3903 | 0.3834 |
| B2 Random K=8 | 0.2613 | 0.3191 | 0.3733 | 0.3602 |
| B2 Random K=16 | 0.2555 | 0.3152 | 0.4102 | 0.4012 |
| **B2 Matched K=4** | **0.2546** | **0.3100** | **0.4970** | **0.4774** |
| B2 Matched K=8 | 0.2558 | 0.3094 | 0.4714 | 0.4597 |
| B2 Matched K=16 | 0.2577 | 0.3144 | 0.5141 | 0.5043 |

因此冻结的 B2 是 **`B2_matched_k4`**。选择只读取 development，未读取 final
test 的 human labels 或响应。

## Final-test 主结果

最终阶段完成 400 道全新题目 × 3 个冻结方法 × 10 次重复，共 12,000 个有效
响应。统计单位为 item，置信区间来自 5,000 次 paired item bootstrap。

| Method | MAE | RMSE | mean JS | Pearson | Spearman | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| B0 Vanilla | 0.2837 | 0.3475 | 0.1225 | 0.3605 | 0.3458 | 0% |
| B1 Persona | 0.2979 | 0.3618 | 0.1338 | 0.3847 | 0.3508 | 0% |
| **B2 Matched K=4** | **0.2446** | **0.2962** | **0.0913** | **0.5696** | **0.5682** | **0%** |

主要对比如下，负数表示 B2 的误差更低：

| Contrast | ΔMAE | Paired 95% CI |
|---|---:|---:|
| B2 − B0 | **−0.0391** | **[−0.0609, −0.0181]** |
| B2 − B1 | **−0.0533** | **[−0.0756, −0.0311]** |

B2−B0 区间的上界低于 0，达到预注册成功条件。这对应预注册的 **Outcome A**：
真实人类行为示例在这一 benchmark 上提供了可泛化的校准信号。B2 相对 B0
将 MAE 降低约 13.8%，相对 B1 降低约 17.9%。这一结论来自冻结后只运行一次的
final test，没有根据最终结果修改假设、检索方式或 K。

## 行为与采样诊断

| Method | First-option rate | PositionBias | BPositionEffect | Degenerate items | Riskier choice | Higher-EV choice |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 35.85% | −0.1415 | −0.2830 | 26.00% | 37.79% | 57.92% |
| B1 | 40.55% | −0.0945 | −0.1890 | 28.25% | 33.86% | 55.86% |
| B2 | 42.00% | −0.0800 | −0.1600 | 22.50% | 39.30% | 61.54% |
| Human | — | — | — | — | 49.25% | 62.16% |

B2 的位置偏差绝对值和退化 item 比例都较低，但位置偏差不是主指标，也不能
单独证明更像人。B2 的 higher-EV choice rate 接近人类聚合率；riskier-choice
rate 仍明显偏低，说明校准没有消除所有行为差距。

相对 B0，B2 在全部预注册分析子组中都降低了 MAE：

| Subgroup | Items | B0 MAE | B2 MAE | ΔMAE B2−B0 |
|---|---:|---:|---:|---:|
| Loss | 265 | 0.2805 | 0.2431 | −0.0374 |
| No loss | 135 | 0.2899 | 0.2475 | −0.0424 |
| Ambiguous | 76 | 0.2953 | 0.2229 | −0.0724 |
| Non-ambiguous | 324 | 0.2810 | 0.2497 | −0.0313 |
| Small EV gap | 134 | 0.3017 | 0.2448 | −0.0569 |
| Medium EV gap | 132 | 0.2895 | 0.2738 | −0.0157 |
| Large EV gap | 134 | 0.2599 | 0.2156 | −0.0444 |
| Strong human consensus | 85 | 0.3054 | 0.2286 | −0.0767 |
| Mixed human preferences | 315 | 0.2778 | 0.2489 | −0.0289 |

这些是预注册的诊断性分组结果；主推断仍是 400 个 item 的整体成对 bootstrap。

## API、审计与结论边界

开发和最终阶段共得到 36,000 个有效响应。API 日志记录 36,001 次 HTTP 尝试：
36,000 次成功，1 次失败后自动重试成功，0 次 repair。输入 token 为
30,898,590，输出 token 为 252,008；中转站没有提供可验证价格，因此不估算费用。
日志和审计产物不含 API key。

可复现性审计通过，确认：

- calibration/development/final-test 数量分别为 600/300/400，交叠为 0；
- development/final 响应数分别为 24,000/12,000；
- B2 的 demonstrations 全部来自 calibration pool；
- target human label 不进入 prompt 或 retrieval；
- 每个 item-method 都保持精确的 5/5 位置平衡；
- split、配置、源代码和 prompt template 哈希与冻结记录一致；
- `selected_method.json` 在 final-test 前已经由 Git 跟踪并提交。

Choices13k 是公开 benchmark，可能出现在模型预训练数据中。实验故意向模型提供
calibration_pool 的人类示例，因此检验的是 in-context behavioral calibration，
而不是干净的 zero-shot 泛化。结果只支持聚合二元风险选择上的人类分布拟合改善，
不能直接证明金融市场 agent、herding、strategy switching 或交易行为已经更像真人。
在进入 LoRA/SFT 或回到内生市场前，应先把 B2 当作已通过 benchmark 的微观决策
组件，而不是完整的人类投资者模型。

## 结果文件

- 正式报告：`outputs/human_benchmark/choices13k_data_driven/evaluation/report.md`
- 完整指标与 bootstrap：`outputs/human_benchmark/choices13k_data_driven/evaluation/summary.json`
- 分组比较：`outputs/human_benchmark/choices13k_data_driven/evaluation/subgroup_comparisons.csv`
- 开发候选比较：`outputs/human_benchmark/choices13k_data_driven/development/candidate_comparison.csv`
- 冻结方法：`outputs/human_benchmark/choices13k_data_driven/selected_method.json`
- API 用量：`outputs/human_benchmark/choices13k_data_driven/api_usage.json`
- 可复现性审计：`outputs/human_benchmark/choices13k_data_driven/evaluation/audit.json`
