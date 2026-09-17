# Choices13k confirmatory experiment 结果

本实验按照预注册协议，在 1,000 条全新 Choices13k `Feedback=False` items 上比较冻结的 P0 与 P1。旧 pilot 的 300 条 calibration/held-out items 全部排除；每个 item/prompt 独立采样 10 次，并严格保持 5 次 A-first、5 次 B-first。

## Primary result

| Prompt | MAE | RMSE | mean JS | Pearson | Spearman | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.2779 | 0.3382 | 0.1159 | 0.4073 | 0.3795 | 0% |
| P1 | 0.2956 | 0.3557 | 0.1302 | 0.4100 | 0.3872 | 0% |

预注册主终点为：

```text
ΔMAE = MAE(P1) - MAE(P0)
     = +0.0177
```

5,000 次 paired item-level bootstrap 的 95% CI 为 **[+0.0059, +0.0297]**。区间完全高于 0，说明 P1 在这批 untouched no-feedback items 上稳定增加了与真人选择分布之间的误差。

因此本实验属于预注册解释规则的 **Case 3**：

> Persona-style human prompting does not provide robust human behavioral alignment.

Pilot 中 P1 的方向性改善没有在 confirmatory population 上复现。当前证据支持进入 data-driven calibration，而不是继续微调一句 persona prompt。

## Subgroup results

所有预注册子组的 ΔMAE 都为正，即 P1 在每个子组中都比 P0 更差。

| Subgroup | Items | P0 MAE | P1 MAE | ΔMAE |
|---|---:|---:|---:|---:|
| Loss | 662 | 0.2588 | 0.2755 | +0.0166 |
| No loss | 338 | 0.3154 | 0.3351 | +0.0197 |
| Ambiguous | 190 | 0.2604 | 0.2796 | +0.0192 |
| Non-ambiguous | 810 | 0.2821 | 0.2994 | +0.0173 |
| Small EV gap | 333 | 0.2800 | 0.2984 | +0.0184 |
| Medium EV gap | 333 | 0.2875 | 0.3027 | +0.0152 |
| Large EV gap | 334 | 0.2663 | 0.2858 | +0.0195 |

这种一致方向说明主结果不是由某一个 loss、ambiguity 或 EV-gap 子组单独造成的。

## Position、risk 与 EV diagnostics

| Metric | P0 | P1 | Human target |
|---|---:|---:|---:|
| First-option rate | 35.86% | 39.97% | — |
| PositionBias | −0.1414 | −0.1003 | 0 |
| BPositionEffect | −0.2828 | −0.2006 | 0 |
| Degenerate item rate | 24.20% | 28.70% | — |
| Riskier-option choice rate | 34.78% | 31.51% | 49.87% |
| Higher-EV choice rate | 56.41% | 56.39% | 60.94% |

P1 减弱了 displayed-second position bias，但增加了 degenerate items，并进一步降低了本就低于真人的 riskier-option choice rate。P1 的 Pearson/Spearman 略高，同时 MAE、RMSE 和 JS 更差，说明排序相关性改善并不等同于人类选择概率校准改善。

Position bootstrap 95% CI：

| Prompt | PositionBias 95% CI | BPositionEffect 95% CI |
|---|---:|---:|
| P0 | [−0.1549, −0.1285] | [−0.3098, −0.2570] |
| P1 | [−0.1128, −0.0876] | [−0.2256, −0.1752] |

## API execution and audit

- Provider/model：`aigc_relay / DeepSeek-V4-Flash`。
- 20,003 次 HTTP attempts：20,000 次成功，3 次失败后重试成功。
- 0 次 JSON repair，最终 invalid rate 为 0%。
- 输入 token 2,380,340；输出 token 139,909。
- 1,000 个 confirmatory items 与旧 pilot 300 items 的交集为 0。
- split hash：`a339149e44014ec1a0ca8404070256c2798973fef5aa27f0aead5b08307ab659`。
- live run 对应冻结 commit：`8fb4de38462971174c3fe3ec96f67ee61f307ab4`。
- 审计通过 config、data、source、prompt、counterbalance、identity 和 secret 检查。

`average_latency_seconds_including_queue` 包含客户端限速队列等待，不能解释为单次 provider 推理延迟。中转站没有提供可核验的价格表，因此不报告成本估算。

## Artifacts

- 主报告：`outputs/human_benchmark/choices13k_confirmatory/evaluation/report.md`
- 完整 summary：`outputs/human_benchmark/choices13k_confirmatory/evaluation/summary.json`
- subgroup comparisons：`outputs/human_benchmark/choices13k_confirmatory/evaluation/subgroup_comparisons.csv`
- bootstrap intervals：`outputs/human_benchmark/choices13k_confirmatory/evaluation/bootstrap_intervals.json`
- API usage：`outputs/human_benchmark/choices13k_confirmatory/api_usage.json`
- 审计：`outputs/human_benchmark/choices13k_confirmatory/evaluation/audit.json`

Choices13k 是公开 benchmark，仍可能存在训练数据污染。本结果只说明轻量 persona prompt 没有可靠降低 aggregate risky-choice gap，不能直接外推到金融 herding、交易频率、strategy switching 或市场宏观真实性。
