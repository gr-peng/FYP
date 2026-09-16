# Memory-market V2 真实 API 流动性筛查结果

运行时间：2026-09-16。源码提交：`de70c5e`。模型：AIGC relay 的
`DeepSeek-V4-Flash`。本轮是预注册的扩展前筛查，不是完整因果实验。

## 做了什么

- 资产改为 MU 与 SNDK。MU 作为 DRAM/HBM/AI memory 上市代理；SNDK
  作为 NAND/flash 相关资产，不把它误称为 DRAM 公司。
- 30 agents，seed 42，内生连续双向竞价，low/high herding，E1-only，
  每个组合 23 个交易日，共 4 个 run、2,760 个 agent-day 订单决策。
- 每日 seeded random arrival，5 agents/批。第一批看到空盘口，后续批次
  看到当日先前批次形成的 best bid、best ask、spread、depth、last trade。
- limit price 限定在当日开盘参考价的 `[0.95 P_t, 1.05 P_t]`。未加入 market
  maker。运行前门槛见 `docs/memory_market_v2_protocol.md`。

## 结果

| Asset | Treatment | Hold | Invalid/rejected | Book visible | Both-side days | Trades | P(follow majority \| signal) | Active-only follow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MU | low | 49.28% | 5.80% | 82.61% | 9/23 | 31 | 47.12% | 96.88% |
| MU | high | 4.64% | 3.04% | 83.33% | 0/23 | 0 | 96.82% | 100.00% |
| SNDK | low | 35.07% | 8.70% | 83.33% | 5/23 | 30 | 64.13% | 99.02% |
| SNDK | high | 4.64% | 4.06% | 83.33% | 0/23 | 0 | 95.76% | 100.00% |

所有 invalid/rejected 都是 portfolio constraint violation；price-band violation
和 API fallback 都是 0。模型看到对手价时，实际买卖报价到对手价的平均距离为
0 bps，即报价不再保守。多数撮合后只剩单边订单，因此完整双边 spread 没有可用
均值；这不是缺失 prompt 字段。

low-herding 的交易集中在少数双边日。MU low 共 31 笔，SNDK low 共 30 笔。
两个 high-herding run 均产生 658 个有效 BUY、0 个 SELL，市场没有对手方，
因此价格始终停在初值。E1 首次可见后没有改变这个结论。

## Herding 解释

按预先指定的总体指标，高 herding 的多数跟随率在两个资产上都明显高于 low
herding，方向符合 manipulation check。这个结果只有一个 seed，不能做显著性或
稳健因果结论。

active-only 指标在 low/high 中都接近 100%，说明差异主要来自 low-herding 更常
HOLD，而不是 active orders 在多数方向上有很大差异。四组的 60 次风格复盘全部
选择 stay，switch rate 仍为 0；这再次说明 switch rate 不适合作为 herding 的主指标。

## 是否扩展

本轮未通过预注册流动性门槛：两个 high-herding run 没有双边日或成交；SNDK low
只有 5 个双边日；MU/SNDK low 的 invalid rate 也超过 5%。因此没有继续调用 API
跑 no-event、E1+E2 或 seeds 43/44。若在当前机制上直接扩展，事件对照仍会被
“无对手盘”混淆。

下一版应把“高 herding 导致单边流动性枯竭”本身作为结果，并另外预注册一个
有背景流动性的价格影响实验。背景流动性需要容量上限，且必须分别报告
agent-agent 与 agent-background 成交，避免强 market maker 冲掉 agent effect。
也可以先加入 treatment-independent、均值为零的 agent-specific private signal，
再做新的独立筛查；该改动必须标记为 post-pilot redesign，不能覆盖本轮结果。

## API 与复现

中转站共记录 3,002 次 HTTP 尝试：3,001 次成功，1 次失败后重试成功，1 次
JSON repair，0 次 fallback，0 次本地缓存命中。输入 4,461,774 tokens，输出
233,611 tokens；平均含排队延迟 6.27 秒，P95 为 12.81 秒。中转站价格规则未归档，
所以不估算费用。

机器可读诊断保存在
`outputs/meta_compute_2026/aigc_relay/memory_market_v2/liquidity_diagnostics.csv`，
审计结果保存在同目录 `evaluation/audit.json`。审计验证 4/4 runs 的输入 hash、
输出 hash、事件时钟、账户/股份守恒、成交量和无未来数据约束。API key 只从本地
忽略的 `.env` 读取，没有写入源码或报告。
