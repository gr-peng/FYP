# 自主原型纯评分器的独立审阅

2026-10-13。`experiments/crypto_autonomous_metrics.py` 与对应测试通过独立合成
审阅，未发现阻塞。此结论只覆盖纯评分算术和失败分母；没有读取真实行情、
真实拟合尺度、M0/M1 路径、开发期或事件期结果。来源桥接和真实评分还须
分别实施、冻结和复核。

作者停编后审阅者运行检查，源码与测试前后 SHA 不变：

| 文件 | SHA-256 |
| --- | --- |
| `experiments/crypto_autonomous_metrics.py` | `e5f3896057c04706c54332c3384ca44c5a4329755b141a1f7b784a97922a1f55` |
| `tests/test_crypto_autonomous_metrics.py` | `4325851a7a3d079ef49c0298088b9c8e8db7d91c75720c7b92bbd91716688562` |

## 指标与边界

七维顺序固定为 log 中间价收益 bps、实际成交 BTC 量、实际发布打印数、平均
价差 bps、平均买/卖侧可见 5bps BTC 深度和成交数量压力。中间价保留半个
价格存储单位；量和打印计数不由拟议意图数代替。深度先用整数/有理数比较
包含恰好 5bps 边界的价位，只计 supplied top-500 的可见数量；两侧是否覆盖
完整 5bps 带另报，不补未见深度。一个分钟块以 600 个 100ms 左端点等权
平均价差和深度，收益另取终点，终点不进入这些均值。

固定变换对 indices 1..5 作 `log1p`，收益与压力保持原值，不中心化。拟合
入口要求 11 月 1—3 日全部 4,320 个 UTC 分钟候选，不允许改日、删候选或
从开发/事件拟合。缺失标签保留候选分母，尺度只用完整标签的 population
standard deviation (`ddof=0`)；零或非有限尺度使拟合失败，不删除维度。

`BlockSummary.complete` 表示声称七维已完整填入且没有 issue，并非数值来源
证明。声称已知却非法或非有限的数字会在变换/拟合时报错，或阻止 joint score
完成，不能先把它们当作缺失而静默剔除。未来来源 bridge 必须独立证明每个
候选来自批准的分钟和完整原始可见视图。

评分固定 20 个期望 seed；公平 energy score 的 pair 项为排除对角的
U-statistic，ordered 分母是 `2*K*(K-1)`。五个分钟块等权，任何 main block
缺少种子或有效指标都不能产生完整窗口主分数。全部期望路径和 100 个
path-block 分母保留，失败不填任意 penalty。price-only P 只提供五个收益，
不能进入七维主评分；已知的单维价格诊断可以单独可评。

每维 90% 区间使用固定 `inverted_cdf` 原始值秩；20 个数的下/上端点分别
是第 1/19 个顺序统计量，包含边界。不完整 seed 组不能按可用子集换分母。
程序没有整体日期聚合、通过阈值或胜者选择；seed 和分钟块不成为独立交易日。

## 独立复算

审阅者运行 **33 项测试通过**，相关 Ruff 检查通过。另一个脚本不用生产
指标/变换/ES helper 产生期望值，采用 Fraction 的精确中间价、5bps 边界和
均值，Decimal80 log-return、Decimal70 population variance 与 Euclidean norm
ordered pair 求和复算：

- 103 个完整分钟块及其 600 左端点、可见深度/coverage，3 个极端半单位收益
  例子和 10 个无效/缺失块；极小收益单独用严格精度，不能舍入成零。
- 4,320 个向量变换；完整和保留 117 个缺失标签的两组拟合，complete 数分别
  为 4,320 与 4,203，候选分母始终 4,320。
- 401 个一维/七维 fair ES 例子，包括 `9/19` 手算例；独立 ordered pair
  公式与生产 unordered pair 实现一致。
- 3 个完整 20×5 窗口的 300 个 path-block、105 个手排序七维区间、3 个 P
  窗口及 12 个缺少/重复/额外/全失败 seed 组。开发/事件/未知 split 在访问
  标签迭代器之前拒绝。

最终凭据：
`outputs/crypto_autonomous_metrics_review_20261013/review_v1/review_receipt_v1.json`，
SHA `59ca2d7ec2f6e6756d3293c06f6e167a46f1c8705cef0c0ec66991f0b469d50b`。
同目录保存独立脚本、复算结果、测试命令、runtime 与前后源码 hash。本阶段
没有市场真实性、训练实际得分或留出泛化结论。
