# 固定分位数与分层汇总：独立审阅

日期：2026-10-13。独立审阅者：`crypto_execution_review_resume`。结论：**新汇总模块通过工程及统计汇总审阅；提交冻结后，可消费已经完成并经独立复核的一小时训练期执行产物。** 本次审阅尚未通过该新模块读取真实逐笔或计算真实尾部分位数。

固定汇总协议已在首次真实冒烟解析前提交于 `a2272add378418dbb2229076995793a995c8fb18`。作者停止修改后，本次捕获以下hash；测试和独立oracle结束时再次核对，字节身份相同。实现没有被审阅者修改，文件在审阅时尚未提交，真实运行仍需满足自身Git冻结门控。

| 文件 | SHA256 |
| --- | --- |
| `experiments/crypto_execution_summary.py` | `5a3b4ade6d50d3ba4d4af8fc4015c64c9b561390085953c00e26ba7c698eb266` |
| `scripts/summarize_crypto_execution.py` | `df3b30d5d574f2963c8f400a0a500c9bc20681b0f70e0a078ac76bb9f32fbf3f` |
| `tests/test_crypto_execution_summary.py` | `cfdd22c79794c7f3006dba2fce0a097bea01dabc90f7bcd5febe21f3c0bec528` |
| `docs/crypto_execution_summary_protocol_20261013.md` | `a5e5ddf08225c38f04d2741ad66ba6bb9669fb019ab163ff0bf67e5e0d944ebf` |

## 独立验证

`.venv/bin/python -m pytest -q tests/test_crypto_execution_summary.py`：本次运行一次，31项通过，0项失败。相关三个Python文件的Ruff检查通过，`git diff --check`通过。

另用固定种子20261013执行独立oracle：

- 2000组有符号、重复值、空样本的分位数，普通Python排序后按整数秩 `ceil(n×百分比/100)−1` 直接取值，没有调用NumPy分位数生成期望值。median/P90/P99均与模块相同，均值用 `math.fsum` 复算相符。
- 200个随机生成的日表，共61,995个print-delay行，独立列表筛选复算候选数、全部排除原因、可见覆盖、共同样本、方向/年龄/数量固定层及成对误差。52,000个组别统计叶子逐项一致。
- 随机表包括跨L1、不足深度、陈旧、未知状态、空层及超过浮点整数精度的数量；整数数量分层与覆盖求和未转换为有损float。

该oracle只使用生成的合成表，验证聚合函数而不是现实盘口。持久证据为 `outputs/crypto_execution_review_20261013/summary_review_v1/independent_summary_review_receipt_v1.json`，SHA256为 `4352747d56c9b8444e690f8c606fc22a14ec75dfb81d29e70340c1fcef506673`；独立oracle脚本同时保留，SHA256为 `ccfd76591708e319d4aa96c482e50d28893ec735b13e50776f1c1c9ecaa27332`。

## 门控与输出判断

程序在文件消费前拒绝开发、事件检验和其他split，仅接受既有固定配置。消费者核对完整/已发布执行manifest、固定日期及窗口、执行commit/当前源码/配置身份、prepared receipt、元数据和完整审计授权；在读取逐笔值前核对每个自然日全部三份产物的实际hash、行数和完整逐笔schema。相对产物路径按项目root解析，再限制在该运行目录指定产物内。

逐笔消费验证五延迟的完整顺序、相同身份、原行号/成交时间顺序、唯一成交ID、固定分层、共同交集和残差可用性。Parquet批边界不需要恰好按五行对齐。空层保留完整候选/排除记录、零样本及缺失误差；部分覆盖不会被纳入完整价格评分。输入/源码处理中变更以及最终输出已存在的竞态均令当前发布失败。

分位数是保留的float64 bps残差在每个日/延迟/同样本组/固定层中的精确经验CDF逆，不抽样、不使用sketch、不插值。它不是无限精度价格有理数的分位数。模块分别给逐笔加权误差和完整自然日的等权日均比较；一小时smoke明确禁用完整日比较，没有逐笔IID显著性或置信区间。

消费者继承经过认证的执行producer原始字节hash，**不会重新打开或rehash原始档案**；当前元数据/审计receipt仍须相同。它验证逐笔配对、分层和残差可用性，不从原始盘口独立恢复状态，也不独立重算整数sweep成本。真实smoke的这两项由另一份独立复核记录覆盖，不能把本模块的输入hash核对称为新的全档案CRC审计。

生产代码与协议相符，未发现阻止冻结后运行smoke汇总的问题。真实汇总完成后仍应独立复算所有层的分母和经验分位数，报告窗口限制；之后才运行预先指定完整训练日。统计汇总正确不会补齐整张主动订单、时钟偏差、不可见流动性、被动队列或代理反馈的识别缺口。
