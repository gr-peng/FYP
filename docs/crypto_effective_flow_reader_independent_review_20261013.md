# 有效流 reader 的独立工程审阅

2026-10-13。最终版本通过独立合成检查；在配置、reader、CLI 及来源文件提交冻结后，
可执行预定的 **2024-11-01 00:10—00:15 UTC 训练提取冒烟**。本审阅没有读取真实
原始行情、开发期、事件期或真实有效流产物，不提供真实生成或市场真实性结论。

审阅对象是 `experiments/crypto_effective_flow_reader.py`、对应测试、
`scripts/extract_crypto_effective_flow.py` 与固定机器配置。作者停编后，审阅者重新
检查全部时钟、分母及发布边界；没有修改这些实现文件。最终 reader SHA 为
`20c6ef9c0cbee34a0314d47b826eb2c19db822e6ca978a1eff6b545c7b97b792`，
测试 SHA 为 `88193768673505964b3e804e9d277cb8240c96b5415b3df616859e8e59a0cbea`。
CLI、配置、纯核心和执行 cursor 依赖的完整身份见凭据；六个直接核验文件在
最终测试、独立复算及凭据写入前后字节不变。

## 时钟与标签检查

reader 复用未消费消息的 raw-order 屏障；保存观察时点已消耗的 prefix，而不从
pending message 取盘口。有效原始 snapshot，包括完全相同的 duplicate，都建立
reset 边界。三个输入分别在 A、A−1s、A−5s 取得自身观察和完整过去一秒成交，
之后才消耗 [A,A+100ms) 标签；状态年龄在各 signal 自己的时点判断。任何延迟
输入跨 segment、错误计数不一致或上下文不足，都保留排除理由，共同队列不变。

独立脚本在临时 Git 仓库生成并完整检查 ZIP/GZIP CRC/EOF，用 9,004 条合成簿
消息和 27,000 条 CSV 打印实际经过提交来源、metadata、audit 与 raw 字节门禁。
900ms 迟到 TS 在固定冒烟内形成 raw-order 屏障；随后较早 TS 的消息不能绕过。
期望值由独立整数 dictionary 前缀状态、严格 bisect-left 选择和 Fraction 算术
产生，不用生产 cursor、特征函数或分解函数生成 oracle。

复算覆盖全部 3,000 个输出候选，包括 2,949 个共同 eligible、8,940 个有效延迟
特征向量、9,000 次打印标签唯一分配和 8,997 个稀疏非零残量/消费价格。
两端和所有延迟信号的前缀时钟、L1、边缘、档数逐项一致。reset 排除理由出现
次数为跨标签端点 1、1 秒延迟跨 segment 10、5 秒延迟跨 segment 50；理由重叠，
不能将这些次数相加作为排除候选数。实际共同排除为 51 行。

全部打印保留原始行序及半开分配。两端共同已知绝对价格域内 q0/q1/C 和相对
模板一致；稀疏省略的未变价位仍须由 hash 绑定的 raw 前缀重建，不能补成零。
被延迟资格排除的候选可以保留分解标签，所以未来拟合必须显式要求
`eligible=true`，不能仅凭 `template` 存在或 row-ID digest 使用记录。

## 发现与修复

独立审阅发现了两处发布身份缺口；初版反例保留在
`outputs/crypto_effective_flow_reader_review_20261013/review_v1/initial_findings_receipt.json`，
SHA `9e1ff43da598a8c59d7e951951cf843160d3b57813945c88fc0abe5e0f3c419d`。

1. 初版在 input audit 期间改写配置后，prepare-only 可以发布与冻结 map 不一致
   的配置 SHA；实际运行直到结束才发现变化。最终版本绑定已解析字节 hash，并在
   parser/prepared 前重验配置、全部冻结来源和所有输入凭据。独立 run/prepare
   反例现在都在打开市场 parser 前失败。
2. 中间版本在 `_write_day` 计算 gzip hash 后若输出被改写，仍发布 complete，
   输出 SHA 已失效。最终版本在全部结束来源检查后、`outputs_published=true` 前
   重新比对每个实际输出 hash。独立改写反例现在留下 failed manifest，保留已写
   输出但不发布。

另外独立故障注入验证了读取前 raw 字节变化、读取前撤销 audit、写入后撤销
audit、写入后配置变化均失败。共七个故障案例通过，actual prepare-only 也完成
验证且没有市场解析。有限 hash fence 提供检查时点的身份绑定；未来消费者仍
须重新核验 complete master、实际输出字节及当前来源凭据。

## 验证凭据与范围

审阅者执行 `reader + pure core` 测试：**60 passed**；相关 Ruff 检查通过。
独立端到端及故障注入脚本通过。最终凭据为
`outputs/crypto_effective_flow_reader_review_20261013/review_v2/review_receipt_v2.json`，
SHA `74b93e364509cd32c1c8669682d3848d7d89592449036d33c8a975ff04c9c69e`；
同目录包含脚本、结果、前后源码身份与命令记录。完整合成 audit 凭据只是临时
生成字节的工程证据，不是交易所来源证明。

CSV 成交时点可见仍是明示的理想假设，服务端 TS/CTS 代理不是已测量的客户
接收时钟。q1−q0+C 只能给有效净残量，不恢复 gross 挂撤单、父订单或 L3 队列。
本次结论只覆盖 train-only 提取组件；真实提取结果、模板拟合、生成器、七维
评分和留出验证都须独立完成并检查，不能用本审阅替代。
