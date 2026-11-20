# TRAIN 目标构建流水线的独立审阅

2026-10-15。最终版本通过独立合成端到端和故障检查。生产源码、CLI、配置及全部
运行依赖提交冻结后，可以执行预定的原始 November 1—3 TRAIN 目标采集。本次
没有读取真实目标文件或开发、事件、人类评测数据，也没有执行自主行情生成。

最终 `experiments/crypto_autonomous_target_pipeline.py` SHA 为
`3c0d45d6a4bf092e30637f8910f9ad4a75384d24fee60d1091dc3f23d1569b87`；
测试 SHA 为 `eba52fb9495a321a4fb68097389d2894ca088510611652bcef884d4b610ac2ac`。
CLI SHA 为 `1b8e65f4355f4ae33791b229aa827c3da182bbced297ae3a14acf87b47834a54`。
最终依赖闭包包含修订后的模板流水线，SHA 为
`9c9c18b353b0efddb5690739d5721e83b064f1e9def6f5396f03682c7ae03699`。
完整来源 map 在最终凭据内；独立执行前后全部字节一致。

## 实际门禁与候选分母

审阅者建立实际临时 Git 仓库并提交源码，生成三日各 1,800 条簿消息及 45 条成交
的 ZIP/GZIP 文件，完整验证 CRC/EOF，生成绑定实际字节的合成 metadata/audit。
主流程没有替换 Git、原始输入门禁、市场 parser 或全天 collector：它真实执行
三个完整 UTC 日的 100ms 网格，然后完成全部产物与固定尺度计算。

合成消息有意只在各日最初 15 分钟更新盘口，剩余时间让盘口变旧，因此全部
**4,320 分钟、72 初始化、864 价格块候选**仍被保存，其中分别有 42、3、6 个
完整候选。这验证了稀疏有效内容不会缩减候选分母，不代表真实数据的缺失率。
三份初始化文件只包含过去盘口、打印、特征和前缀证据，标签与价格块分别输出。
全部 10 个产物的实际字节 hash、序列化字段、Fraction 表示和完整候选计数一致。

审阅者用单独的 `math.log1p` 转换、均值和平方偏差计算核对了七个总体标准差；
训练候选数保持 4,320，完整标签数为 42。实际 prepare-only 执行全部来源门禁，
在禁止 parser 和拟合回调的条件下成功完成。

这些合成 audit 是人工构造文件的完整性测试凭据，不能当作交易所来源证明。
临时文件在测试后清理；生成脚本、运行 manifest、各失败 manifest、源码身份及
核验结果均留存在审阅输出目录，可重建合成实验。

## 发现与修复

独立审阅先复现了落盘 source-preflight 与已验证返回状态不一致的问题：在
preflight 返回后将其 manifest 改为 failed 并替换 inputs，旧 builder 仍会 hash
变更后的字节并发布 complete。修复后，对两份来源凭据使用相同读取字节进行
精确预期状态比较和 hash 绑定，并在耗时输入检查后、每一日 parser 前和最终
发布前重验引用。初始反例保留在 `review_v1`。

随后团队补查了 builder 自身的 prepared receipt。原版本在首次 parser 前、
prepare-only 写入后以及第一日处理中被改写时，仍会继续或发布 complete。
三例独立复现保留在 `review_v2/own_prepared_findings_receipt.json`。最终版本
写入后立即核对精确预期字节，master 保存 `prepared_receipt_sha256`；prepared
自身不包含自引用 hash，并在所有分支的当前与最终检查中核验该引用。

最终独立脚本通过 **19 个故障案例**：来源 preflight 两文件的普通/prepare-only
变更、长输入检查期间变更、读取前 raw/audit/配置/源码变化、写入后 audit/输出/
preflight 变化、缺失三类固定候选，以及自身 prepared 的两种读取前变更、第一日
中变更和全部写入后变更。读取前问题都未进入 parser；第一日中自身 prepared
变更阻止了第二日 parser；所有失败保留 failed/unpublished 结果。另验证了三个
关闭 split 在创建输出目录前拒绝，以及既有输出不会被覆盖。故障实验复用了
第一次实际 collector 产生的合成日结果以避免重复全天网格；来源门禁仍实际执行。

## 凭据与结论范围

审阅者执行作者测试：**22 passed**；对应 Ruff 检查通过。最终独立凭据为
`outputs/crypto_autonomous_target_pipeline_review_20261015/review_v3/review_receipt.json`；
同目录包含完整脚本、日志、source map 和命令记录。旧版本的局部通过凭据与
后续发现均保留，最终判断以 `review_v3` 为准。

本审阅支持固定 TRAIN 目标、过去初始化和尺度构建的工程流程。纯桥接器的
时钟与数值审阅另见 `crypto_autonomous_targets_independent_review_20261015.md`。
真实采集产物仍须在消费前核验完整 master、prepared、来源凭据及实际输出
字节，并独立复算。模板拟合、真实自主生成和留出评估不由本审阅替代。
