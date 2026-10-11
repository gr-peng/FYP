# 辅助周活动实验：独立设计审查

日期：2026-10-11。当前状态：**代码与纯合成独立检查已通过，见末节冻结补记；真实训练冻结产物尚待独立核验，不能据此开启开发行情或宣布模拟完成。** 本审查读取设计文档、代码、既有 November v1 证据及辅助数据配置，并运行纯合成检查；没有读取 October 辅助真实训练或 development 行情、特征、分布或模型分数，也没有读取 November event_test 市场统计。

审查对象为 `docs/crypto_activity_next_model_20261011.md` 的五模型、三延迟和两阶段方案；训练日固定为 October 14—20，开发日为 October 21—27。方法选择参考过 November 结果，故这是按参数信息顺序向后的历史开发检验，不是研究者在当时实施的前瞻试验。

## 设计判断与冻结前必须消除的歧义

N0/N1 与 H0/H1 的交叉比较能够分别检查分布规格和盘口输入；Hnoise 是额外参数的负对照。保留全部 15 个对象、共同样本、三个延迟、所有七日和失败结果，是这轮能够被反驳的关键。没有必要在这轮同时增加长滞后、交互项或搜索正则。

1. **不把“稳定”留给结果出来后解释。** 当前文档没有预注册最小实际效应或等价容差，这与只报告诊断相容。协议应明确本轮没有“真实模拟通过”标签，也不基于某个最漂亮的延迟自动选胜者。如确需一个候选进入后续实验的二元规则，须在开发访问前另行冻结阈值、逐日一致性及多个比较的处理方式；不能用这七日事后估计门槛再宣布通过。
2. **Hnoise 不是可用于显著性检验的零分布。** 四维独立噪声只控制额外特征数量，不能匹配盘口特征的自相关、边际分布或有效自由度。单种子结果不能给 p 值，噪声偶然获胜也不单独证明泄漏。固定 seed、hash/PRNG/正态转换版本、特征序号及 UTC signal bucket；要求按行重排、分批、重复运行时同一信号仍得到完全相同的噪声。同一延迟内五模型不得因噪声构造而改变样本。
3. **分布结构获益不等于机制识别。** H0 比 N0 增加独立的零门控及参数；H0−N0 支持该统计规格更适合这批条件计数，不能单独识别真实市场中存在两个独立交易机制。H1−H0 才是同一分布族内的盘口增量比较。
4. **冻结完整预测定义。** 中位数使用 `inf{k: F(k) >= 0.5}`；均值 MAE 与中位数 MAE 分列。固定计数组用逐样本 CDF/SF 积分后平均，`51+` 使用尾概率而非有限求和。不得用已实现最大计数裁剪预测分布，或根据开发标签调整门控截距。

## Hurdle 实现的数学验收

令 `g=P(Y>0)`、`q=f_NB(0)`、`mu` 为截断前 NB2 均值。必须验证：

```text
P(0)=1-g
P(k)=g*f_NB(k)/(1-q), k>=1
E[Y]=g*mu/(1-q)
F(k)=1-g+g*(F_NB(k)-q)/(1-q), k>=1
```

`g*mu` 是错误的完整均值；删零后普通 NB 拟合也是错误的正计数似然。正计数部分的梯度必须包含截断归一项。若 `l0=log(q)`，对任一参数 θ：

```text
d log f_truncated(k)/dθ
  = d log f_NB(k)/dθ + q/(1-q) * d l0/dθ
```

目标为**全体 N 行平均**完整负 log likelihood，加 `lambda/2` 乘两个子模型的斜率平方和；两个子模型分别优化时仍保持同一 N 分母。改变零样本比例的合成夹具应能发现误用 `N_positive` 的实现。两部分都使用全部训练共同样本计算的同一组标准化统计，截距与 alpha 不惩罚。

必须测试概率归一、上述均值/CDF、正计数 score 加二元 score 等于完整 score、NB2 对照的同分解、数值梯度与解析梯度、alpha 边界投影符号、保存再加载预测不变。还应覆盖很小 `mu`、极端门控概率、厚尾和零样本；采用稳定的 `log(1-exp(l0))`/`log1p`/`expm1` 形式，不能把舍入成零的正概率悄悄裁剪成另一模型。全零或全正训练时，未惩罚 logistic 截距没有有限内点 MLE，应明确失败或采用**事前冻结**的处理，不能隐蔽回退。

## 训练与评价的独立边界

- fit 入口的 reader 包装层必须在读取文件之前拒绝 development、November 日及未知 split；不能只靠调用者“保证不会传错”。训练时只验证训练输入、训练 features 和各自当前成功审计。辅助 development 的完整性审计不等于开放其市场统计。
- 五模型逐个共享同一个已冻结的 sample frame/hash。最新成交秒截至原始 signal available-at；额外延迟期间不刷新。信号、活动和目标在同一 UTC 日与连续 joint run；目标秒唯一。任何模型不得用不同深度筛选获得更容易的评价样本。
- 冻结包必须包含 15 个对象、完整特征顺序、训练归一化、所有参数、固定噪声、训练样本及输入 hash、源码/协议/配置 hash、优化状态和边界信息。缺失对象、非有限参数、非正 scale 或失败拟合不能发布为 complete。
- 模型内容 hash 应先写入独立 manifest/审查 receipt，评价明确绑定这个已冻结 hash；仅由模型文件自报自己的身份不够。评价需在打开第一个开发特征之前完成全部结构/hash 校验，禁止部分检查通过就先评分。评价过程发生源码或协议变化应拒绝成功发布。
- evaluate 只加载冻结参数；将所有 fit 和 optimizer 替换成报错函数，合成评价仍须通过。不得计算开发归一化、alpha、训练分位数或任何标签驱动校准。不可覆盖已存在的模型包/结果；中途失败不能保留冒充本轮成功的旧 receipt。
- 在首次允许消费开发市场信息前记录访问状态；所有七日、全部固定模型及延迟统一完成后报告。失败日期保留，不能按结果删除；修改模型后这个开发周不再是该新版本的未见评价。

## 完成度上限

这轮若数值实现、隔离和复算通过，可完成“已知状态下下一秒成交计数分布”的组件诊断。改善零比例但损害完整 log score、盘口增量随日期/延迟变号、或与噪声对照同量级，均是需要保留的反例。

即便所有描述指标改善，也没有验证自主生成的价格/深度联合路径、限价单排队、真实成交执行或代理干预的因果冲击，更没有验证真人行为。100 ms 聚合 L2 加逐笔成交的可识别边界没有因更换 NB2 分布族而改变；整体目标继续未完成。

## 已完成的代码与纯合成审查（冻结补记）

本节更新前述“实现待核验”状态：代码与纯合成层面的独立检查已经通过，核验内容与提交 `1f4211f` 一致；**尚未审阅辅助真实训练模型包，因此本节不授权开放开发行情**。本轮没有读取真实 train/development 数据或 November event_test 市场数据。

- 最终核心 33 项、Pipeline 22 项测试合计 **55 项通过，9.48 秒**。测试包括 full-N 正则与截断似然、PMF/均值/中位数、完整模型包、训练门控、评价中禁用 fit/optimizer、不可覆盖发布、receipt 替换、重审与 UTC 时序。
- 独立 SciPy/有限差分检查：200 组截断 NB2 与 logistic 梯度；截断梯度最大绝对差 `1.7785239947443188e-9`。1,200 组中位数与计数组概率检查通过；极小均值下概率质量误差不超过 `1.1102230246251565e-16`。
- mpmath 70 位参考：45 组边界案例，`alpha∈{1e-4,1,1000}`、`eta∈{-35,-15,0,15,35}`、`y∈{1,3,1000}`；目标及梯度最大绝对差均为 `1.3096723705530167e-10`。
- 独立暴力时间 oracle：420 个随机场景、8,998 个有效样本全部吻合，包括已知成交选择、共同延迟、断档/切段、目标去重和单位；噪声在行重排、分批和重复 signal 时精确不变。

审查期间发现的三个 Pipeline 缺口已修复并有测试：reader 前后绑定同一审计字节；开发导出完成记录绑定全部 14 个逐文件 receipt 并在评价前后核验；开放前验证显式 UTC 时间顺序。此外，训练 raw/features/receipt 在冻结模型使用前重验。

后续核心改动增加正计数均值报告，并以 `logsumexp(log_mean)-logsumexp(log_positive_probability)` 计算总体正计数混合均值，避免概率下溢；最终 55 项测试覆盖 `eta=700, gate=-1000` 的反例。这与把各行条件均值不加权平均后冒充总体正计数均值不同。

追溯说明：开发中的首轮 oracle 没有记录运行开始时的整文件 hash，不能倒填当前身份。保留的 `independent_code_review_v1.json` 如实记录了这一限制；最终冻结时另行重跑三份 oracle，逐次捕获并核对前后源码 hash，形成 `outputs/crypto_activity_aux_20261011/independent_code_review_v2.json`。最终 oracle 与 55 项测试绑定同一份源码，补齐该缺口，没有修改旧记录，也没有进行参数或数据搜索。

最终机器凭证仍仅是 `independent_auxiliary_activity_code_review`，不是允许开放开发集的训练冻结审查凭证。后者还须独立核验完整真实训练模型包、参数、输入与样本身份。

以下为最终检查前后均一致、且逐项核对提交 `1f4211f` 的准确身份；全部 hash 也保存在 v2 机器凭证中。

| 文件 | SHA256 |
| --- | --- |
| `experiments/crypto_activity_aux.py` | `022f2bef695e1532aa4ec7a9b3d982f7658e781de69ffa6a924d1f525bc46836` |
| `experiments/crypto_activity_aux_pipeline.py` | `4a27a203ab7cd0b83ba0c5b0ca2e3163b65fc21819d9d454b8224c680c646581` |
| `experiments/crypto_activity_model.py` | `dd196f4c56835c1262864ce0e656fbb0e91daada908d1117710e7fb825c5c0af` |
| `experiments/crypto_features.py` | `748efd70ad6b43e9286c50ee483621eca956275c307ca76b631682f119228d03` |
| `experiments/crypto_profile.py` | `3640c6a9d864f9937ce68b168ae6247c7dda1e8b42f06d3536f96d9a09b7fac0` |
| `experiments/bybit_reconstruction.py` | `06ba5bc7739690e35b723b07037f9ce375cdc36816e41cb71530166b2d253094` |
| `experiments/bybit_trades.py` | `842e992937d67e8178d894da15338f077f6e6fd4695461a814f40bb0e7ec9200` |
| `src/behavioral_market/data/bybit_book.py` | `e884a10ccb5a38d585d05787b02b180aebad5804276323767be3348980890c9b` |
| `src/behavioral_market/data/archive.py` | `5be97fffd84042b717a771673eb5de2e7f7e99203e9909ec76887f2c1fc4f44e` |
| `scripts/run_crypto_activity_aux.py` | `8c60ceb3d0aeb63050182b13ab3dd15eb2a93de7822f582156ea3f519ff71995` |
| `scripts/audit_crypto_microstructure.py` | `57c86cc85041365c63c22c00fcb075bb9392a3d7251679b8fa30f5e6275bf77b` |
| `config/crypto_activity_aux_20261011.json` | `2ae7b9303927e69a1ff11d653a06d87ff1cf9df50c1ee63bf12f0fab230b1334` |
| `config/crypto_microstructure_aux_20261011.json` | `ed325ecb674361c72ad55af5b9b2c35728c2f6bce222ede3584e2b26c2aff52d` |
| `tests/test_crypto_activity_aux.py` | `8486db7ac3968b2c1f9df5d0e604ba6a65cb2f2f51662dfe0d8be510a40540db` |
| `tests/test_crypto_activity_aux_pipeline.py` | `3241349b11abd6a70db6d8373c67b60ce3acbfba4871640648b471587f8e5158` |

机器凭证 v2 文件 SHA256：`3f4b83e8957c590255fad61c282b07849da87e9226d40ec5558c82a515d6dea0`。
