# 有效聚合流纯核心：独立工程审阅

日期：2026-10-13。独立审阅者：`crypto_execution_review_resume`。结论：**修复端点一致性、抽样随机流及窄接口之后，纯分解/模板池组件通过本次工程审阅。** 真实提取reader仍须另审，尚不能凭本组件宣称端点已来自真实已消费原始前缀，或延迟特征已经过真实时钟证明。未读取真实行情、辅助、开发或事件检验数据，没有修改实现源码。

## 最终身份与工程检查

作者停止编辑后捕获源码与测试hash，独立测试及oracle结束后再次核对相同：

| 文件 | SHA256 |
| --- | --- |
| `experiments/crypto_effective_flow.py` | `f14d4f8f5076ddaf3463c48a4904640480448e07826f7f9605925fd620bb2f08` |
| `tests/test_crypto_effective_flow.py` | `4e73b52e3f19d4e8e77049397a6bbe2019778877e62848a64a254f4b06f63126` |

`.venv/bin/python -m pytest -q tests/test_crypto_effective_flow.py`：最终38项通过，0项失败。相关Ruff及 `git diff --check` 通过。计划文档在审阅期间由root随 `a76e789` 更新了实施状态，已核对差异只有状态与已完成组件说明，没有变更本次分解/分格规范；没有把文档hash变化误称为执行源码未变。

## 独立审阅发现并修复的问题

初版26项测试虽通过，独立反例发现三个端点一致性缺口：相同原始message index却改变盘口会制造有效新增；同segment ID可以改变snapshot起点；同segment的累计发布TS可以下降而仍合格。还发现相同96行总池与32行条件池使用 `randrange()` 时，在相同种子5下消耗不同父随机状态，不能保证预定共同随机流。最后，`TrainingRow` 没有验证template类型，带合法row ID、未来价格和历史路径的可变 `SimpleNamespace` 可以进入fit/select，突破窄接口。

root修复后，同prefix的全部状态/来源时钟必须一致，同segment起点固定且累计TS不下降，segment/error counters不回退，有效message index为正，CTS/snapshot时钟保留毫秒精度。endpoint、feature、template和pool入口要求准确的不可变类型。pool构造重新验证批准且唯一有序的共同cohort、零延迟精确spread中位数、三个延迟全部18格及row digest。

抽样现在每步固定消耗父 `rng.getrandbits(128)`，再以该seed建立局部 `Random`，对所选池作 `randrange()`；各池局部拒绝采样不再改变父随机流。这个具体算法和Python runtime仍须写入未来冻结模型/运行receipt。

## 独立合成oracle结果

- 1200对生成的100ms端点，独立字典按绝对价格求共同可见域，复算53,588个有效价位、29,006个打印归属/对齐及4800个端点截断统计，逐项一致。
- 每个有效价位另验证：给最小新增和移除同时加任意相同正量，端点及打印证据不变。因此净剩余分解不能识别gross新增/撤量。共53,588个代数检查。
- 20个相邻半开区间含1204笔生成打印，包含精确边界，全部恰好分配一次；原始同时间戳行序保留。
- 150个随机有效模板池，独立计算精确 `Fraction` 中位数、三个延迟的18格划分及固定32行fallback；6000次父draw、12,000次整行选择与独立局部随机期望一致。M0/M1父随机状态始终相同，联合行没有拆开重组。
- 28个时钟、类型、日期与池篡改反例均被拒绝，合法无新消息的后续观察仍被接受。端点、证据及模板中的输入对象保持不变。

打印价格只留在分解证据，合成IOC意图只保留方向与大小；共同域之外的打印仍留在打印/意图总量和对齐标签，未被静默删除。有效增减按该侧起点最优价的相对整数偏移保存，不保存生成用的未来绝对盘口或收益。打印是否处于两个CTS之间另作诊断；观察者网格的打印标签与引擎跨度可能不一致，不能把本净余额映射叫作真实挂撤单恢复。

## 尚缺的真实端点证明

`BookObservation` 只能证明所给字段内部一致；它没有独立看过原始档案。未来reader必须先核验冻结配置、源码、当前完整审计及实际raw字节hash，然后从已消费原始订阅前缀拷贝状态，不能使用尚未可用pending消息的盘口，也不能用按秒特征冒充100ms端点。要保留原始prefix index、segment、CTS、累计TS/available和invalid计数，并在有效/无效/reset边界证明其一致性。

`TrainingRow` 只存三个延迟的标量特征，不能自己证明它们来自 `O(A)`、`O(A−1s)`、`O(A−5s)`，还是误取终点标签。reader须保留对应观察时刻与来源、past-print窗口及共同cohort资格。5秒信息延迟的年龄应相对实际滞后观察时刻计算，不能把过去合法观测直接标成当前时刻的5秒陈旧状态；这需要独立时钟/反例测试。

`row_identity_sha256` 只覆盖row IDs，不能代替完整template内容hash。未来提取与模型产物须绑定全部内容、来源、执行源码、随机算法、runtime和失败发布门控；真实数据消费与自主生成仍待这些reader/runner审阅和完整冻结。本次没有评价实际模板池分布、M0/M1生成分数或任何留出结果。

初版发现凭证为 `outputs/crypto_effective_flow_review_20261013/review_v1/review_receipt_v1.json`，SHA256 `7780f4b6b4f57f39d7120d008cd50e7ec42594316cbb7f952975d508f5806256`。最终凭证为 `outputs/crypto_effective_flow_review_20261013/review_v2/review_receipt_v2.json`，SHA256 `5d96bf09a977a5720a81f214bba3e2dcf4e39f835daee48ff98d29cdf20645e3`；目录保留两份实际独立oracle脚本及前后身份。工程检查通过没有补齐真实gross流、整单结构、隐藏队列或反事实他人反应。
