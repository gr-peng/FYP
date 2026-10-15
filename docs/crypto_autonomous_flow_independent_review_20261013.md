# 自治有效流生成器独立合成审查（2026-10-13）

结论：`pass_for_synthetic_engineering_only`。当前固定实验规则下未发现需要修改生成器的工程错误。本结论只证明以下合成检查覆盖内的内部一致性，不证明真实市场分布逼真、历史交易所制度正确，项目的真实评价目标仍未完成。

审阅者 `/root/crypto_aux_review_resume` 没有参与被审生成器实现；本轮未编辑其实现，也未导入或自审自己编写的 metrics 模块。未读取任何真实/raw、辅助开发、事件或 human final 数据，未调用 LLM。模板和日期标签均为本审阅脚本构造的合成对象。

## 绑定的实现与证据

- generator SHA256：`97bde524244a04468867adfb170ce7d590a016254a374c5ce067c3df557158c5`。
- effective flow SHA256：`f14d4f8f5076ddaf3463c48a4904640480448e07826f7f9605925fd620bb2f08`。
- perpetual adapter SHA256：`6a31d49e5d284350787e44108f0893f3fecf9cc0a4b8605a5f6bef17ad72d43d`。
- 另外绑定 ledger、order book、matching engine、order、trade 及五份测试文件，完整 SHA 见机器 receipt。测试和独立复算开始、完成及发布前的全部依赖源 SHA 一致。
- receipt：`outputs/crypto_autonomous_flow_20261013/independent_generator_review_v1.json`，SHA256 `336a420316956c40b1a8b6eeea6d54eb8440499e2806197184398ca7eb98ea6d`。
- 成功 oracle、脚本、pytest 元数据和日志，以及三个失败 oracle 尝试的原脚本/日志均保留于 `outputs/crypto_autonomous_flow_20261013/independent_generator_v1/`，receipt 逐文件绑定 SHA。

## 实际覆盖

相关模块的 122 项合成测试通过。另用不调用生成器内部私有函数的整数/Fraction oracle 重放 146 条 receipt：106 条 complete，20 条 missing-side failure，20 条 initialization failure。覆盖 10,714 次抽样、10,820 个定义点、28,801 个动作、10,309 条发布 prints 和 12,418 个物理 fills。

主要矩阵为 M0/M1 × S1/S2 × 0/1000/5000ms × 8 个 seed，每条 80 步，共 96 条。每一步从动作和实际 prints 独立维护 FIFO 订单队列、数量和账户头寸/成本，再核对下一时点、完整模型簿、可见 top500、深度、mark、四账户 equity 与剩余挂单风险；不以 receipt 自报守恒量替代复算。

合成反例还验证：净撤量 -1000 加实际买入 1000，只从初始 ask 5000 扣至 3000；同价多个 fill 只按同一 intent 合并，不跨 intent 合并；后续 partial 风险失败会撤销前面预演成交且不消耗实际 trade ID；第 501 个模型档位不泄露到 top500 路径；半价单位 mid 保留为 bid+ask，同时财务 mark 按固定 half-up 规则取整数。

两组各 20 个固定 seed 的 300 秒失败路径，每条都保留 3,001 点和全部计划步数；缺边时点保留实际侧，随后 undefined 尾部不 carry forward。另核对一条完整 3,000 步 quiet 路径。20 seed 数量在这里由审阅 caller 保留，生成器 API 每次只返回一个 seed，正式 orchestrator 仍须证明完整分母。

独立状态复算使用 own book 和已发布的过去打印；逐次核对 feature_us = decision_us - delay。启动上下文严格要求 51 个 [-5s,0] 网格点、打印 [start-1s,start) 及末点一致性。通过 19 个无效输入/日期/未来接口拒绝反例，并在生成期间禁用 builtins.open 与 Path.open。另进行 1,800 次 parent RNG 状态检查：M0、M1、31/32/33 行池和 fallback 每步都恰好消耗一次 128-bit parent draw，局部 randrange 不影响配对 parent 流。

## 固定假设与尚待证明的部分

被动净价格事件固定在步初映射，按 bid-desc→ask-asc 执行；主动 intent 保留源行顺序。DAY 对全部未成交挂单预留保守风险；IOC 对全部可见 partial fills、双方账户与剩余挂单风险作原子验证，不为即刻取消的不可见余量预留。上述两条由 parent 在真实数据尚未打开时明确，并修订前期提案文字，不是依据真实得分选择的规则。

G 为初始可见簿价格×数量总额，四账户各 G，杠杆 1、费用 0、无补资。Decimal precision80、step-start+1ms 发布、own-mid half-up mark 都是固定实验政策；没有依据宣称它们为历史 venue 制度。缺边后的上一有效财务 mark 被标 stale，不能替代缺失的报价标签。真实原始订单、真实 parent order、未来参考路径、清算、MMR、ADL 和 funding 制度均未在本原型恢复。

row_identity_sha256 只绑定共同训练行身份，不能充当模板内容/来源的完整哈希。完整模板内容、批准训练来源、initial book/51 个过去特征的真实因果性、固定窗口/日期/20 seed 分母，以及七维评分与训练尺度应由后续独立 pipeline 审阅证明。本轮不对该 pipeline、真实目标读取或 metrics 发布批准。

四账户角色单向持仓、费用为零，因此本 oracle 对头寸与 entry cost 的复算是精确 Fraction；一般平仓/反手规则仅在相关现有 ledger 测试中覆盖，本报告没有把它们扩大为已经独立重放。top500 之外模型新增档位与每档一个合成初始 FIFO 订单均是模型机制，不能认作真实深度或真实队列。

## 保留的 oracle 失败尝试

v1/v2 在首条 print 对比处因审阅脚本的 list/tuple 表示不一致失败；v3 主矩阵与重放通过后，新增原子回滚 fixture 错把前置 add 的 actions[0] 当 intent。v4 修复审阅表示与 fixture 索引后通过，产品源码与实验规则均未改。三次失败日志没有覆盖，全部随机器 receipt 保存。

审查没有通过阈值、胜者选择、随机种子当独立交易日的统计宣称，也没有宣称真实模拟效果已经验证。
