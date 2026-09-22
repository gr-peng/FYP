# Behavioral LLM Market

可复现的 LLM 投资者行为与内生金融市场模拟研究原型。核心问题是：微观决策更接近
真人的 agent，是否会生成更真实的金融市场？截至 2026-09-22，市场核心、早期市场
实验、Choices13k 校准与 CPC18 外部验证均已有结果；金融行为和微观到宏观验证仍在推进。
最新进展与运行状态见[持续研究状态](docs/active_research_status.md)，原始系统设计见
[技术设计文档](docs/TDD.md)。

CPC18 的 175 道外部风险题上，校准 LLM 的 MAE 为 **0.1739**，优于原始 LLM 的
**0.1923**，但统计模型达到 **0.1135**；真实标签相对中性/打乱标签的机制检验未通过。
完整区间、限制与图表见[外部验证报告](docs/cpc18_external_results_20260922.md)。这还不能
证明金融交易行为或市场真实性改善。当前真实模型调用使用已有 AIGC 路由；Jev 尚无真实
性能结果。

## 已实现

- 单资产连续双向竞价（CDA）：价格优先、时间优先、挂单价成交、部分成交。
- DAY 限价单、取消及日终过期；禁止自成交、卖空、透支和重复占用资金/持仓。
- 使用 `Decimal` 做成交和组合核算；只有成交会改变市场价格。
- 可复现的 N 个脚本代理、交易日志、账户快照、配置快照和运行元数据。
- 行为代理、历史回放、滞后群体信号、LLM 接口，以及独立的人类选择校准实验。
- 金融数据按完整场次冻结划分；持续决策和制度匹配的市场验证为后续重点。

## MVP-0 示例运行

需要 Python 3.11+。在本目录执行：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src tests
behavioral-market-demo --config config/mvp0.json --output outputs/demo
```

默认示例是 30 个脚本代理、30 个交易期，随机种子固定为 42。
输出在 `outputs/demo/`，包括 `orders.csv`、`trades.csv`、`market.csv`、
`agents.csv`、`config_snapshot.json` 和 `run_metadata.json`。资金与份额在整个
运行中守恒；日末未成交订单会失效。脚本代理的报价仅用于软件验收，不构成
投资策略或真实市场标定。

## 同步到 GitHub

使用 [scripts/sync.py](scripts/sync.py) 将当前项目提交并同步到
`https://github.com/gr-peng/FYP`（默认使用等价的 SSH remote）。脚本默认目标 branch 为 `main`，`-m/--comment`
是本次本地提交和真正 merge commit 使用的说明；也可以用 `--comment-file` 提供多行
说明。先用 `--no-push` 做本地预检查：

```bash
python scripts/sync.py --no-push \
  --comment "Implement deterministic market core"
python scripts/sync.py --branch research \
  --comment-file /path/to/merge-message.txt
```

脚本不接受 token 参数，不把 token 写入 remote URL，也不打印 Git 凭据；认证交给
本机 Git credential helper 或 SSH agent。新 remote 默认使用 `git@github.com:gr-peng/FYP.git`；已有 HTTPS
remote 不会被静默改写。`git add --all` 后会检查疑似 API key、token、
password、私钥文件和常见凭据文件，发现后停止且不执行 commit/push。真正的远程分支
分叉会用指定 comment 创建 merge commit；可以快进时不会制造多余的 merge commit。
发生冲突时脚本保留现场并退出，不会强制 push 或自动丢弃冲突；解决冲突并执行
`git add` 后重新运行即可继续完成 merge。

## Meta Compute 2026 真实实验

新增真实行情归档、事件时钟、SiliconFlow agent、历史回放和内生 CDA 实验。
完整运行顺序、数据来源、执行假设及结论边界见
[真实实验协议](docs/meta_compute_2026_protocol.md)。首轮采用 30 agents、3 seeds，
规则/Vanilla/低羊群/高羊群四组，内生场景测试 0%、-2%、-5% 基础价值冲击。
配置位于 `config/meta_compute_2026.json`，结果与报告保存在
`outputs/meta_compute_2026/siliconflow/`，不会由同步脚本上传。

## Choices13k 真人行为 benchmark

Choices13k pilot 用公开真人二元风险选择分布检验 LLM 的微观行为差距，并与
CDA 市场模块隔离。实验固定 200 道 calibration、100 道 held-out、四个预注册
prompt、每题 10 次严格 A/B 位置平衡；held-out 只有在 calibration prompt 冻结后
才能运行。完整设计与数据泄漏边界见
[Choices13k pilot 协议](docs/choices13k_pilot_protocol.md)。

Choices13k pilot 已完成。实际运行、API 用量、审计和 held-out 结论见
[Choices13k pilot 结果](docs/choices13k_pilot_results.md)。旧 held-out 已经观察，后续
不再用于选择或调整 P1。

Confirmatory experiment 也已完成：冻结 P0/P1，在排除旧 300 题的 1,000 道全新
no-feedback 题目上进行验证。P1 的 MAE 比 P0 高 0.0177，item-bootstrap 95% CI
为 `[0.0059, 0.0297]`，说明 persona-style prompt 没有稳健改善本任务的人类行为校准。
预注册设计和完整结果分别见
[Choices13k confirmatory 协议](docs/choices13k_confirmatory_protocol.md)与
[Choices13k confirmatory 结果](docs/choices13k_confirmatory_results.md)。离线查看阶段状态：

```bash
python scripts/run_choices13k_confirmatory.py --stage plan
```

运行必须显式指定 `prepare`、`run`、`evaluate` 或 `all`；没有默认执行阶段。

Data-driven calibration experiment 也已完成。开发集从六个 B2 候选中冻结了
`B2_matched_k4`；在 400 道全新 final-test 题目上，B2 的 MAE 为 0.2446，低于
B0 的 0.2837 和 B1 的 0.2979。B2−B0 的配对 MAE 差为 −0.0391，5,000 次
item-bootstrap 95% CI 为 `[−0.0609, −0.0181]`，达到预注册成功条件。设计、
泄漏控制、冻结 split 和执行门控见
[Choices13k data-driven calibration 协议](docs/choices13k_data_driven_protocol.md)，
实际运行、审计和结论见
[Choices13k data-driven calibration 结果](docs/choices13k_data_driven_results.md)。

## 参考项目

参考仓库位于 `third_party/`，保留上游 Git 历史，且不导入我们的 Python 包。
参考概念、复现实验假设和未来阶段分别见
[参考说明](docs/third_party_notes.md)、[架构](docs/architecture.md)、
[实验协议](docs/experiment_protocol.md) 与 [ADR](docs/adr/README.md)。
