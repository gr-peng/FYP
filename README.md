# Behavioral LLM Market

可复现的 LLM 投资者行为与内生金融市场模拟研究原型。研究目标和后续阶段见
[技术设计文档](docs/TDD.md)。项目包含 **MVP-0：确定性单资产市场核心**
和基于真实行情、事件新闻及 SiliconFlow API 的 Meta Compute 2026 首轮实验。
首轮实验仍在进行中；已完成范围和结论边界见[实验状态](docs/meta_compute_2026_status.md)。

## 已实现

- 单资产连续双向竞价（CDA）：价格优先、时间优先、挂单价成交、部分成交。
- DAY 限价单、取消及日终过期；禁止自成交、卖空、透支和重复占用资金/持仓。
- 使用 `Decimal` 做成交和组合核算；只有成交会改变市场价格。
- 可复现的 N 个脚本代理、交易日志、账户快照、配置快照和运行元数据。
- Meta Compute 2026 实验已实现行为代理、历史回放、滞后均值场与 LLM 接口；
  人类行为数据校准仍属后续阶段。

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

下一阶段 confirmatory experiment 冻结 P0/P1，在排除旧 300 题的 1,000 道全新
no-feedback 题目上进行验证；预注册设计见
[Choices13k confirmatory 协议](docs/choices13k_confirmatory_protocol.md)。离线查看调用量和阶段状态：

```bash
python scripts/run_choices13k_confirmatory.py --stage plan
```

运行必须显式指定 `prepare`、`run`、`evaluate` 或 `all`；没有默认执行阶段。

## 参考项目

参考仓库位于 `third_party/`，保留上游 Git 历史，且不导入我们的 Python 包。
参考概念、复现实验假设和未来阶段分别见
[参考说明](docs/third_party_notes.md)、[架构](docs/architecture.md)、
[实验协议](docs/experiment_protocol.md) 与 [ADR](docs/adr/README.md)。
