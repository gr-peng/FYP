# 中期实验数据

这份数据随代码一起发布。下表中的 CSV、JSON 和压缩轨迹可直接从 GitHub 下载。外部原始资料单列为 145 个文件，共 6,612,464,690 字节，逐一列出研究时使用的下载地址、目标路径、字节数和 SHA-256 校验值。[来源清单](sources.csv) · [研究历程](../docs/research_history_zh.md)

| 目录 | 数据单位与用途 |
|---|---|
| [meta_48_runs](meta_48_runs) | 首批学校接口实验逐次指标、分组均值、消息窗口及不确定性。一次运行包含多个模拟账户 |
| [risk_choices](risk_choices) | Choices13k、CPC18 和 Jev 的风险选择结果。CPC18 逐题 CSV 含人群选择比例和各方法预测，可重算平均绝对误差 |
| [behavior](behavior) | 51 场金融实验的统计模型得分、误差拆分、受限账户动作实验、16 场开发比较及概率修正，另含 48 次虚构记忆实验的汇总 |
| [two_events](two_events) | 两个新闻场景中各次运行的结果、同期真实收益和价格路径图。股票和比特币采用各自的观察窗口 |
| [crypto](crypto) | 三个训练日、五种信息延迟的盘口执行价格误差汇总，及另一个开发周的成交笔数比较 |
| [strategy](strategy) | 替代策略表现及人群比例变化时的模型选择次数。各格是同一情境下的重复询问 |
| [market_synthetic_240](market_synthetic_240) | 240 次构造市场的完整压缩轨迹及逐条指标。index.csv 记录方法、处理顺序、延迟、随机起点和原始压缩文件校验值 |
| [market_20x30](market_20x30) | 固定策略组与模型选策略组各 30 天的市场、账户、订单、成交、盘口、群体比例和策略切换 CSV。目录内另有三组配对的初始资产分配实验汇总 |
| [memory_market](memory_market) | 两个虚构账户、三档可见历史下的 24 份请求与回答，三条独立账户路径，询问覆盖和抽签波动诊断。partial_fill_local_demo.json 是另一个手算例 |

## 从仓库复核

在仓库根目录安装项目依赖。下面的 Python 片段只读取已发布的虚构账户实验，重新生成模型每次看见的输入、订单和账户。将三条完整路径逐一与存档比较，成功时输出三档的成交笔数。

```bash
python -m pip install -e .
python - <<'PY'
import json
from pathlib import Path
from experiments.conditional_market_pilot import replay

base = Path('research_data/memory_market')
def rows(name):
    return [json.loads(s) for s in (base / name).read_text().splitlines()]

paths = replay(rows('requests.jsonl'), rows('responses.jsonl'))
for arm, history in paths.items():
    expected = json.loads((base / (arm + '.json')).read_text())
    assert json.loads(json.dumps(history, allow_nan=False)) == expected
    print(arm, len(history['trades']))
PY
```

三行都应为 0。另一项 20 户实验可以直接打开两组 trades.csv，分别数到 54 和 64 笔，按成交数量求和得到 129 和 161 股。合成轨迹的 .json.gz 用 gzip 解压后可直接按 JSON 读取。核对已发布文件可在本目录运行 `sha256sum -c SHA256SUMS`。

## 取得原始数据

在 sources.csv 中选 dataset 或一个具体路径。脚本把文件放回实验代码期待的位置，按长度和 SHA-256 检查，遇到上游内容变化会报告具体文件。动态行情接口可能返回修订值，上游也可能改变访问方式。所有 145 个地址同时恢复约需 6.61 GB。本目录提供项目产生的模拟逐笔记录，真人参与者逐行记录由原作者维护的来源提供。开发集和最终检验集的划分见各任务协议。

```bash
# 两个风险选择输入文件
python scripts/restore_research_data.py --dataset choices13k
# 仅取一个日期的真实交易所档案
python scripts/restore_research_data.py --path data/raw/crypto_microstructure_20261011/trades/BTCUSDT2024-11-01.csv.gz
# 已有源数据的机器上核对所有清单文件
python scripts/restore_research_data.py --verify-only
```

Choices13k 的两个文件固定到同一个上游提交。CPC18 的 210 题汇总来自竞赛官网，逐题预测结果是本项目计算值，其原始答题记录及字典来自 Zenodo。实验市场复现包包含原实验程序、说明和交易研究资料，Scopic 另有独立的持仓数据出处。[Choices13k 原项目](https://github.com/jcpeterson/choices13k/tree/821ae7e88386b508ebb46fae76fac63cb62ec876) · [CPC18 原始资料](https://zenodo.org/records/2571510) · [实验市场原项目](https://osf.io/5ypku/) · [Scopic 持仓数据](https://data.mendeley.com/datasets/jfg8s32xdm/1)

盘口来源为 Bybit BTCUSDT 永续合约，清单含 14 个主窗口和 28 个辅助窗口的盘口或逐笔档案。另按来源列出事件研究用的 Binance、Yahoo、SEC 和 GDELT 请求。校验值代表归档时使用的版本。研究使用的原始输入由相应提供方维护。[行情获取代码](../scripts/download_crypto_microstructure.py) · [数据时间与划分](../docs/active_research_status.md)

旧实验脚本中有些命令会核对当时工作目录下的源码快照、请求日志或中间缓存。此数据包供阅读、分析和上述离线重放使用。完整原始网络日志、本机配置及旧迭代工作目录保存在原实验机器。
