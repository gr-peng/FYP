# 进度：让不同方法在相同条件下比较

**已做出可运行的通用比较工具，不再只适用于之前的两道题。** 它让不同方法使用相同原题、四种显示格式和同样次数，再逐题比较选择比例与同一个目标的差距。

这能解决一个具体问题：以后比较“加真人参考”和“打乱参考”时，不能让一边总把某方案放前面，另一边放后面，再把显示差异算成参考信息的进步。

工具会同时核对题号、原题文字、字母与位置的对应关系。某个方法改了提交记录中的收益或概率、计划缺一格、同一个回答机会被记了两遍，都会被拒绝。计划完整但实际缺少回答，或收到无效回答，则保留未知；四格没有全部齐全时，不给该题完整比例。只要预定的成对题目有一题不完整，总体主比较也保留未知，不偷偷删题求平均。已经齐全的局部比较仍然显示。

同一提问多次返回相同答案，或前后选择相反，都会正常保留。工具分别报告字母和位置的变化，不会因为合并后的比例接近50%，就认定模型像真人。真实发送是否符合计划，需要后续模型收集工具另外核对。

本轮已运行一个可以手算的虚构演示：

| 演示情形 | 计划题数 | 完整成对题数 | 方法比较 |
|---|---:|---:|---|
| 32份假回答齐全 | 2 | 2 | 左方法平均差5个百分点，右方法差30个百分点，左减右为－25个百分点 |
| 删掉其中1份假回答 | 2 | 1 | 总体主比较未知，仍保留两题、32个计划机会 |

**这些是假题、假目标和预设回答。** “真人参考”“打乱参考”在此只是两个方法名称，较小误差也是预先构造的，不能当成真人记忆有效或模型进步的证据。

本地45项检查通过，独立审阅者也运行了45项检查，并重新运行演示，8份输出与已保存结果完全相同；还绕开比较工具手算核对了表中结果。工具和演示已通过独立验收，没有必须修复的问题。它只使用虚构数据，没有加载新的真人记录，没有连接模型或使用凭据。对照RP，本轮补上了以后做可靠方法比较的工具；“AI更像真人、市场更真实”仍需真实实验来证明。

可用以下命令在一个新目录运行同样的演示，已有目录不会被覆盖：

```bash
.venv/bin/python -m scripts.demo_presentation_comparison --output /tmp/fyp_comparison_demo
```

本轮新增模型请求0次，累计仍为75/100次，剩25次。

独立审阅者建议下一轮开始真实小检查：固定一道目标题，两种方法都看到相同的三道参考题和相同的一组人群选择比例。一种把比例配对正确，另一种按事先定好的顺序打乱；两边使用相同的四种显示格式，每种问两次，共最多16次。目标题的人群比例只留在本地核对，不能给模型看。若全部发出，累计为91/100次。

这个方案已写入下一轮计划，尚未读取新来源或发出请求。它能检查参考信息是否改变了这道题的模型回答；一道题的结果不足以证明整体更像真人，更不能证明市场模拟已变真实。

- [已运行的虚构演示](/home/bld/data/data4/admin/grp/fyp/iterations/presentation_comparison_20261022/demo/REPORT.md)
- [完整演示结果](/home/bld/data/data4/admin/grp/fyp/iterations/presentation_comparison_20261022/demo/complete_evaluation.json)
- [缺一份回答的结果](/home/bld/data/data4/admin/grp/fyp/iterations/presentation_comparison_20261022/demo/incomplete_evaluation.json)
- [通用工具代码](/home/bld/data/data4/admin/grp/fyp/behavioral-llm-market/src/behavioral_market/benchmarks/presentation_comparison.py)

- [独立验收记录](/home/bld/data/data4/admin/grp/fyp/iterations/presentation_comparison_20261022/independent_review.json)
- [下一轮固定方案](/home/bld/data/data4/admin/grp/fyp/iterations/presentation_comparison_20261022/next_iteration_plan.json)
