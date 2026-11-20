# 缓存控制检查：同一问题收到了不同选择

本轮4条请求已实际完成。这里的缓存，是接口复用先前答案。**两条普通请求返回了同一编号、生成时间和正文；两条带缓存控制的请求返回了不同编号，分别选择技术策略、基本面策略。** 后两条发送的完整请求内容完全相同，原问题和模型设置均未改。这提供了新的差异记录，但还不能把它们当成已确认彼此独立生成的回答。

| 次序 | 请求设置 | 返回目标 | 返回编号、时间和正文的关系 |
|---|---|---|---|
| 1 | 原设置 | 基本面 | 与第4条相同 |
| 2 | 同时要求跳过缓存读取、不写入缓存 | 技术 | 与其余三条不同 |
| 3 | 同时要求跳过缓存读取、不写入缓存 | 基本面 | 与其余三条不同 |
| 4 | 原设置 | 基本面 | 与第1条相同 |

四条都使用原B输入：原策略基本面、指定参考价105、当前价95，账户、记忆和问题相同。接口根据[学校说明](https://docs.hpc.hkust-gz.edu.cn/docs/aigc/api-usage/)使用LiteLLM代理；新增选项来自[LiteLLM缓存文档](https://docs.litellm.ai/docs/proxy/caching_controls)的专门小节。这次没有加随机文字、换模型或重问求某种答案。

这组现象与文档所述的联合缓存控制预期一致。不过，4条都没有出现这次记录的接口缓存标记，无法借它直接确认内部缓存状态；缺少这个标记也不代表没有命中。HTTP成功本身不能证明选项被执行，两个选项同时使用也不能分别证明各自的作用。记录里共有3个不同编号、3段不同正文、4种完整返回；第1和第4条仅在附加元数据上不同，不能写成完整返回逐字节相同。

第2条理由中有一处明确事实错误：它声称技术策略收益更好，但输入给的基本面基准是−0.5%，技术基准是−4.5%，前者表现较好。这条错误与原策略选择都保留，没有纠正后重问。参考价仍只是研究中指定的数值，不能当成真实价值证明；其他回答提到“已经证明的做法”或未来回归，也不是本次记录能够证实的结论。

工具会直接记录实际发送的完整请求内容，并在每条返回后检查已有记录。56项本地检查通过，其中25项新检查由独立子agent再次验证。它发现并修复了一处误报：缺少编号或生成时间时，现在明确记为未知，不会把空值拼成完整复用证据；原版本和初次48项验证均保留。实际4条均成功返回，本地没有重试、格式修复或补发；接口上游内部尝试次数仍未知。

独立子agent已从原始4条完成复算并验收，确认执行、上述返回关系和理由中的事实错误都与记录一致。累计已发送**59/100次**，剩41次，沿用已有授权。

对照RP，这轮让我们能更准确地区分重复副本、不同回答和理由里的事实错误，尚未证明AI更像真人或整个市场更真实。独立子agent建议下一步回到旧风险选择校准记录：预先固定原始提示（P0）的20道题，只核对每题10份AI返回，并按两种选项呈现顺序分别检查复用线索。不读真人答案或保留测试，不重新挑提示，也不把去重结果直接叫作真实选择概率。

- [4条实际执行与重复分组](/home/bld/data/data4/admin/grp/fyp/iterations/style_cache_control_20261022/live_execution/REPORT.md)
- [实际观察与理由事实核对](/home/bld/data/data4/admin/grp/fyp/iterations/style_cache_control_20261022/actual_observations.json)
- [事先固定的请求](/home/bld/data/data4/admin/grp/fyp/iterations/style_cache_control_20261022/package/expected_requests.jsonl)
- [修复后的本地验证](/home/bld/data/data4/admin/grp/fyp/iterations/style_cache_control_20261022/revision2_verification.json)
- [官方说明核对与边界](/home/bld/data/data4/admin/grp/fyp/iterations/style_cache_control_20261022/documentation_evidence.json)
- [执行后独立审查](/home/bld/data/data4/admin/grp/fyp/iterations/style_cache_control_20261022/independent_review.json)
- [下一步方向](/home/bld/data/data4/admin/grp/fyp/iterations/style_cache_control_20261022/next_iteration_plan.json)
- [累计用量](/home/bld/data/data4/admin/grp/fyp/iterations/style_cache_control_20261022/budget_ledger.json)
