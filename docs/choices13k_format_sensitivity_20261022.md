# 进度：平均改善确实存在，但还不是稳定能力

这轮把已有24份回答重新逐格核算，**保留上一轮的正向结果：正确参考在本题的原定平均指标上更接近人群。** 同时把它的边界说明白，没有重新问模型，也没有更换主要指标。

与总体人群选择比例65%相比，正确参考合并后的差距仍是2.5个百分点，无参考是22.5，打乱参考是27.5。

但把四种显示方式分别看，结果并不一致：

| 正确参考与谁比较 | 更接近总体65% | 相同 | 更远 |
|---|---:|---:|---:|
| 无参考 | 1种 | 2种 | 1种 |
| 打乱参考 | 1种 | 1种 | 2种 |

这两种结果并不矛盾。不同显示方式下观察到的比例有高有低，合并后偏差可以互相抵消。因此，本题的平均比例更接近人群，尚不能写成“换个字母或位置也能稳定改善”。12组相同输入的重复里，有5组前后选择不同，这些回答都保留。

我们只有人群总体65%的选择比例，没有真人在这四种显示方式下分别作答的比例。表格只是在同一个总体参照下检查结果是否稳定，不能当作四种显示方式各自的真人对照实验。它是看到结果后做的补充检查，不是新的主要指标。

本轮已实现可重复运行的离线程序，从固定请求中的字母对应关系和已验收的24条选择重新计算，输出完整表格和CSV。本地和独立审阅各运行30项检查，全部通过。独立审阅者从三份派生文件重建24条选择、12格和8项配对差值，核对两份CSV和通俗报告；结果已验收通过。只读取三份已验收的派生文件，没有读取新的真人资料或原始模型日志。

对照RP，现在能保留的结论是“正确参考条件下本题收到的回答，按既定平均指标更接近人群”。总体真人相似度、个人交易记忆学习、策略切换和市场真实性仍没有被这项实验证明。

本轮新增模型调用0次，累计仍为115/115次。

- [完整通俗报告与逐格差值](/home/bld/data/data4/admin/grp/fyp/iterations/choices13k_format_sensitivity_20261022/results/REPORT.md)
- [机器可读计算结果](/home/bld/data/data4/admin/grp/fyp/iterations/choices13k_format_sensitivity_20261022/results/format_sensitivity.json)
- [三组逐格表格](/home/bld/data/data4/admin/grp/fyp/iterations/choices13k_format_sensitivity_20261022/results/cells.csv)
- [两项逐格比较](/home/bld/data/data4/admin/grp/fyp/iterations/choices13k_format_sensitivity_20261022/results/pairs.csv)

- [独立验收记录](/home/bld/data/data4/admin/grp/fyp/iterations/choices13k_format_sensitivity_20261022/independent_review.json)
