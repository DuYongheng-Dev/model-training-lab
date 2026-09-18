# 阶段 4：固定 32 张训练图的记忆实验

运行：`04-overfit32-f206c27f589e`。固定训练划分的前 32 个索引，从 seed=42 的初始 Tiny ViT 开始；batch=32，FP32，AdamW lr=3e-4，weight_decay=0，dropout=0，无增强。

| 项目 | 实测结果 |
| --- | --- |
| 初始 loss / accuracy | 2.318534 / 3/32（9.375%） |
| 第一个准确率 100% 的评估点 | step=40，loss=0.442440 |
| 连续满足全部目标的评估点 | 130、140、150 |
| 停止位置 | 150 步；预算上限为 1000 步 |
| 最终 loss / accuracy | 0.037204 / 32/32（100%） |
| 权重保存后重载 | logits 最大绝对差 0，指标一致 |
| 验证集 / 测试集 | 均未评估 |

![训练曲线](learning-curves.png)

指标来自训练所用的同一组 32 张图片，不能代表未见图片上的表现。固定子集覆盖 9 类、没有 ship；未根据结果重新挑选样本。

- [summary.json](summary.json)：设置、实际进度、停止原因、运行检查和资源观测。
- [evaluations.csv](evaluations.csv)：step=0 及每 10 步更新后的固定训练子集指标。
- [subset.json](subset.json)：预先固定的索引、标签和类别分布。
- [predictions.csv](predictions.csv) / [逐图预测](predictions.png)：最终 32 张图片的预测。
- [学习笔记](../../notes/04-overfit32.md)：曲线解释、复现入口与学习问题。

完整日志、每步更新前的 loss、代码和环境快照以及最终权重仅保存在 SSD。本次基于提交 `fb2361c` 加当前阶段实现，精确代码以运行目录的 source 快照和 manifest 为准。
