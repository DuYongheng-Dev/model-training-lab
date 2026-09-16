# 阶段 3：单步训练与梯度观察

运行：`03-step-ea5732fd1e68`；已指定 GPU，FP32，batch=128，seed=42，AdamW lr=3e-4、weight_decay=0。

| 观察 | 实测结果 |
| --- | --- |
| 同一训练 batch 的 loss | 2.294613 → 一次 step 后 2.119175 |
| backward 前 `.grad` | 全部 None |
| backward 后 `.grad` | 56 个参数张量均有有限梯度 |
| backward 后参数 | 全部逐位不变 |
| step 后参数 | 56 个张量均有元素改变 |
| 主模型 optimizer step 次数 | 1 |
| 独立副本：每次清零的梯度倍数 | 1、1、1 |
| 独立副本：不清零的梯度倍数 | 1、2、约 3 |
| 副本 optimizer step 次数 | 0 |

这是一批训练图片上的机制观察，未运行完整 epoch 或评估验证集、测试集。

- [summary.json](summary.json)：配置摘要、实际批次索引、loss、shape、参数例子和检查结果。
- [parameters.csv](parameters.csv)：所有参数张量的梯度大小与更新量。
- [accumulation.csv](accumulation.csv)：梯度清零对照及误差。
- [学习笔记](../../notes/03-step.md)：解释、复现命令与三个问题。

![梯度累积对照](gradient-accumulation.png)

完整日志、源代码快照、依赖和观测张量保存在 SSD 的独立运行目录。运行基于提交 `47e3a5f` 加本阶段新增文件，具体内容以运行目录的 source 快照及 manifest 为准。
