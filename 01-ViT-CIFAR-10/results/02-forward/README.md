# 阶段 2 的精选结果

运行：`02-forward-e4cef5b6ad62`；seed=42；CPU、float32、随机初始化 Tiny ViT；没有训练或测试集评估。

- [summary.json](summary.json)：B=2 / B=128 的实际 shape、参数量、初始 logits 和检查结果。
- [patch-to-token.png](patch-to-token.png)：真实训练图的 patch 划分，以及第一个 patch 的实际投影向量。
- [first-image-logits.csv](first-image-logits.csv)：首张训练图对十个类别的初始分数，未经 softmax。

实测参数量为 809,354。两种 batch 大小均输出有限的 float32 logits，参数状态保持不变，所有参数 grad 为 None。7 项模型检查通过，覆盖显式 patch 投影等价性、batch 独立性、图片信息传入 CLS、独立 block、无更新 forward、初始化复现和错误输入。

图片数据来自 [CIFAR-10 官方数据集](https://www.cs.toronto.edu/~kriz/cifar.html)。图中原图用于展示，embedding 使用标准化到 [-1,1] 的输入计算；颜色表示 embedding 数值，不是注意力或像素概率。

完整记录在 SSD 对应 run 中，实验目录 `.local/latest-forward-run.json` 保存本地定位信息。耗时字段涵盖记录、数据读取与校验、forward 和绘图，不是独立性能基准。初始化权重未写入公开仓库，可用配置、种子、依赖和源码快照重建。
