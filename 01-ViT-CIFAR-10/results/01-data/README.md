# 阶段 1 的精选结果

- 运行 ID：`01-data-538ab412b224`。
- [summary.json](summary.json)：数据规模、shape、类型、完整性检查、种子、划分/数据校验和及软件版本。
- [batch-preview.png](batch-preview.png)：首个训练 batch 的前 16 张图，标题是真实类别。

数据来源为 [CIFAR-10 官方数据集](https://www.cs.toronto.edu/~kriz/cifar.html)，仅展示少量样本用于学习数据处理。图像保持原始内容，通过最近邻放大便于查看。

这是数据读取与划分的结果。未执行模型 forward、参数更新、验证预测或测试集评估。

摘要中的 `duration_seconds` 为已准备好归档后该次观察程序的单调时钟耗时，包含读取/解压、校验和输出，不包含环境安装与此前下载，不应作为 DataLoader 吞吐基准。

完整日志、配置、数据划分索引及源码快照保存在仓库外的对应 run 中；本地定位信息见实验根目录下被忽略的 `.local/latest-data-run.json`。首次提交前源码状态为 `unversioned`，通过运行快照及哈希追溯。
