# 首次 GPU 运行与环境修复

## 实验设备约定

用户已为本实验指定一张 GPU，指定在后续阶段持续有效。设备身份保存在本机 `.local/device.json`（不提交）；启动脚本按其中 UUID 设置 `CUDA_VISIBLE_DEVICES`，每次检查该卡的计算进程与可用显存。

进程只看见这一张卡，因此代码中使用 `cuda:0`。这个编号表示进程内的第一个可见设备，不能直接当作主机卡号。后续 agent 阅读 README 和本机选择记录即可，不需要用户每次重新指定。

`inspect_forward.sh` 默认在已指定 GPU 上执行；`--cpu` 用于 CPU 复现。`inspect_data.sh` 继续在 CPU 上读取和展示数据；这是数据准备工作。后续模型 forward/backward/optimizer 默认使用已指定 GPU。

## 失败证据与修复

原环境为 torch `2.13.0+cu130`、torchvision `0.28.0+cu130`、cuDNN `9.20.0.48`。CPU 导入和模型检查成功，`torch.cuda.is_available()` 也返回 True，但 GPU 上的第一个 Conv2d 失败：

```text
CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
```

检查 `/proc/self/maps` 发现，主体 cuDNN 来自实验环境，而 `libcudnn_engines_tensor_ir` 来自系统另一版本。预加载环境内已有 cuDNN 子库仍不能解决缺失库混用。这个现象与 PyTorch 仓库中的同类报告相符。[上游问题记录](https://github.com/pytorch/pytorch/issues/188892)

最终在本实验独立环境升级到官方 CUDA 13.0 wheel 的配套版本：

| 依赖 | 原版本 | 当前版本 |
| --- | --- | --- |
| torch | 2.13.0+cu130 | 2.14.0+cu130 |
| torchvision | 0.28.0+cu130 | 0.29.0+cu130 |
| nvidia-cudnn-cu13 | 9.20.0.48 | 9.24.0.43 |
| nvidia-nccl-cu13 | 2.29.7 | 2.30.7 |
| triton | 3.7.1 | 3.8.0 |

版本来自已安装 wheel 元数据，完整解析结果见 [requirements-lock.txt](../requirements-lock.txt)。本次没有更改模型架构；原运行目录保留当时版本与代码快照。新旧框架可能选择不同实现，跨版本不承诺逐位一致。

## 验证

- `pip check` 通过。
- 7 项模型检查通过。
- GPU 上真实训练图片 B=2、128 的 forward 通过，模型状态不变且没有梯度：`02-forward-a82a1eb6235d`。
- 升级后再次运行阶段 1：`01-data-6bad4dba5af4`。保留原划分文件及其生成版本，直接复用索引，检查划分互斥、完整、平衡与首批可复现。现有划分不会因框架版本改变被重写。
- 计时用 CPU/GPU 两边相同的新环境、权重和输入，避免拿旧环境 CPU 数据比较新环境 GPU；详见 [CPU/GPU 对比](02-device-benchmark.md)。

安装日志、前后依赖、实际设备和加载库路径仅保存在 SSD 的本地运行记录中。环境修复记录入口是 `.local/gpu-repair-run.txt`。
