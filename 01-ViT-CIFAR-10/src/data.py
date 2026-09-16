"""阶段 1：图片与标签来自 Dataset，DataLoader 负责按索引拼成 batch。"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def load_training_data(root: Path, download: bool) -> CIFAR10:
    # train=True 指官方的 50,000 张训练图。官方 test 不参与这里的划分。
    # ToTensor: [H,W,C] uint8 0..255 -> [C,H,W] float32 0..1。
    return CIFAR10(root=str(root), train=True, download=download, transform=ToTensor())


def stratified_split(labels, seed: int, validation_per_class: int) -> dict:
    """各类独立打乱后抽取验证索引，剩余索引用于训练。"""
    labels = torch.tensor(labels, dtype=torch.long)
    classes = torch.unique(labels, sorted=True)
    if classes.tolist() != list(range(10)):
        raise ValueError('Expected CIFAR-10 labels 0 through 9')
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = [], []
    for label in classes:
        indices = torch.where(labels == label)[0]
        if not 0 < validation_per_class < len(indices):
            raise ValueError('Each class needs both training and validation samples')
        indices = indices[torch.randperm(len(indices), generator=generator)]
        val_indices.extend(indices[:validation_per_class].tolist())
        train_indices.extend(indices[validation_per_class:].tolist())
    # 去掉按类别分组的顺序；DataLoader 随后还会独立 shuffle 训练索引。
    train_indices = torch.tensor(train_indices)
    val_indices = torch.tensor(val_indices)
    return {
        'schema_version': 1,
        'dataset': 'CIFAR-10 official training set',
        'algorithm': 'per-class torch.randperm, then split-level torch.randperm',
        'torch_version': torch.__version__,
        'seed': seed,
        'validation_per_class': validation_per_class,
        'train_indices': train_indices[torch.randperm(len(train_indices), generator=generator)].tolist(),
        'val_indices': val_indices[torch.randperm(len(val_indices), generator=generator)].tolist(),
    }


def save_split(split: dict, path: Path) -> None:
    """复用已有相同划分；不同划分应使用新版本路径，避免覆盖实验依据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != split:
            raise ValueError('Existing split differs; use a new split version instead of overwriting')
    else:
        with path.open('x') as handle:
            json.dump(split, handle, indent=2)
            handle.write('\n')


def make_train_loader(dataset, split: dict, config: dict) -> DataLoader:
    train_dataset = Subset(dataset, split['train_indices'])
    generator = torch.Generator().manual_seed(config['seed'])
    return DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],
        num_workers=config['num_workers'],
        drop_last=config['drop_last'],
        generator=generator,
        pin_memory=False,  # 当前阶段在 CPU 上观察数据。
    )
