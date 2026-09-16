"""读取一个真实 CIFAR-10 batch，检查并保存可阅读的观察结果；不创建模型。"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')  # 服务器没有桌面；输出独立 PNG。
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import load_training_data, make_train_loader, save_split, stratified_split


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split-path', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='Download the official archive if needed')
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    config = json.loads(args.config.read_text())
    if config['stage'] != '01-data' or config['transform'] != 'ToTensor':
        raise ValueError('This entry point implements stage 01 with ToTensor only')
    if config['num_workers'] != 0 or config['batch_size'] < 1 or config['cpu_threads'] < 1:
        raise ValueError('Stage 01 requires worker=0 and positive batch/thread counts')
    if not 1 <= config['preview_count'] <= min(config['batch_size'], 16):
        raise ValueError('Preview count must be between 1 and min(batch size, 16)')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Stage 01 runs on CPU; launch with CUDA_VISIBLE_DEVICES empty')
    lab_root = Path(os.environ['LAB_ROOT']).resolve(strict=True)
    experiment = Path(__file__).resolve().parents[1]
    for path in (args.data_root, args.split_path, args.run_dir):
        resolved = path.resolve()
        if not resolved.is_relative_to(lab_root) or resolved.is_relative_to(experiment):
            raise ValueError('Data, splits and full outputs must be under external LAB_ROOT')
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'config.json').exists():
        raise FileExistsError('Use a fresh run directory')
    for directory in ('logs', 'metrics', 'outputs', 'source'):
        (run / directory).mkdir(exist_ok=True)
    write_json(run / 'config.json', config)

    torch.set_num_threads(config['cpu_threads'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    socket.setdefaulttimeout(60)

    # 每次运行留存小型源码快照，首次 Git 提交之前也可以追溯实际执行代码。
    source_hashes = {}
    files = [experiment / name for name in ('README.md', 'PLAN.md')]
    files += list(experiment.glob('requirements*.txt'))
    for directory in ('src', 'scripts', 'configs'):
        files += [p for p in (experiment / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for source in sorted(files):
        relative = source.relative_to(experiment)
        destination = run / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_hashes[str(relative)] = file_hash(source)
    write_json(run / 'source-manifest.json', source_hashes)
    git_head = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=experiment,
                              capture_output=True, text=True)
    git_status = subprocess.run(['git', 'status', '--porcelain'], cwd=experiment,
                                check=True, capture_output=True, text=True).stdout
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True,
                            capture_output=True, text=True).stdout
    (run / 'requirements-resolved.txt').write_text(freeze)
    versions = {name: importlib.metadata.version(name)
                for name in ('torch', 'torchvision', 'numpy', 'matplotlib', 'tqdm')}
    write_json(run / 'metadata.local.json', {
        'machine_time_utc': datetime.now(timezone.utc).isoformat(),
        'clock_note': 'Machine wall clock is not verified against real time; duration uses perf_counter',
        'python': sys.version, 'executable': sys.executable,
        'command': [sys.executable, *sys.argv], 'data_root': str(args.data_root.resolve()),
        'split_path': str(args.split_path.resolve()), 'run_dir': str(run.resolve()),
        'device': 'cpu', 'gpu_uuid': None, 'versions': versions,
        'git_commit': git_head.stdout.strip() if git_head.returncode == 0 else None,
        'code_status': 'unversioned' if git_head.returncode else 'versioned; see git_status',
        'git_status': git_status,
    })

    print('Loading official CIFAR-10 training data ...', flush=True)
    dataset = load_training_data(args.data_root, args.download)
    if args.split_path.exists():
        # Treat saved indices as the dataset version, including across torch upgrades.
        split = json.loads(args.split_path.read_text())
        assert split['schema_version'] == 1
        assert split['dataset'] == 'CIFAR-10 official training set'
        assert split['seed'] == config['seed']
        assert split['validation_per_class'] == config['validation_per_class']
    else:
        split = stratified_split(dataset.targets, config['seed'], config['validation_per_class'])
    train_indices, val_indices = split['train_indices'], split['val_indices']
    train_set, val_set = set(train_indices), set(val_indices)
    assert len(dataset) == 50_000
    assert len(train_set) == len(train_indices) and len(val_set) == len(val_indices)
    assert not train_set & val_set
    assert train_set | val_set == set(range(len(dataset)))
    target_tensor = torch.tensor(dataset.targets)
    train_counts = torch.bincount(target_tensor[train_indices], minlength=10)
    val_counts = torch.bincount(target_tensor[val_indices], minlength=10)
    assert val_counts.tolist() == [config['validation_per_class']] * 10
    assert train_counts.tolist() == [5000 - config['validation_per_class']] * 10
    save_split(split, args.split_path)
    shutil.copyfile(args.split_path, run / 'split.json')

    # Dataset[i] 取一张图片；loader 将多个样本沿新的 batch 维堆叠。
    image, label = dataset[train_indices[0]]
    loader = make_train_loader(dataset, split, config)
    images, labels = next(iter(loader))
    assert image.shape == (3, 32, 32) and isinstance(label, int)
    assert images.shape == (min(config['batch_size'], len(train_indices)), 3, 32, 32)
    assert labels.shape == (len(images),)
    assert images.dtype == torch.float32 and labels.dtype == torch.int64
    assert torch.isfinite(images).all() and 0 <= images.min() <= images.max() <= 1
    assert 0 <= labels.min() <= labels.max() < 10
    expected = torch.from_numpy(dataset.data[train_indices[0]].copy()).permute(2, 0, 1).float() / 255
    assert torch.equal(image, expected)
    assert label == dataset.targets[train_indices[0]]
    # 重建相同 generator 的 loader，确认首批采样顺序可复现。
    repeated_images, repeated_labels = next(iter(make_train_loader(dataset, split, config)))
    assert torch.equal(images, repeated_images) and torch.equal(labels, repeated_labels)

    count = config['preview_count']
    figure, axes = plt.subplots(4, 4, figsize=(8, 8), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        if index < count:
            axis.imshow(images[index].permute(1, 2, 0).numpy(), interpolation='nearest')
            category = int(labels[index])
            axis.set_title(f'{category}: {dataset.classes[category]}', fontsize=10)
        axis.axis('off')
    figure.suptitle(f"CIFAR-10 | first training batch | true labels | seed {config['seed']}")
    figure.savefig(run / 'outputs' / 'batch-preview.png', dpi=160)
    plt.close(figure)

    data_hashes = {path.name: file_hash(path) for path in sorted(
        (args.data_root / dataset.base_folder).iterdir()) if path.is_file()}
    summary = {
        'stage': '01-data', 'run_id': run.name, 'seed': config['seed'],
        'device': 'cpu', 'cuda_initialized': torch.cuda.is_initialized(),
        'archive_md5_expected': dataset.tgz_md5,
        'official_train_size': len(dataset), 'train_size': len(train_indices),
        'validation_size': len(val_indices), 'test_evaluated': False,
        'classes': dataset.classes, 'train_class_counts': train_counts.tolist(),
        'validation_class_counts': val_counts.tolist(),
        'raw_sample_shape': list(dataset.data[0].shape), 'raw_sample_dtype': str(dataset.data.dtype),
        'sample_image_shape': list(image.shape), 'sample_label_type': type(label).__name__,
        'batch_image_shape': list(images.shape), 'batch_label_shape': list(labels.shape),
        'image_dtype': str(images.dtype), 'label_dtype': str(labels.dtype),
        'pixel_min': float(images.min()), 'pixel_max': float(images.max()),
        'first_16_labels': labels[:16].tolist(), 'train_batches_per_epoch': len(loader),
        'last_batch_size': len(train_indices) % config['batch_size'] or config['batch_size'],
        'split_sha256': file_hash(args.split_path), 'data_sha256': data_hashes,
        'source_manifest_sha256': file_hash(run / 'source-manifest.json'),
        'checks': {'disjoint_complete_split': True, 'balanced_classes': True,
                   'image_label_alignment': True, 'batch_shape_dtype_range': True,
                   'same_seed_same_first_batch': True},
        'versions': versions, 'duration_seconds': round(time.perf_counter() - started, 3),
    }
    write_json(run / 'metrics' / 'data-summary.json', summary)
    write_json(run / 'status.json', {'status': 'complete', 'stage': '01-data'})
    print(json.dumps({key: summary[key] for key in (
        'train_size', 'validation_size', 'sample_image_shape', 'batch_image_shape',
        'batch_label_shape', 'image_dtype', 'label_dtype', 'pixel_min', 'pixel_max',
        'train_batches_per_epoch', 'last_batch_size', 'checks')}, indent=2))
    print(f'Observation saved: {run}', flush=True)


if __name__ == '__main__':
    main()
