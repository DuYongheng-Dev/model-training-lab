"""阶段 2：在真实图片上观察 Tiny ViT forward；不计算 loss，不反传。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import Normalize

from data import load_training_data, make_train_loader
from model import TinyViT


def file_hash(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def draw_patches(raw_image, normalized_image, model, destination):
    """用真实像素与当前投影展示 patch；embedding 色条不是 attention map。"""
    image = raw_image.permute(1, 2, 0).numpy()
    patch_size = model.patch_size
    grid_size = model.image_size // patch_size
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(image, interpolation='nearest')
    for border in range(patch_size, model.image_size, patch_size):
        axes[0].axhline(border - 0.5, color='white', linewidth=0.8)
        axes[0].axvline(border - 0.5, color='white', linewidth=0.8)
    for row in range(grid_size):
        for column in range(grid_size):
            axes[0].text(column * patch_size + (patch_size - 1) / 2,
                         row * patch_size + (patch_size - 1) / 2,
                         str(row * grid_size + column + 1), color='white',
                         ha='center', va='center', fontsize=6,
                         bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 0.3, 'edgecolor': 'none'})
    axes[0].set_title('32 x 32 image -> 8 x 8 patches')
    axes[0].axis('off')
    axes[1].imshow(image[:patch_size, :patch_size], interpolation='nearest')
    axes[1].set_title('Patch 1: 3 x 4 x 4 = 48 values')
    axes[1].axis('off')
    with torch.inference_mode():
        token = model.patch_embed(normalized_image.unsqueeze(0).to(next(model.parameters()).device))[0, :, 0, 0]
    plot = axes[2].imshow(token.cpu().numpy()[None, :], aspect='auto', cmap='coolwarm')
    axes[2].set_title('Patch 1 embedding: 128 numbers\nRandom initialization; before CLS/position')
    axes[2].set_xlabel('Embedding coordinate')
    axes[2].set_yticks([])
    figure.colorbar(plot, ax=axes[2], shrink=0.65)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'data-root', 'split-path', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    started = time.perf_counter()
    project = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    data_config = json.loads((project / 'configs/01-data.json').read_text())
    prior = json.loads((project / 'results/01-data/summary.json').read_text())
    if config['stage'] != '02-forward' or config['batch_sizes'] != [2, 128]:
        raise ValueError('This teaching stage observes B=2 followed by B=128')
    if args.device == 'cuda':
        selection = json.loads((project / '.local/device.json').read_text())
        if os.environ.get('CUDA_VISIBLE_DEVICES') != selection['gpu_uuid']:
            raise ValueError('Use the recorded GPU assignment via scripts/inspect_forward.sh')
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.backends.fp32_precision = 'ieee'
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
        torch.backends.cudnn.benchmark = False
    elif os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU reproduction requires CUDA_VISIBLE_DEVICES empty')
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    lab_root = Path(os.environ['LAB_ROOT']).resolve(strict=True)
    for path in (args.data_root, args.split_path, args.run_dir):
        if not path.resolve().is_relative_to(lab_root):
            raise ValueError('Data, splits and full outputs must be under external LAB_ROOT')
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'config.json').exists():
        raise FileExistsError('Use a new run directory')
    for name in ('metrics', 'outputs', 'source', 'logs'):
        (run / name).mkdir(exist_ok=True)
    write_json(run / 'config.json', config)

    # 代码、依赖、命令等完整运行依据只存入 SSD。
    manifest = {}
    sources = [project / name for name in ('README.md', 'PLAN.md')]
    sources += list(project.glob('requirements*.txt'))
    for directory in ('src', 'scripts', 'configs', 'tests'):
        sources += [p for p in (project / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for source in sorted(sources):
        relative = source.relative_to(project)
        destination = run / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest[str(relative)] = file_hash(destination)
    write_json(run / 'source-manifest.json', manifest)
    head = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=project, capture_output=True, text=True)
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=project, text=True)
    (run / 'requirements-resolved.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'matplotlib')}
    write_json(run / 'metadata.local.json', {
        'machine_time_utc': datetime.now(timezone.utc).isoformat(),
        'clock_note': 'Wall clock not calibrated; duration uses perf_counter',
        'python': sys.version, 'executable': sys.executable, 'command': [sys.executable, *sys.argv],
        'device': str(device), 'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES') or None,
        'data_root': str(args.data_root.resolve()),
        'split_path': str(args.split_path.resolve()), 'run_dir': str(run.resolve()),
        'versions': versions, 'git_commit': head.stdout.strip() if head.returncode == 0 else None,
        'code_status': 'unversioned' if head.returncode else 'versioned; see git_status', 'git_status': status,
    })

    torch.set_num_threads(config['cpu_threads'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    # 复用阶段 1 的实际索引，不重新随机划分，也不下载数据。
    assert file_hash(args.split_path) == prior['split_sha256'], 'Stage 01 split changed'
    split = json.loads(args.split_path.read_text())
    shutil.copyfile(args.split_path, run / 'split.json')
    dataset = load_training_data(args.data_root, download=False)
    data_hashes = {name: file_hash(args.data_root / dataset.base_folder / name)
                   for name in prior['data_sha256'] if name.startswith('data_batch_') or name == 'batches.meta'}
    assert all(value == prior['data_sha256'][name] for name, value in data_hashes.items())
    raw_images, labels = next(iter(make_train_loader(dataset, split, data_config)))
    assert labels[:16].tolist() == prior['first_16_labels']
    images = Normalize(config['normalize_mean'], config['normalize_std'])(raw_images)
    assert torch.equal(images, raw_images * 2 - 1)

    model = TinyViT(**config['model']).eval().to(device)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    parameter_counts = {'total': sum(p.numel() for p in model.parameters()),
                        'patch_embed': sum(p.numel() for p in model.patch_embed.parameters()),
                        'cls_token': model.cls_token.numel(),
                        'position_embedding': model.position_embedding.numel(),
                        'blocks': [sum(p.numel() for p in block.parameters()) for block in model.blocks],
                        'final_norm': sum(p.numel() for p in model.norm.parameters()),
                        'head': sum(p.numel() for p in model.head.parameters())}
    traces, outputs = {}, {}
    with torch.inference_mode():
        for batch_size in config['batch_sizes']:
            trace = {}
            logits = model(images[:batch_size].to(device), trace=trace)
            assert logits.shape == (batch_size, config['model']['num_classes'])
            assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
            traces[str(batch_size)] = trace
            outputs[batch_size] = logits
            print(f'\nBatch size = {batch_size}')
            for name, shape in trace.items():
                print(f'{name:22s} {shape}')
        torch.testing.assert_close(outputs[2], outputs[128][:2], atol=1e-5, rtol=1e-5)
    state_unchanged = all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    no_gradients = all(parameter.grad is None for parameter in model.parameters())
    assert state_unchanged and no_gradients
    if args.device == 'cpu':
        assert not torch.cuda.is_initialized()
    draw_patches(raw_images[0], images[0], model, run / 'outputs/patch-to-token.png')
    first_logits = outputs[2][0].tolist()
    with (run / 'outputs/first-image-logits.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['class_index', 'class_name', 'logit'])
        writer.writerows((i, name, first_logits[i]) for i, name in enumerate(dataset.classes))
    summary = {
        'stage': '02-forward', 'run_id': run.name, 'seed': config['seed'], 'device': args.device,
        'cuda_initialized': torch.cuda.is_initialized(), 'pretrained': False, 'training_steps': 0,
        'test_evaluated': False, 'model_config': config['model'], 'versions': versions,
        'initialization': 'truncated normal: mean=0, std=0.02, bounds=[-0.04,0.04]; biases=0; LayerNorm weight=1',
        'parameter_counts': parameter_counts, 'shapes': traces,
        'raw_range': [float(raw_images.min()), float(raw_images.max())],
        'normalized_range': [float(images.min()), float(images.max())],
        'normalization': {'mean': config['normalize_mean'], 'std': config['normalize_std']},
        'first_image_label': int(labels[0]), 'first_image_class': dataset.classes[int(labels[0])],
        'first_image_logits': first_logits,
        'batch_consistency_max_abs_error': float((outputs[2] - outputs[128][:2]).abs().max()),
        'split_sha256': file_hash(args.split_path), 'training_data_sha256': data_hashes,
        'source_manifest_sha256': file_hash(run / 'source-manifest.json'),
        'checks': {'finite_float32_logits': True, 'same_images_across_batch_sizes': True,
                   'state_unchanged': state_unchanged, 'all_parameter_grad_none': no_gradients,
                   'same_split_and_training_data_as_stage_01': True},
        'duration_seconds': round(time.perf_counter() - started, 3),
    }
    write_json(run / 'metrics/forward-summary.json', summary)
    write_json(run / 'status.json', {'status': 'complete', 'stage': '02-forward'})
    print(f'\nParameters: {parameter_counts["total"]:,}; no parameter update or gradients.')
    print(f'Observation saved: {run}', flush=True)


if __name__ == '__main__':
    main()
