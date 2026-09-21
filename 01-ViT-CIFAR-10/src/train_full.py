"""阶段 5：完整 train/validation、最佳模型选择和 epoch 边界恢复。"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from checkpoint import atomic_save, capture_rng, cpu_snapshot, load_checkpoint, restore_rng
from evaluate import evaluate_loader
from model import TinyViT
from run_record import file_hash, record_run, write_json


class IndexedView(Dataset):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, slot):
        index = self.indices[slot]
        image, label = self.dataset[index]
        return image, label, index


def build_loaders(args, config, forward, project):
    prior = json.loads((project / 'results/01-data/summary.json').read_text())
    assert file_hash(args.split_path) == prior['split_sha256']
    split = json.loads(args.split_path.read_text())
    train_indices, val_indices = split['train_indices'], split['val_indices']
    assert len(train_indices) == 45000 and len(val_indices) == 5000
    assert len(set(train_indices)) == 45000 and len(set(val_indices)) == 5000
    assert not set(train_indices).intersection(val_indices)
    assert set(train_indices).union(val_indices) == set(range(50000))
    train_indices = train_indices[:config['train_samples']]
    val_indices = val_indices[:config['validation_samples']]
    indices = {'train': train_indices, 'validation': val_indices}
    loaders, generators = {}, {}
    for offset, (name, selected) in enumerate(indices.items()):
        # Independent views/transforms prevent future training augmentation leaking into validation.
        transform = Compose([ToTensor(), Normalize(forward['normalize_mean'], forward['normalize_std'])])
        dataset = CIFAR10(root=str(args.data_root), train=True, download=False, transform=transform)
        generator = torch.Generator().manual_seed(config['seed'] + offset)
        generators[name] = generator
        loaders[name] = DataLoader(IndexedView(dataset, selected), batch_size=config['batch_size'],
                                   shuffle=name == 'train', drop_last=config['drop_last'],
                                   num_workers=config['num_workers'], pin_memory=config['pin_memory'],
                                   generator=generator)
    hashes = {name: file_hash(args.data_root / dataset.base_folder / name)
              for name in prior['data_sha256'] if name.startswith('data_batch_') or name == 'batches.meta'}
    assert all(value == prior['data_sha256'][name] for name, value in hashes.items())
    return loaders, generators, {'split_sha256': file_hash(args.split_path),
                                 'training_data_sha256': hashes, 'indices': indices}


def train_epoch(model, optimizer, loader, expected_indices):
    model.train()
    loss_sum, correct, count, batches, data_wait = 0.0, 0, 0, 0, 0.0
    observed_indices = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    iterator = iter(loader)
    while True:
        waiting = time.perf_counter()
        try:
            images, labels, indices = next(iterator)
        except StopIteration:
            break
        data_wait += time.perf_counter() - waiting
        images = images.to('cuda:0', non_blocking=True)
        labels = labels.to('cuda:0', non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        if not torch.stack([p.grad.isfinite().all() for p in model.parameters()]).all():
            raise FloatingPointError('Non-finite training gradient')
        optimizer.step()

        # These predictions are from BEFORE this batch's update, with changing weights across batches.
        loss_sum += float(loss.detach()) * len(labels)
        correct += int((logits.detach().argmax(dim=1) == labels).sum())
        count += len(labels)
        batches += 1
        last_batch_size = len(labels)
        observed_indices.extend(indices.tolist())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert count == len(expected_indices) and sorted(observed_indices) == sorted(expected_indices)
    assert batches == math.ceil(count / loader.batch_size)
    if not torch.stack([p.isfinite().all() for p in model.parameters()]).all():
        raise FloatingPointError('Non-finite parameter after epoch')
    optimizer.zero_grad(set_to_none=True)
    order_hash = hashlib.sha256(np.asarray(observed_indices, dtype='<i8').tobytes()).hexdigest()
    return {'train_loss': loss_sum / count, 'train_accuracy': correct / count,
            'train_correct': correct, 'train_count': count, 'train_batches': batches,
            'train_last_batch_size': last_batch_size, 'train_order_sha256': order_hash,
            'train_seconds': elapsed, 'data_wait_seconds': data_wait,
            'train_samples_per_second': count / elapsed}


def write_history(run, history):
    with (run / 'metrics/epochs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(history)


def draw_history(history, destination):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    epochs = [row['epoch'] for row in history]
    for split in ('train', 'val'):
        label = 'Train (during updates)' if split == 'train' else 'Validation (epoch-end model)'
        axes[0].plot(epochs, [row[f'{split}_loss'] for row in history], 'o-', label=label, markersize=3)
        axes[1].plot(epochs, [100 * row[f'{split}_accuracy'] for row in history], 'o-', label=label, markersize=3)
    axes[0].set_ylabel('Mean cross-entropy loss')
    axes[1].set_ylabel('Accuracy (%)')
    for axis in axes:
        axis.set_xlabel('Completed epoch')
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle('Tiny ViT / CIFAR-10: training and held-out validation')
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'data-root', 'split-path', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--until', type=int, help='Last completed epoch in this invocation, within the fixed budget')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    forward = json.loads((args.config.parent / config['forward_config']).read_text())
    selection = json.loads((project / '.local/device.json').read_text())
    assert selection['device'] == 'cuda' and os.environ.get('CUDA_VISIBLE_DEVICES') == selection['gpu_uuid']
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert config['stage'] in ('05-train', '05-resume-smoke')
    expected_sizes = (45000, 5000) if config['stage'] == '05-train' else (257, 129)
    assert (config['train_samples'], config['validation_samples']) == expected_sizes
    assert config['batch_size'] == 128 and config['num_workers'] == 0 and not config['drop_last']
    assert config['augmentation'] == 'none' and forward['model']['dropout'] == 0
    assert config['precision'] == 'float32_ieee' and config['deterministic_algorithms']
    until = args.until if args.until is not None else config['max_epochs']
    assert 1 <= until <= config['max_epochs'] <= 20
    if args.resume and not args.resume.resolve(strict=True).is_relative_to(Path(os.environ['LAB_ROOT']).resolve()):
        raise ValueError('Resume checkpoint must be under LAB_ROOT')
    versions = record_run(project, args, config)
    run = args.run_dir
    (run / 'checkpoints').mkdir()
    started = time.perf_counter()
    try:
        torch.set_num_threads(config['cpu_threads'])
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.backends.fp32_precision = 'ieee'
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        loaders, generators, data_identity = build_loaders(args, config, forward, project)
        write_json(run / 'data-identity.json', data_identity)
        identity = {'config': config, 'forward': forward, 'data': data_identity, 'versions': versions,
                    'source_hashes': {name: file_hash(project / 'src' / name) for name in
                                      ('model.py', 'data.py', 'train_full.py', 'evaluate.py', 'checkpoint.py')},
                    'runtime': {'gpu_uuid': selection['gpu_uuid'], 'torch_cuda': str(torch.version.cuda),
                                'cudnn': torch.backends.cudnn.version(),
                                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}}
        model = TinyViT(**forward['model']).to('cuda:0')
        assert sum(p.numel() for p in model.parameters()) == 809354
        optimizer = torch.optim.AdamW(model.parameters(), **config['optimizer'])
        history, completed, global_step, best = [], 0, 0, None
        parent_run = None
        if args.resume:
            restored = load_checkpoint(args.resume, identity)
            completed, global_step = restored['completed_epoch'], restored['global_step']
            if completed >= until:
                raise ValueError('--until must exceed the checkpoint completed epoch')
            model.load_state_dict(restored['model'])
            optimizer.load_state_dict(restored['optimizer'])
            history = restored['history']
            best = restored.get('best_checkpoint', restored)
            parent_run = restored['run_id']
            atomic_save(best, run / 'checkpoints/best.pt')
            write_json(run / 'resume.local.json', {'checkpoint': str(args.resume.resolve()),
                'checkpoint_sha256': file_hash(args.resume), 'parent_run': parent_run,
                'completed_epoch': completed, 'next_epoch': completed + 1})
            # Restore after constructors, IO, and all other preparation that may consume randomness.
            restore_rng(restored['rng'], generators)
        first_epoch = completed + 1
        print(f"run={run.name} train={len(loaders['train'].dataset)} validation={len(loaders['validation'].dataset)} "
              f"epochs={first_epoch}..{until} resumed={args.resume is not None}", flush=True)
        for epoch in range(first_epoch, until + 1):
            torch.cuda.reset_peak_memory_stats()
            train = train_epoch(model, optimizer, loaders['train'], data_identity['indices']['train'])
            global_step += train['train_batches']
            assert all(int(state['step']) == global_step for state in optimizer.state.values())
            state_before_eval = cpu_snapshot(model.state_dict())
            torch.cuda.synchronize()
            validating = time.perf_counter()
            validation = evaluate_loader(model, loaders['validation'], 'cuda:0')
            torch.cuda.synchronize()
            validation_seconds = time.perf_counter() - validating
            assert validation['count'] == config['validation_samples']
            assert model.training and all(p.grad is None for p in model.parameters())
            assert all(torch.equal(state_before_eval[name], value.cpu()) for name, value in model.state_dict().items())
            assert all(int(state['step']) == global_step for state in optimizer.state.values())
            row = {'epoch': epoch, 'global_step': global_step, **train,
                   'val_loss': validation['loss'], 'val_accuracy': validation['accuracy'],
                   'val_correct': validation['correct'], 'val_count': validation['count'],
                   'validation_seconds': validation_seconds,
                   'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
            history.append(row)
            improved = best is None or validation['loss'] < best['best_val_loss']
            best_epoch = epoch if improved else best['best_epoch']
            best_loss = validation['loss'] if improved else best['best_val_loss']
            payload = {'schema_version': 1, 'run_id': run.name, 'parent_run': parent_run,
                       'identity': identity, 'completed_epoch': epoch, 'global_step': global_step,
                       'best_epoch': best_epoch, 'best_val_loss': best_loss,
                       'model': cpu_snapshot(model.state_dict()), 'optimizer': cpu_snapshot(optimizer.state_dict()),
                       'rng': capture_rng(generators), 'history': cpu_snapshot(history)}
            if improved:
                best = payload
                atomic_save(best, run / 'checkpoints/best.pt')
            # Embed the best checkpoint so last/epoch checkpoints are self-contained across run ancestry.
            resumable = {**payload, 'best_checkpoint': best}
            atomic_save(resumable, run / f'checkpoints/epoch-{epoch:03d}.pt')
            atomic_save(resumable, run / 'checkpoints/last.pt')
            write_history(run, history)
            write_json(run / 'status.json', {'status': 'running', 'stage': config['stage'],
                                            'completed_epoch': epoch, 'global_step': global_step})
            print(f"epoch={epoch:02d} step={global_step} train_loss={train['train_loss']:.6f} "
                  f"train_acc={train['train_accuracy']:.4%} val_loss={validation['loss']:.6f} "
                  f"val_acc={validation['accuracy']:.4%} train_s={train['train_seconds']:.2f} "
                  f"data_wait_s={train['data_wait_seconds']:.2f} best_epoch={best_epoch}", flush=True)

        disk_last = load_checkpoint(run / 'checkpoints/last.pt', identity)
        assert all(torch.equal(disk_last['model'][name], value.cpu()) for name, value in model.state_dict().items())
        disk_best = load_checkpoint(run / 'checkpoints/best.pt', identity)
        model.load_state_dict(disk_best['model'])
        best_validation = evaluate_loader(model, loaders['validation'], 'cuda:0')
        expected_best = history[disk_best['completed_epoch'] - 1]
        assert best_validation['loss'] == expected_best['val_loss']
        assert best_validation['accuracy'] == expected_best['val_accuracy']
        draw_history(history, run / 'outputs/learning-curves.png')
        summary = {'stage': config['stage'], 'run_id': run.name, 'parent_run': parent_run,
                   'config': config, 'model': forward['model'], 'parameters': 809354,
                   'versions': versions, 'split_sha256': data_identity['split_sha256'],
                   'training_data_sha256': data_identity['training_data_sha256'],
                   'source_hashes': identity['source_hashes'],
                   'source_manifest_sha256': file_hash(run / 'source-manifest.json'),
                   'first_epoch_this_run': first_epoch, 'completed_epoch': until, 'global_step': global_step,
                   'best_epoch': disk_best['completed_epoch'], 'best_validation': best_validation,
                   'last': history[-1], 'history': history, 'test_evaluated': False,
                   'elapsed_this_run_seconds': time.perf_counter() - started,
                   'checkpoint_sha256': {name: file_hash(run / 'checkpoints' / name) for name in ('best.pt', 'last.pt')},
                   'checks': {'fixed_disjoint_split_and_original_data': True, 'all_train_samples_seen_once_per_epoch': True,
                              'finite_training_loss_gradients_parameters': True, 'optimizer_counters_match': True,
                              'validation_preserves_weights_and_creates_no_gradients': True,
                              'last_checkpoint_weights_match': True, 'best_checkpoint_reproduces_validation': True,
                              'checkpoints_load_with_weights_only': True},
                   'metric_note': 'Train metrics use changing pre-update parameters; validation uses the epoch-end model.',
                   'timing_note': 'Train time includes loading, transfer, finite checks and metric synchronization; not a pure GPU benchmark.',
                   'memory_note': 'PyTorch peak allocated memory; excludes driver overhead.',
                   'resume_scope': 'Epoch boundary, same code/config/data/software/GPU; cross-device equivalence is not claimed.'}
        write_json(run / 'metrics/summary.json', summary)
        write_json(run / 'status.json', {'status': 'complete', 'stage': config['stage'],
                                        'completed_epoch': until, 'global_step': global_step})
        write_json(project / '.local/latest-full-run.json', {'run_id': run.name, 'run_dir': str(run),
                                                            'stage': config['stage']})
        print(f"Completed: {run.name}; best_epoch={summary['best_epoch']} "
              f"best_val_loss={best_validation['loss']:.6f} best_val_accuracy={best_validation['accuracy']:.4%}", flush=True)
    except BaseException as exc:
        write_json(run / 'status.json', {'status': 'failed', 'stage': config['stage'],
                                        'error_type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
