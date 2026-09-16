"""Compare CPU and the assigned GPU using identical FP32 inference workloads."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import Normalize

from data import load_training_data, make_train_loader
from inspect_forward import file_hash, write_json
from model import TinyViT


def snapshot(project, run, config, args):
    for name in ('metrics', 'outputs', 'source', 'logs'):
        (run / name).mkdir(exist_ok=True)
    write_json(run / 'config.json', config)
    files = [project / name for name in ('README.md', 'PLAN.md')]
    files += list(project.glob('requirements*.txt'))
    for folder in ('src', 'scripts', 'configs', 'tests'):
        files += [p for p in (project / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    manifest = {}
    for source in sorted(files):
        relative = source.relative_to(project)
        destination = run / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest[str(relative)] = file_hash(destination)
    write_json(run / 'source-manifest.json', manifest)
    shutil.copyfile(args.split_path, run / 'split.json')
    (run / 'requirements-resolved.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    head = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=project, capture_output=True, text=True)
    write_json(run / 'metadata.local.json', {
        'machine_time_utc': datetime.now(timezone.utc).isoformat(),
        'clock_note': 'Machine wall clock is not calibrated; timings use perf_counter.',
        'command': [sys.executable, *sys.argv], 'executable': sys.executable,
        'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
        'gpu_name': torch.cuda.get_device_name(0),
        'gpu_snapshot': subprocess.check_output(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
            '--query-gpu=index,uuid,name,driver_version,memory.used,memory.free,temperature.gpu,power.draw',
            '--format=csv'], text=True),
        'data_root': str(args.data_root), 'run_dir': str(run),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'git_commit': head.stdout.strip() if head.returncode == 0 else None,
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=project, text=True),
        'code_status': 'unversioned' if head.returncode else 'versioned; see git_status',
    })


def measure(operation, iterations):
    # Synchronization includes actual GPU completion; wall time includes Python dispatch.
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        operation()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def plot_results(rows, destination):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels = {'cpu': 'CPU (4 threads)', 'gpu': 'GPU (input already on GPU)',
              'gpu_with_transfer': 'GPU + pageable host-to-device copy'}
    for mode, label in labels.items():
        selected = [r for r in rows if r['mode'] == mode]
        sizes = [r['batch_size'] for r in selected]
        axes[0].plot(sizes, [r['median_ms'] for r in selected], 'o-', label=label)
        axes[1].plot(sizes, [r['images_per_second'] for r in selected], 'o-', label=label)
    for axis in axes:
        axis.set_xscale('log', base=2)
        axis.set_yscale('log')
        axis.set_xlabel('Batch size (32 x 32 images)')
        axis.grid(True, which='both', alpha=0.2)
    axes[0].set_ylabel('Milliseconds per batch (median of round means)')
    axes[1].set_ylabel('Images / second')
    axes[0].legend(fontsize=7)
    figure.suptitle('Tiny ViT: FP32 inference, no training or disk I/O')
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'data-root', 'split-path', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    forward_config = json.loads((args.config.parent / config['forward_config']).read_text())
    selection = json.loads((project / '.local/device.json').read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES') != selection['gpu_uuid']:
        raise ValueError('Use the recorded GPU assignment via scripts/benchmark_devices.sh')
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    lab_root = Path(os.environ['LAB_ROOT']).resolve(strict=True)
    for path in (args.data_root, args.split_path, args.run_dir):
        if not path.resolve().is_relative_to(lab_root):
            raise ValueError('Assets and full outputs must remain under LAB_ROOT')
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'config.json').exists():
        raise FileExistsError('Use a new run directory')
    snapshot(project, run, config, args)
    write_json(run / 'status.json', {'status': 'running'})
    torch.set_num_threads(config['cpu_threads'])
    torch.set_num_interop_threads(1)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.fp32_precision = 'ieee'
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'ieee'
    torch.backends.cudnn.benchmark = False
    assert torch.backends.cudnn.enabled

    prior = json.loads((project / 'results/01-data/summary.json').read_text())
    assert file_hash(args.split_path) == prior['split_sha256']
    split = json.loads(args.split_path.read_text())
    dataset = load_training_data(args.data_root, download=False)
    hashes = {name: file_hash(args.data_root / dataset.base_folder / name)
              for name in prior['data_sha256'] if name.startswith('data_batch_') or name == 'batches.meta'}
    assert all(value == prior['data_sha256'][name] for name, value in hashes.items())
    data_config = json.loads((project / 'configs/01-data.json').read_text())
    data_config['batch_size'] = max(config['batch_sizes'])
    raw_images, labels = next(iter(make_train_loader(dataset, split, data_config)))
    assert labels[:16].tolist() == prior['first_16_labels']
    images = Normalize(forward_config['normalize_mean'], forward_config['normalize_std'])(raw_images)
    cpu_model = TinyViT(**forward_config['model']).eval()
    gpu_model = copy.deepcopy(cpu_model).to('cuda:0').eval()
    before = {name: value.clone() for name, value in cpu_model.state_dict().items()}
    rows, checks = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_size in config['batch_sizes']:
            host_input = images[:batch_size].contiguous()
            gpu_input = host_input.to('cuda:0')
            cpu_logits = cpu_model(host_input)
            gpu_logits = gpu_model(gpu_input).cpu()
            assert torch.isfinite(gpu_logits).all()
            torch.testing.assert_close(cpu_logits, gpu_logits, atol=1e-5, rtol=1e-4)
            checks.append({'batch_size': batch_size,
                           'cpu_gpu_max_abs_error': float((cpu_logits - gpu_logits).abs().max())})
            operations = {
                'cpu': lambda: cpu_model(host_input),
                'gpu': lambda: gpu_model(gpu_input),
                'gpu_with_transfer': lambda: gpu_model(host_input.to('cuda:0', non_blocking=False)),
            }
            for operation in operations.values():
                for _ in range(config['warmup_iterations']):
                    operation()
            torch.cuda.synchronize()
            samples = {mode: [] for mode in operations}
            peaks = {mode: 0 for mode in operations}
            modes = list(operations)
            for round_index in range(config['rounds']):
                # Rotate execution order to reduce a fixed ordering advantage.
                order = modes[round_index % len(modes):] + modes[:round_index % len(modes)]
                for mode in order:
                    torch.cuda.reset_peak_memory_stats()
                    samples[mode].append(measure(operations[mode], config['iterations_per_round']))
                    if mode != 'cpu':
                        peaks[mode] = max(peaks[mode], torch.cuda.max_memory_allocated())
            for mode in modes:
                median = statistics.median(samples[mode])
                q25, q75 = np.percentile(samples[mode], [25, 75]).tolist()
                row = {'batch_size': batch_size, 'mode': mode, 'median_ms': median,
                       'p25_ms': q25, 'p75_ms': q75, 'images_per_second': batch_size * 1000 / median,
                       'gpu_peak_allocated_mib': peaks[mode] / 2**20 if mode != 'cpu' else None,
                       'round_mean_ms': samples[mode]}
                rows.append(row)
                print(f'B={batch_size:3d} {mode:18s}: {median:9.3f} ms/batch; '
                      f'{row["images_per_second"]:10.1f} images/s', flush=True)
            del operations, gpu_input, cpu_logits, gpu_logits
    assert all(torch.equal(before[name], value) for name, value in cpu_model.state_dict().items())
    assert all(torch.equal(before[name], value.cpu()) for name, value in gpu_model.state_dict().items())
    assert all(p.grad is None for model in (cpu_model, gpu_model) for p in model.parameters())
    write_json(run / 'metadata-libraries.local.json', sorted({
        line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcudnn' in line
    }))
    summary = {
        'stage': config['stage'], 'run_id': run.name, 'config': config,
        'model_config': forward_config['model'],
        'parameters': sum(p.numel() for p in cpu_model.parameters()),
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'nvidia-cudnn-cu13')},
        'cuda_build': torch.version.cuda, 'cudnn_runtime': torch.backends.cudnn.version(),
        'precision': 'FP32 IEEE; TF32/AMP/compile disabled', 'mode': 'eval + inference_mode',
        'cpu_threads': torch.get_num_threads(), 'cpu_interop_threads': torch.get_num_interop_threads(),
        'split_sha256': file_hash(args.split_path), 'training_data_sha256': hashes,
        'source_manifest_sha256': file_hash(run / 'source-manifest.json'),
        'first_16_labels': labels[:16].tolist(),
        'normalization': {'mean': forward_config['normalize_mean'], 'std': forward_config['normalize_std']},
        'method': 'Synchronize, perf_counter, repeated forwards, synchronize. Median of 5 round means; not individual request latency.',
        'transfer': 'Pageable CPU memory, blocking .to(cuda:0), no overlap or pinned memory.',
        'excluded': ['disk I/O', 'DataLoader', 'normalization', 'backward', 'optimizer', 'first-call initialization'],
        'memory_note': 'PyTorch peak allocated GPU memory includes model and resident GPU input; CPU mode left null.',
        'training_steps': 0, 'test_evaluated': False,
        'checks': {'same_weights_and_inputs': True, 'state_unchanged': True, 'no_gradients': True,
                   'cpu_gpu_atol': 1e-5, 'cpu_gpu_rtol': 1e-4, 'batches': checks},
        'rows': rows, 'measurement_duration_seconds': time.perf_counter() - started,
    }
    write_json(run / 'metrics/summary.json', summary)
    with (run / 'metrics/timings.csv').open('w', newline='') as handle:
        fields = [name for name in rows[0] if name != 'round_mean_ms']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    plot_results(rows, run / 'outputs/device-comparison.png')
    write_json(run / 'status.json', {'status': 'complete'})
    write_json(project / '.local/latest-device-benchmark.json', {'run_dir': str(run), 'run_id': run.name})
    print(f'Benchmark saved: {run}', flush=True)


if __name__ == '__main__':
    main()
