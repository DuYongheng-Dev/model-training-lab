"""阶段 3：一个 batch、一次参数更新，以及独立的梯度清零对照。"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision.transforms import Normalize

from data import load_training_data, make_train_loader
from model import TinyViT
from run_record import file_hash, record_run, write_json


class IndexedDataset(torch.utils.data.Dataset):
    """Include original training indices so the observed batch can be reconstructed."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


def parameter_snapshot(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def all_parameters_equal(model, reference):
    return all(torch.equal(value, reference[name]) for name, value in model.named_parameters())


def inspect_one_step(model, images, labels, optimizer_config):
    """学习入口：五行训练主干之间插入观测；整个函数只调用一次 step。"""
    model.train()  # 与阶段 2 的 inference_mode 不同：这里需要建立计算图。
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    before = parameter_snapshot(model)

    optimizer.zero_grad(set_to_none=True)                 # 1. 清理旧梯度
    assert all(p.grad is None for p in model.parameters())
    trace = {}
    logits = model(images, trace=trace)                   # 2. [B,3,32,32] -> [B,10]
    logits.retain_grad()  # 仅为了核对 dloss/dlogits；正式训练不需要这一行。
    loss = criterion(logits, labels)                     # 3. [B,10] + [B] -> 标量
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all_parameters_equal(model, before)
    assert all(p.grad is None for p in model.parameters())

    loss.backward()                                     # 4. 计算并累加 .grad
    assert all_parameters_equal(model, before), 'Backward must not update parameters'
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    # 独立公式核对：CrossEntropyLoss = 真实类别的负 log 概率的均值。
    with torch.no_grad():
        log_probs = logits.log_softmax(dim=1)
        per_image_loss = -log_probs[torch.arange(len(labels), device=labels.device), labels]
        expected_logit_grad = (logits.softmax(dim=1) - nn.functional.one_hot(labels, logits.shape[1])) / len(labels)
        torch.testing.assert_close(loss, per_image_loss.mean(), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(logits.grad, expected_logit_grad, atol=1e-7, rtol=1e-5)

    optimizer.step()                                    # 5. 用梯度更新参数
    after = parameter_snapshot(model)
    assert not all_parameters_equal(model, before), 'Step must update at least one parameter'
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.equal(p.grad, gradients[name]) for name, p in model.named_parameters())
    assert len(optimizer.state) == len(before)
    assert all(int(state['step'].item()) == 1 for state in optimizer.state.values())

    # 只复查同一训练 batch 的 loss；保持 train 模式和同一 forward 路径。
    # dropout=0 且没有 BatchNorm；这里只读，不再 backward 或 step。
    logits_after = model(images)
    loss_after = criterion(logits_after, labels)
    optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters())
    assert all_parameters_equal(model, after), 'zero_grad must not reset learned weights'

    rows = []
    for name, weight in before.items():
        gradient = gradients[name]
        delta = after[name] - weight
        rows.append({'parameter': name, 'numel': weight.numel(),
                     'grad_l2': float(gradient.norm()), 'grad_mean_abs': float(gradient.abs().mean()),
                     'grad_max_abs': float(gradient.abs().max()), 'backward_delta_max_abs': 0.0,
                     'step_delta_l2': float(delta.norm()), 'step_delta_max_abs': float(delta.abs().max()),
                     'changed_elements': int(torch.count_nonzero(delta))})
    # 选取分类头中梯度绝对值最大的一个参数，便于展示数值变化。
    flat = int(gradients['head.weight'].abs().argmax())
    index = [flat // model.head.in_features, flat % model.head.in_features]
    selected = {'name': 'head.weight', 'index': index,
                'before': float(before['head.weight'].flatten()[flat]),
                'after_backward': float(before['head.weight'].flatten()[flat]),
                'gradient': float(gradients['head.weight'].flatten()[flat]),
                'after_step': float(after['head.weight'].flatten()[flat])}
    observation = {
        'loss_before': float(loss.detach()), 'loss_after_one_step_same_batch': float(loss_after.detach()),
        'loss_shape': list(loss.shape), 'logits_shape': list(logits.shape), 'labels_shape': list(labels.shape),
        'shapes': trace, 'head_grad_l2': float(gradients['head.weight'].norm()),
        'global_grad_l2': float(torch.stack([g.square().sum() for g in gradients.values()]).sum().sqrt()),
        'parameter_tensors': len(before), 'tensors_with_gradient': len(gradients),
        'tensors_changed_by_step': sum(row['changed_elements'] > 0 for row in rows),
        'selected_parameter': selected,
        'first_image': {'label': int(labels[0]), 'logits': logits[0].detach().cpu().tolist(),
                        'true_class_probability': float(log_probs[0, labels[0]].exp()),
                        'negative_log_probability': float(per_image_loss[0])},
        'logit_gradient_formula_max_abs_error': float((logits.grad - expected_logit_grad).abs().max()),
        'checks': {'grad_none_before_backward': True, 'parameters_unchanged_after_forward_and_backward': True,
                   'all_gradients_finite': True, 'parameters_changed_after_step': True,
                   'step_does_not_clear_gradients': True, 'zero_grad_preserves_updated_parameters': True,
                   'grad_none_after_final_zero': True, 'optimizer_state_steps_all_one': True,
                   'cross_entropy_and_logit_gradient_match_independent_formulas': True},
    }
    return observation, rows, before, after, gradients


def inspect_accumulation(model, images, labels, reference_gradients, config):
    """R1：固定初始权重；每次重新 forward，再 backward，不调用 step。"""
    model.train()
    before = parameter_snapshot(model)
    criterion = nn.CrossEntropyLoss()
    rows = []
    for mode in ('zero_each_time', 'accumulate_without_zero'):
        model.zero_grad(set_to_none=True)
        for repeat in range(1, config['accumulation_repeats'] + 1):
            if mode == 'zero_each_time':
                model.zero_grad(set_to_none=True)
            logits = model(images)  # 每轮建立新的图，不对同一 loss 重复反传。
            loss = criterion(logits, labels)
            loss.backward()
            multiplier = 1 if mode == 'zero_each_time' else repeat
            max_error = 0.0
            for name, parameter in model.named_parameters():
                expected = reference_gradients[name] * multiplier
                torch.testing.assert_close(parameter.grad, expected,
                                           atol=config['comparison_atol'], rtol=config['comparison_rtol'])
                max_error = max(max_error, float((parameter.grad - expected).abs().max()))
            head_norm = float(model.head.weight.grad.norm())
            rows.append({'mode': mode, 'backward_number': repeat, 'loss': float(loss.detach()),
                         'expected_multiplier': multiplier, 'head_grad_l2': head_norm,
                         'head_grad_norm_ratio': head_norm / float(reference_gradients['head.weight'].norm()),
                         'all_parameters_gradient_max_abs_error': max_error})
            assert all_parameters_equal(model, before)
        model.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters())
    return rows


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_accumulation(rows, destination):
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for mode, label in [('zero_each_time', 'Zero gradients each time'),
                        ('accumulate_without_zero', 'Zero only before the first backward')]:
        selected = [row for row in rows if row['mode'] == mode]
        axis.plot([r['backward_number'] for r in selected],
                  [r['head_grad_norm_ratio'] for r in selected], 'o-', label=label)
    axis.set(xlabel='Fresh forward + backward count', ylabel='Head gradient norm / single-backward norm',
             xticks=[1, 2, 3], yticks=[1, 2, 3], title='Same batch and fixed weights; no optimizer step')
    axis.grid(alpha=0.25)
    axis.legend()
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
    if selection['device'] != 'cuda' or os.environ.get('CUDA_VISIBLE_DEVICES') != selection['gpu_uuid']:
        raise ValueError('Use the assigned GPU via scripts/inspect_step.sh')
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert config['stage'] == '03-step' and config['optimizer_steps'] == 1
    assert config['accumulation_repeats'] == 3 and forward_config['model']['dropout'] == 0
    assert config['precision'] == 'float32_ieee' and config['optimizer']['weight_decay'] == 0
    versions = record_run(project, args, config)
    run = args.run_dir
    try:
        started = time.perf_counter()
        torch.set_num_threads(config['cpu_threads'])
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.backends.fp32_precision = 'ieee'
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(config['deterministic_algorithms'])
        prior = json.loads((project / 'results/01-data/summary.json').read_text())
        assert file_hash(args.split_path) == prior['split_sha256']
        split = json.loads(args.split_path.read_text())
        dataset = load_training_data(args.data_root, download=False)
        hashes = {name: file_hash(args.data_root / dataset.base_folder / name)
                  for name in prior['data_sha256'] if name.startswith('data_batch_') or name == 'batches.meta'}
        assert all(value == prior['data_sha256'][name] for name, value in hashes.items())
        data_config = json.loads((project / 'configs/01-data.json').read_text())
        data_config['batch_size'] = config['batch_size']
        raw_images, labels, indices = next(iter(make_train_loader(IndexedDataset(dataset), split, data_config)))
        assert labels[:16].tolist() == prior['first_16_labels']
        assert set(indices.tolist()) <= set(split['train_indices'])
        assert not (set(indices.tolist()) & set(split['val_indices']))
        images = Normalize(forward_config['normalize_mean'], forward_config['normalize_std'])(raw_images).to('cuda:0')
        labels = labels.to('cuda:0')
        assert images.shape == (config['batch_size'], 3, 32, 32) and images.dtype == torch.float32
        assert labels.shape == (config['batch_size'],) and labels.dtype == torch.int64
        model = TinyViT(**forward_config['model']).to('cuda:0')
        demo_model = copy.deepcopy(model)  # 独立模型保留初始权重，主模型只训练一步。
        step, parameter_rows, before, after, gradients = inspect_one_step(model, images, labels, config['optimizer'])
        accumulation = inspect_accumulation(demo_model, images, labels, gradients, config)
        summary = {
            'stage': '03-step', 'run_id': run.name, 'seed': config['seed'], 'device': 'cuda',
            'optimizer_steps': 1, 'accumulation_optimizer_steps': 0, 'validation_evaluated': False,
            'test_evaluated': False, 'batch_size': config['batch_size'], 'batch_indices': indices.tolist(),
            'first_16_labels': labels[:16].tolist(), 'model_config': forward_config['model'],
            'parameters': sum(p.numel() for p in model.parameters()), 'optimizer': config['optimizer'],
            'normalization': {'mean': forward_config['normalize_mean'], 'std': forward_config['normalize_std']},
            'precision': config['precision'], 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'versions': versions, 'split_sha256': file_hash(args.split_path), 'training_data_sha256': hashes,
            'source_manifest_sha256': file_hash(run / 'source-manifest.json'), 'step': step,
            'accumulation': accumulation, 'gradient_comparison_tolerance': {
                'atol': config['comparison_atol'], 'rtol': config['comparison_rtol']},
            'checks': {'same_training_split_and_data': True, 'accumulation_all_parameters_match': True,
                       'accumulation_weights_unchanged': True, 'accumulation_final_grad_none': True},
            'duration_seconds': time.perf_counter() - started,
            'timing_note': 'Whole observation duration, including diagnostics; not a training speed benchmark.',
        }
        write_json(run / 'metrics/summary.json', summary)
        write_csv(run / 'metrics/parameters.csv', parameter_rows)
        write_csv(run / 'metrics/accumulation.csv', accumulation)
        draw_accumulation(accumulation, run / 'outputs/gradient-accumulation.png')
        # 保存教学观测张量，不作为支持继续训练的 checkpoint（后者在阶段 5 实现）。
        torch.save({'before': {k: v.cpu() for k, v in before.items()},
                    'after_one_step': {k: v.cpu() for k, v in after.items()},
                    'single_backward_gradients': {k: v.cpu() for k, v in gradients.items()},
                    'batch_indices': indices, 'labels': labels.cpu()}, run / 'outputs/observations.pt')
        write_json(run / 'status.json', {'status': 'complete', 'stage': '03-step'})
        write_json(project / '.local/latest-step-run.json', {'run_id': run.name, 'run_dir': str(run)})
        print(json.dumps({'run_id': run.name, 'step': step, 'accumulation': accumulation}, indent=2))
        print(f'Observation saved: {run}', flush=True)
    except Exception as exc:
        write_json(run / 'status.json', {'status': 'failed', 'stage': '03-step',
                                       'error_type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
