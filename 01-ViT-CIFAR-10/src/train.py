"""阶段 4：固定 32 张训练图，显式训练循环；阶段 5 的完整训练尚未实现。"""
import argparse
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

from data import load_training_data
from evaluate import evaluate_batch
from model import TinyViT
from run_record import file_hash, record_run, write_json


def train_fixed_batch(model, images, labels, config, run):
    """学习主干：同一批图片每步重新 forward/backward，清零并更新一次。"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), **config['optimizer'])
    initial_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    initial, _ = evaluate_batch(model, images, labels)
    assert all(torch.equal(p, initial_parameters[name]) for name, p in model.named_parameters())
    assert model.training and all(p.grad is None for p in model.parameters())
    history = [{'step': 0, **initial, 'elapsed_seconds': 0.0, 'target_streak': 0}]
    target_streak, first_perfect_step = 0, None
    stop_reason = 'max_steps_reached'
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    print(f"step=0 loss={initial['loss']:.6f} accuracy={initial['correct']}/32", flush=True)

    with (run / 'metrics/steps.csv').open('w', newline='') as step_handle, \
            (run / 'metrics/evaluations.csv').open('w', newline='') as eval_handle:
        steps = csv.DictWriter(step_handle, fieldnames=['step', 'loss_before_update'])
        evaluations = csv.DictWriter(eval_handle, fieldnames=list(history[0]))
        steps.writeheader()
        evaluations.writeheader()
        evaluations.writerow(history[0])
        for step in range(1, config['max_steps'] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)      # 每步只用这一次 backward 的梯度
            logits = model(images)                    # 固定的同一批 32 张图
            loss = criterion(logits, labels)
            assert torch.isfinite(loss), f'Non-finite loss at step {step}'
            loss.backward()
            assert all(p.grad is not None for p in model.parameters())
            assert torch.stack([p.grad.isfinite().all() for p in model.parameters()]).all(), 'Non-finite gradients'
            if step == 1:
                assert all(torch.equal(p, initial_parameters[name]) for name, p in model.named_parameters())
            optimizer.step()
            if step == 1:
                assert any(not torch.equal(p, initial_parameters[name]) for name, p in model.named_parameters())
                del initial_parameters
            steps.writerow({'step': step, 'loss_before_update': float(loss.detach())})
            del logits, loss

            if step % config['evaluate_every'] == 0 or step == config['max_steps']:
                optimizer.zero_grad(set_to_none=True)
                # 评估的还是这 32 张训练图，不是 validation 集。
                metrics, _ = evaluate_batch(model, images, labels)
                assert model.training and all(p.grad is None for p in model.parameters())
                assert torch.stack([p.isfinite().all() for p in model.parameters()]).all()
                if metrics['correct'] == len(labels) and first_perfect_step is None:
                    first_perfect_step = step
                qualifies = (metrics['accuracy'] >= config['target']['accuracy']
                             and metrics['loss'] <= config['target']['loss_at_most'])
                target_streak = target_streak + 1 if qualifies else 0
                row = {'step': step, **metrics, 'elapsed_seconds': time.perf_counter() - started,
                       'target_streak': target_streak}
                history.append(row)
                evaluations.writerow(row)
                step_handle.flush()
                eval_handle.flush()
                print(f"step={step} loss={metrics['loss']:.6f} accuracy={metrics['correct']}/32 "
                      f"target_streak={target_streak}", flush=True)
                if target_streak >= config['target']['consecutive_evaluations']:
                    stop_reason = 'target_met_three_consecutive_evaluations'
                    break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    assert all(int(state['step'].item()) == step for state in optimizer.state.values())
    assert len(optimizer.state) == len(list(model.parameters()))
    return history, {
        'completed_steps': step, 'stop_reason': stop_reason,
        'first_observed_perfect_accuracy_step': first_perfect_step,
        'training_and_evaluation_seconds': elapsed,
        'gpu_peak_allocated_mib': peak_mib,
        'timing_note': 'Includes finite-value checks, metric synchronization, and CSV writes; not a throughput benchmark.',
        'memory_note': 'PyTorch peak allocated memory during training/measurement, including diagnostics; excludes driver overhead.',
    }


def draw_curves(history, target_loss, destination):
    steps = [row['step'] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(steps, [row['loss'] for row in history], 'o-', markersize=3)
    axes[0].axhline(target_loss, color='gray', linestyle='--', label=f'Target loss = {target_loss}')
    axes[0].set_ylabel('Cross-entropy loss')
    axes[0].legend()
    axes[1].plot(steps, [100 * row['accuracy'] for row in history], 'o-', markersize=3)
    axes[1].set(ylabel='Accuracy (%)', ylim=(0, 105))
    for axis in axes:
        axis.set_xlabel('Completed optimizer steps')
        axis.grid(alpha=0.25)
    figure.suptitle('Same 32 training images evaluated after updates (not validation)')
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def save_predictions(raw_images, labels, logits, indices, classes, run):
    probabilities = logits.softmax(dim=1).cpu()
    predictions = probabilities.argmax(dim=1)
    labels = labels.cpu()
    rows = []
    figure, axes = plt.subplots(4, 8, figsize=(14, 8), constrained_layout=True)
    for slot, axis in enumerate(axes.flat):
        truth, predicted = int(labels[slot]), int(predictions[slot])
        row = {'slot': slot, 'training_index': indices[slot], 'true_label': truth,
               'true_class': classes[truth], 'predicted_label': predicted,
               'predicted_class': classes[predicted], 'correct': truth == predicted,
               'true_class_probability': float(probabilities[slot, truth])}
        rows.append(row)
        axis.imshow(raw_images[slot].permute(1, 2, 0).numpy(), interpolation='nearest')
        axis.set_title(f"#{slot + 1} true: {classes[truth]}\npred: {classes[predicted]}", fontsize=8,
                       color='darkgreen' if truth == predicted else 'darkred')
        axis.axis('off')
    figure.suptitle('Final predictions on the 32 images used for training')
    figure.savefig(run / 'outputs/predictions.png', dpi=160)
    plt.close(figure)
    with (run / 'metrics/predictions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
        raise ValueError('Launch with the assigned GPU via scripts/overfit32.sh')
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert config['stage'] == '04-overfit32' and config['batch_size'] == 32
    assert config['subset_selection'] == 'first_32_saved_train_indices'
    assert 1 <= config['max_steps'] <= 1000 and config['evaluate_every'] == 10
    assert config['target'] == {'accuracy': 1.0, 'loss_at_most': 0.05, 'consecutive_evaluations': 3}
    assert config['precision'] == 'float32_ieee'
    assert config['optimizer']['weight_decay'] == 0 and forward_config['model']['dropout'] == 0
    versions = record_run(project, args, config)
    run = args.run_dir
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
        torch.use_deterministic_algorithms(config['deterministic_algorithms'])
        prior = json.loads((project / 'results/01-data/summary.json').read_text())
        assert file_hash(args.split_path) == prior['split_sha256']
        split = json.loads(args.split_path.read_text())
        indices = split['train_indices'][:32]
        assert len(indices) == len(set(indices)) == 32
        assert not set(indices).intersection(split['val_indices'])
        dataset = load_training_data(args.data_root, download=False)
        hashes = {name: file_hash(args.data_root / dataset.base_folder / name)
                  for name in prior['data_sha256'] if name.startswith('data_batch_') or name == 'batches.meta'}
        assert all(value == prior['data_sha256'][name] for name, value in hashes.items())
        samples = [dataset[index] for index in indices]
        raw_images = torch.stack([image for image, _ in samples])
        labels = torch.tensor([label for _, label in samples], dtype=torch.long)
        class_counts = torch.bincount(labels, minlength=10).tolist()
        write_json(run / 'subset.json', {'selection': config['subset_selection'], 'training_indices': indices,
                                         'labels': labels.tolist(), 'class_counts': class_counts,
                                         'split_sha256': file_hash(args.split_path)})
        images = Normalize(forward_config['normalize_mean'], forward_config['normalize_std'])(raw_images).to('cuda:0')
        labels = labels.to('cuda:0')
        assert images.shape == (32, 3, 32, 32) and images.dtype == torch.float32
        model = TinyViT(**forward_config['model']).to('cuda:0')
        assert sum(p.numel() for p in model.parameters()) == 809354
        history, training = train_fixed_batch(model, images, labels, config, run)
        final, final_logits = evaluate_batch(model, images, labels)
        assert final == {key: history[-1][key] for key in final}
        state_before_eval = {name: value.detach().clone() for name, value in model.state_dict().items()}
        evaluate_batch(model, images, labels)
        assert all(torch.equal(state_before_eval[name], value) for name, value in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        del state_before_eval

        # 权重供后续查看；完整 optimizer/RNG 恢复协议在阶段 5 实现。
        weights_path = run / 'outputs/final-weights.pt'
        temporary = weights_path.with_suffix('.tmp')
        torch.save({name: value.cpu() for name, value in model.state_dict().items()}, temporary)
        temporary.replace(weights_path)
        reloaded = TinyViT(**forward_config['model']).to('cuda:0')
        reloaded.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        reloaded_metrics, reloaded_logits = evaluate_batch(reloaded, images, labels)
        torch.testing.assert_close(reloaded_logits, final_logits, atol=1e-6, rtol=1e-5)
        assert reloaded_metrics == final
        summary = {
            'stage': config['stage'], 'run_id': run.name, 'seed': config['seed'], 'device': 'cuda',
            'config': config, 'model_config': forward_config['model'], 'parameters': 809354,
            'initialization': 'fresh random initialization with seed; does not resume stage 03 weights',
            'normalization': {'mean': forward_config['normalize_mean'], 'std': forward_config['normalize_std']},
            'subset_indices': indices, 'class_counts': class_counts, 'class_names': dataset.classes,
            'subset_sha256': file_hash(run / 'subset.json'), 'split_sha256': file_hash(args.split_path),
            'training_data_sha256': hashes, 'versions': versions,
            'source_manifest_sha256': file_hash(run / 'source-manifest.json'),
            'initial': {key: history[0][key] for key in final}, 'final': final,
            'history': history, **training,
            'target_met': history[-1]['target_streak'] >= config['target']['consecutive_evaluations'],
            'validation_evaluated': False, 'test_evaluated': False,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'final_weights_sha256': file_hash(weights_path),
            'weight_reload_max_abs_logit_error': float((final_logits - reloaded_logits).abs().max()),
            'checks': {'fixed_unique_training_samples': True, 'validation_excluded': True,
                       'original_split_and_training_data_unchanged': True, 'finite_losses_gradients_and_parameters': True,
                       'first_backward_preserves_parameters_and_step_changes_them': True,
                       'all_optimizer_counters_match_completed_steps': True,
                       'evaluation_preserves_parameters_and_restores_train_mode': True,
                       'evaluation_does_not_create_gradients': True, 'weight_reload_matches_final_predictions': True},
        }
        write_json(run / 'metrics/summary.json', summary)
        draw_curves(history, config['target']['loss_at_most'], run / 'outputs/learning-curves.png')
        save_predictions(raw_images, labels, final_logits, indices, dataset.classes, run)
        write_json(run / 'status.json', {'status': 'complete', 'stage': config['stage'],
                                        'target_met': summary['target_met'], 'stop_reason': training['stop_reason']})
        write_json(project / '.local/latest-overfit32-run.json', {'run_id': run.name, 'run_dir': str(run)})
        print(json.dumps({'run_id': run.name, **training, 'final': final, 'target_met': summary['target_met']}, indent=2))
        print(f'Run saved: {run}', flush=True)
    except Exception as exc:
        write_json(run / 'status.json', {'status': 'failed', 'stage': config['stage'],
                                        'error_type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
