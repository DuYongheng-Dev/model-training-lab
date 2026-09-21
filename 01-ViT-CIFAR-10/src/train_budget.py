"""阶段 6：从第 20 轮 last 延长至 50 轮，增加不影响训练状态的 train_eval。"""
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
from torch.utils.data import DataLoader

from checkpoint import atomic_save, capture_rng, cpu_snapshot, load_checkpoint, restore_rng
from compare_resume import compare_tree
from evaluate import evaluate_loader
from model import TinyViT
from run_record import file_hash, record_run, write_json
from train_full import build_loaders, train_epoch


CORE_SOURCES = ('model.py', 'data.py', 'train_full.py', 'evaluate.py', 'checkpoint.py')
SOURCES = (*CORE_SOURCES, 'train_budget.py', 'compare_resume.py', 'run_record.py')


def require_equal(left, right, label):
    report = compare_tree(left, right)
    if not report['exactly_equal']:
        raise ValueError(f'{label}: {report["first_mismatches"]}')
    return report


def validate_parent(parent, identity):
    """One explicit migration, then strict stage-6 resumes; no general mismatch bypass."""
    old = parent['identity']
    if old['config']['stage'] == '06-budget':
        if old != identity:
            raise ValueError('Stage 6 resume identity differs')
        kind = 'same_identity_resume'
    else:
        if old['config']['stage'] != '05-train' or parent['completed_epoch'] != 20:
            raise ValueError('Stage 6 must start from the full stage-5 epoch-20 last checkpoint')
        expected = cpu_snapshot(identity)
        expected['config'].update(stage='05-train', max_epochs=20)
        expected['runtime']['gpu_uuid'] = old['runtime']['gpu_uuid']
        expected['source_hashes'] = {key: identity['source_hashes'][key] for key in CORE_SOURCES}
        if old != expected:
            raise ValueError('Migration changes fields beyond stage, budget and assigned GPU')
        kind = 'stage5_budget_extension'
    epoch = parent['completed_epoch']
    history = parent['history']
    if [row['epoch'] for row in history] != list(range(1, epoch + 1)):
        raise ValueError('Incomplete checkpoint history')
    if parent['global_step'] != epoch * 352:
        raise ValueError('Unexpected optimizer progress')
    best = parent['best_checkpoint']
    selected = min(history, key=lambda row: row['val_loss'])
    if (best['completed_epoch'] != selected['epoch'] or parent['best_epoch'] != selected['epoch']
            or parent['best_val_loss'] != selected['val_loss']
            or best['best_val_loss'] != selected['val_loss']):
        raise ValueError('Historical best selection does not match validation history')
    return kind


def evaluate_readonly(model, optimizer, loader, generators, device='cuda:0'):
    """Extra observations must preserve weights, optimizer, mode, and every RNG stream."""
    before_rng = capture_rng(generators)
    before_model = cpu_snapshot(model.state_dict())
    before_optimizer = cpu_snapshot(optimizer.state_dict())
    was_training = model.training
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Observe only at an epoch boundary after clearing gradients')
    try:
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        started = time.perf_counter()
        metrics = evaluate_loader(model, loader, device)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    finally:
        restore_rng(before_rng, generators)
    require_equal(before_model, cpu_snapshot(model.state_dict()), 'Observation changed model')
    require_equal(before_optimizer, cpu_snapshot(optimizer.state_dict()), 'Observation changed optimizer')
    require_equal(before_rng, capture_rng(generators), 'Observation changed RNG')
    assert model.training == was_training and all(p.grad is None for p in model.parameters())
    return {**metrics, 'seconds': seconds}


def check_validation(actual, expected):
    difference = abs(actual['loss'] - expected['val_loss'])
    if difference > 1e-5 or actual['correct'] != expected['val_correct']:
        raise ValueError('Restored checkpoint validation exceeds the predeclared migration tolerance')
    return {'loss_absolute_difference': difference,
            'correct_count_equal': actual['correct'] == expected['val_correct'],
            'loss_absolute_tolerance': 1e-5}


def write_history(run, history):
    # Earlier epochs intentionally have blank train_eval columns, not invented measurements.
    columns = list(dict.fromkeys(key for row in history for key in row))
    with (run / 'metrics/epochs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(history)


def draw_history(history, baseline, destination):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, label in [('train', 'Train (during updates)'), ('val', 'Validation (epoch-end)')]:
        for axis, metric, scale in zip(axes, ('loss', 'accuracy'), (1, 100)):
            axis.plot([r['epoch'] for r in history], [scale*r[f'{name}_{metric}'] for r in history],
                      'o-', markersize=3, label=label)
    for axis, metric, scale in zip(axes, ('loss', 'accuracy'), (1, 100)):
        measured = [r for r in history if f'train_eval_{metric}' in r]
        axis.plot([20] + [r['epoch'] for r in measured],
                  [scale*baseline['train_eval'][metric]] + [scale*r[f'train_eval_{metric}'] for r in measured],
                  'o-', markersize=3, label='Train eval (epoch-end, starts at 20)')
        axis.axvline(20, color='gray', linestyle=':', label='Continuation / device change')
        axis.set_xlabel('Completed epoch')
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    axes[0].set_ylabel('Mean cross-entropy loss')
    axes[1].set_ylabel('Accuracy (%)')
    figure.suptitle('Tiny ViT / CIFAR-10: fixed recipe, total budget 50 epochs')
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'data-root', 'split-path', 'run-dir', 'resume'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--until', type=int, default=50)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    locked = json.loads((project / 'configs/05-train.json').read_text())
    locked.update(stage='06-budget', max_epochs=50)
    if config != locked:
        raise ValueError('Stage 6 only extends the stage-5 budget')
    assert config['max_epochs'] == 50 and 21 <= args.until <= 50
    forward = json.loads((args.config.parent / config['forward_config']).read_text())
    selection = json.loads((project / '.local/device.json').read_text())
    assert selection['device'] == 'cuda' and os.environ.get('CUDA_VISIBLE_DEVICES') == selection['gpu_uuid']
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    if not args.resume.resolve(strict=True).is_relative_to(Path(os.environ['LAB_ROOT']).resolve()):
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
        # Separate sequential loader sharing the same deterministic train dataset/view.
        eval_generator = torch.Generator().manual_seed(config['seed'] + 2)
        train_eval_loader = DataLoader(loaders['train'].dataset, batch_size=config['batch_size'],
            shuffle=False, drop_last=False, num_workers=0, pin_memory=config['pin_memory'], generator=eval_generator)
        observation_generators = {**generators, 'train_eval': eval_generator}
        write_json(run / 'data-identity.json', data_identity)
        identity = {'config': config, 'forward': forward, 'data': data_identity, 'versions': versions,
                    'source_hashes': {name: file_hash(project / 'src' / name) for name in SOURCES},
                    'runtime': {'gpu_uuid': selection['gpu_uuid'], 'torch_cuda': str(torch.version.cuda),
                                'cudnn': torch.backends.cudnn.version(),
                                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}}
        parent_hash = file_hash(args.resume)
        parent = load_checkpoint(args.resume)
        kind = validate_parent(parent, identity)
        completed, global_step = parent['completed_epoch'], parent['global_step']
        if completed >= args.until:
            raise ValueError('--until must exceed the checkpoint completed epoch')
        model = TinyViT(**forward['model']).to('cuda:0')
        assert sum(p.numel() for p in model.parameters()) == 809354
        optimizer = torch.optim.AdamW(model.parameters(), **config['optimizer'])
        model.load_state_dict(parent['model'])
        optimizer.load_state_dict(parent['optimizer'])
        history = cpu_snapshot(parent['history'])
        best = parent['best_checkpoint']
        atomic_save(best, run / 'checkpoints/best.pt')
        restore_rng(parent['rng'], generators)
        restore_checks = {
            'model': require_equal(parent['model'], cpu_snapshot(model.state_dict()), 'Restored model'),
            'optimizer': require_equal(parent['optimizer'], cpu_snapshot(optimizer.state_dict()), 'Restored optimizer'),
            'rng': require_equal(parent['rng'], capture_rng(generators), 'Restored RNG'),
        }
        assert len(optimizer.state) == len(list(model.parameters()))
        assert all(int(state['step']) == global_step for state in optimizer.state.values())
        write_json(run / 'resume.local.json', {'checkpoint': str(args.resume.resolve()),
            'checkpoint_sha256': parent_hash, 'parent_run': parent['run_id'], 'kind': kind,
            'source_gpu_uuid': parent['identity']['runtime']['gpu_uuid'],
            'target_gpu_uuid': selection['gpu_uuid'], 'completed_epoch': completed, 'next_epoch': completed + 1})

        if kind == 'stage5_budget_extension':
            # These two validations and the training-set baseline consume no training RNG.
            validation = evaluate_readonly(model, optimizer, loaders['validation'], observation_generators)
            restored_val = check_validation(validation, history[-1])
            train_eval = evaluate_readonly(model, optimizer, train_eval_loader, observation_generators)
            baseline = {'epoch': 20, 'train_eval': train_eval, 'validation': validation,
                        'restored_validation_check': restored_val}
            model.load_state_dict(best['model'])
            best_val = evaluate_readonly(model, optimizer, loaders['validation'], observation_generators)
            baseline['historical_best_validation_check'] = check_validation(best_val, history[best['completed_epoch']-1])
            baseline['historical_best_epoch'] = best['completed_epoch']
            baseline['historical_best_validation'] = best_val
            model.load_state_dict(parent['model'])
            lineage = {'stage5_parent_run': parent['run_id'], 'stage5_last_sha256': parent_hash,
                       'device_changed_at_epoch20': parent['identity']['runtime']['gpu_uuid'] != selection['gpu_uuid']}
        else:
            baseline, lineage = parent['baseline20'], parent['lineage']
        require_equal(parent['model'], cpu_snapshot(model.state_dict()), 'Pre-training model')
        require_equal(parent['optimizer'], cpu_snapshot(optimizer.state_dict()), 'Pre-training optimizer')
        require_equal(parent['rng'], capture_rng(generators), 'Pre-training RNG')
        write_json(run / 'metrics/resume-checks.json', {'kind': kind, 'checks': restore_checks,
                   'completed_epoch': completed, 'global_step': global_step, 'baseline20': baseline,
                   'scope': 'State loading checked exactly; no cross-GPU training trajectory equivalence claim.'})
        first_epoch = completed + 1
        print(f'run={run.name} epochs={first_epoch}..{args.until} resume={kind}; state checks passed', flush=True)
        for epoch in range(first_epoch, args.until + 1):
            torch.cuda.reset_peak_memory_stats()
            train = train_epoch(model, optimizer, loaders['train'], data_identity['indices']['train'])
            global_step += train['train_batches']
            train_eval = evaluate_readonly(model, optimizer, train_eval_loader, observation_generators)
            assert train_eval['count'] == 45000
            # Preserve the original stage-5 validation iteration and its generator advancement.
            torch.cuda.synchronize()
            validating = time.perf_counter()
            validation = evaluate_loader(model, loaders['validation'], 'cuda:0')
            torch.cuda.synchronize()
            validation_seconds = time.perf_counter() - validating
            assert validation['count'] == 5000 and model.training
            assert all(p.grad is None for p in model.parameters())
            assert all(int(state['step']) == global_step for state in optimizer.state.values())
            row = {'epoch': epoch, 'global_step': global_step, **train,
                   **{f'train_eval_{key}': value for key, value in train_eval.items()},
                   'val_loss': validation['loss'], 'val_accuracy': validation['accuracy'],
                   'val_correct': validation['correct'], 'val_count': validation['count'],
                   'validation_seconds': validation_seconds,
                   'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20}
            history.append(row)
            improved = validation['loss'] < best['best_val_loss']
            payload = {'schema_version': 1, 'run_id': run.name, 'parent_run': parent['run_id'],
                       'identity': identity, 'completed_epoch': epoch, 'global_step': global_step,
                       'best_epoch': epoch if improved else best['best_epoch'],
                       'best_val_loss': validation['loss'] if improved else best['best_val_loss'],
                       'model': cpu_snapshot(model.state_dict()), 'optimizer': cpu_snapshot(optimizer.state_dict()),
                       'rng': capture_rng(generators), 'history': cpu_snapshot(history),
                       'baseline20': baseline, 'lineage': lineage}
            if improved:
                best = payload
                atomic_save(best, run / 'checkpoints/best.pt')
            resumable = {**payload, 'best_checkpoint': best}
            atomic_save(resumable, run / f'checkpoints/epoch-{epoch:03d}.pt')
            atomic_save(resumable, run / 'checkpoints/last.pt')
            write_history(run, history)
            write_json(run / 'status.json', {'status': 'running', 'stage': config['stage'],
                                            'completed_epoch': epoch, 'global_step': global_step})
            print(f"epoch={epoch:02d} step={global_step} train_loss={train['train_loss']:.6f} "
                  f"train_acc={train['train_accuracy']:.2%} train_eval_loss={train_eval['loss']:.6f} "
                  f"train_eval_acc={train_eval['accuracy']:.2%} val_loss={validation['loss']:.6f} "
                  f"val_acc={validation['accuracy']:.2%} best_epoch={payload['best_epoch']}", flush=True)

        disk_last = load_checkpoint(run / 'checkpoints/last.pt', identity)
        require_equal(disk_last['model'], cpu_snapshot(model.state_dict()), 'Saved last model')
        require_equal(disk_last['optimizer'], cpu_snapshot(optimizer.state_dict()), 'Saved last optimizer')
        require_equal(disk_last['rng'], capture_rng(generators), 'Saved last RNG')
        # Inherited best retains its original identity; never relabel an old GPU checkpoint.
        disk_best = load_checkpoint(run / 'checkpoints/best.pt', best['identity'])
        require_equal(best, disk_best, 'Saved best checkpoint')
        model.load_state_dict(disk_best['model'])
        best_validation = evaluate_readonly(model, optimizer, loaders['validation'], observation_generators)
        best_check = check_validation(best_validation, history[disk_best['completed_epoch']-1])
        assert disk_best['completed_epoch'] == min(history, key=lambda r: r['val_loss'])['epoch']
        assert file_hash(args.resume) == parent_hash
        draw_history(history, baseline, run / 'outputs/learning-curves.png')
        summary = {'stage': config['stage'], 'run_id': run.name, 'parent_run': parent['run_id'],
                   'config': config, 'model': forward['model'], 'parameters': 809354,
                   'versions': versions, 'split_sha256': data_identity['split_sha256'],
                   'training_data_sha256': data_identity['training_data_sha256'],
                   'source_hashes': identity['source_hashes'], 'source_manifest_sha256': file_hash(run / 'source-manifest.json'),
                   'lineage': lineage, 'baseline20': baseline, 'first_epoch_this_run': first_epoch,
                   'completed_epoch': args.until, 'global_step': global_step,
                   'best_epoch': disk_best['completed_epoch'], 'best_validation': best_validation,
                   'best_validation_check': best_check, 'last': history[-1], 'history': history,
                   'test_evaluated': False, 'elapsed_this_run_seconds': time.perf_counter() - started,
                   'checkpoint_sha256': {name: file_hash(run / 'checkpoints' / name) for name in ('best.pt', 'last.pt')},
                   'checks': {'parent_training_identity_validated': True, 'loaded_state_exactly_equal': True,
                              'extra_observations_preserve_model_optimizer_rng': True,
                              'same_split_and_training_code': True, 'all_training_samples_once_per_epoch': True,
                              'finite_training_loss_gradients_parameters': True, 'optimizer_counters_match': True,
                              'saved_last_model_optimizer_rng_match': True, 'best_reproduces_validation_within_tolerance': True,
                              'source_checkpoint_unchanged': True},
                   'metric_note': 'Online train uses changing weights; train_eval and validation use the same epoch-end weights.',
                   'missing_data_note': 'train_eval measured from epoch 20; no backfilled earlier measurements.',
                   'resume_scope': 'Explicit budget/device migration at epoch 20; exact cross-device trajectory is not claimed.',
                   'timing_note': 'perf_counter; training includes loading, transfers and finite checks. Not a GPU speed comparison.'}
        write_json(run / 'metrics/summary.json', summary)
        write_json(run / 'status.json', {'status': 'complete', 'stage': config['stage'],
                                        'completed_epoch': args.until, 'global_step': global_step})
        write_json(project / '.local/latest-budget-run.json', {'run_id': run.name, 'run_dir': str(run), 'stage': config['stage']})
        print(f"Completed: {run.name}; best_epoch={summary['best_epoch']} val_accuracy={best_validation['accuracy']:.2%}", flush=True)
    except BaseException as exc:
        write_json(run / 'status.json', {'status': 'failed', 'stage': config['stage'],
                                        'error_type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
