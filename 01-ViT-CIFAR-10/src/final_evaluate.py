"""Freeze the validation-selected model, then evaluate official CIFAR-10 test once."""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from checkpoint import atomic_save, cpu_snapshot, load_checkpoint
from compare_resume import compare_tree
from evaluate import evaluate_loader
from final_metrics import draw_confusion, draw_examples, predict_loader, select_examples, summarize_predictions, write_csv
from model import TinyViT
from run_record import file_hash, record_run, write_json
from train_full import IndexedView


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'data-root', 'split-path', 'run-dir', 'source-run'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    assert config['stage'] == '07-final' and config['test_samples'] == 10000
    assert config['selection_metric'] == 'minimum_validation_loss' and config['selection_epochs'] == [1, 50]
    assert config['batch_size'] == 128 and config['num_workers'] == 0
    assert config['precision'] == 'float32_ieee' and config['deterministic_algorithms']
    assert not config['download'] and config['examples_per_outcome'] == 10
    assert config['example_selection'] == 'first_in_official_test_order'
    assert args.source_run.resolve(strict=True).is_relative_to(Path(os.environ['LAB_ROOT']).resolve())
    assert args.source_run.name == config['training_run_id']
    selection = json.loads((project / '.local/device.json').read_text())
    assert selection['device'] == 'cuda' and os.environ['CUDA_VISIBLE_DEVICES'] == selection['gpu_uuid']
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    versions = record_run(project, args, config)
    run = args.run_dir
    started = time.perf_counter()
    try:
        unit_checks = json.loads((run / 'metrics/unit-checks.json').read_text())
        assert unit_checks['passed'] and not unit_checks['official_test_used']
        assert unit_checks['source_sha256'] == file_hash(project / unit_checks['entry'])
        source = json.loads((args.source_run / 'metrics/summary.json').read_text())
        assert source['completed_epoch'] == 50 and source['global_step'] == 17600
        assert file_hash(args.source_run / 'metrics/summary.json') == file_hash(project / 'results/06-budget/summary.json')
        assert [row['epoch'] for row in source['history']] == list(range(1, 51))
        chosen = min(source['history'], key=lambda row: row['val_loss'])
        assert source['best_epoch'] == config['selected_epoch'] == chosen['epoch']
        checkpoint_path = args.source_run / 'checkpoints/best.pt'
        digest = file_hash(checkpoint_path)
        assert digest == config['checkpoint_sha256'] == source['checkpoint_sha256']['best.pt']
        checkpoint = load_checkpoint(checkpoint_path)
        assert checkpoint['completed_epoch'] == chosen['epoch'] and checkpoint['best_val_loss'] == chosen['val_loss']
        assert checkpoint['global_step'] == chosen['global_step']
        forward = json.loads((args.config.parent / config['forward_config']).read_text())
        assert forward == checkpoint['identity']['forward']
        assert versions == source['versions'] == checkpoint['identity']['versions']
        for name in ('model.py', 'data.py', 'train_full.py', 'evaluate.py', 'checkpoint.py'):
            assert file_hash(project / 'src' / name) == checkpoint['identity']['source_hashes'][name]
        assert file_hash(args.split_path) == source['split_sha256'] == checkpoint['identity']['data']['split_sha256']
        data_record = json.loads((project / 'results/01-data/summary.json').read_text())
        assert data_record['split_sha256'] == source['split_sha256']
        selection_record = {'rule': config['selection_metric'], 'candidate_epochs': [1, 50],
            'source_training_run': source['run_id'], 'checkpoint_source_run': checkpoint['run_id'],
            'selected_epoch': chosen['epoch'], 'checkpoint_sha256': digest,
            'validation_loss': chosen['val_loss'], 'validation_accuracy': chosen['val_accuracy'],
            'frozen_before_test_predictions': True}
        write_json(run / 'metrics/model-selection.json', selection_record)
        write_json(run / 'checkpoint-source.local.json', {'path': str(checkpoint_path.resolve()),
            'source_gpu_uuid': checkpoint['identity']['runtime']['gpu_uuid'], 'target_gpu_uuid': selection['gpu_uuid'],
            'scope': 'Frozen inference; training state is not resumed.'})

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
        model = TinyViT(**forward['model']).to('cuda:0')
        model.load_state_dict(checkpoint['model'], strict=True)
        model.requires_grad_(False)
        model.eval()
        assert sum(p.numel() for p in model.parameters()) == 809354
        before = cpu_snapshot(model.state_dict())
        assert compare_tree(checkpoint['model'], before)['exactly_equal']
        transform = Compose([ToTensor(), Normalize(forward['normalize_mean'], forward['normalize_std'])])
        split = json.loads(args.split_path.read_text())
        assert split['val_indices'] == checkpoint['identity']['data']['indices']['validation']
        for name, expected in source['training_data_sha256'].items():
            assert file_hash(args.data_root / CIFAR10.base_folder / name) == expected
        validation_data = CIFAR10(root=str(args.data_root), train=True, download=False, transform=transform)
        validation_loader = DataLoader(IndexedView(validation_data, split['val_indices']),
            batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=config['pin_memory'],
            generator=torch.Generator().manual_seed(config['seed']+1))
        validation = evaluate_loader(model, validation_loader, 'cuda:0')
        assert validation['count'] == 5000 and validation['correct'] == chosen['val_correct']
        assert abs(validation['loss']-chosen['val_loss']) <= 1e-5
        write_json(run / 'metrics/preflight.json', {'unit_checks': unit_checks,
            'validation': validation, 'expected_validation_loss': chosen['val_loss'],
            'loss_absolute_difference': abs(validation['loss']-chosen['val_loss']), 'loss_tolerance': 1e-5,
            'loaded_weights_exact': True, 'test_inference_started': False})
        del validation_loader, validation_data
        print(f"Frozen epoch {chosen['epoch']}; validation recheck passed; starting official test", flush=True)

        # Only this dataset/loader is used for the one full official-test forward pass.
        test_hashes = {name: file_hash(args.data_root / CIFAR10.base_folder / name)
                       for name in ('test_batch', 'batches.meta')}
        assert all(value == data_record['data_sha256'][name] for name, value in test_hashes.items())
        dataset = CIFAR10(root=str(args.data_root), train=False, download=False, transform=transform)
        assert dataset.classes == data_record['classes'] and len(dataset) == config['test_samples']
        assert np.bincount(dataset.targets, minlength=10).tolist() == [1000]*10
        loader = DataLoader(IndexedView(dataset, list(range(len(dataset)))), batch_size=config['batch_size'],
            shuffle=False, drop_last=False, num_workers=0, pin_memory=config['pin_memory'],
            generator=torch.Generator().manual_seed(config['seed']+2))
        rows, logits, performance = predict_loader(model, loader, 'cuda:0')
        metrics = summarize_predictions(rows, dataset.classes)
        assert [row['index'] for row in rows] == list(range(10000))
        assert performance['batches'] == 79 and performance['last_batch_size'] == 16
        assert logits.shape == (10000, 10)
        matrix = np.asarray(metrics['confusion_matrix'])
        assert int(matrix.sum()) == metrics['count'] == 10000
        assert matrix.sum(axis=1).tolist() == [1000]*10
        assert int(matrix.trace()) == metrics['correct']
        assert compare_tree(before, cpu_snapshot(model.state_dict()))['exactly_equal']
        assert not model.training and all(p.grad is None for p in model.parameters())
        assert all(not p.requires_grad for p in model.parameters())
        assert file_hash(checkpoint_path) == digest
        write_csv(run / 'outputs/predictions.csv', rows)
        atomic_save({'logits': logits, 'indices': torch.tensor([r['index'] for r in rows]),
                     'labels': torch.tensor([r['true_label'] for r in rows])}, run / 'outputs/predictions.pt')
        write_csv(run / 'metrics/per-class.csv', metrics['per_class'])
        write_csv(run / 'metrics/confusion-matrix.csv', [
            {'true_class': name, **dict(zip(dataset.classes, counts))}
            for name, counts in zip(dataset.classes, metrics['confusion_matrix'])])
        examples = select_examples(rows, config['examples_per_outcome'])
        assert all(len(items) == 10 for items in examples.values())
        write_json(run / 'metrics/examples.json', {'selection': config['example_selection'],
            'classes': dataset.classes, 'confidence_note': 'Maximum softmax score; not calibrated correctness probability.',
            **examples})
        draw_confusion(metrics, dataset.classes, run / 'outputs/confusion-matrix.png')
        for name, items in examples.items():
            draw_examples(dataset.data, items, dataset.classes, name.capitalize(), run / f'outputs/examples-{name}.png')
        summary = {'stage': config['stage'], 'run_id': run.name, 'config': config, 'selection': selection_record,
            'parameters': 809354, 'model': forward['model'], 'versions': versions, 'classes': dataset.classes,
            'split_sha256': source['split_sha256'], 'test_data_sha256': test_hashes, 'validation_recheck': validation,
            'test': metrics, 'test_evaluated': True, 'test_passes_this_run': 1, 'training_updates_this_run': 0,
            'performance': performance, 'elapsed_this_run_seconds': time.perf_counter()-started,
            'source_hashes': {name: file_hash(project / 'src' / name) for name in
                              ('model.py', 'final_metrics.py', 'final_evaluate.py', 'evaluate.py', 'checkpoint.py', 'run_record.py')},
            'source_manifest_sha256': file_hash(run / 'source-manifest.json'),
            'predictions_sha256': file_hash(run / 'outputs/predictions.csv'),
            'checks': {'selection_fixed_before_test': True, 'validation_reproduced': True,
                       'original_official_test_and_class_order': True, 'all_test_indices_once': True,
                       'confusion_counts_consistent': True, 'weights_and_checkpoint_unchanged': True,
                       'no_gradients_or_training_updates': True, 'finite_outputs': True},
            'scope': 'One seed and one validation-selected model. Test results are descriptive, not used for tuning.',
            'timing_note': 'Test time includes loading, transfer, forward, softmax and prediction collection; not pure GPU latency.'}
        write_json(run / 'metrics/summary.json', summary)
        write_json(run / 'status.json', {'status': 'complete', 'stage': config['stage'],
            'test_samples': 10000, 'selected_epoch': chosen['epoch'], 'training_updates': 0})
        write_json(project / '.local/latest-final-run.json', {'run_id': run.name, 'run_dir': str(run), 'stage': config['stage']})
        print(f"Completed {run.name}: test_loss={metrics['loss']:.6f} test_accuracy={metrics['accuracy']:.2%} "
              f"correct={metrics['correct']}/10000; no training updates", flush=True)
    except BaseException as exc:
        write_json(run / 'status.json', {'status': 'failed', 'stage': config['stage'],
            'error_type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
