"""Compare saved trajectories without running another model or touching checkpoints."""
import argparse
import json
from pathlib import Path

import torch

from checkpoint import load_checkpoint
from run_record import file_hash, write_json


def compare_tree(left, right):
    errors = []
    max_abs_error = 0.0
    tensor_count = 0

    def visit(a, b, location):
        nonlocal max_abs_error, tensor_count
        if isinstance(a, torch.Tensor):
            tensor_count += 1
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                errors.append(location + ': tensor shape/type differs')
                return
            if a.numel() and a.is_floating_point():
                difference = float((a.double() - b.double()).abs().max())
                max_abs_error = max(max_abs_error, difference)
            if not torch.equal(a, b):
                errors.append(location + ': tensor values differ')
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                errors.append(location + ': keys differ')
                return
            for key in a:
                visit(a[key], b[key], f'{location}.{key}')
        elif isinstance(a, (list, tuple)):
            if not isinstance(b, type(a)) or len(a) != len(b):
                errors.append(location + ': sequence differs')
                return
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                visit(item_a, item_b, f'{location}[{index}]')
        elif a != b:
            errors.append(location + ': scalar differs')

    visit(left, right, 'state')
    return {'exactly_equal': not errors, 'max_abs_tensor_error': max_abs_error,
            'tensor_count': tensor_count, 'mismatch_count': len(errors), 'first_mismatches': errors[:5]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--resumed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    left, right = load_checkpoint(args.reference), load_checkpoint(args.resumed)
    timing_keys = {'train_seconds', 'data_wait_seconds', 'train_samples_per_second',
                   'validation_seconds', 'gpu_peak_allocated_mib'}
    metrics = lambda rows: [{k: v for k, v in row.items() if k not in timing_keys} for row in rows]
    comparisons = {name: compare_tree(left[name], right[name]) for name in ('identity', 'model', 'optimizer', 'rng')}
    comparisons['metrics_and_sample_order'] = compare_tree(metrics(left['history']), metrics(right['history']))
    comparisons['progress_and_selection'] = compare_tree(
        {k: left[k] for k in ('completed_epoch', 'global_step', 'best_epoch', 'best_val_loss')},
        {k: right[k] for k in ('completed_epoch', 'global_step', 'best_epoch', 'best_val_loss')})
    comparisons['best_model'] = compare_tree(left.get('best_checkpoint', left)['model'],
                                             right.get('best_checkpoint', right)['model'])
    report = {'reference_run': left['run_id'], 'resumed_run': right['run_id'],
              'completed_epoch': left['completed_epoch'], 'global_step': left['global_step'],
              'reference_checkpoint_sha256': file_hash(args.reference),
              'resumed_checkpoint_sha256': file_hash(args.resumed), 'comparisons': comparisons,
              'all_exactly_equal': all(item['exactly_equal'] for item in comparisons.values()),
              'scope': 'Same GPU, software, code, config and data; elapsed time/memory excluded.'}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report['all_exactly_equal']:
        raise SystemExit('Resume comparison failed')


if __name__ == '__main__':
    main()
