"""Export small stage-6 results; checkpoints, full logs and machine identity stay in LAB_ROOT."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_record import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    for run in (args.preflight, args.run):
        assert run.resolve().is_relative_to(Path(os.environ['LAB_ROOT']).resolve())
        status = json.loads((run / 'status.json').read_text())
        assert status['status'] == 'complete'
    summary = json.loads((args.run / 'metrics/summary.json').read_text())
    preflight = json.loads((args.preflight / 'metrics/summary.json').read_text())
    unit_checks = json.loads((args.preflight / 'metrics/unit-checks.json').read_text())
    assert unit_checks['exit_code'] == 0 and unit_checks['tests_run'] == 3
    assert unit_checks['source_sha256'] == file_hash(project / unit_checks['entry'])
    assert summary['completed_epoch'] == 50 and summary['global_step'] == 17600
    assert summary['first_epoch_this_run'] == 22 and preflight['first_epoch_this_run'] == 21
    assert preflight['completed_epoch'] == 21 and summary['parent_run'] == preflight['run_id']
    assert summary['source_hashes'] == preflight['source_hashes']
    assert all(summary['checks'].values()) and all(preflight['checks'].values())
    assert not summary['test_evaluated'] and not preflight['test_evaluated']
    history = summary['history']
    assert [r['epoch'] for r in history] == list(range(1, 51))
    assert summary['best_epoch'] == min(history, key=lambda row: row['val_loss'])['epoch']
    assert preflight['history'] == history[:21]
    old = json.loads((project / 'results/05-train/summary.json').read_text())
    assert history[:20] == old['history']
    assert all('train_eval_loss' not in row for row in history[:20])
    assert all(row['train_eval_count'] == 45000 for row in history[20:])
    for run in (args.preflight, args.run):
        assert json.loads((run / 'source-manifest.json').read_text())['src/train_budget.py'] == summary['source_hashes']['train_budget.py']
    for name, digest in summary['checkpoint_sha256'].items():
        assert file_hash(args.run / 'checkpoints' / name) == digest

    destination = project / 'results/06-budget'
    destination.mkdir(exist_ok=False)
    for source, name in [('metrics/summary.json', 'summary.json'), ('metrics/epochs.csv', 'epochs.csv'),
                         ('outputs/learning-curves.png', 'learning-curves.png'),
                         ('metrics/resume-checks.json', 'resume-checks.json')]:
        shutil.copyfile(args.run / source, destination / name)
    shutil.copyfile(args.preflight / 'metrics/resume-checks.json', destination / 'migration-checks.json')
    write_json(destination / 'unit-checks.json', unit_checks)
    write_json(destination / 'baseline20.json', summary['baseline20'])
    added = history[20:]
    resources = {'additional_full_training_epochs': len(added), 'additional_optimizer_steps': 30*352,
                 'trajectory_epochs': 50, 'trajectory_optimizer_steps': 17600,
                 'training_seconds': sum(r['train_seconds'] for r in added),
                 'train_eval_seconds': sum(r['train_eval_seconds'] for r in added),
                 'validation_seconds': sum(r['validation_seconds'] for r in added),
                 'epoch20_train_eval_seconds': summary['baseline20']['train_eval']['seconds'],
                 'wall_seconds_two_runs': summary['elapsed_this_run_seconds'] + preflight['elapsed_this_run_seconds'],
                 'train_seconds_median': statistics.median(r['train_seconds'] for r in added),
                 'train_samples_per_second_median': statistics.median(r['train_samples_per_second'] for r in added),
                 'gpu_peak_allocated_mib': max(r['gpu_peak_allocated_mib'] for r in added),
                 'data_wait_fraction': sum(r['data_wait_seconds'] for r in added)/sum(r['train_seconds'] for r in added),
                 'timing_note': 'Training includes data/transfer/checks; wall includes setup, extra validation and checkpoint IO.',
                 'scope': 'One seed; no repeated full training epochs in stage 6. Not a cross-device speed benchmark.'}
    write_json(destination / 'resources.json', resources)
    comparisons = {'epoch20': history[19], 'epoch20_train_eval': summary['baseline20']['train_eval'],
                   'epoch50': history[-1], 'global_best': history[summary['best_epoch']-1],
                   'maximum_validation_accuracy_epoch': max(history, key=lambda r: r['val_accuracy'])['epoch'],
                   'windows': {}}
    for low, high in [(11, 20), (21, 30), (31, 40), (41, 50)]:
        rows = history[low-1:high]
        comparisons['windows'][f'{low}-{high}'] = {
            key: statistics.mean(r[key] for r in rows) for key in ('train_loss', 'train_accuracy', 'val_loss', 'val_accuracy')}
        if low >= 21:
            comparisons['windows'][f'{low}-{high}'].update({key: statistics.mean(r[key] for r in rows)
                                                         for key in ('train_eval_loss', 'train_eval_accuracy')})
    write_json(destination / 'comparison.json', comparisons)
    write_json(destination / 'run-index.json', {
        'stage5_parent': summary['lineage']['stage5_parent_run'], 'preflight_epoch21': preflight['run_id'],
        'continued_epochs22_to50': summary['run_id'], 'targeted_checks_passed': unit_checks['tests_run'],
        'checks_entry': unit_checks['entry'],
        'exporter_sha256': file_hash(Path(__file__)),
        'main_trajectory': 'Stage 5 epochs 1-20, stage 6 preflight epoch 21, continuation epochs 22-50.',
        'device_changed_at_epoch20': summary['lineage']['device_changed_at_epoch20']})
    lines = ['# 阶段 6 · 训练预算 20 → 50', '',
             f"最终运行：`{summary['run_id']}`；第 21 轮试跑：`{preflight['run_id']}`。", '',
             '从阶段 5 的 epoch 20 last 恢复；新增 30 轮、10,560 次更新，总计 50 轮、17,600 次更新。', '',
             '| Epoch | Online train loss | Online train acc | Train eval loss | Train eval acc | Val loss | Val acc |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for epoch in sorted(set([20, 30, 40, 50, summary['best_epoch']])):
        row = history[epoch-1]
        ev = summary['baseline20']['train_eval'] if epoch == 20 else {
            'loss': row.get('train_eval_loss'), 'accuracy': row.get('train_eval_accuracy')}
        eval_loss = f"{ev['loss']:.6f}" if ev['loss'] is not None else '未测'
        eval_acc = f"{ev['accuracy']:.2%}" if ev['accuracy'] is not None else '未测'
        label = f'{epoch}（best）' if epoch == summary['best_epoch'] else str(epoch)
        lines.append(f"| {label} | {row['train_loss']:.6f} | {row['train_accuracy']:.2%} | {eval_loss} | {eval_acc} | {row['val_loss']:.6f} | {row['val_accuracy']:.2%} |")
    lines += ['', '![六条指标曲线](learning-curves.png)', '',
              '- best 仍按全程最低 val_loss 选择；accuracy 最高的轮次可能不同。',
              '- 第 20 轮 train_eval 是迁移后的补测；第 1–19 轮未测，CSV 的第 1–20 轮保持原历史，基线另见 `baseline20.json`。',
              '- 原训练代码、模型、数据划分、优化器和学习率均保持一致；按用户指定更换物理设备，不能声称同设备单变量对照或跨设备逐位等价。',
              '- `migration-checks.json` 记录迁移加载与历史模型验证，`resume-checks.json` 记录第 21→22 轮完整状态恢复；新增监测不改变权重、优化器与随机状态。',
              '- 三项专项检查通过；官方 test 未评估，未增加增强或学习率调度。',
              '- `comparison.json` 包含 10 轮窗口均值；`resources.json` 是实测资源与耗时；完整检查点与日志在外部资产目录。',
              '- 分析与复现命令见 [阶段 6 笔记](../../notes/06-budget.md)。', '']
    (destination / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'exported': str(destination.relative_to(project)), 'best_epoch': summary['best_epoch'],
                      'best_validation': summary['best_validation'], 'resources': resources}, indent=2))


if __name__ == '__main__':
    main()
