"""Export selected stage-5 metrics/plots after all runs and comparisons succeed."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True, help='JSON mapping six role names to run IDs')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = project / 'results/05-train'
    if output.exists():
        raise FileExistsError('Result directory already exists; inspect before replacing a published result')
    roles = ('small_reference', 'small_resumed', 'full_preflight', 'full_reference', 'full_resumed', 'full_final')
    raw = json.loads(args.runs.read_text())
    run_ids = {role: raw[role] for role in roles}
    assert all(re.fullmatch(r'05-train-[0-9a-f]{12}', name) for name in run_ids.values())
    runs = {role: Path(os.environ['LAB_RUNS_DIR']) / project.name / name for role, name in run_ids.items()}
    summaries = {}
    for role, run in runs.items():
        assert json.loads((run / 'status.json').read_text())['status'] == 'complete'
        summary = json.loads((run / 'metrics/summary.json').read_text())
        assert summary['run_id'] == run_ids[role] and all(summary['checks'].values())
        assert not summary['test_evaluated']
        summaries[role] = summary
    final = summaries['full_final']
    assert final['completed_epoch'] == 20 and final['global_step'] == 7040
    assert final['parent_run'] == run_ids['full_resumed']
    assert summaries['full_resumed']['parent_run'] == run_ids['full_reference']
    history = final['history']
    assert [row['epoch'] for row in history] == list(range(1, 21))
    assert all(row['train_count'] == 45000 and row['val_count'] == 5000 and row['train_batches'] == 352
               and row['train_last_batch_size'] == 72 and row['global_step'] == row['epoch'] * 352 for row in history)
    best = min(history, key=lambda row: row['val_loss'])
    assert best['epoch'] == final['best_epoch']
    assert best['val_loss'] == final['best_validation']['loss']
    comparisons = {role: json.loads((runs[role] / 'metrics/resume-comparison.json').read_text())
                   for role in ('small_resumed', 'full_resumed')}
    assert all(report['all_exactly_equal'] for report in comparisons.values())
    assert comparisons['full_resumed']['global_step'] == 2112
    full_actual_rows = [row for role, summary in summaries.items() if role.startswith('full_')
                        for row in summary['history'] if row['epoch'] >= summary['first_epoch_this_run']]
    assert len(full_actual_rows) == 22
    resource_summary = {
        'main_trajectory_epochs': 20, 'main_trajectory_optimizer_steps': 7040,
        'executed_full_data_epochs_including_checks': len(full_actual_rows),
        'executed_small_subset_epochs': 7,
        'main_trajectory_train_seconds': sum(row['train_seconds'] for row in history),
        'main_trajectory_validation_seconds': sum(row['validation_seconds'] for row in history),
        'main_trajectory_data_wait_seconds': sum(row['data_wait_seconds'] for row in history),
        'median_train_epoch_seconds': statistics.median(row['train_seconds'] for row in history),
        'median_train_samples_per_second': statistics.median(row['train_samples_per_second'] for row in history),
        'max_gpu_peak_allocated_mib': max(row['gpu_peak_allocated_mib'] for row in history),
        'all_full_data_train_seconds_including_checks': sum(row['train_seconds'] for row in full_actual_rows),
        'note': 'Times include metric/finite checks; train also includes loading/transfer. Excludes checkpoint IO and framework startup.',
    }
    output.mkdir()
    for name in ('summary.json', 'epochs.csv'):
        shutil.copyfile(runs['full_final'] / 'metrics' / name, output / name)
    shutil.copyfile(runs['full_final'] / 'outputs/learning-curves.png', output / 'learning-curves.png')
    for name, value in [('run-index.json', run_ids), ('resources.json', resource_summary),
                        ('resume-comparison.json', comparisons['full_resumed']),
                        ('smoke-resume-comparison.json', comparisons['small_resumed']),
                        ('preflight-summary.json', summaries['full_preflight'])]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    rows = '\n'.join(f"| {row['epoch']} | {row['train_loss']:.6f} | {row['train_accuracy']:.2%} | "
                     f"{row['val_loss']:.6f} | {row['val_accuracy']:.2%} |"
                     for row in history if row['epoch'] in {1, 5, 6, 10, 15, 20, final['best_epoch']})
    readme = f'''# 阶段 5：完整训练、验证与恢复结果

Tiny ViT 从 seed=42 随机初始化，在固定 45,000/5,000 train/validation 划分上完成 20 epoch、7,040 次参数更新。官方测试集未评估。

| Epoch | Train loss | Train accuracy | Val loss | Val accuracy |
| --- | --- | --- | --- | --- |
{rows}

按最低 val_loss 选择 **epoch {final['best_epoch']}**：验证 loss={final['best_validation']['loss']:.6f}，accuracy={final['best_validation']['accuracy']:.2%}。重新加载 best checkpoint 后，验证指标完全一致。

![训练和验证曲线](learning-curves.png)

train 指标来自每个 batch 更新前、不断变化的模型；validation 来自 epoch 结束模型。两者测量条件不同，不等同于最终模型分别对整个 train/validation 做一次评估。

## 恢复检查

连续训练 6 轮的参考与从第 5 轮检查点在新进程重启、训练第 6 轮的结果完全一致：56 个模型参数张量、168 个优化器状态张量、随机状态、指标与样本顺序均通过逐项比较，参数最大绝对差为 0。范围限于本次相同设备、软件、代码、配置与数据条件；耗时/显存不参与相等检查。

- [完整恢复对照](resume-comparison.json)
- [257/129 小子集恢复对照](smoke-resume-comparison.json)
- [完整数据一轮试跑](preflight-summary.json)
- [正式指标摘要](summary.json)、[逐轮 CSV](epochs.csv)、[资源观测](resources.json)
- [运行 ID 与职责](run-index.json)
- [学习笔记与复现命令](../../notes/05-training.md)

主轨迹的 epoch 1–5 来自连续参考，epoch 6 来自验证通过的恢复分支，epoch 7–20 从恢复分支继续。20 轮是主轨迹预算；包含首轮试跑和重复第 6 轮，共实际执行了 22 个完整数据 epoch，另有 7 个小子集 epoch。

最终运行：`{run_ids['full_final']}`。完整 checkpoint、日志、数据索引、源代码快照和本机信息保存在仓库外。代码基于阶段 4 已发布提交加本次改动，精确训练代码哈希见 summary 与各运行的 source manifest。本结果只包含一个种子，未开展超参数搜索或跨设备复现。
'''
    (output / 'README.md').write_text(readme)
    print(json.dumps({'output': str(output), 'best_epoch': final['best_epoch'],
                      'best_validation': final['best_validation'], 'resources': resource_summary}, indent=2))


if __name__ == '__main__':
    main()
