"""Export selected stage-7 figures and metrics, leaving full predictions on the SSD."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_record import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    assert args.run.resolve().is_relative_to(Path(os.environ['LAB_ROOT']).resolve())
    project = Path(__file__).resolve().parents[1]
    summary = json.loads((args.run / 'metrics/summary.json').read_text())
    assert json.loads((args.run / 'status.json').read_text())['status'] == 'complete'
    assert all(summary['checks'].values()) and summary['test_passes_this_run'] == 1
    assert summary['training_updates_this_run'] == 0 and summary['test']['count'] == 10000
    assert file_hash(args.run / 'outputs/predictions.csv') == summary['predictions_sha256']
    checks = json.loads((args.run / 'metrics/unit-checks.json').read_text())
    assert checks['passed'] and checks['tests_run'] == 2 and not checks['official_test_used']
    assert checks['source_sha256'] == file_hash(project / checks['entry'])
    destination = project / 'results/07-final'
    destination.mkdir(exist_ok=False)
    exports = {
        'summary.json': 'metrics/summary.json', 'model-selection.json': 'metrics/model-selection.json',
        'preflight.json': 'metrics/preflight.json', 'unit-checks.json': 'metrics/unit-checks.json',
        'per-class.csv': 'metrics/per-class.csv', 'confusion-matrix.csv': 'metrics/confusion-matrix.csv',
        'examples.json': 'metrics/examples.json', 'confusion-matrix.png': 'outputs/confusion-matrix.png',
        'examples-correct.png': 'outputs/examples-correct.png', 'examples-incorrect.png': 'outputs/examples-incorrect.png'}
    for name, source in exports.items():
        shutil.copyfile(args.run / source, destination / name)
    write_json(destination / 'export-manifest.json', {
        'run_id': summary['run_id'], 'exporter_sha256': file_hash(Path(__file__)),
        'files_sha256': {name: file_hash(destination / name) for name in exports},
        'full_predictions_exported': False})
    test = summary['test']
    lines = ['# 阶段 7 · 官方测试集最终评估', '', f"运行：`{summary['run_id']}`。", '',
        f"模型：按 epoch 1–50 的最低 val_loss 选择第 **{summary['selection']['selected_epoch']}** 轮，读取 test 前固定。", '',
        '| 数据 | 样本数 | Loss | Accuracy | 正确数量 |', '| --- | ---: | ---: | ---: | ---: |']
    for name, metrics in [('Validation（重测）', summary['validation_recheck']), ('Official test', test)]:
        lines.append(f"| {name} | {metrics['count']:,} | {metrics['loss']:.6f} | {metrics['accuracy']:.2%} | {metrics['correct']:,} |")
    lines += ['', '## 混淆矩阵与每类指标', '', '行是真实类别，列是预测类别；左图为数量，右图为各真实类别内的百分比。', '',
              '![混淆矩阵](confusion-matrix.png)', '', '| 类别 | 正确 / 实际样本 | Recall | Precision |',
              '| --- | ---: | ---: | ---: |']
    for row in test['per_class']:
        precision = f"{row['precision']:.2%}" if row['precision'] is not None else '未定义'
        lines.append(f"| {row['class']} | {row['correct']} / {row['support']} | {row['recall']:.2%} | {precision} |")
    lines += ['', '## 预测示例', '', '分别取官方测试索引顺序中最先出现的 10 张正确、10 张错误图片，展示规则在评估前固定；不代表各类错误的频率。', '',
        '![预测正确的示例](examples-correct.png)', '', '![预测错误的示例](examples-incorrect.png)', '',
        'confidence 是预测类别的 softmax 分数，没有进行概率校准。', '',
        '## 检查与边界', '',
        '- 两项合成数据检查通过；测试前重测 validation，通过预定的 loss 容差与正确数量检查。',
        '- 完整 test 仅一遍 forward，79 批、尾批 16 张；每张图片恰好计入一次。',
        '- 混淆矩阵总和 10,000、每行 1,000、对角线和等于正确数；模型和源检查点不变，没有参数更新。',
        '- 所有图表来自同一遍预测；未评估其他 checkpoint 的 test，未用 test 结果调参。',
        '- 单种子、单个验证集选定模型的结果，不代表架构上限或所有真实场景。',
        '- 完整预测与日志留在外部资产目录；可审计摘要见本目录 JSON/CSV。', '',
        '阅读 [阶段 7 笔记](../../notes/07-final.md) 和 [完整实验总结](../../notes/experiment-summary.md)。', '']
    (destination / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'exported': str(destination.relative_to(project)), 'run_id': summary['run_id'],
        'selected_epoch': summary['selection']['selected_epoch'], 'test_loss': test['loss'],
        'test_accuracy': test['accuracy'], 'correct': test['correct'], 'top_confusions': test['top_confusions'][:5]}, indent=2))


if __name__ == '__main__':
    main()
