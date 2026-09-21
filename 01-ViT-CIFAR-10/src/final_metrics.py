"""Single-pass predictions and descriptive reports for a frozen classifier."""
import csv
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


@torch.inference_mode()
def predict_loader(model, loader, device):
    was_training = model.training
    model.eval()
    rows, all_logits, batches = [], [], 0
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for images, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            logits = model(images)
            if logits.ndim != 2 or len(logits) != len(labels):
                raise ValueError('Invalid classifier output shape')
            losses = nn.functional.cross_entropy(logits, labels_device, reduction='none')
            if not torch.isfinite(logits).all() or not torch.isfinite(losses).all():
                raise FloatingPointError('Non-finite evaluation output')
            predictions = logits.argmax(dim=1)
            confidence = logits.softmax(dim=1).gather(1, predictions[:, None]).squeeze(1)
            for index, label, pred, score, loss in zip(indices.tolist(), labels.tolist(), predictions.tolist(),
                                                     confidence.tolist(), losses.tolist()):
                rows.append({'index': index, 'true_label': label, 'predicted_label': pred,
                             'confidence': score, 'loss': loss})
            all_logits.append(logits.detach().cpu())
            batches += 1
            last_batch_size = len(labels)
        if not rows:
            raise ValueError('Cannot evaluate an empty dataset')
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        return rows, torch.cat(all_logits), {
            'batches': batches, 'last_batch_size': last_batch_size,
            'seconds': time.perf_counter() - started,
            'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20
                if str(device).startswith('cuda') else None}
    finally:
        model.train(was_training)


def summarize_predictions(rows, classes):
    if not rows or len({row['index'] for row in rows}) != len(rows):
        raise ValueError('Empty predictions or repeated sample indices')
    size = len(classes)
    confusion = np.zeros((size, size), dtype=np.int64)
    for row in rows:
        truth, predicted = row['true_label'], row['predicted_label']
        if not (0 <= truth < size and 0 <= predicted < size):
            raise ValueError('Class outside the configured label space')
        if not math.isfinite(row['loss']) or row['loss'] < 0 or not 0 <= row['confidence'] <= 1:
            raise ValueError('Invalid sample metric')
        confusion[truth, predicted] += 1  # Rows = true labels, columns = predicted labels.
    correct = int(np.trace(confusion))
    per_class = []
    for index, name in enumerate(classes):
        support, predicted = int(confusion[index].sum()), int(confusion[:, index].sum())
        true_positive = int(confusion[index, index])
        per_class.append({'label': index, 'class': name, 'support': support,
                          'predicted_count': predicted, 'correct': true_positive,
                          'recall': true_positive / support if support else None,
                          'precision': true_positive / predicted if predicted else None})
    mistakes = [{'true_class': classes[i], 'predicted_class': classes[j], 'count': int(confusion[i, j]),
                 'fraction_of_true_class': float(confusion[i, j] / confusion[i].sum())}
                for i in range(size) for j in range(size) if i != j and confusion[i, j] > 0]
    mistakes.sort(key=lambda item: (-item['count'], item['true_class'], item['predicted_class']))
    return {'loss': math.fsum(row['loss'] for row in rows) / len(rows),
            'accuracy': correct / len(rows), 'correct': correct, 'count': len(rows),
            'confusion_matrix': confusion.tolist(), 'per_class': per_class, 'top_confusions': mistakes[:10],
            'matrix_orientation': 'rows=true labels; columns=predicted labels'}


def select_examples(rows, per_outcome):
    ordered = sorted(rows, key=lambda row: row['index'])
    return {name: [row for row in ordered if (row['true_label'] == row['predicted_label']) == correct][:per_outcome]
            for name, correct in [('correct', True), ('incorrect', False)]}


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def draw_confusion(metrics, classes, destination):
    counts = np.asarray(metrics['confusion_matrix'])
    percentages = 100 * counts / counts.sum(axis=1, keepdims=True)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    for axis, values, title, fmt, limit in zip(axes, (counts, percentages),
            ('Number of images', 'Within each true class (%)'), ('d', '.1f'), (counts.max(), 100)):
        picture = axis.imshow(values, cmap='Blues', vmin=0, vmax=limit)
        axis.set_xticks(range(len(classes)), classes, rotation=45, ha='right')
        axis.set_yticks(range(len(classes)), classes)
        axis.set_xlabel('Predicted class')
        axis.set_ylabel('True class')
        axis.set_title(title)
        for row in range(len(classes)):
            for col in range(len(classes)):
                value = values[row, col]
                axis.text(col, row, format(value, fmt), ha='center', va='center', fontsize=7,
                          color='white' if value > limit / 2 else 'black')
        figure.colorbar(picture, ax=axis, fraction=0.04)
    figure.suptitle('CIFAR-10 official test / frozen validation-best checkpoint')
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def draw_examples(raw_images, examples, classes, outcome, destination):
    figure, axes = plt.subplots(2, 5, figsize=(12, 5.6), constrained_layout=True)
    for axis, row in zip(axes.flat, examples):
        axis.imshow(raw_images[row['index']], interpolation='nearest')
        axis.set_title(f"#{row['index']}  true: {classes[row['true_label']]}\n"
                       f"pred: {classes[row['predicted_label']]}  ({row['confidence']:.1%})", fontsize=9)
        axis.axis('off')
    for axis in list(axes.flat)[len(examples):]:
        axis.axis('off')
    figure.suptitle(f'{outcome}: first 10 in official test order\nConfidence = max softmax score (not calibrated)')
    figure.savefig(destination, dpi=160)
    plt.close(figure)
