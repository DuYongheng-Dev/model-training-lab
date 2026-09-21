"""Evaluate an already loaded batch; the caller determines which dataset it is."""
import torch
from torch import nn


@torch.inference_mode()
def evaluate_batch(model, images, labels):
    """Read-only metrics; eval mode does not mean the inputs are validation data."""
    was_training = model.training
    model.eval()
    try:
        logits = model(images)
        assert logits.shape == (len(labels), model.head.out_features)
        assert torch.isfinite(logits).all()
        # Sum first, then divide by the number of actual samples.
        loss_sum = nn.functional.cross_entropy(logits, labels, reduction='sum')
        predictions = logits.argmax(dim=1)
        correct = int((predictions == labels).sum())
        return {
            'loss': float(loss_sum) / len(labels),
            'correct': correct,
            'count': len(labels),
            'accuracy': correct / len(labels),
        }, logits.detach()
    finally:
        model.train(was_training)


@torch.inference_mode()
def evaluate_loader(model, loader, device):
    """Weight metrics by sample count, including the short final batch."""
    was_training = model.training
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    try:
        for images, labels, _indices in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            losses = nn.functional.cross_entropy(logits, labels, reduction='sum')
            if not torch.isfinite(losses):
                raise FloatingPointError('Non-finite evaluation loss')
            loss_sum += float(losses)
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += len(labels)
        if count == 0:
            raise ValueError('Cannot evaluate an empty dataset')
        return {'loss': loss_sum / count, 'accuracy': correct / count,
                'correct': correct, 'count': count}
    finally:
        model.train(was_training)
