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
