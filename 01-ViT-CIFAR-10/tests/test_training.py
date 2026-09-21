"""Checks for uneven final batches, RNG restoration and immutable best weights."""
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from checkpoint import atomic_save, capture_rng, cpu_snapshot, load_checkpoint, restore_rng
from compare_resume import compare_tree
from evaluate import evaluate_loader


class TrainingChecks(unittest.TestCase):
    def test_validation_weights_short_batch_and_does_not_train(self):
        self.assertTrue(torch.cuda.is_available())
        images = torch.tensor([[6.0, 0.0], [6.0, 0.0], [6.0, 0.0]])
        labels = torch.tensor([0, 0, 1])
        loader = DataLoader(TensorDataset(images, labels, torch.arange(3)), batch_size=2)
        model = nn.Linear(2, 2, bias=False).cuda()
        with torch.no_grad():
            model.weight.copy_(torch.eye(2, device='cuda'))
        before = model.weight.detach().clone()
        actual = evaluate_loader(model, loader, 'cuda:0')
        expected = nn.functional.cross_entropy(images.cuda(), labels.cuda()).item()
        self.assertAlmostEqual(actual['loss'], expected, places=6)
        self.assertEqual(actual['accuracy'], 2 / 3)
        self.assertEqual(actual['count'], 3)
        self.assertTrue(torch.equal(before, model.weight))
        self.assertIsNone(model.weight.grad)
        self.assertTrue(model.training)

    def test_rng_roundtrip_and_weights_only_loading(self):
        generators = {'train': torch.Generator().manual_seed(42), 'validation': torch.Generator().manual_seed(43)}
        state = capture_rng(generators)

        def draw():
            return {'python': random.random(), 'numpy': np.random.rand(),
                    'cpu': torch.rand(4), 'cuda': torch.rand(4, device='cuda').cpu(),
                    'loaders': {key: torch.randperm(17, generator=g) for key, g in generators.items()}}

        expected = draw()
        with tempfile.TemporaryDirectory(prefix='vit-checkpoint-') as temporary:
            path = Path(temporary) / 'rng.pt'
            atomic_save({'schema_version': 1, 'identity': {'split': 'original'}, 'rng': state}, path)
            loaded = load_checkpoint(path, {'split': 'original'})
            with self.assertRaises(ValueError):
                load_checkpoint(path, {'split': 'different'})
            restore_rng(loaded['rng'], generators)
            self.assertTrue(compare_tree(expected, draw())['exactly_equal'])

    def test_snapshot_independence_and_comparison_detects_change(self):
        original = {'weights': torch.ones(2, device='cuda')}
        saved = cpu_snapshot(original)
        original['weights'].add_(1)
        self.assertTrue(torch.equal(saved['weights'], torch.ones(2)))
        self.assertFalse(compare_tree(saved, cpu_snapshot(original))['exactly_equal'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
