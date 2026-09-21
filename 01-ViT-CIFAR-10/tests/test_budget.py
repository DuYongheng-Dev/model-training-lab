"""Stage-6 migration rejects confounds; extra observations preserve the next update."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from checkpoint import capture_rng, cpu_snapshot, restore_rng
from train_budget import CORE_SOURCES, evaluate_readonly, require_equal, validate_parent


def identities():
    identity = {'config': {'stage': '06-budget', 'max_epochs': 50, 'optimizer': {'lr': 3e-4}},
                'runtime': {'gpu_uuid': 'target', 'cudnn': 123}, 'data': {'split': 'same'},
                'forward': {'dim': 128}, 'versions': {'torch': 'same'},
                'source_hashes': {**{name: name for name in CORE_SOURCES}, 'train_budget.py': 'new'}}
    old = deepcopy(identity)
    old['config'].update(stage='05-train', max_epochs=20)
    old['runtime']['gpu_uuid'] = 'source'
    old['source_hashes'].pop('train_budget.py')
    history = [{'epoch': i, 'val_loss': 1 / i} for i in range(1, 21)]
    parent = {'identity': old, 'completed_epoch': 20, 'global_step': 7040, 'history': history,
              'best_epoch': 20, 'best_val_loss': .05,
              'best_checkpoint': {'completed_epoch': 20, 'best_val_loss': .05}}
    return identity, parent


class BudgetChecks(unittest.TestCase):
    def test_migration_only_allows_budget_stage_and_device(self):
        identity, parent = identities()
        self.assertEqual(validate_parent(parent, identity), 'stage5_budget_extension')
        for section, key, value in [('config', 'optimizer', {'lr': 1e-3}), ('data', 'split', 'changed'),
                                    ('versions', 'torch', 'changed'), ('forward', 'dim', 256),
                                    ('source_hashes', 'train_full.py', 'changed'), ('runtime', 'cudnn', 999)]:
            with self.subTest(section=section):
                changed = deepcopy(identity)
                changed[section][key] = value
                with self.assertRaises(ValueError):
                    validate_parent(parent, changed)
        parent['completed_epoch'] = 19
        with self.assertRaises(ValueError):
            validate_parent(parent, identity)

    def test_stage6_resume_is_strict_and_best_must_match_history(self):
        identity, parent = identities()
        parent['identity'] = deepcopy(identity)
        self.assertEqual(validate_parent(parent, identity), 'same_identity_resume')
        parent['identity']['runtime']['gpu_uuid'] = 'unexpected'
        with self.assertRaises(ValueError):
            validate_parent(parent, identity)
        parent['identity'] = deepcopy(identity)
        parent['best_checkpoint']['completed_epoch'] = 19
        with self.assertRaises(ValueError):
            validate_parent(parent, identity)

    def test_observation_preserves_next_adamw_update_and_rng(self):
        self.assertTrue(torch.cuda.is_available())
        model = nn.Linear(2, 2).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False, fused=False)
        generator = torch.Generator().manual_seed(42)
        eval_generator = torch.Generator().manual_seed(44)
        generators = {'train': generator, 'train_eval': eval_generator}
        images = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10
        labels = torch.tensor([0, 1, 0, 1, 1])
        dataset = TensorDataset(images, labels, torch.arange(5))
        loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=generator)
        observed_loader = DataLoader(dataset, batch_size=2, generator=eval_generator)

        def step():
            x, y, indices = next(iter(loader))
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.cuda()), y.cuda())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return {'model': cpu_snapshot(model.state_dict()), 'optimizer': cpu_snapshot(optimizer.state_dict()),
                    'rng': capture_rng(generators), 'indices': indices, 'loss': float(loss.detach())}

        step()  # Populate the AdamW moments before the observation experiment.
        before = {'model': cpu_snapshot(model.state_dict()), 'optimizer': cpu_snapshot(optimizer.state_dict()),
                  'rng': capture_rng(generators)}
        expected = step()
        model.load_state_dict(before['model'])
        optimizer.load_state_dict(before['optimizer'])
        restore_rng(before['rng'], generators)
        actual = evaluate_readonly(model, optimizer, observed_loader, generators)
        self.assertEqual(actual['count'], 5)
        require_equal(expected, step(), 'Extra evaluation changed the next training update')


if __name__ == '__main__':
    unittest.main(verbosity=2)
