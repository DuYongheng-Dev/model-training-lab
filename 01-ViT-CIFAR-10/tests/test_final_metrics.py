"""Check final evaluation on synthetic examples before opening the official test set."""
import argparse
import json
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from checkpoint import cpu_snapshot
from final_metrics import predict_loader, select_examples, summarize_predictions
from run_record import file_hash


class FinalMetricChecks(unittest.TestCase):
    def test_uneven_batches_predictions_and_unchanged_weights(self):
        self.assertTrue(torch.cuda.is_available())
        logits = torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 4., 0.],
                               [0., 0., 5.], [3., 0., 0.]])
        labels = torch.tensor([0, 0, 1, 2, 2])
        model = nn.Linear(3, 3, bias=False).cuda()
        with torch.no_grad():
            model.weight.copy_(torch.eye(3, device='cuda'))
        before = cpu_snapshot(model.state_dict())
        loader = DataLoader(TensorDataset(logits, labels, torch.arange(5)), batch_size=2)
        rows, actual_logits, timing = predict_loader(model, loader, 'cuda:0')
        report = summarize_predictions(rows, ['a', 'b', 'c'])
        self.assertEqual(report['confusion_matrix'], [[1, 1, 0], [0, 1, 0], [1, 0, 1]])
        self.assertEqual(report['accuracy'], 3 / 5)
        expected = nn.functional.cross_entropy(logits, labels).item()
        self.assertAlmostEqual(report['loss'], expected, places=6)
        self.assertEqual(report['per_class'][1]['recall'], 1)
        self.assertEqual(report['per_class'][1]['precision'], .5)
        self.assertEqual(timing['last_batch_size'], 1)
        self.assertEqual(timing['batches'], 3)
        self.assertTrue(torch.equal(actual_logits, logits))
        self.assertAlmostEqual(rows[-1]['confidence'], logits[-1].softmax(0).max().item(), places=6)
        self.assertTrue(model.training)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(torch.equal(before[k], v.cpu()) for k, v in model.state_dict().items()))

    def test_example_order_and_missing_class_denominators(self):
        rows = [{'index': i, 'true_label': y, 'predicted_label': p, 'confidence': .8, 'loss': .5}
                for i, y, p in [(7, 0, 0), (2, 0, 1), (5, 0, 0), (1, 0, 1)]]
        selected = select_examples(rows, 1)
        self.assertEqual(selected['correct'][0]['index'], 5)
        self.assertEqual(selected['incorrect'][0]['index'], 1)
        report = summarize_predictions(rows, ['a', 'b', 'c'])
        self.assertIsNone(report['per_class'][2]['recall'])
        self.assertIsNone(report['per_class'][2]['precision'])
        with self.assertRaises(ValueError):
            summarize_predictions(rows + [rows[0]], ['a', 'b', 'c'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FinalMetricChecks))
    if args.report:
        args.report.write_text(json.dumps({'entry': 'tests/test_final_metrics.py',
            'source_sha256': file_hash(Path(__file__)), 'tests_run': result.testsRun,
            'failures': len(result.failures), 'errors': len(result.errors), 'passed': result.wasSuccessful(),
            'official_test_used': False}, indent=2) + '\n')
    sys.exit(0 if result.wasSuccessful() else 1)
