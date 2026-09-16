"""CPU forward checks: patch projection, batch isolation, initialization and input contract."""
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from model import TinyViT


class TinyViTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyViT().eval()
        self.images = torch.randn(2, 3, 32, 32)

    def test_patch_convolution_matches_explicit_linear_projection(self):
        with torch.inference_mode():
            patches = F.unfold(self.images, kernel_size=4, stride=4).transpose(1, 2)
            explicit = F.linear(patches, self.model.patch_embed.weight.flatten(1), self.model.patch_embed.bias)
            actual = self.model.patch_embed(self.images).flatten(2).transpose(1, 2)
            torch.testing.assert_close(actual, explicit, atol=1e-6, rtol=1e-5)

    def test_batch_members_do_not_attend_to_other_images(self):
        with torch.inference_mode():
            batched = self.model(self.images)
            separate = torch.cat([self.model(image.unsqueeze(0)) for image in self.images])
            torch.testing.assert_close(batched, separate, atol=1e-5, rtol=1e-5)

    def test_distinct_images_reach_cls_output(self):
        with torch.inference_mode():
            output = self.model(self.images)
        self.assertFalse(torch.allclose(output[0], output[1]))

    def test_blocks_have_independent_parameters_and_initial_values(self):
        weights = [block.attention.in_proj_weight for block in self.model.blocks]
        self.assertEqual(len({weight.data_ptr() for weight in weights}), 4)
        self.assertFalse(torch.equal(weights[0], weights[1]))

    def test_forward_trace_is_observational_and_does_not_update_state(self):
        before = {name: tensor.clone() for name, tensor in self.model.state_dict().items()}
        trace = {}
        with torch.inference_mode():
            actual = self.model(self.images, trace=trace)
            repeated = self.model(self.images)
        torch.testing.assert_close(actual, repeated, atol=0, rtol=0)
        self.assertEqual(trace['patch_tokens'], [2, 64, 128])
        self.assertEqual(trace['with_position'], [2, 65, 128])
        self.assertEqual(tuple(actual.shape), (2, 10))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertTrue(all(torch.equal(before[name], tensor) for name, tensor in self.model.state_dict().items()))

    def test_seed_recreates_identical_initial_model(self):
        torch.manual_seed(42)
        repeated = TinyViT()
        for name, tensor in self.model.state_dict().items():
            self.assertTrue(torch.equal(tensor, repeated.state_dict()[name]), name)

    def test_invalid_shapes_and_dimensions_are_rejected(self):
        for kwargs in ({'patch_size': 3}, {'embed_dim': 127}, {'depth': 0}):
            with self.assertRaises(ValueError):
                TinyViT(**kwargs)
        with self.assertRaises(ValueError):
            self.model(torch.zeros(2, 32, 32, 3))


if __name__ == '__main__':
    unittest.main()
