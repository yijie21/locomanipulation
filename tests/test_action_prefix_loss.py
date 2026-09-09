import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "action_loss", Path(__file__).resolve().parents[1] / "src/gr00t/model/modules/action_loss.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
masked_action_loss = module.masked_action_loss


class ActionLossTests(unittest.TestCase):
    def test_default_loss_and_gradient_are_identical(self):
        torch.manual_seed(3)
        p = torch.randn(2, 40, 65, requires_grad=True)
        target = torch.randn_like(p)
        mask = torch.ones_like(p)
        mask[:, :, 64] = 0
        old = (F.mse_loss(p, target, reduction="none") * mask).sum() / (mask.sum() + 1e-6)
        new, _, _ = masked_action_loss(p, target, mask)
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(torch.autograd.grad(old, p, retain_graph=True)[0], torch.autograd.grad(new, p)[0]))

    def test_prefix_gradient_is_four_times_tail_and_padding_stays_zero(self):
        p = torch.ones(1, 40, 3, requires_grad=True)
        mask = torch.ones_like(p)
        mask[:, :, 2] = 0
        loss, _, effective = masked_action_loss(p, torch.zeros_like(p), mask, 10, 4.0)
        loss.backward()
        self.assertAlmostEqual(float(p.grad[0, 0, 0] / p.grad[0, 20, 0]), 4.0)
        self.assertEqual(float(p.grad[:, :, 2].abs().sum()), 0.0)
        self.assertAlmostEqual(float(loss.detach()), 1.0, places=6)
        self.assertEqual(float(effective.sum()), 140.0)

    def test_frozen_and_empty_masks_are_respected(self):
        p = torch.ones(1, 40, 2, requires_grad=True)
        mask = torch.ones_like(p)
        mask[:, :5] = 0
        loss, _, _ = masked_action_loss(p, torch.zeros_like(p), mask, 10, 4.0)
        loss.backward()
        self.assertEqual(float(p.grad[:, :5].abs().sum()), 0.0)
        empty, _, _ = masked_action_loss(p, torch.zeros_like(p), torch.zeros_like(p), 10, 4.0)
        self.assertEqual(float(empty.detach()), 0.0)

    def test_invalid_configuration_is_rejected(self):
        p = torch.ones(1, 40, 2)
        for weight in [0.0, -1.0, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                masked_action_loss(p, p, p, 10, weight)


if __name__ == "__main__":
    unittest.main()
