"""CPU-only regression checks of the real policy forward reduction; no model download."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'submodules/lerobot/src'))
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def forward(losses, mask=None):
    stub = SimpleNamespace(
        config=SimpleNamespace(adapt_to_pi_aloha=False, max_action_dim=losses.shape[-1]),
        normalize_inputs=lambda b: b, normalize_targets=lambda b: b,
        prepare_images=lambda b: (None, None), prepare_state=lambda b: None,
        prepare_language=lambda b: (None, None), prepare_action=lambda b: None,
        model=SimpleNamespace(forward=lambda *args: losses),
    )
    batch = {} if mask is None else {'action_is_pad': mask}
    return SmolVLAPolicy.forward(stub, batch)[0]


class PaddingTest(unittest.TestCase):
    def test_padding_has_no_loss_or_gradient(self):
        values = torch.tensor([[[2., 4.], [100., 200.]]], requires_grad=True)
        loss = forward(values, torch.tensor([[False, True]]))
        self.assertEqual(loss.item(), 3.)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([[[.5, .5], [0., 0.]]]))

    def test_adding_padding_does_not_dilute_loss(self):
        a = forward(torch.ones(1, 1, 7))
        b = forward(torch.ones(1, 50, 7), torch.tensor([[False] + [True]*49]))
        self.assertEqual(a.item(), b.item())

    def test_unpadded_unchanged(self):
        values = torch.arange(24.).reshape(2, 3, 4)
        self.assertEqual(forward(values).item(), values.mean().item())
        self.assertEqual(forward(values, torch.zeros(2, 3, dtype=torch.bool)).item(), values.mean().item())

    def test_invalid_masks_rejected(self):
        for mask in (torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2), torch.zeros(1, 3, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                forward(torch.ones(1, 2, 7), mask)


if __name__ == '__main__':
    unittest.main()
