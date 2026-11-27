import unittest

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

from batch_support import sequence_ce


@unittest.skipIf(torch is None, 'PyTorch is needed for loss/gradient equivalence tests')
class SequenceCrossEntropyTests(unittest.TestCase):
    def compare(self, dtype):
        torch.manual_seed(42)
        reference = torch.randn(3, 13, 37, dtype=dtype, requires_grad=True)
        candidate = reference.detach().clone().requires_grad_(True)
        labels = torch.randint(0, 37, (3, 13))
        labels[0, :9] = -100
        labels[1, :3] = -100
        labels[1, 7:9] = -100
        labels[2, :12] = -100
        expected = torch.stack([
            F.cross_entropy(scores[:-1].float(), target[1:], ignore_index=-100)
            for scores, target in zip(reference, labels)
        ])
        actual = sequence_ce(candidate, labels)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        expected.mean().backward()
        actual.mean().backward()
        torch.testing.assert_close(candidate.grad, reference.grad, rtol=1e-6, atol=1e-7)
        ignored = labels[:, 1:] == -100
        self.assertTrue(torch.all(candidate.grad[:, :-1][ignored] == 0))

    def test_float32_loss_and_gradient_match_original(self):
        self.compare(torch.float32)

    def test_bfloat16_loss_and_gradient_match_original(self):
        self.compare(torch.bfloat16)

    def test_all_ignored_sequence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No shifted supervised tokens'):
            sequence_ce(torch.zeros(1, 4, 9), torch.full((1, 4), -100))


if __name__ == '__main__':
    unittest.main()
