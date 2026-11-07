"""Verify custom convolution preserves native outputs and gradients."""
import torch

from genomics.cnn import SequenceCNN


def test_convolution_matches_native_outputs_and_gradients():
    torch.manual_seed(7)
    model = SequenceCNN()
    x = torch.randn(3, 4, 200, requires_grad=True)
    actual = model(x)
    expected = model.network(x).squeeze(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual_grad = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-6)
