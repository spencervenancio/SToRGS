"""DFIM sequence classifier; input float32 (batch, A/C/G/T, 200)."""
from torch import nn
from torch.nn import functional as F


class SequenceCNN(nn.Module):
    def __init__(self, channels=(32, 64), kernels=(15, 7), hidden=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(4, channels[0], kernels[0], padding=kernels[0] // 2),
            nn.ReLU(),
            nn.Conv1d(channels[0], channels[1], kernels[1], padding=kernels[1] // 2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[1], hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # Exact convolution semantics via unfold + GEMM; much faster on this CPU.
        for index in (0, 2):
            layer = self.network[index]
            padded = F.pad(x, (layer.padding[0], layer.padding[0]))
            windows = padded.unfold(2, layer.kernel_size[0], 1)
            windows = windows.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
            x = F.linear(windows, layer.weight.flatten(1), layer.bias).transpose(1, 2)
            x = self.network[index + 1](x)
        for layer in self.network[4:]:
            x = layer(x)
        return x.squeeze(-1)
