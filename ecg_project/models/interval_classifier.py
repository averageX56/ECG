"""Inception-style morphology encoder with interval fusion and record MIL."""
import torch
from torch import nn


class InceptionBlock(nn.Module):
    def __init__(self, channels, width=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(channels, width, 1, bias=False)
        self.branches = nn.ModuleList(nn.Conv1d(width, width, k, padding=k//2, bias=False)
                                      for k in (9, 19, 39))
        self.pool = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(channels, width, 1, bias=False))
        self.norm = nn.GroupNorm(8, width*4)
        self.skip = nn.Conv1d(channels, width*4, 1, bias=False)

    def forward(self, x):
        z = self.bottleneck(x)
        return torch.nn.functional.silu(self.norm(torch.cat([b(z) for b in self.branches] + [self.pool(x)], 1)) + self.skip(x))


class IntervalClassifier(nn.Module):
    """Record labels supervise bags, never incorrectly assigned to each beat.

    Input: [records, beats, samples], [records, beats, features], valid beat mask.
    """
    def __init__(self, classes, interval_dim, use_intervals=True, input_channels=1):
        super().__init__()
        self.use_intervals = use_intervals
        self.input_channels = input_channels
        self.encoder = nn.Sequential(InceptionBlock(input_channels), InceptionBlock(128), InceptionBlock(128),
                                     nn.AvgPool1d(2), InceptionBlock(128), InceptionBlock(128), InceptionBlock(128))
        self.interval = nn.Sequential(nn.Linear(interval_dim, 64), nn.LayerNorm(64), nn.SiLU())
        dim = 256 + (64 if use_intervals else 0)
        self.attention = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.LayerNorm(dim*2), nn.Dropout(.3), nn.Linear(dim*2, classes))

    def forward(self, wave, interval, mask):
        if not mask.any(1).all():
            raise ValueError('Every record needs at least one valid beat')
        # Encode only real beats; padding cannot change normalization or pooling.
        real = wave[mask]
        if real.ndim == 2:
            real = real.unsqueeze(1)
        if real.shape[1] != self.input_channels:
            raise ValueError('Classifier input channel count mismatch')
        encoded = self.encoder(real)
        features = torch.cat([encoded.mean(-1), encoded.amax(-1)], -1)
        if self.use_intervals:
            features = torch.cat([features, self.interval(interval[mask])], -1)
        z = features.new_zeros((*mask.shape, features.shape[-1]))
        z[mask] = features
        weights = self.attention(z).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(1)
        pooled = (z * weights[..., None]).sum(1)
        maximum = z.masked_fill(~mask[..., None], -torch.inf).amax(1)
        return self.head(torch.cat([pooled, maximum], -1))
