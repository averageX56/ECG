"""
1D ResNet for raw ECG classification.

Replaces the CNN+Transformer (CTN) + hand-crafted HRV/template features
pipeline with a plain ResNet-style 1D CNN trained from scratch on the raw
signal. A small side-input (age, sex) is concatenated with the pooled
CNN embedding before the final linear classifier -- everything else
(wide HRV/template features) is dropped.

Works for any number of input leads (1 for Lead I only, 12 for all leads);
only the first conv layer's in_channels changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1d(nn.Module):
    """Standard ResNet basic block, 1D version."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=kernel_size,
                                stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size,
                                stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_planes, planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class ResNet1d(nn.Module):
    """
    ResNet-18-style 1D CNN for ECG.

    in_channels: number of leads (1 or 12)
    layers: number of BasicBlocks per stage, e.g. [2, 2, 2, 2] = ResNet18-style
    base_planes: channel width of the first stage (doubles each stage)
    deepfeat_sz: size of the pooled embedding fed into the classifier
    nb_demo: size of the side-input (age, sex) = 2
    """

    def __init__(self, classes, in_channels=12, layers=(2, 2, 2, 2),
                 base_planes=64, deepfeat_sz=128, nb_demo=2,
                 kernel_size=15, dropout_rate=0.3):
        super().__init__()
        self.classes = classes
        num_classes = len(classes)

        # Stem: large-kernel conv + stride to downsample the raw 500Hz signal
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_planes, kernel_size=15, stride=2,
                      padding=7, bias=False),
            nn.BatchNorm1d(base_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        planes = base_planes
        self.stage1 = self._make_stage(planes, planes, layers[0], stride=1,
                                        kernel_size=kernel_size)
        self.stage2 = self._make_stage(planes, planes * 2, layers[1], stride=2,
                                        kernel_size=kernel_size)
        self.stage3 = self._make_stage(planes * 2, planes * 4, layers[2], stride=2,
                                        kernel_size=kernel_size)
        self.stage4 = self._make_stage(planes * 4, planes * 8, layers[3], stride=2,
                                        kernel_size=kernel_size)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_embed = nn.Linear(planes * 8, deepfeat_sz)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(deepfeat_sz + nb_demo, num_classes)

    @staticmethod
    def _make_stage(in_planes, planes, num_blocks, stride, kernel_size):
        blocks = [BasicBlock1d(in_planes, planes, stride=stride, kernel_size=kernel_size)]
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock1d(planes, planes, stride=1, kernel_size=kernel_size))
        return nn.Sequential(*blocks)

    def forward(self, x, wide_feats):
        """
        x: [batch, in_channels, seq_len] raw (filtered+normalized) ECG window
        wide_feats: [batch, nb_demo] -- normalized age + sex only
        """
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.global_pool(out).squeeze(-1)     # [batch, planes*8]
        out = self.dropout(F.relu(self.fc_embed(out)))
        out = self.fc_out(torch.cat([wide_feats, out], dim=1))
        return out


def build_resnet1d(classes, leads='all', **kwargs):
    """
    leads: 'all' -> 12 input channels, 'lead1' -> 1 input channel
    """
    in_channels = 12 if leads == 'all' else 1
    return ResNet1d(classes, in_channels=in_channels, **kwargs)
