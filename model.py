import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):   
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add position encodings to embeddings
        # x: embedding vects, [B x L x d_model]
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class Transformer(nn.Module):
    '''
    Transformer encoder processes convolved ECG samples
    Stacks a number of TransformerEncoderLayers
    '''
    def __init__(self, d_model, h, d_ff, num_layers, dropout):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.pe = PositionalEncoding(d_model, dropout=0.1)
        
        # batch_first=True: вход/выход в формате [B, L, d_model].
        # Убирает UserWarning про enable_nested_tensor и включает более быстрый
        # путь инференса. На веса НЕ влияет — чекпоинты остаются совместимыми.
        encode_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.h,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encode_layer, self.num_layers)

    def forward(self, x):
        # x: [B, d_model, L] (выход свёрточного энкодера)
        out = x.permute(0, 2, 1)              # -> [B, L, d_model]
        out = self.pe(out)                    # позиц. кодирование, [B, L, d_model]
        out = self.transformer_encoder(out)   # batch_first -> [B, L, d_model]
        out = out.mean(1)                     # global pooling по длине -> [B, d_model]
        return out

# 15 second model
class CTN(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout_rate, deepfeat_sz, nb_feats, nb_demo, classes):
        super(CTN, self).__init__()
        
        self.encoder = nn.Sequential( # downsampling factor = 20
            nn.Conv1d(12, 128, kernel_size=14, stride=3, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=14, stride=3, padding=0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=10, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        self.transformer = Transformer(d_model, nhead, d_ff, num_layers, dropout=0.1)
        self.fc1 = nn.Linear(d_model, deepfeat_sz)
        self.fc2 = nn.Linear(deepfeat_sz+nb_feats+nb_demo, len(classes))
        self.dropout = nn.Dropout(dropout_rate)
            
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        #self.apply(_weights_init)
            
    def forward(self, x, wide_feats):
        z = self.encoder(x)          # encoded sequence is batch_sz x nb_ch x seq_len
        out = self.transformer(z)    # transformer output is batch_sz x d_model
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(torch.cat([wide_feats, out], dim=1))
        return out


# ==========================================================================
# ResNet1d — 1D ResNet-18-подобная свёрточная сеть
# ==========================================================================
class _BasicBlock1d(nn.Module):
    def __init__(self, in_planes, planes, stride=1, kernel_size=15):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm1d(planes))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1d(nn.Module):
    """1D ResNet: свёрточный энкодер сырого сигнала + демография/ручные фичи."""
    def __init__(self, classes, in_channels=12, nb_feats=0, nb_demo=2,
                 layers=(2, 2, 2, 2), base=64, deepfeat_sz=128, kernel_size=15, dropout=0.3):
        super().__init__()
        self.classes = classes
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))

        def stage(inp, out, n, stride):
            blocks = [_BasicBlock1d(inp, out, stride, kernel_size)]
            blocks += [_BasicBlock1d(out, out, 1, kernel_size) for _ in range(1, n)]
            return nn.Sequential(*blocks)

        self.stage1 = stage(base, base, layers[0], 1)
        self.stage2 = stage(base, base * 2, layers[1], 2)
        self.stage3 = stage(base * 2, base * 4, layers[2], 2)
        self.stage4 = stage(base * 4, base * 8, layers[3], 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_embed = nn.Linear(base * 8, deepfeat_sz)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(deepfeat_sz + nb_feats + nb_demo, len(classes))

    def forward(self, x, wide_feats):
        out = self.stem(x)
        out = self.stage4(self.stage3(self.stage2(self.stage1(out))))
        out = self.pool(out).squeeze(-1)
        out = self.dropout(F.relu(self.fc_embed(out)))
        return self.fc_out(torch.cat([wide_feats, out], dim=1))


# ==========================================================================
# UNet1d — 1D U-Net (энкодер-декодер), пулинг признаков -> классификатор
# ==========================================================================
class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=9):
        super().__init__()
        pad = k // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class UNet1d(nn.Module):
    """1D U-Net для классификации: энкодер-декодер, глобальный пулинг выхода +
    демография/ручные фичи -> классификатор."""
    def __init__(self, classes, in_channels=12, nb_feats=0, nb_demo=2,
                 base=32, deepfeat_sz=128, dropout=0.3):
        super().__init__()
        self.classes = classes
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.inc = _DoubleConv(in_channels, c1)
        self.d1, self.e1 = nn.MaxPool1d(2), _DoubleConv(c1, c2)
        self.d2, self.e2 = nn.MaxPool1d(2), _DoubleConv(c2, c3)
        self.d3, self.e3 = nn.MaxPool1d(2), _DoubleConv(c3, c4)
        self.d4, self.bott = nn.MaxPool1d(2), _DoubleConv(c4, c5)
        self.u4, self.de4 = nn.Conv1d(c5, c4, 1), _DoubleConv(c4 + c4, c4)
        self.u3, self.de3 = nn.Conv1d(c4, c3, 1), _DoubleConv(c3 + c3, c3)
        self.u2, self.de2 = nn.Conv1d(c3, c2, 1), _DoubleConv(c2 + c2, c2)
        self.u1, self.de1 = nn.Conv1d(c2, c1, 1), _DoubleConv(c1 + c1, c1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_embed = nn.Linear(c1, deepfeat_sz)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(deepfeat_sz + nb_feats + nb_demo, len(classes))

    @staticmethod
    def _up(x, skip):
        x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x, wide_feats):
        x1 = self.inc(x)
        x2 = self.e1(self.d1(x1))
        x3 = self.e2(self.d2(x2))
        x4 = self.e3(self.d3(x3))
        xb = self.bott(self.d4(x4))
        y = self.de4(self._up(self.u4(xb), x4))
        y = self.de3(self._up(self.u3(y), x3))
        y = self.de2(self._up(self.u2(y), x2))
        y = self.de1(self._up(self.u1(y), x1))
        out = self.pool(y).squeeze(-1)
        out = self.dropout(F.relu(self.fc_embed(out)))
        return self.fc_out(torch.cat([wide_feats, out], dim=1))


# ==========================================================================
# Фабрика моделей
# ==========================================================================
def _ctn_one_lead(model):
    old = model.encoder[0]
    model.encoder[0] = nn.Conv1d(1, old.out_channels, kernel_size=old.kernel_size,
                                 stride=old.stride, padding=old.padding, bias=False)
    return model


def build_model(name, classes, in_channels=12, nb_feats=0, nb_demo=2):
    """Создаёт модель по имени: 'ctn' | 'resnet1d' | 'unet1d'.

    Все модели имеют одинаковый интерфейс forward(x, wide_feats) -> [B, n_classes],
    где wide_feats = [age, sex, (ручные фичи)]. in_channels=12 (все отведения) или 1.
    """
    name = name.lower()
    if name == 'ctn':
        m = CTN(256, 8, 2048, 8, 0.2, 64, nb_feats, nb_demo, classes)
        return _ctn_one_lead(m) if in_channels == 1 else m
    if name == 'resnet1d':
        return ResNet1d(classes, in_channels=in_channels, nb_feats=nb_feats, nb_demo=nb_demo)
    if name == 'unet1d':
        return UNet1d(classes, in_channels=in_channels, nb_feats=nb_feats, nb_demo=nb_demo)
    raise ValueError(f"Неизвестная модель '{name}' (ожидалось ctn/resnet1d/unet1d)")
