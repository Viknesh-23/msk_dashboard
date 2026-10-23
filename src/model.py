"""Module 3 - 2D U-Net."""
import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """4-level U-Net. `return_features=True` exposes the bottleneck features,
    which the CORAL domain-adaptation loss (Module 6) will use later."""

    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.inc = DoubleConv(in_ch, c[0])
        self.downs = nn.ModuleList([DoubleConv(c[i], c[i + 1]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(c[i] * 2, c[i]) for i in reversed(range(4))])
        self.outc = nn.Conv2d(c[0], out_ch, 1)

    def encode(self, x):
        feats = [self.inc(x)]
        for down in self.downs:
            feats.append(down(self.pool(feats[-1])))
        return feats[-1], feats[:-1]          # bottleneck, skip connections

    def forward(self, x, return_features=False):
        bottleneck, skips = self.encode(x)
        y = bottleneck
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            y = dec(torch.cat([skip, up(y)], dim=1))
        logits = self.outc(y)
        return (logits, bottleneck) if return_features else logits
