"""A small U-Net: (path, head dot) in, motif cells out.

Input  (batch, 2, 128, 128): channel 0 = the drawn path, channel 1 = a dot at the head end.
Output (batch, 1, 128, 128): one score per cell; sigmoid(score) = chance the cell is stitched motif.

The U shape: the left side shrinks the picture step by step (128 → 64 → 32 → 16 → 8)
so each cell "sees" a wider area (the whole figure, not only its neighbours); the right
side grows it back to 128. Skip connections hand each size's details straight across,
so the output keeps stitch-level sharpness while knowing the figure as a whole.
"""
import torch
from torch import nn


class Block(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, c_in=2, width=32, depth=4):
        super().__init__()
        widths = [width * 2 ** i for i in range(depth + 1)]  # 32, 64, 128, 256, 512
        self.down = nn.ModuleList([Block(c_in, widths[0])] +
                                  [Block(widths[i], widths[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2)
                                 for i in reversed(range(depth))])
        self.merge = nn.ModuleList([Block(widths[i] * 2, widths[i]) for i in reversed(range(depth))])
        self.out = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            if i:
                x = nn.functional.max_pool2d(x, 2)  # halve the size
            x = block(x)
            skips.append(x)
        skips.pop()  # the smallest level is the bottom of the U, not a skip
        for up, merge in zip(self.up, self.merge):
            x = merge(torch.cat([up(x), skips.pop()], dim=1))  # double the size, add the details back
        return self.out(x)


if __name__ == "__main__":
    net = UNet()
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters")
    print(net(torch.zeros(1, 2, 128, 128)).shape)
