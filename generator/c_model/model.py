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


class PatchDiscriminator(nn.Module):
    """The pix2pix critic: (path, head dot, motif) in, a 14x14 grid of real-or-fake scores out.

    Each score judges one ~70-cell patch, so it learns what stitched figures look like up
    close (crisp edges, white line work) rather than the figure as a whole. Spectral norm
    keeps it from overpowering the generator, which matters with only a few hundred figures.
    """

    def __init__(self, c_in=3, width=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        layers, c = [], c_in
        for w in [width, width * 2, width * 4]:  # 128 → 64 → 32 → 16
            layers += [sn(nn.Conv2d(c, w, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, inplace=True)]
            c = w
        layers += [sn(nn.Conv2d(c, width * 8, 4, padding=1)), nn.LeakyReLU(0.2, inplace=True),  # 16 → 15
                   sn(nn.Conv2d(width * 8, 1, 4, padding=1))]  # 15 → 14
        self.layers = nn.Sequential(*layers)

    def forward(self, x, motif):
        return self.layers(torch.cat([x, motif], dim=1))


class Generator(nn.Module):
    """Unconditional generator: random noise in, one figure's motif cells out.

    The reverse of the U-Net's left side: noise becomes an 8x8 picture with many
    channels, then each step doubles its size (8 → 16 → 32 → 64 → 128) and halves the
    channels. Different noise gives a different figure.
    Output (batch, 1, 128, 128): one score per cell, like the U-Net's.
    """

    def __init__(self, z_dim=128, width=32):
        super().__init__()
        widths = [width * 2 ** i for i in reversed(range(5))]  # 512, 256, 128, 64, 32
        self.start = nn.Linear(z_dim, widths[0] * 8 * 8)
        self.start_norm = nn.Sequential(nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        self.up = nn.ModuleList([Block(widths[i], widths[i + 1]) for i in range(4)])
        self.out = nn.Conv2d(widths[-1], 1, 1)

    def forward(self, z):
        x = self.start_norm(self.start(z).view(len(z), -1, 8, 8))
        for block in self.up:
            x = block(nn.functional.interpolate(x, scale_factor=2, mode="nearest"))  # double the size
        return self.out(x)


class Critic(nn.Module):
    """Judges a motif on its own (no stroke): real figure or generated?

    Two answers from one network: a 16x16 grid of patch scores halfway down (are the
    stitches and white lines right up close?) and one score at the bottom (is the whole
    thing a believable human?). Spectral norm keeps it from overpowering the generator.
    """

    def __init__(self, width=64):
        super().__init__()
        sn = nn.utils.spectral_norm

        def down(c_in, c_out):
            return [sn(nn.Conv2d(c_in, c_out, 4, stride=2, padding=1)), nn.LeakyReLU(0.2, inplace=True)]

        self.local = nn.Sequential(*down(1, width), *down(width, width * 2), *down(width * 2, width * 4))  # → 16
        self.patch = sn(nn.Conv2d(width * 4, 1, 3, padding=1))
        self.whole = nn.Sequential(*down(width * 4, width * 8), *down(width * 8, width * 8),  # → 4
                                   nn.Flatten(), sn(nn.Linear(width * 8 * 4 * 4, 1)))

    def forward(self, motif):
        x = self.local(motif)
        return self.patch(x), self.whole(x)


if __name__ == "__main__":
    net = UNet()
    print(f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters")
    print(net(torch.zeros(1, 2, 128, 128)).shape)
    critic = PatchDiscriminator()
    print(f"{sum(p.numel() for p in critic.parameters()) / 1e6:.1f}M parameters")
    print(critic(torch.zeros(1, 2, 128, 128), torch.zeros(1, 1, 128, 128)).shape)
    gen, free_critic = Generator(), Critic()
    print(f"{sum(p.numel() for p in gen.parameters()) / 1e6:.1f}M, "
          f"{sum(p.numel() for p in free_critic.parameters()) / 1e6:.1f}M parameters")
    print(gen(torch.zeros(2, 128)).shape, [s.shape for s in free_critic(torch.zeros(2, 1, 128, 128))])
