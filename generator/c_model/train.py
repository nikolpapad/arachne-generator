"""Train the U-Net on dataset.npz.  Run:  python generator/c_model/train.py

Each time a figure is used it is changed a little (augmentation), so 346 figures
look like many more:
  - mirrored left-right (half the time)
  - shifted to a random spot on the canvas (staying fully inside)
  - path corners nudged by a few cells, so the model copes with imprecise strokes

Loss = BCE (each cell right or wrong) + Dice (overlap of the whole shape; keeps the
model from just predicting "all ground", since motif is only ~8% of the canvas).

Every --every epochs it scores the held-out test figures (no augmentation) and writes:
  checkpoints/last.pt, checkpoints/best.pt   (best = lowest test loss)
  out/train_log.csv                          epoch, train loss, test loss, test IoU
  out/samples_<epoch>.png                    per test figure: path | model | model thresholded | real
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from model import UNet
from prepare import BACKGROUND, GOLD, INK, RED, SIZE, draw_path

HERE = Path(__file__).resolve().parent
JITTER = 2.0  # path corners move by about this many cells (standard deviation)


def inputs(points):
    """Path line + head dot as a (2, SIZE, SIZE) float array."""
    x = np.zeros((2, SIZE, SIZE), np.float32)
    x[0] = draw_path(points)
    y, xx = np.clip(points[0], 1, SIZE - 2)
    x[1, y - 1:y + 2, xx - 1:xx + 2] = 1
    return x


class Figures(torch.utils.data.Dataset):
    def __init__(self, data, which, augment):
        self.motifs = data["motifs"][which].astype(bool)
        self.points = [p[:n] for p, n in zip(data["points"][which], data["n_points"][which])]
        self.augment = augment
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.motifs)

    def __getitem__(self, i):
        motif, points = self.motifs[i], self.points[i].astype(float)
        if self.augment:
            if self.rng.random() < 0.5:
                motif = motif[:, ::-1]
                points = points * [1, -1] + [0, SIZE - 1]
            # shift: anywhere that keeps the whole figure on the canvas
            ys, xs = np.nonzero(motif)
            dy = self.rng.integers(-ys.min(), SIZE - ys.max())
            dx = self.rng.integers(-xs.min(), SIZE - xs.max())
            motif = np.roll(motif, (dy, dx), axis=(0, 1))
            points = points + [dy, dx] + self.rng.normal(0, JITTER, points.shape)
        points = np.clip(np.round(points), 0, SIZE - 1).astype(int)
        return torch.from_numpy(inputs(points)), torch.from_numpy(motif[None].astype(np.float32))


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    overlap = (p * target).sum((1, 2, 3))
    return (1 - (2 * overlap + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()


def loss_fn(logits, target):
    return nn.functional.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def sample_sheet(x, logits, target, path, n=12, scale=2):
    """Rows of: path input | model probability | thresholded | real figure."""
    def rgb(mask_or_prob, line=None):
        a = np.asarray(mask_or_prob, np.float32)[..., None]
        img = (np.array(BACKGROUND) * (1 - a) + np.array(INK) * a).astype(np.uint8)
        if line is not None:
            img[line[0] > 0] = RED
            img[line[1] > 0] = GOLD
        return img

    prob = torch.sigmoid(logits[:n, 0]).cpu().numpy()
    rows = []
    for i in range(min(n, len(prob))):
        xi, real = x[i].cpu().numpy(), target[i, 0].cpu().numpy()
        cells = [rgb(np.zeros((SIZE, SIZE)), xi), rgb(prob[i]), rgb(prob[i] > 0.5, xi), rgb(real, xi)]
        rows.append(np.concatenate([np.pad(c, ((2, 2), (2, 2), (0, 0)), constant_values=200) for c in cells], 1))
    sheet = Image.fromarray(np.concatenate(rows, 0))
    sheet.resize((sheet.width * scale, sheet.height * scale), Image.NEAREST).save(path)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--epochs", type=int, default=300)
    args.add_argument("--batch", type=int, default=16)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--every", type=int, default=10, help="test, save and draw samples every N epochs")
    args.add_argument("--width", type=int, default=32, help="channels in the first layer (model size)")
    args.add_argument("--quick", action="store_true", help="2 epochs of 3 batches, to check the code runs")
    args = args.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(HERE / "dataset.npz")
    test = data["test"]
    train_set, test_set = Figures(data, ~test, augment=True), Figures(data, test, augment=False)
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True)
    x_test, y_test = (torch.stack(t).to(device) for t in zip(*[test_set[i] for i in range(len(test_set))]))

    net = UNet(width=args.width).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    epochs = 2 if args.quick else args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)  # lr slides down to 0 by the end
    (HERE / "checkpoints").mkdir(exist_ok=True)
    (HERE / "out").mkdir(exist_ok=True)
    log = open(HERE / "out" / "train_log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "train_loss", "test_loss", "test_iou"])
    print(f"{len(train_set)} train, {len(test_set)} test figures on {device}, "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters")

    best, start = float("inf"), time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        losses = []
        for step, (x, y) in enumerate(loader):
            if args.quick and step == 3:
                break
            x, y = x.to(device), y.to(device)
            loss = loss_fn(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        if epoch % args.every and epoch != epochs:
            continue
        net.eval()
        with torch.no_grad():
            logits = torch.cat([net(x_test[i:i + args.batch]) for i in range(0, len(x_test), args.batch)])
            test_loss = loss_fn(logits, y_test).item()
            pred = logits > 0
            iou = ((pred & (y_test > 0)).sum() / (pred | (y_test > 0)).sum()).item()
        train_loss = float(np.mean(losses))
        writer.writerow([epoch, round(train_loss, 4), round(test_loss, 4), round(iou, 4)])
        log.flush()
        print(f"epoch {epoch:4d}  train {train_loss:.3f}  test {test_loss:.3f}  IoU {iou:.3f}  "
              f"{time.time() - start:.0f}s")
        state = {"model": net.state_dict(), "width": args.width, "epoch": epoch, "test_loss": test_loss}
        torch.save(state, HERE / "checkpoints" / "last.pt")
        if test_loss < best:
            best = test_loss
            torch.save(state, HERE / "checkpoints" / "best.pt")
        sample_sheet(x_test, logits, y_test, HERE / "out" / f"samples_{epoch:04d}.png")
    log.close()
    print(f"done; best test loss {best:.3f}. Samples in {HERE / 'out'}")


if __name__ == "__main__":
    main()
