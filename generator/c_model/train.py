"""Train the U-Net on dataset.npz.  Run:  python generator/c_model/train.py

--data picks another data file with the same keys, e.g. catalog.npz (pieces of the whole
catalog, from prepare_catalog.py) for pretraining. Stroke input needs dataset.npz: only it has paths.

What the model is given (--input):
  shape   (default) a rough blocky silhouette: the figure seen through blocks of 5-8
          cells, the way a user would paint it with a big brush. The model adds the detail.
  stroke  the head-to-feet line plus a dot at the head. Too little to go on: the
          model draws the same robed figure for every stroke.

Each time a figure is used it is changed a little (augmentation), so 346 figures
look like many more:
  - mirrored left-right (half the time)
  - shifted to a random spot on the canvas (staying fully inside)
  - shape: block size and how full a block must be are picked at random, so the
    model copes with coarser or finer, fatter or thinner painting
  - stroke: path corners nudged by a few cells, so the model copes with imprecise strokes

Loss = BCE (each cell right or wrong) + Dice (overlap of the whole shape; keeps the
model from just predicting "all ground", since motif is only ~8% of the canvas).

At the start it writes out/train_settings.json: arguments, data file, sizes, device, start time.
Every --every epochs it scores the held-out test figures (no augmentation) and writes:
  checkpoints/last.pt, checkpoints/best.pt   (best = lowest test loss)
  out/train_log.csv                          epoch, train loss, test loss, test IoU
  out/samples_<epoch>.png                    per test figure: input | model | thresholded, specks removed | real
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.measure import label
from torch import nn

from model import UNet
from prepare import BACKGROUND, GOLD, INK, RED, SIZE, draw_path

HERE = Path(__file__).resolve().parent
JITTER = 2.0  # path corners move by about this many cells (standard deviation)
CHANNELS = {"shape": 1, "stroke": 2}
BLOCK, FILL = 6, 0.35            # shape input for test figures: block side, share of a block that must be motif
BLOCKS, FILLS = (5, 8), (0.2, 0.5)  # ranges picked from during training
SPECK = 3                        # despeckle: pieces of motif or holes of ground smaller than this go
TINT = (0xF0, 0xC8, 0xC0)


def inputs(points):
    """Path line + head dot as a (2, SIZE, SIZE) float array."""
    x = np.zeros((2, SIZE, SIZE), np.float32)
    x[0] = draw_path(points)
    y, xx = np.clip(points[0], 1, SIZE - 2)
    x[1, y - 1:y + 2, xx - 1:xx + 2] = 1
    return x


def rough_shape(motif, block=BLOCK, fill=FILL):
    """The figure as a blocky silhouette: a block is on where at least `fill` of its cells are motif."""
    n = -(-SIZE // block)  # blocks per side, rounded up
    padded = np.zeros((n * block, n * block), np.float32)
    padded[:SIZE, :SIZE] = motif
    on = padded.reshape(n, block, n, block).mean((1, 3)) >= fill
    return on.repeat(block, 0).repeat(block, 1)[:SIZE, :SIZE]


def despeckle(mask, smallest=SPECK):
    """Remove motif specks and fill ground pinholes smaller than `smallest` cells; they don't crochet well."""
    for value in (True, False):
        labels = label(mask == value, connectivity=1)
        sizes = np.bincount(labels.ravel())
        small = (sizes < smallest)[labels] & (labels > 0)
        mask = mask ^ small
    return mask


class Figures(torch.utils.data.Dataset):
    def __init__(self, data, which, augment, input="shape"):
        if input == "stroke" and "points" not in data:
            raise SystemExit("stroke input needs dataset.npz: this data file has no head-to-feet paths")
        self.motifs = data["motifs"][which].astype(bool)
        if input == "stroke":
            self.points = [p[:n] for p, n in zip(data["points"][which], data["n_points"][which])]
        self.augment = augment
        self.input = input
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.motifs)

    def __getitem__(self, i):
        motif = self.motifs[i]
        points = self.points[i].astype(float) if self.input == "stroke" else np.zeros((0, 2))  # no path to move
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
        target = torch.from_numpy(motif[None].astype(np.float32))
        if self.input == "stroke":
            points = np.clip(np.round(points), 0, SIZE - 1).astype(int)
            return torch.from_numpy(inputs(points)), target
        block, fill = BLOCK, FILL
        if self.augment:
            block, fill = self.rng.integers(BLOCKS[0], BLOCKS[1] + 1), self.rng.uniform(*FILLS)
        return torch.from_numpy(rough_shape(motif, block, fill)[None].astype(np.float32)), target


def progress(epoch, epochs, start, width=30):
    """Redraw one line in place: bar, epoch count, elapsed and estimated time left."""
    elapsed = time.time() - start
    left = elapsed / epoch * (epochs - epoch)
    filled = round(width * epoch / epochs)
    print(f"\r[{'#' * filled}{'.' * (width - filled)}] epoch {epoch}/{epochs}  "
          f"{elapsed / 60:.0f} min done, ~{left / 60:.0f} min left ", end="", flush=True)


def save_settings(path, args, data_path, n_train, n_test, device):
    """Write how this run was made (every argument, data, sizes, device, start time) next to its results."""
    settings = {**vars(args), "data_file": str(data_path), "train_items": n_train, "test_items": n_test,
                "device": device, "started": datetime.now().isoformat(timespec="seconds")}
    path.write_text(json.dumps(settings, indent=1, default=str))


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    overlap = (p * target).sum((1, 2, 3))
    return (1 - (2 * overlap + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()


def loss_fn(logits, target):
    return nn.functional.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def sample_sheet(x, logits, target, path, n=12, scale=2):
    """Rows of: input | model probability | thresholded and despeckled | real figure.
    A stroke is drawn in red (gold head); a rough shape tints the ground it covers."""
    def rgb(mask_or_prob, x=None):
        a = np.asarray(mask_or_prob, np.float32)[..., None]
        img = (np.array(BACKGROUND) * (1 - a) + np.array(INK) * a).astype(np.uint8)
        if x is not None and len(x) == 2:
            img[x[0] > 0] = RED
            img[x[1] > 0] = GOLD
        elif x is not None:
            img[(x[0] > 0) & (a[..., 0] < 0.5)] = TINT
        return img

    prob = torch.sigmoid(logits[:n, 0]).cpu().numpy()
    rows = []
    for i in range(min(n, len(prob))):
        xi, real = x[i].cpu().numpy(), target[i, 0].cpu().numpy()
        cells = [rgb(np.zeros((SIZE, SIZE)), xi), rgb(prob[i]), rgb(despeckle(prob[i] > 0.5), xi), rgb(real, xi)]
        rows.append(np.concatenate([np.pad(c, ((2, 2), (2, 2), (0, 0)), constant_values=200) for c in cells], 1))
    sheet = Image.fromarray(np.concatenate(rows, 0))
    sheet.resize((sheet.width * scale, sheet.height * scale), Image.NEAREST).save(path)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--input", choices=CHANNELS, default="shape", help="what the model is given (see top of file)")
    args.add_argument("--data", type=Path, default="dataset.npz", help="data file in generator/c_model/ or a full path")
    args.add_argument("--epochs", type=int, default=300)
    args.add_argument("--batch", type=int, default=16)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--every", type=int, default=10, help="test, save and draw samples every N epochs")
    args.add_argument("--width", type=int, default=32, help="channels in the first layer (model size)")
    args.add_argument("--quick", action="store_true", help="2 epochs of 3 batches, to check the code runs")
    args = args.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = HERE / args.data  # a full path stays as it is
    data = np.load(data_path)
    test = data["test"]
    train_set, test_set = Figures(data, ~test, True, args.input), Figures(data, test, False, args.input)
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True)
    x_test, y_test = (torch.stack(t).to(device) for t in zip(*[test_set[i] for i in range(len(test_set))]))

    net = UNet(c_in=CHANNELS[args.input], width=args.width).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    epochs = 2 if args.quick else args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)  # lr slides down to 0 by the end
    (HERE / "checkpoints").mkdir(exist_ok=True)
    (HERE / "out").mkdir(exist_ok=True)
    save_settings(HERE / "out" / "train_settings.json", args, data_path, len(train_set), len(test_set), device)
    log = open(HERE / "out" / "train_log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "train_loss", "test_loss", "test_iou"])
    print(f"{len(train_set)} train, {len(test_set)} test items from {data_path.name} on {device}, input {args.input}, "
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
        progress(epoch, epochs, start)

        if epoch % args.every and epoch != epochs:
            continue
        print()
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
        state = {"model": net.state_dict(), "width": args.width, "input": args.input, "epoch": epoch, "test_loss": test_loss}
        torch.save(state, HERE / "checkpoints" / "last.pt")
        if test_loss < best:
            best = test_loss
            torch.save(state, HERE / "checkpoints" / "best.pt")
        sample_sheet(x_test, logits, y_test, HERE / "out" / f"samples_{epoch:04d}.png")
    log.close()
    print(f"done; best test loss {best:.3f}. Samples in {HERE / 'out'}")


if __name__ == "__main__":
    main()
