"""Train the U-Net as a pix2pix generator.  Run:  python generator/c_model/train_gan.py

train.py's U-Net hedges: when a stroke fits many figures it paints grey wherever a figure
might be, so it draws a blurry average. Here a second network, the critic, looks at
(stroke, motif) pairs and learns to tell real figures from generated ones. The U-Net is
trained to fool it, which pushes it to commit to one sharp, believable figure.

Generator loss = adversarial (fool the critic) + --l1 x L1 to the real figure. L1 keeps
the figure in the right place along the stroke; the critic makes it look stitched.
Same data, augmentation and test figures as train.py.

Every --every epochs it writes, next to train.py's files without touching them:
  checkpoints/gan_last.pt                     generator + critic
  out/gan_log.csv                             epoch, losses, test L1, test IoU
  out/gan_samples_<epoch>.png                 per test figure: path | model | model thresholded | real
There is no "best" checkpoint: GAN losses don't say which epoch looks best, the samples do.
IoU will likely go *down* compared with train.py; a sharp figure in a different pose
overlaps the real one less than a blur does.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import PatchDiscriminator, UNet
from train import Figures, progress, sample_sheet

HERE = Path(__file__).resolve().parent


def adversarial(scores, real):
    """BCE pushing every patch score towards real (1) or fake (0)."""
    return nn.functional.binary_cross_entropy_with_logits(scores, torch.full_like(scores, float(real)))


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--epochs", type=int, default=300)
    args.add_argument("--batch", type=int, default=16)
    args.add_argument("--lr", type=float, default=2e-4)
    args.add_argument("--l1", type=float, default=100, help="weight of L1 against the adversarial loss")
    args.add_argument("--every", type=int, default=10, help="test, save and draw samples every N epochs")
    args.add_argument("--width", type=int, default=32, help="channels in the generator's first layer")
    args.add_argument("--start-from", type=Path, help="U-Net checkpoint from train.py to start the generator from")
    args.add_argument("--quick", action="store_true", help="2 epochs of 3 batches, to check the code runs")
    args = args.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(HERE / "dataset.npz")
    test = data["test"]
    train_set, test_set = Figures(data, ~test, augment=True), Figures(data, test, augment=False)
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True)
    x_test, y_test = (torch.stack(t).to(device) for t in zip(*[test_set[i] for i in range(len(test_set))]))

    gen = UNet(width=args.width).to(device)
    if args.start_from:
        state = torch.load(args.start_from, map_location=device)
        if state["width"] != args.width:
            raise SystemExit(f"{args.start_from} has width {state['width']}; pass --width {state['width']}")
        gen.load_state_dict(state["model"])
    critic = PatchDiscriminator().to(device)
    # betas (0.5, 0.999): the pix2pix setting; less momentum keeps the two networks from overshooting
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.999))
    epochs = 2 if args.quick else args.epochs
    scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, epochs) for o in (opt_g, opt_d)]
    (HERE / "checkpoints").mkdir(exist_ok=True)
    (HERE / "out").mkdir(exist_ok=True)
    log = open(HERE / "out" / "gan_log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "critic_loss", "gen_adv_loss", "gen_l1", "test_l1", "test_iou"])
    print(f"{len(train_set)} train, {len(test_set)} test figures on {device}, "
          f"generator {sum(p.numel() for p in gen.parameters()) / 1e6:.1f}M, "
          f"critic {sum(p.numel() for p in critic.parameters()) / 1e6:.1f}M parameters"
          + (f", generator from {args.start_from}" if args.start_from else ""))

    start = time.time()
    for epoch in range(1, epochs + 1):
        gen.train()
        critic.train()
        losses = []
        for step, (x, y) in enumerate(loader):
            if args.quick and step == 3:
                break
            x, y = x.to(device), y.to(device)
            fake = torch.sigmoid(gen(x))

            # critic: real pairs → 1, generated pairs → 0 (detach: this step doesn't train the generator)
            d_loss = (adversarial(critic(x, y), True) + adversarial(critic(x, fake.detach()), False)) / 2
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # generator: make the critic say real, while staying close to the real figure
            g_adv = adversarial(critic(x, fake), True)
            g_l1 = (fake - y).abs().mean()
            opt_g.zero_grad()
            (g_adv + args.l1 * g_l1).backward()
            opt_g.step()
            losses.append([d_loss.item(), g_adv.item(), g_l1.item()])
        for s in scheds:
            s.step()
        progress(epoch, epochs, start)

        if epoch % args.every and epoch != epochs:
            continue
        print()
        gen.eval()
        with torch.no_grad():
            logits = torch.cat([gen(x_test[i:i + args.batch]) for i in range(0, len(x_test), args.batch)])
            test_l1 = (torch.sigmoid(logits) - y_test).abs().mean().item()
            pred = logits > 0
            iou = ((pred & (y_test > 0)).sum() / (pred | (y_test > 0)).sum()).item()
        d_loss, g_adv, g_l1 = np.mean(losses, axis=0)
        writer.writerow([epoch] + [round(float(v), 4) for v in (d_loss, g_adv, g_l1, test_l1, iou)])
        log.flush()
        print(f"epoch {epoch:4d}  critic {d_loss:.3f}  gen adv {g_adv:.3f}  L1 {g_l1:.3f}  "
              f"test L1 {test_l1:.3f}  IoU {iou:.3f}  {time.time() - start:.0f}s")
        torch.save({"model": gen.state_dict(), "critic": critic.state_dict(), "width": args.width,
                    "epoch": epoch, "test_l1": test_l1}, HERE / "checkpoints" / "gan_last.pt")
        sample_sheet(x_test, logits, y_test, HERE / "out" / f"gan_samples_{epoch:04d}.png")
    log.close()
    print(f"done. Samples in {HERE / 'out'}")


if __name__ == "__main__":
    main()
