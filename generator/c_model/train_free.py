"""Train a GAN that draws new human motifs from nothing (no stroke).  Run:
  python generator/c_model/train_free.py                     train
  python generator/c_model/train_free.py --generate 32       draw 32 new figures from checkpoints/free_last.pt

The question here is only "can a model learn to draw convincing filet-crochet humans?";
following a user's stroke comes later. The generator turns random noise into a figure;
the critic (model.Critic) judges each figure, both up close and as a whole.

384 figures is very few for a GAN: the critic would soon memorise them and stop teaching
the generator anything. DiffAugment (Zhao et al. 2020) prevents that: every image the
critic sees, real or generated, is first shifted and has a random square blanked out.
The generator's figures go through the same changes, so it can't win by drawing them.

All 384 figures are used (no test set: nothing here is predicted). They are already
centred, so only left-right mirroring is applied to the figures themselves.

Every --every epochs it writes:
  checkpoints/free_last.pt              generator, its averaged copy, critic
  checkpoints/free_<epoch>.pt           the averaged generator alone, every --snapshot epochs
  out/free_log.csv                      epoch, losses, ink share, nearest-real IoU
  out/free_samples_<epoch>.png          16 new figures (the same noise each time), each
                                        beside the most similar real figure
Samples come from an averaged copy of the generator (weights averaged over recent steps),
which makes steadier figures than the generator of any single step.

Nearest-real IoU: how much each new figure overlaps its most similar real one. Around
0.3-0.6 means new figures in the style of the real ones; above ~0.8 means it is copying.
"""
import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from model import Critic, Generator
from prepare import BACKGROUND, INK, SIZE

HERE = Path(__file__).resolve().parent
Z_DIM = 128
EMA = 0.999  # the averaged generator moves 0.1% of the way towards the trained one each step...
# ...but only once training is well under way; at first it follows closely (see ema_decay)


def ema_decay(step):
    """0.1 at step 0, 0.99 at step 1000, capped at EMA: a slow average of a young generator
    would still be mostly its random starting weights, and draw blank canvases."""
    return min(EMA, (1 + step) / (10 + step))


def diff_augment(motif):
    """Shift by up to 1/8 of the canvas, then blank a random half-size square (DiffAugment)."""
    b, shift, hole = len(motif), SIZE // 8, SIZE // 2
    padded = nn.functional.pad(motif, [shift] * 4)
    dy, dx = torch.randint(0, 2 * shift + 1, (2, b)).tolist()
    motif = torch.stack([padded[i, :, dy[i]:dy[i] + SIZE, dx[i]:dx[i] + SIZE] for i in range(b)])
    keep = torch.ones_like(motif)
    hy, hx = torch.randint(0, SIZE - hole + 1, (2, b)).tolist()
    for i in range(b):
        keep[i, :, hy[i]:hy[i] + hole, hx[i]:hx[i] + hole] = 0
    return motif * keep


def critic_loss(critic, real, fake):
    """Hinge loss: push real scores above +1 and generated ones below -1, for both heads."""
    loss = 0
    for r, f in zip(critic(diff_augment(real)), critic(diff_augment(fake))):
        loss = loss + nn.functional.relu(1 - r).mean() + nn.functional.relu(1 + f).mean()
    return loss / 2


def generator_loss(critic, fake):
    return -sum(s.mean() for s in critic(diff_augment(fake))) / 2


def nearest_real(made, real):
    """For each made figure (n, SIZE, SIZE bool): the index and IoU of the most similar real one."""
    made, both = made.flatten(1).float(), torch.cat([real, real.flip(-1)]).flatten(1).float()
    overlap = made @ both.T
    iou = overlap / (made.sum(1, keepdim=True) + both.sum(1) - overlap).clamp(min=1)
    best, index = iou.max(1)
    flipped = index >= len(real)
    return index % len(real), flipped, best


def sheet(made, real, path, per_row=4, scale=2):
    """Pairs of: new figure | most similar real figure (mirrored if that matched better)."""
    index, flipped, iou = nearest_real(made, real)

    def rgb(mask):
        a = mask.cpu().numpy().astype(np.float32)[..., None]
        img = (np.array(BACKGROUND) * (1 - a) + np.array(INK) * a).astype(np.uint8)
        return np.pad(img, ((2, 2), (2, 2), (0, 0)), constant_values=200)

    pairs = []
    for m, i, f in zip(made, index, flipped):
        twin = real[i].flip(-1) if f else real[i]
        pairs.append(np.pad(np.concatenate([rgb(m), rgb(twin)], 1), ((0, 0), (0, 8), (0, 0)), constant_values=255))
    rows = [np.concatenate(pairs[r:r + per_row], 1) for r in range(0, len(pairs), per_row)]
    img = Image.fromarray(np.concatenate(rows, 0))
    img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(path)
    return iou.mean().item()


def generate(args, real, device):
    state = torch.load(args.checkpoint, map_location=device)
    gen = Generator(Z_DIM, state["width"]).to(device).eval()
    gen.load_state_dict(state["ema"])
    torch.manual_seed(args.seed)
    with torch.no_grad():
        made = torch.sigmoid(gen(torch.randn(args.generate, Z_DIM, device=device)))[:, 0] > 0.5
    name = HERE / "out" / f"free_generated_{args.checkpoint.stem}_seed{args.seed}"
    np.savez_compressed(name.with_suffix(".npz"), motifs=made.cpu().numpy().astype(np.uint8))
    iou = sheet(made, real, name.with_suffix(".png"))
    print(f"{args.generate} figures from {args.checkpoint} (epoch {state['epoch']}), "
          f"nearest-real IoU {iou:.3f}: {name}.png and .npz (0/1 grids)")


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--epochs", type=int, default=1000)
    args.add_argument("--batch", type=int, default=16)
    args.add_argument("--lr", type=float, default=2e-4)
    args.add_argument("--every", type=int, default=25, help="save and draw samples every N epochs")
    args.add_argument("--snapshot", type=int, default=100, help="keep a copy of the generator every N epochs")
    args.add_argument("--width", type=int, default=32, help="channels in the generator's last layer")
    args.add_argument("--quick", action="store_true", help="2 epochs of 3 batches, to check the code runs")
    args.add_argument("--generate", type=int, metavar="N", help="don't train; draw N new figures")
    args.add_argument("--checkpoint", type=Path, default=HERE / "checkpoints" / "free_last.pt",
                      help="for --generate: which saved generator to use")
    args.add_argument("--seed", type=int, default=0, help="for --generate: same seed, same figures")
    args = args.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    real = torch.from_numpy(np.load(HERE / "dataset.npz")["motifs"] > 0).to(device)
    (HERE / "out").mkdir(exist_ok=True)
    if args.generate:
        return generate(args, real, device)

    gen, critic = Generator(Z_DIM, args.width).to(device), Critic().to(device)
    ema = copy.deepcopy(gen).eval()
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.5, 0.999))
    fixed_z = torch.randn(16, Z_DIM, device=device, generator=torch.Generator(device).manual_seed(0))
    epochs = 2 if args.quick else args.epochs
    (HERE / "checkpoints").mkdir(exist_ok=True)
    log = open(HERE / "out" / "free_log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "critic_loss", "gen_loss", "ink_share", "nearest_real_iou"])
    print(f"{len(real)} figures on {device}, generator {sum(p.numel() for p in gen.parameters()) / 1e6:.1f}M, "
          f"critic {sum(p.numel() for p in critic.parameters()) / 1e6:.1f}M parameters; "
          f"real figures cover {real.float().mean():.3f} of the canvas")

    start, steps = time.time(), 0
    for epoch in range(1, epochs + 1):
        gen.train()
        losses = []
        order = torch.randperm(len(real), device=device)
        for step in range(len(real) // args.batch):
            if args.quick and step == 3:
                break
            y = real[order[step * args.batch:(step + 1) * args.batch], None].float()
            mirror = torch.rand(len(y), 1, 1, 1, device=device) < 0.5
            y = torch.where(mirror, y.flip(-1), y)
            fake = torch.sigmoid(gen(torch.randn(len(y), Z_DIM, device=device)))

            d_loss = critic_loss(critic, y, fake.detach())  # detach: this step doesn't train the generator
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            g_loss = generator_loss(critic, fake)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            with torch.no_grad():
                decay = ema_decay(steps)
                for p_ema, p in zip(ema.parameters(), gen.parameters()):
                    p_ema.lerp_(p, 1 - decay)
                for b_ema, b in zip(ema.buffers(), gen.buffers()):
                    b_ema.copy_(b)  # batch-norm running statistics
            losses.append([d_loss.item(), g_loss.item()])
            steps += 1

        if epoch % args.every and epoch != epochs:
            continue
        with torch.no_grad():
            made = torch.sigmoid(ema(fixed_z))[:, 0] > 0.5
        iou = sheet(made, real, HERE / "out" / f"free_samples_{epoch:04d}.png")
        d_loss, g_loss = np.mean(losses, axis=0)
        ink = made.float().mean().item()
        writer.writerow([epoch] + [round(float(v), 4) for v in (d_loss, g_loss, ink, iou)])
        log.flush()
        print(f"epoch {epoch:4d}  critic {d_loss:.3f}  gen {g_loss:.3f}  ink {ink:.3f}  "
              f"nearest-real IoU {iou:.3f}  {time.time() - start:.0f}s")
        state = {"ema": ema.state_dict(), "width": args.width, "epoch": epoch}
        torch.save({**state, "model": gen.state_dict(), "critic": critic.state_dict()},
                   HERE / "checkpoints" / "free_last.pt")
        if epoch % args.snapshot == 0:
            torch.save(state, HERE / "checkpoints" / f"free_{epoch:04d}.pt")
    log.close()
    print(f"done. Samples in {HERE / 'out'}; draw more with --generate 32")


if __name__ == "__main__":
    main()
