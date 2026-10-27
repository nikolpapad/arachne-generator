# Method C, first model: what we built and why

> **The current plan is [generation-plan.md](generation-plan.md).** This report records the first model (the U-Net) and the reasons behind it.

Status on 2026-10-25: the data is prepared and the training code works (tested on CPU).
The first real training run is next, on the GTX 960M laptop.

**Goal.** A user draws one stroke on the canvas, from the head to the feet. The model
draws a filet-crochet human that follows the stroke, in the style of the real catalog,
without copying one pattern.

---

## 1. The pipeline at a glance

```
tags.json          which patterns contain a human            (search page, catalog/serve.py)
   │
motif_boxes.json   where each human is, and what kind        (box page, boxes.py)
   │
dataset.npz        384 (stroke, figure) pairs, 128×128       (prepare.py)
   │
checkpoints/       the trained network                       (train.py + model.py)
   │
generated grids    stroke in → figure out                    (sample.py, not written yet)
```

| File | What it does |
|---|---|
| `generator/c_model/boxes.py` + `boxes.html` | Review page: put a box around each human, say what kind it is |
| `generator/c_model/prepare.py` | Cuts, cleans and centres each figure; computes its head-to-feet stroke |
| `generator/c_model/model.py` | The network (a U-Net) |
| `generator/c_model/train.py` | The training loop: augmentation, loss, test score, sample pictures |

---

## 2. Decisions, and why

### Which motif: human
`count.py` showed **human 286**, putti 188, animal 106, woman 16. The handoff set a
gate of about 200 examples before a trained model is worth trying, and only human
passed it.

### Boxes reviewed by hand
A tag says that a pattern *contains* a human, not *where*. CLIP suggested boxes, and
every box was checked by eye: 285 patterns reviewed, 597 boxes. Automatic boxes would
have let borders, animals and lettering into the training data.

### Kinds, and size measured automatically
Every box got a kind: full figure, half figure, head only, or other human (groups,
lying down, partly hidden). Size was not labelled by hand because it is simply the box
height in cells.

| kind | small <40 | medium 40–99 | large 100+ | all |
|---|---|---|---|---|
| Full figure | 6 | 295 | 148 | **449** |
| Half figure | 0 | 9 | 0 | 9 |
| Head only | 9 | 3 | 2 | 14 |
| Other human | 5 | 47 | 73 | 125 |

**Why it matters:** with few examples, a model that sees very different things for
the same kind of input learns their *average*, which comes out as a blurry blob.
Training on one consistent kind gives it a fair chance.

### Training set: full figures up to 128 cells, kept at their real size
- Full figures are the biggest consistent group (449).
- 80% of them are 53–136 cells tall. A 128×128 canvas holds about 400 of them, and the
  65 larger ones are skipped.
- **No resizing.** A filet figure 40 cells tall is drawn differently from one 120
  cells tall (fewer details, blockier). Shrinking big figures would blur their
  stitches, and stretching small ones would make fake blocky details. Each figure is
  placed on the canvas at 1 cell = 1 cell.
- The handoff suggested 64×64, but most figures would have had to shrink.

### Cleaning what the rectangle caught
A rectangle also catches nearby letters, crests, foliage or ground lines. The cleanup
keeps every *connected piece* of stitches with at least 15% of the cells of the biggest
piece, and drops the rest. On the review sheets the dropped cells (grey) were almost
all decoration. Things that *touch* the figure (a held sword, a horse under a rider)
can't be separated this way and stay in.

### The stroke: head to feet, smoothed, head marked
The model learns "stroke → figure". At use time the stroke comes from a person, so in
training it has to look like a person's stroke and mean the same thing every time.

1. **First version (rejected):** the longest line through the figure's skeleton. It
   wandered through arms, shields and swords, so its meaning changed from figure to
   figure and it looked nothing like a hand-drawn stroke.
2. **Current version:**
   - **Spine:** the route along the skeleton from the topmost point (taken as the head)
     to the bottommost point (the feet).
   - **Smoothed** into 3–17 straight segments, so it looks hand-drawn.
   - **Head dot:** a second input channel marks the head end, so the model knows
     which way up the figure goes.

This sets the rule for the app: **"draw one stroke from the head to the feet."**

Known weakness: in about 1 figure in 10, a raised arm or object is higher than the
head, so the stroke starts at the hand (section 5).

### Testing on unseen patterns
About 10% of *patterns* (21 patterns, 38 figures) are kept out of training. All
figures from one pattern stay on the same side, because figures from the same page
often look alike and would make the test too easy.

### Augmentation (applied fresh every time, in `train.py`)
346 training figures is small, so each one is varied randomly each time it is used:
- **Mirrored left-right**, half the time. Upside-down is not used, since an upside-down
  human isn't a valid figure.
- **Shifted** to a random place on the canvas, staying fully inside.
- **Stroke corners nudged** by about 2 cells, so the model copes with imprecise strokes.

### The network: a small U-Net
- **In:** 2 channels (stroke, head dot). **Out:** one score per cell, "how likely this
  cell is stitched".
- **The U shape:** the left half shrinks the picture 128 → 64 → 32 → 16 → 8, so deep
  layers see the whole figure at once. The right half grows it back. *Skip
  connections* pass each size's details across, so the output can stay sharp at the
  stitch level.
- **Size:** 7.8M parameters (`--width 32`). That's small by modern standards and fits
  a 960M.

### The loss: BCE + Dice
- **BCE** scores each cell as right or wrong.
- **Dice** scores the overlap of the whole shape. It is needed because motif is only
  about 8% of the canvas, so with BCE alone "everything is ground" already scores well.

### Optimiser and schedule
Adam, learning rate 1e-3, sliding down to 0 over the run (cosine schedule). Big steps
early, fine adjustments at the end. 300 epochs by default.

### Train on everything first
Putti and horse riders are still in the set. Train once with everything as a baseline,
then try without them and compare (section 6).

---

## 3. Training on the GTX 960M laptop

**Is it worth it? Yes.** On this machine's 4-CPU WSL one training step takes about 7
seconds, so 300 epochs would take **about 12 hours**. A 960M should be many times
faster. My guess is under an hour, but I haven't measured it, so time a short run
first.

**What the laptop needs:** the code, `catalog/load.py` and `dataset.npz`. The grids
and CLIP files aren't needed.

```bash
# from this machine (replace user@laptop and the target folder)
rsync -av --relative generator/c_model/*.py generator/c_model/dataset.npz catalog/load.py \
      user@laptop:~/arachne-generator/
```

On the laptop:
```bash
cd ~/arachne-generator
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
```

**Check the GPU is really used.** The 960M is an old (Maxwell) card. It needs
`sm_50` in the printed arch list. Newer PyTorch CUDA builds have been dropping old
cards. If `sm_50` is missing, `is_available()` is False, or training stops with "no
kernel image is available", install an older CUDA build of torch (`cu118`, or an older
torch version). The NVIDIA driver on the laptop must also be new enough for the CUDA
version you pick (`nvidia-smi` shows it).

**Run it.** Use `tmux` or `nohup`, so closing the SSH connection doesn't stop training:
```bash
cd ~/arachne-generator/generator/c_model
../../.venv/bin/python train.py --quick                # 1 minute: does it run on the GPU?
../../.venv/bin/python train.py --epochs 20            # time this, then multiply by 15 for 300 epochs
nohup ../../.venv/bin/python train.py > train.out 2>&1 &   # the full run
tail -f train.out
```
If the GPU runs out of memory (the 960M has 2 or 4 GB), use `--batch 8`.

Copy the results back:
```bash
rsync -av user@laptop:~/arachne-generator/generator/c_model/{out,checkpoints} generator/c_model/
```

---

## 4. How to read the results

Every 10 epochs `train.py` prints a line and writes `out/samples_<epoch>.png`.

**The numbers** (`out/train_log.csv`):
- **train loss**: how wrong it is on the figures it learns from. It should go down.
- **test loss**: how wrong it is on the 38 figures it never sees. This is what counts.
  `best.pt` is saved where it was lowest.
- **IoU**: overlap between the predicted and real figures on the test set, from 0 to 1.
  It will never be close to 1, because many different figures fit the same stroke.
  Treat it as a trend, not a target.

**The sample pictures:** each row is one test figure, with four columns: the stroke,
the model's guess (grey = unsure), the guess as stitches, and the real figure. You
judge these by eye.

| What you see | What it means |
|---|---|
| Grey fog, no shape (early epochs) | Normal at the start |
| A human-ish silhouette along the stroke, soft edges | Working: it learned "a figure goes along the stroke" |
| Blobby figure, no details, same for every stroke | **Averaging**: too few or too varied examples (section 6) |
| Train loss keeps falling, test loss rises | **Overfitting**: memorising the training figures. Use the earlier `best.pt` |
| Sharp output that matches one training figure exactly | **Copying**, which the owner doesn't want. `evaluate.py` will check this |

Expect the first run to look soft and blobby. A plain U-Net gives one fixed answer per
stroke, so where several figures fit, it blends them. If the outline and pose are right
but the inside is mush, the model is on the right track, and the next steps below
target the mush.

---

## 5. Known weaknesses of the current data

- **Raised arms or objects:** about 1 in 10 strokes starts at a hand, not the head,
  because the highest point is taken as the head.
- **Lying or flying figures:** they get a sideways stroke, for the same reason.
- **Horse riders** are labelled full figure, and the horse stays in because it touches
  the rider.
- **Putti and adults are mixed.** The `human` tag overlaps `putti`, so winged children
  and adult figures in armour or dresses are learned together.
- **65 tall figures** (over 128 cells) aren't used.

---

## 6. Future improvements, roughly in order

1. **Compare without putti and riders.** Rerun `prepare.py` with them excluded
   (putti via `tags.json`, riders relabelled "Other human" in the box page) and train
   again. Less data but more consistent. Compare the sample sheets side by side.
2. **Click the head on the wrong strokes.** A small addition to the box page: about 40
   clicks fix the raised-arm cases.
3. **`sample.py`:** stroke in, `{w, h, rows, palette}` out, with motif = `"0"` and
   ground = `"1"`, so it imports back into Arachne Studio.
4. **`evaluate.py`:** for each generated figure, find the closest training figure and
   show them side by side. This is the "no plain copy-paste" check from the handoff.
5. **Clean up specks:** remove single isolated cells from the output, which are hard to
   crochet.
6. **Pretraining on the whole catalog** (handoff 4.3). First teach the network "what
   filet looks like" on all 2283 patterns by hiding a patch and asking it to fill it
   in, then train on the figures. More data for the "texture" part of the job.
7. **Sharper output:**
   - Add a small *discriminator* (pix2pix GAN), a second network that penalises
     outputs that don't look like real filet. Classic fix for blur, but trickier to
     train.
   - Or move to a **small diffusion model**, which picks one plausible figure instead
     of averaging them. The handoff's planned next step if the U-Net is promising but
     blurry. Slower, and more work on a 960M.
8. **More data:**
   - Tag more patterns (putti is at 188, close to the gate).
   - Use the 65 tall figures with a 192 or 256 canvas.
   - Add "other human" boxes once the basics work.
9. **Control over size:** give the model the intended figure height as an extra input,
   so a short stroke gives a small, simple figure and a long one a detailed figure.
