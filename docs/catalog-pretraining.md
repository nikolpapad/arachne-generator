# Phase 2: pretraining on the whole catalog

**Status (2026-10-27): built and checked on CPU; the three Colab runs are next.** Branch: `catalog-pretrain`.
Context and history: [generation-plan.md](generation-plan.md).

## Why

The last run (rough shape → pix2pix) got the pose and outline of each figure
right, but filled it with **random white dots** instead of the clean white lines
of real filet: 1 cell wide, continuous, following arms, folds and outlines. The
model never learned what that line work looks like. 346 human figures are too
few to learn it from, and the critic learned from the same 346, so it can't
teach it either.

The whole catalog is full of that line work, in vines, flowers, animals,
borders and figures. And our task ("here's a blocky outline, fill in the
detail") works on **any** piece of any pattern, with no labelling. So:

1. **Pretrain** on about 4600 pieces from the whole catalog: learn filet detail.
2. **Fine-tune** on the 346 human figures: learn what human detail looks like.

This is "pretraining on the whole catalog" from the original handoff, in its
simplest form: same task, same input and same networks, only more data.

## What changes

| File | Change |
|---|---|
| `generator/c_model/prepare_catalog.py` | **New.** Cuts 128×128 pieces from every pattern → `catalog.npz` + review sheets |
| `generator/c_model/train.py` | `--data` option (default `dataset.npz`); stroke input needs `dataset.npz`; saves the run's settings |
| `generator/c_model/train_gan.py` | `--data` option; `--start-from` also loads the critic when the checkpoint has one; saves the run's settings |
| `generator/c_model/colab_train.ipynb` | A `DATA` setting: which data file to fetch from Drive and pass as `--data` |
| `.gitignore` | `generator/c_model/catalog.npz` |
| `docs/generation-plan.md` | Tick Phase 2 items; record results |

Nothing else changes. `model.py`, `prepare.py`, `dataset.npz`, and the
existing behaviour without `--data` all stay the same.

## 1. `prepare_catalog.py` → `catalog.npz`

Run: `python generator/c_model/prepare_catalog.py` (about 5–15 minutes on CPU).

**Which patterns**
- Every pattern from `load.all_ids()`.
- Skip the copies listed in `catalog/duplicates.json`, so each design is used once.
- Skip grids under `load.MIN_SIDE` cells on a side.
- **Leak guard (important):** skip every pattern that holds a human *test*
  figure in `dataset.npz` (`ids[test]`, 21 patterns), *and* any duplicate of
  those patterns. Otherwise the model would see the test figures during
  pretraining and the comparison would be unfair. Print how many were skipped
  and `assert` that none slipped through.
  *As built:* `ids[test]` is 18 patterns, not 21: 3 of prepare.py's 21 held-out
  patterns only have figures taller than 128, so none of their figures are in
  `dataset.npz`. `duplicates.json` only lists pixel-identical copies, so
  `NEAR_COPIES` in `prepare_catalog.py` adds two more found by comparing every
  pattern with the test ones: 006_page_19 (same design as 006_page_18, ~9% of
  cells differ) and 008_page_70 (the same putti redrawn). 22 patterns are
  skipped in all. Both near-copies are *training* patterns in `dataset.npz`,
  so that test figure was already seen in Phase 1 and will be in fine-tuning:
  read 006_page_18's samples with that in mind.
- Hold out about 5% of the remaining patterns (seed 0) as the pretraining test
  set. They're only used for test loss and sample sheets during pretraining.

**Cutting pieces**
- `mask = load.motif_mask(grid)`. Use the raw mask, with **no** `prepare.clean()`:
  for pretraining we want all the line work, including small bits.
- If a side is under 128, pad the mask with ground to 128, centred.
- Cut 128×128 pieces every 64 cells. Add one last piece flush with the right
  and bottom edges, so the edges are covered too.
- Keep pieces with **3–85% motif**. Emptier ones teach nothing; fuller ones are
  solid black.
- Expect about 4600 pieces. *As built: 7147* (6736 train, 411 test). The 4600
  counted only the pieces on the 64-cell grid (4587); the extra pieces flush
  with the edges add about 2560. Pieces overlap, so neighbours share half their
  cells. That's fine, and both stay on the same side of the split because the
  split is by pattern.

**Saved file:** the same key names as `dataset.npz`, so the training code reads both:
- `motifs` (N, 128, 128) uint8
- `ids` (N,) pattern id
- `boxes` (N, 4): `[y, x, 128, 128]`, where the piece sits in the padded mask
- `test` (N,) bool

Use `np.savez_compressed`. Print the file size. If it's over about 100 MB,
switch to `np.packbits`; we expect roughly 10–40 MB. *As built: 10.2 MB.*
Pieces are stored in shuffled order (seed 0), so the first 12 test pieces on the
training sample sheets come from different patterns.

**Review sheets:** `out/catalog_*.png`, a fixed random sample of about 96
pieces, each with its rough shape tinted underneath (reuse `train.rough_shape`
and the colours/`tile` style of `prepare.py`). Look at them before training:
the pieces should show real line work, not mostly empty or solid squares.

**Printed summary:** patterns used, skipped (copies, too small, leak guard),
pieces kept and dropped (too empty / too full), train/test pieces, file size.

## 2. Training scripts

**`--data`** (both scripts): file name in `generator/c_model/` or a full path.
The default `dataset.npz` keeps today's behaviour exactly.
- `Figures` reads `points`/`n_points` only for `--input stroke`. With stroke
  input and a file without points, stop with a clear message ("stroke input
  needs dataset.npz").
- Shift augmentation works unchanged. Pieces often fill the canvas, so they
  just shift less.

**Run settings file** (both scripts, for bookkeeping): at start, write
`out/<log prefix>settings.json` (`train_settings.json` / `gan_settings.json`)
with every argument, the data file, the number of train/test items, the device
and the start time. Every result folder then says how it was made.

**`train_gan.py --start-from`:** if the checkpoint also has `"critic"`, load it
too and print that it did. When fine-tuning, the critic already knows what real
filet looks like. A U-Net `best.pt` from `train.py` has no critic, so only the
generator loads, as today. Keep the existing width and input checks.

Checkpoints and logs keep their names (`best.pt`, `gan_last.pt`, …). On Colab
every run has its own folder, so nothing is overwritten. Locally, copy
`out/` and `checkpoints/` aside between runs if you want to keep them.

## 3. Colab notebook

- New setting next to `SCRIPT`/`ARGS`: `DATA = "dataset.npz"` (or `"catalog.npz"`).
- The data cell fetches `MyDrive/arachne/<DATA>`, or asks to upload it once,
  and copies it next to the code.
- The run command adds `--data {DATA}` itself, so the setting can't disagree
  with `ARGS`.
- Name the run folder `<date>_<branch>_<script>_<data stem>`, for example
  `2026-10-28_1430_catalog-pretrain_train_gan_catalog`.

## 4. How to run (Colab, three runs)

Once: make `catalog.npz` on your machine and put it in `MyDrive/arachne/`.
Push the branch, and set `BRANCH = "catalog-pretrain"` in the notebook.

| Run | SCRIPT | DATA | ARGS |
|---|---|---|---|
| 1. Pretrain U-Net | `train.py` | `catalog.npz` | `--epochs 30 --every 2` |
| 2. Pretrain pix2pix | `train_gan.py` | `catalog.npz` | `--epochs 30 --every 2 --start-from <run 1>/checkpoints/best.pt` |
| 3. Fine-tune on humans | `train_gan.py` | `dataset.npz` | `--epochs 150 --start-from <run 2>/checkpoints/gan_last.pt` |

An epoch on the catalog is about 19 times longer than on the human figures
(6736 pieces against 346). So 30 catalog epochs are about as many training
steps as 580 human epochs; if the time estimate is too long, 20 catalog epochs
(about 390 human epochs) is the closest to what was planned. Start each run with `--quick` once, and
read the progress bar's time estimate before committing to a long run.

## 5. How to judge

- **Run 2 sheets (catalog test pieces):** does the model fill blocky shapes
  with continuous white lines like the real column? If not even here, more
  human data won't help: rethink first (diffusion is parked in the plan).
- **Run 3 sheets (the 38 human test figures)** against the Phase 1 rough-shape
  sheet (same figures, same layout): clean lines along limbs and drapery
  instead of dots? Faces and hair kept?
- **Risk to watch:** the catalog has lots of geometric lace and borders. If the
  human figures come out filled with lace patterns, fine-tune longer or with a
  lower pretraining share.
- **Copying:** outputs shouldn't match their real figure line for line. The
  copy check (`evaluate.py`) comes in Phase 3.

## 6. Checks before handing over (CPU is fine)

1. `prepare_catalog.py` runs; summary numbers are sensible; review sheets look right; leak guard assert passes.
2. `train.py --quick --data catalog.npz` and `train.py --quick` (default data) both run.
3. `train_gan.py --quick --data catalog.npz --start-from <that best.pt>` runs.
4. `train_gan.py --quick --start-from <that gan_last.pt>` runs and says the critic was loaded.
5. `train.py --quick --input stroke --data catalog.npz` stops with the clear message.
6. Settings files are written. The notebook is still valid JSON
   (`python -c "import json; json.load(open(...))"`).
7. `git status` shows no data, `out/` or `checkpoints/` files.

## 7. Commits (on `catalog-pretrain`, never pushed by Claude)

Small, one logical change each, written as the user's own: **no Co-Authored-By
or any Claude line.** Before each one: `git status`, stage files **by name**
(never `git add -A` / `.`), and read `git diff --cached`. Leave unrelated
files alone.

1. `Plan pretraining on the whole catalog`: `docs/generation-plan.md`, `docs/catalog-pretraining.md` (if not committed yet)
2. `Cut training pieces from the whole catalog`: `prepare_catalog.py`, `.gitignore`
3. `Train on any data file and resume pix2pix with its critic`: `train.py`, `train_gan.py`
4. `Choose the data file in the Colab notebook`: `colab_train.ipynb`
5. `Record catalog pretraining as built`: docs (status line here, Phase 2 ticks in the plan)

Then the user runs `git push -u origin catalog-pretrain`. Merge into `main`
with `git merge` only after the results are judged.
