# Generation plan: rough shape → filet crochet figure

**This is the current plan.** Older docs (`c-model-report.md`,
`handoff-neural-network.md`) record how we got here; this file says where we are
and what comes next.

**Goal.** On an empty grid, the user paints a rough, blocky silhouette and gets
back a filet crochet human figure that looks like it belongs in the Arachne
catalog: not a copy of one pattern and not a blob of noise, but a coherent new
figure faithful to the collection's look. Vines, objects and decoration touching
the figure are part of the style.

**Output format:** `{w, h, rows, palette}`, motif = `"0"`, ground = `"1"`.

---

## Where we are (2026-10-27)

### Data

| Asset | Count | Notes |
|---|---|---|
| Catalog patterns | 2283 unique | All of them, unlabelled |
| Tagged patterns | 442 | human 286, putti 188, animal 106, woman 16 |
| Reviewed bounding boxes | 597 | `motif_boxes.json`, 285 patterns |
| Full-figure training pairs (128×128) | **384** | `dataset.npz`: 346 train, 38 test (21 held-out patterns) |
| ↳ from putti-tagged pages | 247 | about 2/3 of the set |
| ↳ adults | 133 | |
| ↳ riders (animal-tagged pages) | 4 | |
| 128×128 pieces in the whole catalog | **7147** | `catalog.npz`: 6736 train, 411 test, 3–85% motif, from 2263 patterns |

### Code (`generator/c_model/`)

| File | What it does |
|---|---|
| `boxes.py` + `boxes.html` | Review page: box each figure, label its kind |
| `prepare.py` | Boxes → cleaned, centred figures + head-to-feet strokes → `dataset.npz` |
| `prepare_catalog.py` | Every catalog pattern → 128×128 pieces → `catalog.npz` (pretraining), with leak guard |
| `model.py` | `UNet` (generator) and `PatchDiscriminator` (critic) |
| `train.py` | Plain U-Net training. `--input shape` (default) or `stroke`; `--data` picks the data file |
| `train_gan.py` | pix2pix training (U-Net + critic), `--start-from` a checkpoint (critic too, if it has one) |
| `colab_train.ipynb` | Runs any training script on a Colab GPU, results to Drive |

`train_free.py` (a GAN with no input at all) stays on the `gan-experiments`
branch only. It isn't on the path to the drawing tool.

### What we learned, run by run

| Run | Input | Result |
|---|---|---|
| U-Net | stroke | Blurry silhouettes along the stroke, no inner detail. One stroke fits many figures, so it paints their average. |
| pix2pix | stroke | Sharp, filet-looking texture, no copying. But **every figure is the same robed person**: no arms, objects or vines. A stroke says too little about the figure. |
| U-Net → pix2pix, 450 epochs | **rough shape** | **Pose and outline now right** (raised arms, putti, dancers), heads with hair. But the **inside is random white dots**, not the clean 1-cell white lines (folds, outlines of limbs) of real filet. Test L1 flat from about epoch 80; critic loss about 0.55, so it barely tells real from fake. |

**What the last run tells us.** The model has not learned what filet line work
looks like. Its 346 human figures are too few to learn it from, and the critic
learned from the same 346, so it can't teach it either. Changing the loss (for
example comparing blurred versions) wouldn't add that missing knowledge. More
examples of line work would.

---

## Next step: Phase 2, pretraining on the whole catalog

**Full spec: [catalog-pretraining.md](catalog-pretraining.md).** Built on the
branch `catalog-pretrain`.

The task the model learns ("here is a blocky outline, fill in the filet
detail") doesn't need human boxes. It works on any piece of any pattern, and
every pattern in the catalog uses the same white-line style: vines, flowers,
animals, borders, figures.

1. Cut 7147 128×128 pieces from all catalog patterns (`catalog.npz`).
2. Pretrain the same U-Net → pix2pix on them: rough shape in, real detail out.
   This is where it should learn lines, folds and curls.
3. Fine-tune on the 346 human figures, so it learns what *human* detail looks
   like on top.
4. Compare with the last run on the same 38 test figures.

Patterns that hold human test figures are kept out of pretraining, so the test
stays fair, along with their copies (including two near-copies of 006_page_18
that `duplicates.json` misses). Those two near-copies are already training
patterns in `dataset.npz`, so 006_page_18's test figures were seen in training
in Phase 1 too: judge them with care.

---

## Roadmap

### Phase 1: first models (done)
- [x] U-Net, stroke input
- [x] pix2pix, stroke input
- [x] Rough-shape input, U-Net → pix2pix

### Phase 2: pretrain on the whole catalog (next, branch `catalog-pretrain`)
- [x] `prepare_catalog.py` → `catalog.npz`, with review sheets
- [x] `--data` option in `train.py` and `train_gan.py`; critic also loaded by `--start-from`
- [x] Notebook: choose which data file to use
- [ ] Pretrain U-Net, then pix2pix, on the catalog (Colab)
- [ ] Fine-tune pix2pix on the human figures
- [ ] Judge against the Phase 1 rough-shape run. Is there real line work now?

### Phase 3: polish and the drawing tool (once the figures look right)
- [ ] Putti / adult label as an extra input, so you can choose which you get
- [ ] `sample.py`: painted shape in → `{w, h, rows, palette}` out, despeckled
- [ ] `evaluate.py`: nearest training figure next to each output (copy check)
- [ ] Drawing page: paint blocks on a grid, get a figure back

### Phase 4: more data (any time, independent)
- [ ] Tag more putti patterns (188 → 200+)
- [ ] Use the 65 figures taller than 128 cells (bigger canvas)
- [ ] Add "other human" boxes (groups, half figures)

### Parked
- **Diffusion model:** can give different figures for the same shape. More
  code and training time; only if pix2pix hits a wall after pretraining.
- **Blurred-L1 loss:** dropped. It changes how mistakes are scored, but doesn't
  give the model the missing knowledge of line work.

---

## Repo and bookkeeping

- **`main`**: working code only.
- **One short branch per step**, merged into `main` with `git merge` (never
  rebase) when the step is done and judged. Now: `catalog-pretrain`.
- **Commits:** small, one logical change each, written as the user's own, with
  no co-author or Claude sign-off line.
- **Not in git** (`.gitignore`): `data/`, `dataset.npz`, `catalog.npz`, `out/`,
  `checkpoints/`. Data files go to Colab through Google Drive
  (`MyDrive/arachne/`).
- **Results:** every Colab run gets its own folder,
  `MyDrive/arachne/runs/<date>_<branch>_<script>/`, so runs never overwrite each other.

---

## Decisions already made

| Decision | Reason |
|---|---|
| 128×128 canvas, no resizing | Filet figures at different sizes are drawn differently; resizing corrupts the stitches |
| Full figures only (for now) | Biggest consistent group (449 boxes, 384 fit the canvas) |
| Rough blocky shape as input (blocks of 6, on at ≥35% motif; random 5–8 and 20–50% in training) | A line says too little; a shape gives pose and outline, and the model supplies the detail |
| Connected-component cleanup, keep pieces ≥15% of the largest | Removes loose border bits; touching context stays, which suits the style |
| Test split by pattern, not by figure | Figures from the same page look alike; splitting by figure would leak |
| Left-right flip + shift | Safe for upright figures |
| BCE + Dice (U-Net); adversarial + L1 ×100 (pix2pix) | Dice avoids "all ground"; L1 keeps the figure on the shape |
| Despeckle outputs (pieces and holes under 3 cells) | Single cells don't crochet well |
| Judge GANs by the sample sheets, not the numbers | GAN numbers bounce and don't measure looks |

---

## Open questions

1. After Phase 2: is the line work good enough, or do we need diffusion?
2. Drawing page: block size the user paints with (6 cells?), and can they
   paint the head separately?
3. Animals: their own model, or a label in the same one?
