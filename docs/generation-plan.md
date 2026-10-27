# Generation plan: stroke → filet crochet figure

**Goal.** Draw a stroke on an empty grid, get back a filet crochet human figure
that looks like it belongs in the Arachne catalog — not a copy of one pattern,
not a blob of noise, but a coherent new figure faithful to the collection's
aesthetic. Vines, objects and decorative context around figures are part of the
style.

**Output format:** `{w, h, rows, palette}`, motif = `"0"`, ground = `"1"`.

---

## What we have

| Asset | Count | Notes |
|---|---|---|
| Catalog patterns | 2283 | All of them, unlabelled |
| Tagged patterns | 442 | human 286, putti 188, animal 106, woman 16 |
| Reviewed bounding boxes | 597 | In `motif_boxes.json`, 285 patterns |
| Full-figure training pairs (128×128) | **384** | In `dataset.npz` |
| ↳ from putti-tagged pages | 247 | ~64% of the set |
| ↳ from animal-tagged pages | 4 | Riders on horses |
| ↳ adults only | 133 | Adults without putti or animal tags |
| Patterns with usable boxes | 184 | Some give 2–6 figures each |

### What we built

| File | Purpose | Status |
|---|---|---|
| `catalog/` (embed, serve, search, count, load) | Tag patterns, find motifs | Done, on main |
| `boxes.py` + `boxes.html` | Review page: box each figure, label its kind | Done, on main |
| `prepare.py` | Cut, clean, spine, build dataset.npz | Done, on main |
| `model.py` | U-Net, PatchDiscriminator, Generator, Critic | Done (U-Net on main, GAN parts on gan-experiments) |
| `train.py` | U-Net training loop | Done, on main |
| `train_gan.py` | pix2pix training loop | Done, on gan-experiments |
| `train_free.py` | Unconditional GAN (DiffAugment) | Done, on gan-experiments |
| `colab_train.ipynb` | Colab notebook for any training script | Done, on gan-experiments |

### What we learned

1. **The U-Net works but blurs.** It learns "a figure goes along the stroke"
   but outputs a soft silhouette — no internal line work, no stitched detail.
   This is normal: one stroke fits many figures, so the U-Net paints their
   average.
2. **The data is mostly putti.** 247 of 384 figures come from putti pages.
   That's not bad — putti are the most consistent group (similar size, poses,
   always with vines). Adults are more varied (armour, dresses, riders).
3. **384 figures is small but workable** for pix2pix with proper
   regularisation. Free GANs (no conditioning) struggle at this scale.
4. **Clutter inside boxes** (vines, neighbour hands) is not hurting much.
   The connected-component cleanup removes separate pieces. Touching context
   stays in, but for putti that *is* the style.

---

## The three options, ranked

### Option 1: pix2pix (recommended starting point)

**What it is.** The U-Net becomes a *generator*. A second network (the
PatchDiscriminator) sees real and generated figures side by side with their
strokes, and learns to tell them apart. The generator is trained to fool it.
The critic forces the generator to commit to one sharp, believable figure
instead of averaging.

**Why it's the best first step:**
- Directly fixes the blur problem we saw.
- The code already exists (`train_gan.py`, `PatchDiscriminator` in `model.py`).
- The L1 term keeps the figure on the stroke, so it's stable even with few
  data.
- Standard technique, well understood, trains in minutes on a T4.

**Limitations:**
- Still one answer per stroke (deterministic once trained). Two identical
  strokes give the same figure.
- With only 384 figures the critic can memorise them, which makes training
  unstable after many epochs. Spectral norm and the L1 weight help.
- The output will look like the training data: mostly putti, some adults.

**Training plan:**
1. Start from the best U-Net checkpoint (`--start-from best.pt`), so the
   generator already knows the rough shape.
2. 300 epochs, `--l1 100` (high L1 keeps it grounded), lr 2e-4.
3. Judge by the sample sheets, not the loss curves — GAN losses bounce.

### Option 2: class-conditional pix2pix (recommended extension)

**What it is.** Same as option 1, but the generator and critic also receive a
**category label** (putti / adult / animal) as an extra input channel.

**Why it helps:**
- Everything about *how filet is drawn* (stitch texture, white line work, how
  edges step) is learned from all 384 figures together.
- Only *what* to draw comes from the label. At use time you choose: "draw a
  putto along this stroke" or "draw an adult."
- The putti figures teach the model filet style; the adults teach it adult
  poses. Both benefit.
- This is the standard answer to "few examples per category."

**What it takes:**
- Add a category label to `prepare.py` (from `tags.json`: if putti-tagged →
  putti, else → adult; riders are too few to separate).
- One extra input channel in the U-Net and critic (a constant plane: 0 for
  putti, 1 for adult).
- About 30 lines of code change.

### Option 3: inpainting pretraining + fine-tuning (recommended if options 1–2 plateau)

**What it is.** Before training on figures, teach the network "what filet
looks like" using **all 2283 patterns** with no labels:
1. Take a random pattern, hide a random rectangle, ask the network to fill
   it in.
2. The network learns stitch texture, line work, symmetry — everything that
   makes filet look like filet — from 6× more data.
3. Then fine-tune on the 384 figure pairs, where it only needs to learn
   *what shape to draw*, not *how filet is drawn*.

**Why it's powerful but second:**
- It's the biggest data leverage we have (2283 vs 384).
- But it's more code (a separate pretraining dataset, a masked input, a two-
  phase training loop) and we should first see if pix2pix alone is enough.
- If pix2pix gives sharp figures but the texture still looks off, this is
  the fix.

### What about diffusion?

A diffusion model can give *different* figures for the same stroke (it samples
from possibilities instead of picking the average). That's the ideal end state.
But it needs more code, more training time, and more data to converge. **Park
it.** If pix2pix works well, the figures are good enough — variety comes from
drawing different strokes. If we hit a wall, diffusion is the escape hatch.

---

## Roadmap

### Phase 1: pix2pix baseline (done)

- [x] Merge `gan-experiments` into `main` (pix2pix + notebook; `train_free.py` left on the branch)
- [x] Train pix2pix on Colab, starting from the U-Net checkpoint
- [x] Judge sample sheets. **Result (2026-10-27):** real filet texture (white line
  work, heads, hats) that follows the stroke, and no copying. But every figure is
  the same robed person: no arms, objects or vines, and lots of single-cell speckle.
  Test L1 stopped improving around epoch 120.

**Why:** a head-to-feet line says where the figure stands, but nothing about its
width, arms or what it holds. The model can't know, so it draws the average.

### Phase 1b: rough-shape input (next)

The user paints a rough, blocky silhouette instead of a line. The model's job
becomes "fill this outline with filet detail" (face, folds, line work, vines),
which is what pix2pix is best at.

- `--input shape` (now the default in `train.py` and `train_gan.py`): each figure
  seen through blocks of 6 cells, a block on where ≥35% of it is motif. In
  training the block size (5–8) and the threshold (20–50%) vary, so coarse or
  sloppy painting still works. Made on the fly from the real figure, so no new
  labelling.
- `--input stroke` keeps the old behaviour for comparison.
- The sample sheets' third column is now despeckled: motif pieces and ground holes
  under 3 cells are removed.

- [ ] Train the U-Net with shape input (`train.py`, 150 epochs)
- [ ] Train pix2pix from it (`train_gan.py --start-from …/best.pt`, 150 epochs)
- [ ] Judge: does the detail look like the catalog and vary between figures?
  Watch for copying: the shape gives the outline, the model must invent the inside.

### Phase 2: class-conditional + polish

- [ ] Add putti/adult label to `prepare.py` and the model
- [ ] Train class-conditional pix2pix
- [ ] Build `sample.py`: stroke in → `{w, h, rows, palette}` out
- [ ] Build `evaluate.py`: nearest-neighbour copy check
- [ ] Integrate into the interactive grid page

### Phase 3: inpainting pretraining (if needed)

- [ ] Build inpainting dataset from all 2283 patterns
- [ ] Pretrain the U-Net encoder on masked fill-in
- [ ] Fine-tune on figure pairs
- [ ] Compare with non-pretrained version

### Phase 4: more data (ongoing, independent)

- [ ] Tag more putti patterns (188 → 200+ for its own category)
- [ ] Use the 65 tall figures with a 192×192 canvas
- [ ] Add "other human" boxes (groups, half figures)
- [ ] Tag and box animals if the model generalises well

---

## Repo plan

### Branches

The `gan-experiments` branch served its purpose as an exploration space. The
useful code (pix2pix, colab notebook) should come to `main`. Going forward:

- **`main`**: all working code. Train scripts, model, data prep.
- **Feature branches**: short-lived, for one change at a time, merged to main
  when done.

No more long-lived experiment branches — they get confusing.

### Files that stay

```
generator/c_model/
  boxes.py + boxes.html     bounding box review tool
  prepare.py                dataset builder
  model.py                  all network architectures
  train.py                  U-Net training (baseline)
  train_gan.py              pix2pix training
  colab_train.ipynb         Colab notebook
  sample.py                 (to build) stroke → grid output
  evaluate.py               (to build) copy-paste check
```

### Files to drop

- `train_free.py` — the unconditional GAN was an experiment. It's not on the
  path to the interactive tool. Keep it on the `gan-experiments` branch for
  reference but don't merge to main.

### Data files (gitignored)

- `generator/c_model/dataset.npz` — the training pairs
- `generator/c_model/out/` — sample sheets, logs
- `generator/c_model/checkpoints/` — model weights

---

## Decisions already made

These are settled — no need to revisit unless something breaks.

| Decision | Reason |
|---|---|
| 128×128 canvas, no resizing | Filet figures at different sizes are drawn differently; resizing corrupts the stitch pattern |
| Full figures only (for now) | Biggest consistent group (449 boxes, 384 fit the canvas) |
| Head-to-feet spine as stroke | Sets a clear rule for the interactive tool: "draw one stroke, head to feet" |
| Connected-component cleanup, keep ≥15% of largest | Removes loose border pieces; touching context (vines, swords) stays in, which is fine for the style |
| 10% held-out patterns (not figures) | Figures from the same page look alike; pattern-level split prevents data leaking |
| Left-right flip + shift + path jitter | Safe augmentations for upright figures |
| BCE + Dice loss | Dice prevents the "all ground" trap on sparse motifs |
| No Claude sign-off in commits | Per user preference |

---

## Open questions

1. **After seeing pix2pix results:** sharp enough, or do we need pretraining?
2. **Interactive page design:** how should the drawing tool work? (Depends on
   `sample.py` being built first.)
3. **Animals:** worth pursuing separately, or only if humans work well?
