"""Cut 128x128 pieces from every catalog pattern, for pretraining.  Run:  python generator/c_model/prepare_catalog.py

The task "blocky outline in, filet detail out" works on any piece of any pattern, so
the whole catalog can teach the model what filet line work looks like before it is
fine-tuned on the human figures in dataset.npz.

Which patterns:
  - every pattern once: copies listed in catalog/duplicates.json are skipped
  - grids under load.MIN_SIDE cells on a side are skipped
  - leak guard: patterns holding a human test figure in dataset.npz are skipped, with
    all their copies (duplicates.json, plus NEAR_COPIES found by hand), so pretraining
    never sees the test figures
  - about 5% of the rest (seed 0) are held out as the pretraining test set

Cutting: the raw motif mask (no cleaning, small line work is what we want), padded
with ground to 128 where a side is shorter, cut every 64 cells plus one piece flush
with the right and bottom edges. Pieces with 3-85% motif are kept.

Writes:
  generator/c_model/catalog.npz          motifs (N, SIZE, SIZE) uint8, ids, boxes [y, x, SIZE, SIZE]
                                         in the padded mask, test; same keys as dataset.npz (no points)
  generator/c_model/out/catalog_*.png    review sheets: a fixed random sample of pieces,
                                         rough shape tinted under the motif
"""
import json

import numpy as np
from PIL import Image, ImageDraw

from prepare import GREY, HERE, ROOT, SIZE
from load import BACKGROUND, INK, MIN_SIDE, all_ids, grid_hash, load_grid, motif_mask  # catalog/, put on the path by prepare
from train import TINT, rough_shape

STEP = 64                     # a new piece starts every STEP cells, so neighbours overlap by half
MIN_MOTIF, MAX_MOTIF = 0.03, 0.85  # emptier pieces teach nothing, fuller ones are solid black
TEST = 0.05                   # share of patterns held out for pretraining test loss and samples
SAMPLE = 96                   # pieces on the review sheets
# Copies of human test patterns that are not pixel-identical, so duplicates.json misses them.
# Found by comparing every pattern with the test ones (CLIP and cell by cell) and looking.
NEAR_COPIES = {
    "006_page_18": ["006_page_19",   # same design, about 9% of the cells differ
                    "008_page_70"],  # the same two putti redrawn at another size
}


def guarded(test_ids, duplicates):
    """The human test patterns plus every copy of them: these must stay out of catalog.npz."""
    out = set(test_ids)
    for pid in test_ids:
        out |= set(NEAR_COPIES.get(pid, []))
    for kept, copies in duplicates.items():
        group = {kept, *copies}
        if group & out:
            out |= group
    return out


def pieces(mask):
    """Yield (y, x, piece): SIZE x SIZE windows of the mask, padded with ground to at least SIZE."""
    h, w = max(mask.shape[0], SIZE), max(mask.shape[1], SIZE)
    padded = np.zeros((h, w), bool)
    y0, x0 = (h - mask.shape[0]) // 2, (w - mask.shape[1]) // 2
    padded[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] = mask

    def starts(side):
        s = list(range(0, side - SIZE + 1, STEP))
        return s if s[-1] == side - SIZE else s + [side - SIZE]  # last one flush with the edge

    for y in starts(h):
        for x in starts(w):
            yield y, x, padded[y:y + SIZE, x:x + SIZE]


def tile(motif, scale=2):
    """The piece in ink, with its rough shape (the model's input) tinted on the ground it covers."""
    rgb = np.full((SIZE, SIZE, 3), BACKGROUND, np.uint8)
    rgb[rough_shape(motif) & ~motif] = TINT
    rgb[motif] = INK
    return Image.fromarray(rgb).resize((SIZE * scale, SIZE * scale), Image.NEAREST)


def review_sheets(motifs, ids, test, out, cols=8, rows=6, scale=2):
    out.mkdir(exist_ok=True)
    for old in out.glob("catalog_*.png"):
        old.unlink()
    pick = np.sort(np.random.default_rng(0).choice(len(motifs), min(SAMPLE, len(motifs)), replace=False))
    side, caption = SIZE * scale, 14
    per_page = cols * rows
    for page in range(0, len(pick), per_page):
        sheet = Image.new("RGB", (cols * side, rows * (side + caption)), BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        for i, k in enumerate(pick[page:page + per_page]):
            x, y = (i % cols) * side, (i // cols) * (side + caption)
            sheet.paste(tile(motifs[k].astype(bool), scale), (x, y))
            draw.rectangle([x, y, x + side - 1, y + side - 1], outline=GREY)
            draw.text((x + 3, y + side), ids[k] + (" test" if test[k] else ""), fill=INK)
        sheet.save(out / f"catalog_{page // per_page:02d}.png")


def main():
    human = np.load(HERE / "dataset.npz")
    test_ids = set(human["ids"][human["test"]])
    duplicates = json.loads((ROOT / "catalog" / "duplicates.json").read_text())["duplicates"]
    copies = {c for cs in duplicates.values() for c in cs}
    keep_out = guarded(test_ids, duplicates)

    used, grids, skipped = [], {}, {"leak guard": 0, "copies": 0, "too small": 0}
    for pid in all_ids():
        if pid in keep_out:
            skipped["leak guard"] += 1
        elif pid in copies:
            skipped["copies"] += 1
        else:
            grid = load_grid(pid)
            if min(grid.shape) < MIN_SIDE:
                skipped["too small"] += 1
            else:
                used.append(pid)
                grids[pid] = grid
    rng = np.random.default_rng(0)
    test_patterns = set(rng.permutation(used)[:round(TEST * len(used))])

    motifs, ids, boxes, test, too_empty, too_full = [], [], [], [], 0, 0
    for pid in used:
        for y, x, piece in pieces(motif_mask(grids[pid])):
            share = piece.mean()
            if share < MIN_MOTIF:
                too_empty += 1
            elif share > MAX_MOTIF:
                too_full += 1
            else:
                motifs.append(piece)
                ids.append(pid)
                boxes.append([y, x, SIZE, SIZE])
                test.append(pid in test_patterns)

    # leak guard, checked twice: by id, and by grid content (catches copies duplicates.json might lack)
    assert not set(ids) & keep_out, f"human test patterns in catalog.npz: {set(ids) & keep_out}"
    test_hashes = {grid_hash(load_grid(pid)) for pid in test_ids}
    assert not any(grid_hash(grids[pid]) in test_hashes for pid in set(ids)), "a copy of a human test pattern slipped in"

    # stored in shuffled order: the training scripts draw the first 12 test items on their
    # sample sheets, and in pattern order those would mostly come from one big pattern
    order = rng.permutation(len(motifs))
    motifs, ids, boxes, test = (np.array(a)[order] for a in (motifs, ids, boxes, test))
    path = HERE / "catalog.npz"
    np.savez_compressed(path, motifs=motifs.astype(np.uint8), ids=ids, boxes=boxes, test=test)
    review_sheets(motifs, ids, test, HERE / "out")

    n_test = sum(test)
    print(f"{len(used)} patterns used; skipped {skipped['copies']} copies, {skipped['too small']} too small, "
          f"{skipped['leak guard']} by the leak guard ({len(test_ids)} human test patterns and their copies)")
    print(f"{len(motifs)} pieces kept; dropped {too_empty} under {MIN_MOTIF:.0%} motif, {too_full} over {MAX_MOTIF:.0%}")
    print(f"{len(motifs) - n_test} train, {n_test} test pieces (from {len(test_patterns)} held-out patterns); "
          f"motif per piece: median {np.median([m.mean() for m in motifs]):.0%}")
    print(f"leak guard: OK. {path} {path.stat().st_size / 1e6:.1f} MB")
    print(f"review sheets: {HERE / 'out'}/catalog_*.png")


if __name__ == "__main__":
    main()
