"""Turn reviewed motif boxes into (path, motif) training pairs.  Run:  python generator/c_model/prepare.py human

For each box in motif_boxes.json of the chosen kind that fits the canvas:
  1. cut the motif mask (load.motif_mask) inside the box
  2. clean it: keep the big connected pieces, drop small separate specks
     (a flower or border corner that the rectangle caught)
  3. path: the head-to-feet spine, standing in for the stroke a user draws
     ("one stroke from the head to the feet"): the skeleton route from the
     figure's topmost to its bottommost point, smoothed into a few straight
     segments so it looks hand-drawn
  4. centre both on a SIZE x SIZE canvas at their real scale (no resizing)

Flips, shifts and path jitter are not stored; train.py applies them on the fly
and redraws the path from its corner points with draw_path().
About 10% of patterns (all their boxes together) are held out for testing.

Writes:
  generator/c_model/dataset.npz   motifs, paths (N, SIZE, SIZE) uint8,
                                  points (N, MAX_POINTS, 2) path corners from the head, -1 padded,
                                  n_points, ids, boxes, test
  generator/c_model/out/prepare_*.png   review sheets: black = kept, grey = dropped,
                                        red = path, gold = head end
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.draw import line
from skimage.measure import approximate_polygon, label
from skimage.morphology import skeletonize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "catalog"))
from load import BACKGROUND, INK, load_grid, motif_mask  # noqa: E402

SIZE = 128        # canvas side in cells; boxes larger than this are skipped
KEEP = 0.15       # keep connected pieces with at least 15% of the largest piece's cells
TEST = 0.10       # share of patterns held out
DROP_WARN = 0.35  # list boxes where cleaning dropped more than this share of cells
SMOOTH = 2.0      # the smoothed path may stray this many cells from the skeleton
MAX_POINTS = 32   # at most this many corner points per path
GREY, RED, GOLD = (0xC9, 0xBD, 0xB0), (0xA0, 0x10, 0x20), (0xD4, 0xAF, 0x37)


def clean(mask):
    """Keep the large 8-connected pieces of the mask."""
    labels = label(mask, connectivity=2)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if sizes.max() == 0:
        return mask
    return (sizes >= KEEP * sizes.max())[labels]


def spine(mask):
    """Head-to-feet line as corner points [(y, x), ...], starting at the head.

    Follows the skeleton of the mask's largest piece from its topmost cell
    (the head) to its bottommost cell (the feet) by breadth-first search, then
    smooths the route into a few straight segments. Figures lying down or
    flying sideways get a sideways line: topmost is only a cheap guess.
    """
    skel = skeletonize(mask)
    labels = label(skel, connectivity=2)
    if labels.max() == 0:
        return np.zeros((0, 2), int)
    biggest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
    ys, xs = np.nonzero(labels == biggest)
    cells = set(zip(ys.tolist(), xs.tolist()))
    mid = np.median(xs)  # of several top (or bottom) cells, take the most central
    head = min(cells, key=lambda c: (c[0], abs(c[1] - mid)))
    feet = max(cells, key=lambda c: (c[0], -abs(c[1] - mid)))

    came_from, queue = {head: None}, [head]
    for y, x in queue:  # the list grows while we walk it
        if (y, x) == feet:
            break
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y + dy, x + dx)
                if q in cells and q not in came_from:
                    came_from[q] = (y, x)
                    queue.append(q)
    route, p = [], feet
    while p is not None:
        route.append(p)
        p = came_from[p]
    route = np.array(route[::-1])

    tolerance, points = SMOOTH, approximate_polygon(route, SMOOTH)
    while len(points) > MAX_POINTS:
        tolerance *= 1.5
        points = approximate_polygon(route, tolerance)
    return np.round(points).astype(int)


def draw_path(points, size=SIZE):
    """Rasterise corner points into a 1-cell-wide line on a size x size canvas."""
    path = np.zeros((size, size), dtype=bool)
    points = np.clip(points, 0, size - 1)
    for (y0, x0), (y1, x1) in zip(points[:-1], points[1:]):
        rr, cc = line(y0, x0, y1, x1)
        path[rr, cc] = True
    if len(points) == 1:
        path[tuple(points[0])] = True
    return path


def offset(shape):
    """Where a mask of this shape starts when centred on the canvas."""
    return np.array([(SIZE - shape[0]) // 2, (SIZE - shape[1]) // 2])


def centre(mask):
    canvas = np.zeros((SIZE, SIZE), dtype=bool)
    (y, x), (h, w) = offset(mask.shape), mask.shape
    canvas[y:y + h, x:x + w] = mask
    return canvas


def tile(raw, kept, points, scale=2):
    rgb = np.full((SIZE, SIZE, 3), BACKGROUND, np.uint8)
    rgb[raw & ~kept] = GREY
    rgb[kept] = INK
    rgb[draw_path(points)] = RED
    if len(points):
        y, x = points[0]
        rgb[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3] = GOLD
    return Image.fromarray(rgb).resize((SIZE * scale, SIZE * scale), Image.NEAREST)


def review_sheets(examples, out, cols=8, rows=6, scale=2):
    out.mkdir(exist_ok=True)
    for old in out.glob("prepare_*.png"):
        old.unlink()
    side, caption = SIZE * scale, 14
    per_page = cols * rows
    for page in range(0, len(examples), per_page):
        chunk = examples[page:page + per_page]
        sheet = Image.new("RGB", (cols * side, rows * (side + caption)), BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        for i, (pid, raw, kept, points) in enumerate(chunk):
            x, y = (i % cols) * side, (i // cols) * (side + caption)
            sheet.paste(tile(centre(raw), centre(kept), points, scale), (x, y))
            draw.rectangle([x, y, x + side - 1, y + side - 1], outline=GREY)
            draw.text((x + 3, y + side), pid, fill=INK)
        sheet.save(out / f"prepare_{page // per_page:02d}.png")


def main():
    args = argparse.ArgumentParser()
    args.add_argument("tag")
    args.add_argument("--kind", default="full", help="which box kind to use (full, half, head, other)")
    args = args.parse_args()

    boxes = json.loads((ROOT / "motif_boxes.json").read_text()).get(args.tag, {})
    patterns = sorted(pid for pid, bs in boxes.items() if any(b["kind"] == args.kind for b in bs))
    rng = np.random.default_rng(0)
    test_ids = set(rng.permutation(patterns)[:round(TEST * len(patterns))])

    motifs, paths, points, ids, kept_boxes, test, examples, warn, too_big = [], [], [], [], [], [], [], [], 0
    for pid in patterns:
        mask = motif_mask(load_grid(pid))
        for b in boxes[pid]:
            if b["kind"] != args.kind:
                continue
            y, x, h, w = b["box"]
            if h > SIZE or w > SIZE:
                too_big += 1
                continue
            raw = mask[y:y + h, x:x + w]
            kept = clean(raw)
            corners = spine(kept) + offset(kept.shape)  # in canvas coordinates
            dropped = 1 - kept.sum() / max(raw.sum(), 1)
            if dropped > DROP_WARN:
                warn.append(f"{pid} {b['box']}: dropped {dropped:.0%}")
            motifs.append(centre(kept))
            paths.append(draw_path(corners))
            padded = np.full((MAX_POINTS, 2), -1)
            padded[:len(corners)] = corners
            points.append(padded)
            ids.append(pid)
            kept_boxes.append(b["box"])
            test.append(pid in test_ids)
            examples.append((pid, raw, kept, corners))

    if not motifs:
        print(f"No '{args.kind}' boxes up to {SIZE} cells for {args.tag}.")
        return
    np.savez_compressed(
        HERE / "dataset.npz",
        motifs=np.array(motifs, np.uint8), paths=np.array(paths, np.uint8),
        points=np.array(points), n_points=(np.array(points)[:, :, 0] >= 0).sum(1),
        ids=np.array(ids), boxes=np.array(kept_boxes), test=np.array(test),
    )
    review_sheets(examples, HERE / "out")

    n_test = sum(test)
    print(f"{len(motifs)} pairs from {len(patterns)} patterns "
          f"({len(motifs) - n_test} train, {n_test} test from {len(test_ids)} held-out patterns); "
          f"{too_big} boxes skipped as larger than {SIZE}")
    print(f"motif cells per example: median {int(np.median([m.sum() for m in motifs]))}, "
          f"path cells: median {int(np.median([p.sum() for p in paths]))}")
    print(f"review sheets: {HERE / 'out'}/prepare_*.png")
    if warn:
        print(f"\ncleaning dropped a lot in {len(warn)} boxes, check these:")
        print("  " + "\n  ".join(warn))


if __name__ == "__main__":
    main()
