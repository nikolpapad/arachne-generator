"""Compute CLIP image embeddings for every catalog pattern (run once).

Each pattern is rendered from its grid with load.render(motif_mask(...)), so
all patterns look alike (dark motif on light background) whatever the scan.
Large or long patterns also get overlapping square crops, so a small figure
inside a big scene still gets a view where it fills the frame.

Writes, next to this file:
  embeddings.npy   one L2-normalised row per view (float16)
  ids.json         pattern id for each row; a pattern's first row is its whole view
  duplicates.json  {"kept_id": [identical copies]} and {"skipped": [too small]}
"""
import json
import time
from pathlib import Path

import numpy as np
import open_clip
import torch

from load import MIN_SIDE, all_ids, grid_hash, load_grid, motif_mask, render

HERE = Path(__file__).resolve().parent
MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
CROP = 128  # max crop side, in cells
BATCH = 64


def crop_boxes(h, w):
    """Overlapping square windows (y, x, side) covering the grid; [] if one view is enough."""
    side = min(h, w, CROP)
    if max(h, w) <= 1.5 * side:
        return []
    step = side // 2
    ys = list(range(0, h - side + 1, step)) or [0]
    xs = list(range(0, w - side + 1, step)) or [0]
    if ys[-1] != h - side:
        ys.append(h - side)
    if xs[-1] != w - side:
        xs.append(w - side)
    return [(y, x, side) for y in ys for x in xs]


def unique_patterns():
    """Pattern ids to embed, plus the duplicates/skipped bookkeeping."""
    kept, by_hash, skipped = [], {}, []
    for pid in all_ids():
        grid = load_grid(pid)
        if min(grid.shape) < MIN_SIDE:
            skipped.append(pid)
            continue
        h = grid_hash(grid)
        if h in by_hash:
            by_hash[h].append(pid)
        else:
            by_hash[h] = [pid]
            kept.append(pid)
    duplicates = {ids[0]: ids[1:] for ids in by_hash.values() if len(ids) > 1}
    return kept, {"duplicates": duplicates, "skipped": skipped}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=device)
    model.eval()

    kept, bookkeeping = unique_patterns()
    print(f"{len(kept)} patterns to embed on {device} "
          f"({sum(map(len, bookkeeping['duplicates'].values()))} duplicates, {len(bookkeeping['skipped'])} too small)")

    def views():
        for pid in kept:
            mask = motif_mask(load_grid(pid))
            yield pid, render(mask)
            for y, x, s in crop_boxes(*mask.shape):
                yield pid, render(mask[y:y + s, x:x + s])

    ids, rows, batch, start = [], [], [], time.time()

    def flush():
        with torch.no_grad():
            feats = model.encode_image(torch.stack([preprocess(im) for _, im in batch]).to(device))
        rows.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy().astype(np.float16))
        ids.extend(pid for pid, _ in batch)
        batch.clear()

    done = set()
    for pid, image in views():
        batch.append((pid, image))
        if len(batch) == BATCH:
            flush()
            done.update(ids[-BATCH:])
            print(f"\r{len(done)}/{len(kept)} patterns, {len(ids)} views, {time.time() - start:.0f}s", end="", flush=True)
    if batch:
        flush()
    print(f"\r{len(kept)}/{len(kept)} patterns, {len(ids)} views, {time.time() - start:.0f}s")

    np.save(HERE / "embeddings.npy", np.concatenate(rows))
    (HERE / "ids.json").write_text(json.dumps(ids))
    (HERE / "duplicates.json").write_text(json.dumps(bookkeeping, indent=1))


if __name__ == "__main__":
    main()
