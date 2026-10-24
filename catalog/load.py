"""Read Arachne pattern grids and render them in one consistent style.

Grids in data/previews/<id>_grid.json store "0"/"1" rows. Across the catalog
"1" is the stitched ground and "0" is the motif, plus "0" for the empty area
outside round or triangular pieces. motif_mask() separates the two: "0" cells
connected to the grid edge are outside, the rest are motif.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(__file__).resolve().parent.parent / "data" / "previews"
MIN_SIDE = 10  # grids smaller than this on either side are skipped

BACKGROUND = (0xF5, 0xF0, 0xEB)
INK = (0x0A, 0x0A, 0x0A)


def all_ids():
    return sorted(p.name[: -len("_grid.json")] for p in DATA.glob("*_grid.json"))


def load_grid(pattern_id):
    """Raw grid as a bool array (True where the file says "1")."""
    d = json.loads((DATA / f"{pattern_id}_grid.json").read_text())
    return np.array([[c != "0" for c in row] for row in d["rows"]], dtype=bool)


def _grow(mask, allowed, steps=None):
    """Grow mask 4-connectedly inside allowed, for steps (or until it stops)."""
    n = 0
    while steps is None or n < steps:
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        grown &= allowed
        if (grown == mask).all():
            break
        mask, n = grown, n + 1
    return mask


def _close(grid, r):
    """Morphological closing with a (2r+1) square; the grid border counts as stitched."""
    g = np.pad(grid, r, constant_values=True)
    for op in (np.logical_or, np.logical_and):  # dilate, then erode
        p = np.pad(g, r, constant_values=True)
        out = g.copy()
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                out = op(out, p[dy:dy + g.shape[0], dx:dx + g.shape[1]])
        g = out
    return g[r:-r, r:-r]


def _beyond_spans(grid):
    """Cells before the first or after the last stitch in BOTH their row and column.

    Catches thin empty slivers in the corners of round pieces, which the
    gap-closing below would otherwise seal off.
    """
    def per_row(g):
        before = np.cumsum(g, axis=1) == 0
        after = np.cumsum(g[:, ::-1], axis=1)[:, ::-1] == 0
        return before | after
    return per_row(grid) & per_row(grid.T).T


def outside_mask(grid, gap=2):
    """The empty area around a shaped piece (round, triangle, ...).

    Flood-fills "0" cells from the grid edge. Motif cells that touch the piece
    outline would let the flood leak inside, so the flood runs on a copy where
    gaps up to 2*gap cells wide are closed. Corner slivers that closing seals
    are added back by _beyond_spans, then the result grows up to gap cells to
    restore the staircase outline.
    """
    empty = ~grid
    sealed = ~_close(grid, gap)
    seed = np.zeros_like(empty)
    seed[[0, -1], :] = sealed[[0, -1], :]
    seed[:, [0, -1]] |= sealed[:, [0, -1]]
    out = _grow(seed, sealed) | _beyond_spans(grid)
    return _grow(out & empty, empty, steps=gap)


def motif_mask(grid):
    """True = motif ("0" inside the piece), False = ground or outside."""
    return ~grid & ~outside_mask(grid)


def grid_hash(grid):
    """Identical grids get identical hashes (used to drop duplicate scans)."""
    return hashlib.md5(np.packbits(grid).tobytes() + str(grid.shape).encode()).hexdigest()


def render(mask, size=224, ink=INK, background=BACKGROUND):
    """Draw a motif mask as ink on background, fitted into a size x size square.

    Nearest-neighbour scaling keeps stitches crisp; aspect ratio is kept and
    the rest of the square is padded with background.
    """
    h, w = mask.shape
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    rgb = np.where(mask[..., None], np.array(ink, np.uint8), np.array(background, np.uint8))
    img = Image.fromarray(rgb.astype(np.uint8)).resize((nw, nh), Image.NEAREST)
    canvas = Image.new("RGB", (size, size), background)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas
