# Arachne Generator

Generating new filet crochet patterns from the real Arachne Studio catalog.

**Step 1 (this code): count how many patterns contain each motif**, so we can
choose between composing real motifs (method B) and training a model (method C).
Nothing is labelled automatically: the counts are your own ticks.

New to this? Start with **[docs/motif-counting-guide.md](docs/motif-counting-guide.md)**:
a step-by-step walkthrough with explanations of how and why it works.

## Setup (once)

```bash
cd ~/arachne-generator
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install open_clip_torch numpy pillow fastapi uvicorn scikit-image
```

`data/previews` is a link to the pattern files in pattern-app
(`apps/api/assets/previews`). Nothing is copied, and `data/` is not committed.

## 1. Embed the catalog (once, about 5–10 minutes on CPU)

```bash
.venv/bin/python catalog/embed.py
```

This draws every pattern the same way (dark motif on light background) and
turns each drawing into a CLIP "fingerprint" that can be compared with text
or with other patterns. Big or long patterns also get zoomed-in square crops,
so a small figure in a large scene can still be found.

- Grids smaller than 10 cells on a side are skipped.
- Identical copies (e.g. book 009 is a re-scan of book 003) are embedded once,
  so they are never counted twice. `catalog/duplicates.json` lists them.

Run it again only if the catalog changes.

## 2. Search and tick

```bash
.venv/bin/python catalog/serve.py
```

Open http://127.0.0.1:8000

1. **Tag**: the name to save, e.g. `putti`.
2. **Phrases**: one per line; they are averaged. For example:
   ```
   cherub angel putto
   a crochet filet pattern of a cherub
   winged baby
   ```
3. **Search** ranks all patterns, best first. Tick each one:
   - `y` / `n` mark the highlighted pattern and move on (or use the buttons,
     or click the picture to cycle yes → no → unmarked)
   - arrow keys move, `u` clears a mark, `o` opens the pattern large
   - `m` or **≈ like** re-ranks by that pattern's picture instead of the text.
     Use it when the text misses things.
   - After 20 "no" in a row a hint suggests you've probably found them all.
4. **Save** (or ctrl+s) writes your ticks.
5. **Hide already-reviewed** hides patterns you've already saved a mark for
   under this tag, so a second phrasing only shows new ones.

The pictures are the normalised drawings, not the original previews:
the motif is always dark.

### Where ticks go

- `tags.json`: `{"<pattern_id>": ["putti", "grapes"]}`. This is the source of
  truth and is committed. Saving only adds or removes the current tag; other
  tags on a pattern are kept. Marking "no" removes only that tag.
- `reviewed.json`: every yes/no per tag, so "hide already-reviewed" survives a
  restart.

## 3. Count

```bash
.venv/bin/python catalog/count.py
```

Prints how many patterns have each tag and groups the tags:
≥200 (worth trying method C), 50–199, <50 (use method B).

## Files

| File | What it is |
|---|---|
| `catalog/load.py` | Read grids, find the motif, draw a grid. Reused later by methods A/B/C. |
| `catalog/embed.py` | Builds `embeddings.npy`, `ids.json`, `duplicates.json` (not committed). |
| `catalog/serve.py` | Local server: search, thumbnails, saving ticks. |
| `catalog/search.html` | The search page. |
| `catalog/count.py` | Counts per tag. |

### How the motif is found (`load.py`)

In the catalog grids, `"1"` is the stitched ground and `"0"` is the motif.
`"0"` also fills the empty area outside round or triangular pieces.
`motif_mask()` removes that outside area:

1. Fill inward from the grid edge through `"0"` cells. Small gaps (up to 4
   cells) are closed first, so the fill can't leak into a motif that touches
   the piece outline.
2. Also count as outside any empty cell beyond the last stitch in both its row
   and its column. This catches thin slivers in the corners of round pieces.
