# Motif counting: a guide

This guide walks you through counting how many patterns contain a motif, and
explains what each step does and why, so you can reuse the approach later on
your own.

---

## 1. Why we're doing this

The goal is a generator that draws a new filet pattern along a path you drag
on a grid, learned from the real Arachne catalog. There are two ways to get
whole motifs (a putto, a grape cluster) into those patterns:

| Method | How it works | What it needs |
|---|---|---|
| **B: motif composition** | Cut real motifs out of your patterns and place, scale or mirror them along the path | A handful of good examples (5–20) |
| **C: trained model** | Teach a neural network to *draw* the motif along a path | Many examples of that motif (~200+) |

The catalog has no motif labels, so you don't know how many examples each
motif has. **This step counts them**, so you can choose B or C per motif
instead of guessing.

You only need **one motif** to start. Pick one that is:
1. **common**: you've seen it many times while browsing
2. **consistent**: looks similar across patterns (putti, grape clusters)
   rather than something broad like "woman"
3. **what you actually want** the generator to draw

---

## 2. What has already been done

| Step | Status | Result |
|---|---|---|
| Profile the catalog | done | 2506 patterns → **2283 unique** (209 duplicate copies, 14 grids under 10 cells skipped) |
| Work out which cells are the motif | done | `catalog/load.py`, checked on 60+ patterns, including the large ones |
| Embed every pattern | done | `catalog/embeddings.npy`: 12,246 views, about 17 minutes on CPU |
| Search page, saving, counting | done | `serve.py`, `search.html`, `count.py` |
| **Ticking your motif** | **your turn** | |
| Decide B or C | after ticking | |

---

## 3. Step by step

### Step 1: Start the search page

Open a terminal (in VS Code: *Terminal → New Terminal*) and run:

```bash
cd ~/arachne-generator
.venv/bin/python catalog/serve.py
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. Loading the model takes
about 10–20 seconds. Then open **http://127.0.0.1:8000** in your browser.

Leave the terminal open while you work. Press **Ctrl+C** in it to stop the
server when you're done.

> If it says *address already in use*, a server is already running (maybe one
> Claude started). Just open the page, or stop the old one with
> `pkill -f catalog/serve.py` and start again.

### Step 2: Search

1. **Tag**: the name your ticks are saved under, e.g. `putti`. Keep it short,
   lowercase, and use the same spelling every time.
2. **Phrases**: one per line. They are averaged into one query. For example:
   ```
   cherub angel putto
   a crochet filet pattern of a cherub
   naked winged child
   ```
3. Click **Search**. All 2283 patterns are ranked, most likely first.

**Tips for phrasing:**
- Describe what is *visible*, not the meaning: "naked child with small
  wings" works better than "cupid".
- Two or three phrasings are better than one, and more than five rarely helps.
- A phrasing that pulls in the wrong things ("winged baby" also found griffins
  and a pegasus) can just be removed.

### Step 3: Tick

Work through the results from the top:

| Key | Action |
|---|---|
| `y` | yes: contains the motif, go to next |
| `n` | no, go to next |
| `u` | clear the mark |
| arrows | move |
| `o` | open the pattern large in a new tab (for big scenes) |
| `m` | **more like this**: re-rank by this pattern's picture |
| ctrl+s | save |

You can also click the buttons under each picture, or click the picture
itself (cycles yes → no → unmarked).

**Before you start, decide what counts as "yes"** and write it down. For
example: "any putto, even small, even inside a crowded scene" or "only
patterns where a putto is a main figure". Stick to it, or the count won't
mean anything.

**Save often** (ctrl+s). The page warns you if you try to close it with
unsaved ticks.

### Step 4: Know when to stop

You don't go through all 2283. The likely matches are at the top and they thin
out further down.

- After **20 "no" in a row** the page shows a hint. That's usually the time
  to stop.
- **Before you stop, do a second pass:**
  1. Turn on **"hide already-reviewed for this tag"**.
  2. Press `m` on one of your best matches, or try a new phrasing.
  3. Only patterns you haven't seen yet appear. Tick any new matches.
- If a second pass finds almost nothing new, you're done.

Expect about 10–20 minutes per motif.

### Step 5: Count

```bash
cd ~/arachne-generator
.venv/bin/python catalog/count.py
```

Example output:

```
tag                    yes  reviewed
putti                   87       240

≥200       worth trying method C (trained model)
  —
50–199     borderline: B now, C maybe later
  putti (87)
<50        use method B (motif composition)
  —
```

- **yes**: patterns you ticked yes. This is the count.
- **reviewed**: how many you looked at (yes + no) for this tag.

### Step 6: Decide

| Count | Decision |
|---|---|
| **≥ 200** | Method C is realistic for this motif. |
| **50–199** | Start with B. Collect more examples (other motif names, more phrasings) before trying C. |
| **< 50** | Method B. You already have the list of patterns to cut motifs from. |

The count is a slight **undercount**: a few matches will be ranked low and
missed. That's fine, because you only need to know which group the motif is
in. If it's close to a boundary (say 180 or 45), do one more pass with new
phrasings before deciding.

---

## 4. How it works

### 4.1 The grids, and the surprise in them

Each pattern's `_grid.json` is rows of `"0"` and `"1"`. The original spec
assumed `"0"` = background, but in this catalog it's the other way round:

- `"1"` = the **stitched ground** (the black areas in the previews)
- `"0"` = the **motif** (the white figures in the previews)
- `"0"` *also* fills the **empty area outside** round or triangular pieces

So "fill ratio > 0.5 means inverted" doesn't work here: 73% of patterns would
count as "inverted". The question to ask isn't "which colour is more common"
but "which cells are the motif".

**Lesson to reuse:** before building anything, look at a sample of the data
with your own eyes. One look at 20 rendered patterns overturned an assumption
the whole plan was based on.

### 4.2 Finding the motif (`motif_mask` in `load.py`)

To separate the motif from the empty outside area:

1. **Fill from the edge.** Start at every `"0"` on the border of the grid and
   spread to neighbouring `"0"` cells, like the paint-bucket tool in an image
   editor. Everything reached is "outside".
2. **Problem: leaks.** If a motif touches the outline of the piece (the
   cherub's foot touching a triangle's diagonal), the fill pours into the
   motif and erases it.
3. **Fix: close small gaps first.** Before filling, gaps up to 4 cells wide
   are sealed. This is called *morphological closing*: grow the stitched area
   by 2 cells, then shrink it back by 2. Narrow gaps get filled and wide areas
   stay open. The fill then can't squeeze through.
4. **New problem: thin corner slivers.** The empty corners of round pieces are
   only 2–3 cells deep, so step 3 sealed them too and they showed up as dark
   marks.
5. **Fix: the row-and-column rule.** An empty cell is also "outside" if it
   lies beyond the last stitch in **both** its row **and** its column. A
   corner sliver passes that test. A motif touching the edge of a border strip
   doesn't, because its column has stitches on both sides.

Each fix was checked by rendering the worst-affected patterns side by side
with the originals (the sheets in `data/review/`). A third idea, letting the
fill run freely along the grid edge, was tested on all patterns and rejected:
it erased real motif in 699 of them.

**Lesson to reuse:** for each rule, find the patterns where it does the most
damage and look at those, not at random ones. A rule that looks fine on
typical patterns can fail badly on unusual ones.

### 4.3 Why the patterns are redrawn

Every pattern is redrawn from its grid as a **dark motif on a light
background** (#0a0a0a on #f5f0eb), using the motif mask. Without this, the
search would partly sort patterns by "mostly black or mostly white" instead of
by what is drawn. Redrawing them all the same way removes that difference.

### 4.4 CLIP, weights and embeddings

**CLIP** is a neural network published by OpenAI. The version used here
(*ViT-B/32*, trained by the LAION group) learned from about 2 billion web
images with captions. It learned one skill: **how well does this picture match
this text?**

It has two halves:
- an **image encoder**: picture (224×224 px) → 512 numbers
- a **text encoder**: sentence → 512 numbers

**Weights** are the millions of internal numbers that make the network do
what it does. Training means adjusting them over billions of examples. We
**don't train** anything: we downloaded LAION's finished weights (about 600 MB,
cached in `~/.cache/huggingface`) and use them as they are.

An **embedding** is the 512 numbers the network gives for one picture or one
sentence. Think of it as a position on a map with 512 directions instead of 2.
CLIP places things with similar meaning close together: a picture of a cherub
lands near the text "a cherub", and two grapevine patterns land near each
other.

`embed.py` computes the embedding of every pattern once and saves them in
`embeddings.npy`. That's the slow part. Searching afterwards is only
arithmetic on that saved table, so it takes 0.3 seconds.

### 4.5 How a search ranks patterns

1. Each phrase goes through the text encoder, giving one point per phrase.
2. The points are **averaged**. Different phrasings of the same idea point
   roughly the same way, and the average is steadier than any single one.
3. For every pattern, the **cosine similarity** between its point and the
   query point is computed. This is a number around 0.1–0.4 here, and higher
   means closer. It's the score shown on each thumbnail.
4. Patterns are sorted by score.

**More like this** does the same with a pattern's *picture* embedding as the
query instead of text. It often works better than text, because CLIP learned
mostly from photos, not crochet charts. Comparing two crochet patterns is
easier for it than comparing a crochet pattern with a sentence.

### 4.6 Crops for big patterns

CLIP always shrinks its input to 224×224 pixels. A 747×175 frieze shrunk that
far leaves each figure only a few pixels tall, too small to recognise. So big
or long patterns also get **overlapping square crops** (up to 128×128 cells,
each overlapping the next by half), and each crop gets its own embedding. That
is why there are 12,246 embeddings for 2283 patterns.

A pattern's score is its **best crop**: one putto anywhere in a large scene is
enough to rank the whole pattern high.

### 4.7 What the search can and can't do

- It only **orders** the patterns. It never tags anything.
- **Wrong results near the top are normal.** CLIP mixes up things that look
  alike at stitch resolution: winged creatures, robed figures, putti. A wrong
  result costs one `n`.
- **Every count comes from your ticks** (`tags.json`), never from the model.

---

## 5. Files

| File | What it is | Commit it? |
|---|---|---|
| `tags.json` | `{"pattern_id": ["putti", ...]}`: your yes ticks, **the source of truth** | yes |
| `reviewed.json` | every yes/no per tag, for "hide already-reviewed" | yes |
| `catalog/load.py` | read grids, find the motif, draw a grid | yes |
| `catalog/embed.py` | build the embeddings (run once) | yes |
| `catalog/serve.py` | local server: search, thumbnails, saving | yes |
| `catalog/search.html` | the search page | yes |
| `catalog/count.py` | counts per tag | yes |
| `catalog/embeddings.npy`, `ids.json`, `duplicates.json` | generated by `embed.py` | no (rebuild with `embed.py`) |
| `data/previews` | link to `pattern-app/apps/api/assets/previews` | no |
| `data/review/*.png` | the check sheets for the motif rule | no |

**How saving works:** saving only adds or removes the **current** tag on each
pattern; other tags are kept. Marking "no" removes only that tag, which is how
you fix a wrong "yes". Identical duplicate scans were merged before embedding,
so nothing is counted twice.

---

## 6. Applying this yourself later

The same recipe works for any "how many of X do I have?" question on an
unlabelled image collection:

1. **Look at the data first.** Count files, check pairs, sizes, duplicates.
   Render a sample and look at it.
2. **Normalise** the images so the thing you care about looks the same
   everywhere (here: dark motif on light background).
3. **Embed** everything once with a pre-trained model such as CLIP. Add crops
   if the things you're looking for are small inside big images.
4. **Rank by text or by an example image**, then have a person confirm. The
   model only decides the order; the person decides the answer.
5. **Stop when matches run out**, and do a second pass with a different query
   to catch what the first one missed.

**When to re-run what:**

| Change | Re-run |
|---|---|
| New patterns added to the catalog | `embed.py` (tags are kept) |
| Change how patterns are drawn (`load.py`) | `embed.py` |
| New motif to count | nothing: just search with a new tag |
| Try a different CLIP model | change `MODEL`/`PRETRAINED` in `embed.py`, re-run it |

**Speed:** `embed.py` used the CPU here (about 17 min). It automatically uses
a GPU if one is present, which would take about a minute. Searching never
needs a GPU. Training for method C will.

---

## 7. What comes after the count

- **Count says B:** next we build a tool to cut out the motif from your "yes"
  patterns (drawing a box or outline around each one you want to use), and
  then the placement along a path.
- **Count says C:** next we cut the motif out of each "yes" pattern. For each
  one, a program traces a centre line through the motif and uses it as "the
  path a user would have drawn", which gives (path → motif) training pairs
  with no manual work. Then we train a model, ideally on a GPU.
- **Either way:** method A (texture fill) needs no tags at all and can be
  built in parallel, because it learns from all patterns.

---

## Glossary

| Term | Meaning |
|---|---|
| **grid** | the pattern as rows of `0`/`1` cells, one per stitch block |
| **motif mask** | the grid with only the motif cells marked |
| **CLIP** | a pre-trained model that matches images and text |
| **weights** | the internal numbers of a neural network, set by training |
| **pre-trained** | already trained by someone else; we only use it |
| **embedding** | the list of numbers a model gives for an image or text; similar things get similar lists |
| **cosine similarity** | how close two embeddings are; higher = more similar |
| **crop / view** | a square piece of a big pattern, embedded on its own |
| **CPU / GPU** | normal processor / graphics card; GPUs run neural networks 10–50× faster |
| **morphological closing** | grow then shrink a shape to seal small gaps |
| **flood fill** | spread from a starting cell to all connected cells of the same kind |
