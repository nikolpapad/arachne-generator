"""Review page for motif boxes.  Run:  python generator/c_model/boxes.py human  → http://127.0.0.1:8001

A tag says a pattern *contains* a motif, not where. For every "yes" pattern of
the tag this suggests boxes from the CLIP crops embed.py already scored (the
crops most like the motif phrasings, tightened to the motif cells inside), and
lets you accept, adjust, draw or reject them, and say what kind each one is.

Saved to motif_boxes.json:
  {"<tag>": {"<pattern_id>": [{"box": [y, x, h, w], "kind": "full"}, ...]}}
An empty list means "reviewed, nothing usable". Patterns not in the file are
not reviewed yet. The box height (h, in cells) is the motif's size.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import open_clip
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "catalog"))
from embed import MODEL, PRETRAINED, crop_boxes  # noqa: E402
from load import load_grid, motif_mask  # noqa: E402

TAGS, BOXES = ROOT / "tags.json", ROOT / "motif_boxes.json"
KINDS = ["full", "half", "head", "other"]  # full figure, half figure / bust, head only, anything else
PHRASES = {
    "human": ["a human figure", "a person standing", "a crochet filet pattern of a man or a woman"],
}
TOP, MARGIN, OVERLAP = 3, 0.03, 0.3  # up to 3 suggestions, within 0.03 of the best, overlapping < 30%

args = argparse.ArgumentParser()
args.add_argument("tag")
args.add_argument("--phrases", nargs="+", help="override the CLIP phrasings for this tag")
args.add_argument("--port", type=int, default=8001)
args = args.parse_args()
TAG = args.tag.strip().lower()

ids = json.loads((ROOT / "catalog" / "ids.json").read_text())
emb = torch.from_numpy(np.load(ROOT / "catalog" / "embeddings.npy").astype(np.float32))
first_row, n_rows = {}, Counter(ids)
for i, pid in enumerate(ids):
    first_row.setdefault(pid, i)

model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
model.eval()
phrases = args.phrases or PHRASES.get(TAG, [f"a {TAG}", f"a crochet filet pattern of a {TAG}"])
with torch.no_grad():
    text = torch.nn.functional.normalize(model.encode_text(open_clip.get_tokenizer(MODEL)(phrases)), dim=-1)
query = torch.nn.functional.normalize(text.mean(0), dim=0)

app = FastAPI()


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def tagged():
    return sorted(pid for pid, t in read_json(TAGS).items() if TAG in t)


def tighten(mask, box):
    """Shrink [y, x, h, w] to the motif cells inside it (unchanged if there are none)."""
    y, x, h, w = box
    ys, xs = np.nonzero(mask[y:y + h, x:x + w])
    if not len(ys):
        return box
    return [int(y + ys.min()), int(x + xs.min()), int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)]


def overlap(a, b):
    """Intersection over the smaller box's area."""
    dy = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    dx = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(dy, 0) * max(dx, 0) / min(a[2] * a[3], b[2] * b[3])


def suggest(pid, mask):
    """Best CLIP crops for the tag, tightened. Patterns without crops get the whole motif."""
    crops = crop_boxes(*mask.shape)
    if pid not in first_row or not crops or n_rows[pid] != 1 + len(crops):
        return [{"box": tighten(mask, [0, 0, *mask.shape]), "score": None}]
    start = first_row[pid] + 1  # row 0 of a pattern is its whole view, crops follow in crop_boxes order
    scores = (emb[start:start + len(crops)] @ query).tolist()
    best = max(scores)
    picked = []
    for i in sorted(range(len(crops)), key=lambda i: -scores[i]):
        y, x, s = crops[i]
        if len(picked) == TOP or scores[i] < best - MARGIN:
            break
        if all(overlap([y, x, s, s], p["raw"]) < OVERLAP for p in picked):
            picked.append({"raw": [y, x, s, s], "score": round(scores[i], 4)})
    return [{"box": tighten(mask, p["raw"]), "score": p["score"]} for p in picked]


class Save(BaseModel):
    id: str
    boxes: list[dict]  # [{"box": [y, x, h, w], "kind": "full" | ... | None}]


@app.get("/")
def page():
    return FileResponse(HERE / "boxes.html")


@app.get("/items")
def items():
    return {"tag": TAG, "kinds": KINDS, "ids": tagged(), "saved": read_json(BOXES).get(TAG, {})}


@app.get("/item/{pid}")
def item(pid: str):
    if pid not in tagged():
        raise HTTPException(404, f"{pid} is not tagged {TAG}")
    mask = motif_mask(load_grid(pid))
    return {"h": mask.shape[0], "w": mask.shape[1],
            "rows": ["".join("1" if c else "0" for c in row) for row in mask],
            "suggestions": suggest(pid, mask)}


@app.post("/save")
def save(req: Save):
    if req.id not in tagged():
        raise HTTPException(404, f"{req.id} is not tagged {TAG}")
    h, w = load_grid(req.id).shape
    clean = []
    for b in req.boxes:
        y, x, bh, bw = (int(v) for v in b["box"])
        if not (0 <= y and 0 <= x and bh > 0 and bw > 0 and y + bh <= h and x + bw <= w):
            raise HTTPException(400, f"box {b['box']} is outside the {h}x{w} grid")
        clean.append({"box": [y, x, bh, bw], "kind": b.get("kind") if b.get("kind") in KINDS else None})
    data = read_json(BOXES)
    data.setdefault(TAG, {})[req.id] = clean
    write_json(BOXES, data)
    return {"reviewed": len(data[TAG])}


if __name__ == "__main__":
    print(f"{TAG}: {len(tagged())} patterns, phrases: {' | '.join(phrases)}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)
