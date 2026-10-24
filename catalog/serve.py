"""Local search-and-tick server.  Run:  python catalog/serve.py  → http://127.0.0.1:8000

Ranks patterns by CLIP similarity to text phrasings (averaged) or to one
pattern ("more like this"). A pattern's score is its best-matching view.

Ticks are saved to:
  tags.json      {"<pattern_id>": ["putti", "grapes", ...]}  — the source of truth
  reviewed.json  {"<tag>": {"<pattern_id>": "yes" | "no"}}   — what you've looked at
Saving merges: other tags on a pattern are never touched. Marking a pattern
"no" for a tag removes only that tag from it.
"""
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import open_clip
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from embed import MODEL, PRETRAINED
from load import load_grid, motif_mask, render

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TAGS, REVIEWED = ROOT / "tags.json", ROOT / "reviewed.json"

ids = json.loads((HERE / "ids.json").read_text())
emb = torch.from_numpy(np.load(HERE / "embeddings.npy").astype(np.float32))
patterns = list(dict.fromkeys(ids))  # unique, in order; first row of each = whole view
_index = {p: i for i, p in enumerate(patterns)}
row_pattern = torch.tensor([_index[p] for p in ids])
whole_row = {p: ids.index(p) for p in patterns}

model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
model.eval()
tokenizer = open_clip.get_tokenizer(MODEL)

app = FastAPI()


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def rank(query_vec):
    """Best-view score per pattern, highest first."""
    sims = emb @ query_vec
    best = torch.full((len(patterns),), -1.0).scatter_reduce(0, row_pattern, sims, reduce="amax")
    order = torch.argsort(best, descending=True)
    return [{"id": patterns[i], "score": round(float(best[i]), 4)} for i in order]


class Search(BaseModel):
    phrases: list[str] = []
    like: str | None = None


class Save(BaseModel):
    tag: str
    marks: dict[str, str]  # pattern_id -> "yes" | "no"


@app.get("/")
def page():
    return FileResponse(HERE / "search.html")


@app.post("/search")
def search(req: Search):
    if req.like:
        if req.like not in whole_row:
            raise HTTPException(404, "unknown pattern")
        return rank(emb[whole_row[req.like]])
    phrases = [p.strip() for p in req.phrases if p.strip()]
    if not phrases:
        raise HTTPException(400, "type at least one phrase")
    with torch.no_grad():
        text = torch.nn.functional.normalize(model.encode_text(tokenizer(phrases)), dim=-1)
    return rank(torch.nn.functional.normalize(text.mean(0), dim=0))


@app.get("/state")
def state():
    return {"tags": read_json(TAGS), "reviewed": read_json(REVIEWED)}


@app.post("/save")
def save(req: Save):
    tag = req.tag.strip().lower()
    if not tag:
        raise HTTPException(400, "tag is empty")
    tags, reviewed = read_json(TAGS), read_json(REVIEWED)
    seen = reviewed.setdefault(tag, {})
    for pid, mark in req.marks.items():
        if pid not in _index or mark not in ("yes", "no"):
            continue
        seen[pid] = mark
        current = set(tags.get(pid, []))
        current.add(tag) if mark == "yes" else current.discard(tag)
        if current:
            tags[pid] = sorted(current)
        else:
            tags.pop(pid, None)
    write_json(TAGS, tags)
    write_json(REVIEWED, reviewed)
    return {"yes": sum(tag in t for t in tags.values()), "reviewed": len(seen)}


@lru_cache(maxsize=4096)
def png(pid, size):
    buf = io.BytesIO()
    render(motif_mask(load_grid(pid)), size).save(buf, "PNG")
    return buf.getvalue()


@app.get("/img/{pid}.png")
def image(pid: str, size: int = 220):
    if pid not in _index:
        raise HTTPException(404)
    return Response(png(pid, min(max(size, 64), 1600)), media_type="image/png",
                    headers={"Cache-Control": "max-age=86400"})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
