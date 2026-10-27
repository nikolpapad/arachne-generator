"""Draw the catalog's CLIP embeddings as one browsable map (run after embed.py).

Each pattern's whole-view embedding (its first row in embeddings.npy) is
projected to 2D with UMAP, so patterns CLIP finds alike sit close together.
K-means groups them into clusters; each cluster is named by CLIP's best
matches from a fixed word list (a guess, not a label). For each pattern the
12 nearest neighbours in the full 512-d space are stored too.

Writes catalog/map.html: one self-contained file, open it in a browser.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
import umap
from PIL import Image
from sklearn.cluster import KMeans

from embed import MODEL, PRETRAINED
from load import load_grid, motif_mask

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLUSTERS = 16
NEIGHBOURS = 12
THUMB = 112  # px, longest side

# Candidate names for clusters; CLIP picks the two closest to each cluster centre.
VOCAB = [
    "angels", "cherubs", "a woman", "a man", "a couple", "children", "dancers",
    "a lady in a long dress", "a knight on horseback", "the Virgin Mary", "a religious scene",
    "a horse", "a deer", "a dog", "a cat", "birds", "a peacock", "swans", "a rooster",
    "butterflies", "fish", "a lion", "an elephant", "a dragon",
    "flowers", "roses", "a vase of flowers", "a basket of flowers", "a flower wreath",
    "a tree", "leaves and vines", "grapes", "fruit",
    "letters of the alphabet", "a monogram", "words and lettering",
    "a geometric pattern", "a lace border", "a narrow edging", "a round doily", "a square motif",
    "stars", "hearts", "a cross", "a church", "a house", "a landscape", "a ship", "a windmill",
    "a Christmas scene", "a crown", "a coat of arms", "a bow and ribbons", "a fan", "a vase",
    "a triangular corner piece", "a long horizontal border", "a tall vertical strip",
    "a tall ornamental panel", "animals", "people in a scene", "a single standing figure",
]


def thumbnail(mask):
    """Motif as grey on cream, box-downsampled so thin lines survive, as a WebP data URL."""
    h, w = mask.shape
    scale = THUMB / max(h, w)
    img = Image.fromarray(np.where(mask, 10, 245).astype(np.uint8))
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BOX)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=70)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def cluster_names(centres):
    model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model.eval()
    tokens = open_clip.get_tokenizer(MODEL)([f"a filet crochet pattern of {v}" for v in VOCAB])
    with torch.no_grad():
        text = torch.nn.functional.normalize(model.encode_text(tokens), dim=-1).numpy()
    # Relative to the average cluster, so words CLIP likes for everything don't win everywhere.
    sims = centres @ text.T
    sims -= sims.mean(0)
    return [[VOCAB[j] for j in np.argsort(-row)[:2]] for row in sims]


def main():
    ids = json.loads((HERE / "ids.json").read_text())
    emb = np.load(HERE / "embeddings.npy").astype(np.float32)
    first = {}
    for row, pid in enumerate(ids):
        first.setdefault(pid, row)
    patterns = list(first)
    x = emb[list(first.values())]
    print(f"{len(patterns)} patterns")

    xy = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=0).fit_transform(x)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0)).max()

    km = KMeans(CLUSTERS, n_init=10, random_state=0).fit(x)
    centres = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    names = cluster_names(centres)

    sims = x @ x.T
    np.fill_diagonal(sims, -1)
    nn = np.argsort(-sims, axis=1)[:, :NEIGHBOURS]

    tags = json.loads((ROOT / "tags.json").read_text())
    duplicates = json.loads((HERE / "duplicates.json").read_text())

    items = []
    for i, pid in enumerate(patterns):
        mask = motif_mask(load_grid(pid))
        items.append({
            "id": pid,
            "book": pid.split("_")[0],
            "x": round(float(xy[i, 0]), 4),
            "y": round(float(xy[i, 1]), 4),
            "c": int(km.labels_[i]),
            "size": list(mask.shape),
            "tags": tags.get(pid, []),
            "nn": [[int(j), round(float(sims[i, j]), 3)] for j in nn[i]],
            "img": thumbnail(mask),
        })
        print(f"\r{i + 1}/{len(patterns)} thumbnails", end="", flush=True)
    print()

    # Representatives: the members closest to each cluster centre.
    to_centre = (x @ centres.T)[np.arange(len(x)), km.labels_]
    clusters = []
    for c in range(CLUSTERS):
        members = np.flatnonzero(km.labels_ == c)
        reps = members[np.argsort(-to_centre[members])][:10]
        clusters.append({"names": names[c], "n": int(len(members)), "reps": reps.tolist()})

    data = {
        "items": items,
        "clusters": clusters,
        "model": f"{MODEL} ({PRETRAINED})",
        "views": len(ids),
        "duplicates": sum(map(len, duplicates["duplicates"].values())),
        "skipped": len(duplicates["skipped"]),
    }
    page = (HERE / "map_template.html").read_text()
    page = page.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    (HERE / "map.html").write_text(page)
    print(f"wrote {HERE / 'map.html'} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
