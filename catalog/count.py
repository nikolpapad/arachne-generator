"""Count patterns per motif tag and group them by which generation method fits.

Reads tags.json (and reviewed.json for how many you looked at). Each pattern
is counted once; identical duplicate scans were already folded together by
embed.py, so they are never counted twice.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUPS = [
    (200, "≥200       worth trying method C (trained model)"),
    (50, "50–199     borderline: B now, C maybe later"),
    (0, "<50        use method B (motif composition)"),
]


def main():
    tags_path, reviewed_path = ROOT / "tags.json", ROOT / "reviewed.json"
    tags = json.loads(tags_path.read_text()) if tags_path.exists() else {}
    reviewed = json.loads(reviewed_path.read_text()) if reviewed_path.exists() else {}
    counts = Counter(t for pattern_tags in tags.values() for t in pattern_tags)
    if not counts:
        print("No tags yet: tick some patterns in the search page and Save.")
        return

    print(f"{len(tags)} tagged patterns, {len(counts)} tags\n")
    print(f"{'tag':<20}{'yes':>6}{'reviewed':>10}")
    for tag, n in counts.most_common():
        print(f"{tag:<20}{n:>6}{len(reviewed.get(tag, {})):>10}")

    for low, title in GROUPS:
        high = min((g for g, _ in GROUPS if g > low), default=None)
        members = [f"{t} ({n})" for t, n in counts.most_common() if n >= low and (high is None or n < high)]
        print(f"\n{title}\n  " + (", ".join(members) or "—"))


if __name__ == "__main__":
    main()
