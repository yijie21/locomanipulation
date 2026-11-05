#!/usr/bin/env python3
"""Write `assets/mujoco_scanned_objects/categories.json`: every Google Scanned Object's category as the dataset labels
it (Shoe, Consumer Goods, Toys, Bottles and Cans and Cups, Bag, Media Cases, Action Figures, Board Games, Legos, ...;
about a fifth carry no label), read from the dataset's home on Gazebo Fuel. kevinzakka's MuJoCo conversion ships
the meshes without these labels; the catalog (scripts/gso_catalog.py) reads this file to keep whole categories out of
the logistics room (user 2026-11-04: no shoes).

    python3 scripts/gso_categories.py [--out assets/mujoco_scanned_objects/categories.json]

Network: GET https://fuel.gazebosim.org/1.0/GoogleResearch/models?per_page=100&page=N until the API answers that the
page does not exist. The file is tracked; rerun it only to refresh the labels (the dataset is frozen since 2020).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/mujoco_scanned_objects/categories.json"
API = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"
PER_PAGE = 100


def fetch(api: str = API, per_page: int = PER_PAGE, timeout: float = 60.0) -> dict[str, str | None]:
    """{model name: its first category, or None} for every GoogleResearch model on Fuel."""
    out: dict[str, str | None] = {}
    page = 1
    while True:
        req = urllib.request.Request(f"{api}?per_page={per_page}&page={page}", headers={"User-Agent": "locomani"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                rows = json.loads(resp.read())
        except urllib.error.HTTPError as e:  # 404 {"errcode": 3017, "msg": "Page not found"}: past the last page
            if e.code == 404 and page > 1:
                break
            raise
        if isinstance(rows, dict):
            raise RuntimeError(f"{api} page {page}: {rows}")
        if not rows:
            break
        for r in rows:
            cats = r.get("categories") or []
            out[r["name"]] = cats[0] if cats else None
        page += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    cats = fetch()
    counts = Counter(c or "(none)" for c in cats.values())
    doc = {
        "source": API,
        "fetched": dt.date.today().isoformat(),
        "counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "categories": dict(sorted(cats.items())),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(a.out), "models": len(cats), "counts": doc["counts"]}, indent=1), file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
