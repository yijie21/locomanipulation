"""TODO 23.2: one video per object for a run_batch tag, round 0 -- each with the magenta ghost gripper at the chosen
grasp (ghost_overlay.py --tag <tag> first) and a caption (object, size, mass, grasp source, outcome, the failure text),
the file name carrying success / fail. Full quality (960x720, 12.5 fps, crf 20, as split_videos.render) to
<root>/cache/videos<tag>_full/, <= 500 KB previews to <root>/cache/videos<tag>/ (locomani.pipeline.video_preview),
index.csv next to the previews. A round without a ghost video (no chosen grasp) uses the plain recording. The tag
_presort_b (the 2026-10-10 best configuration) keeps its old directories, cache/videos_best_full/ and videos_best/.

    PYTHONPATH=src "$VENV_SIM_PY" scripts/experiments/graspnet_bottle_20261008/best_videos.py outputs/graspnet_bottle_20261008 [--tag _presort_b] [--jobs 6]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_videos import REPO, outcome  # noqa: E402
from split_videos import render  # noqa: E402


def one(job) -> dict:
    root, tag, o = job
    name = "videos_best" if tag == "_presort_b" else f"videos{tag}"
    oid = o["id"]
    run = REPO / "outputs" / "runs" / f"gnb_obj_{oid:03d}{tag}"
    rec_path = run / "grasp_planned" / "round_00_ik" / "record.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
    mrow = None
    mf = run / "skills" / "manifest.jsonl"
    if mf.exists() and mf.read_text().strip():
        mrow = json.loads(mf.read_text().splitlines()[0])
    ok, why = outcome(rec, mrow)
    ghost = root / tag.strip("_") / "ghost_rounds" / f"obj_{oid:03d}" / "round_00.mp4"
    src = ghost if ghost.exists() else run / "videos" / "round_00.mp4"
    row = dict(object=oid, outcome="success" if ok else "fail", why=why, ghost=ghost.exists(), files="")
    if not src.exists():
        row["why"] = row["why"] or "no video"
        return row
    stem = f"obj_{oid:03d}_{'success' if ok else 'fail'}"
    full = root / "cache" / f"{name}_full" / f"{stem}.mp4"
    g = rec.get("grasp") if rec else None
    source = (rec or {}).get("source", "cache")
    title = f"obj {oid:03d}  {100 * o['height']:.0f}x{100 * o['width']:.0f} cm  {o['mass']:.2f} kg  {source} grasp + IK"
    status = "LIFTED" if ok else f"FAIL: {why.replace('[grasp_planned] ', '')}"
    ghost_text = (
        f"magenta ghost = {source} grasp: tier {g.get('tier')}, score {g.get('score', 0):.2f}, width {100 * g.get('width', 0):.1f} cm"
        if g
        else "no grasp chosen"
    )
    render(src, full, title, status, ok, ghost_text)
    preview_dir = root / "cache" / name
    preview_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "locomani.pipeline.video_preview", str(full), str(preview_dir / stem), "--width", "960"],
        check=True,
        capture_output=True,
    )
    row["files"] = " ".join(p.name for p in sorted(preview_dir.glob(f"{stem}*.mp4")))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--tag", default="_presort_b")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    root = args.root.resolve()
    objects = json.loads((root / "objects.json").read_text())["objects"]
    name = "videos_best" if args.tag == "_presort_b" else f"videos{args.tag}"
    (root / "cache" / f"{name}_full").mkdir(parents=True, exist_ok=True)
    (root / "cache" / name).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(one, [(root, args.tag, o) for o in objects]))
    with open(root / "cache" / name / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["object", "outcome", "why", "ghost", "files"])
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["outcome"] == "success" for r in rows)
    print(
        f"{len(rows)} objects ({n_ok} success, {len(rows) - n_ok} fail, {sum(r['ghost'] for r in rows)} with ghost, "
        f"{sum(1 for r in rows if not r['files'])} without video) -> {root / 'cache' / name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
