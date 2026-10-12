"""TODO 21.3.4: the planner comparison as one video per round instead of side by side (user, 2026-10-10).

For every object whose run has both rounds: each round's video -- the ghost-overlaid one from ghost_overlay.py
(ghost_rounds/obj_XXX/round_NN.mp4) when the round chose a grasp, else the sim's own -- with make_videos.py's label bar
(object and planner, outcome, what the ghost is), written full quality to videos_single_full/ and as <= 500 KB previews
to videos_single/ by locomani.pipeline.video_preview. The file name carries the planner and the outcome:
obj_XXX_ik_success.mp4, obj_XXX_curobo_fail.mp4 (a long round splits into _partNofM). index.csv lists every file with
its failure reason. --tag _fixed reads the fixed-grasp runs and writes under <root>/fixed/.

    PYTHONPATH=src python3 scripts/experiments/graspnet_bottle_20261008/split_videos.py outputs/graspnet_bottle_20261008 [--tag _fixed] [--jobs 8] [ids...]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_videos import FONT, PLANNERS, REPO, label, outcome  # noqa: E402

SHORT = {"round_00_ik": "ik", "round_01_curobo": "curobo"}


def render(src: Path, out: Path, title: str, status: str, ok: bool, ghost: str) -> None:
    colour = "0x30d040" if ok else "0xff5050"
    graph = (
        "scale=960:720,fps=12.5,"
        "drawbox=x=0:y=0:w=960:h=114:color=black@0.6:t=fill,"
        f"drawtext=fontfile={FONT}:text='{label(title)}':x=18:y=12:fontsize=30:fontcolor=white,"
        f"drawtext=fontfile={FONT}:text='{label(status)}':x=18:y=50:fontsize=24:fontcolor={colour},"
        f"drawtext=fontfile={FONT}:text='{label(ghost)}':x=18:y=82:fontsize=20:fontcolor=0xff40c0"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(src), "-vf", graph, "-threads", "1",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out),
        ],
        check=True,
    )  # fmt: skip


def one_round(job) -> dict | None:
    base, oid, run, k, mrow = job
    key, name, video = PLANNERS[k]
    ghosts = base / "ghost_rounds" / f"obj_{oid:03d}"
    shared = (run / "grasp_planned" / "target.json").exists()
    rec_path = run / "grasp_planned" / key / "record.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
    ok, why = outcome(rec, mrow)
    src = ghosts / video if (ghosts / video).exists() else run / "videos" / video
    if not src.exists():
        return None
    if src.parent == ghosts:
        ghost = f"magenta ghost = the {'shared ' if shared else ''}graspnet grasp{'' if shared else ' it chose'} (zoom top right)"
    else:
        ghost = "no grasp chosen, no ghost"
    stem = f"obj_{oid:03d}_{SHORT[key]}_{'success' if ok else 'fail'}"
    full = base / "videos_single_full" / f"{stem}.mp4"
    render(src, full, f"object {oid:03d}   {name}", "LIFTED" if ok else f"FAILED - {why[:60]}", ok, ghost)
    preview_dir = base / "videos_single"
    for old in glob.glob(str(preview_dir / f"obj_{oid:03d}_{SHORT[key]}_*.mp4")):
        Path(old).unlink()  # a re-run whose outcome changed must not leave the other name behind
    subprocess.run(
        [sys.executable, "-m", "locomani.pipeline.video_preview", str(full), str(preview_dir / stem), "--width", "960"],
        check=True, cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        capture_output=True,
    )  # fmt: skip
    files = sorted(Path(p).name for p in glob.glob(str(preview_dir / f"{stem}*.mp4")))
    print(f"{stem}: {' '.join(files)}", flush=True)
    return dict(
        object=f"{oid:03d}", planner=SHORT[key], outcome="success" if ok else "fail", why=why, files=" ".join(files)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--tag", default="", help="run-name suffix (run_batch.sh's tag); outputs go to <root>/<tag>/")
    ap.add_argument("ids", nargs="*", type=int, help="only these objects (index.csv then lists only them)")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    root = args.root.resolve()
    base = root / args.tag.strip("_") if args.tag else root
    objects = json.loads((root / "objects.json").read_text())["objects"]
    work = []
    for o in objects:
        oid = o["id"]
        if args.ids and oid not in args.ids:
            continue
        run = REPO / "outputs" / "runs" / f"gnb_obj_{oid:03d}{args.tag}"
        manifest = run / "skills" / "manifest.jsonl"
        if not manifest.exists():
            continue
        mrows = [json.loads(line) for line in manifest.read_text().splitlines()]
        if len(mrows) < 2:
            continue
        work += [(base, oid, run, k, mrows[k]) for k in range(len(PLANNERS))]
    with ProcessPoolExecutor(args.jobs) as pool:
        index = [row for row in pool.map(one_round, work) if row is not None]
    with open(base / "videos_single" / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["object", "planner", "outcome", "why", "files"])
        w.writeheader()
        w.writerows(index)
    n_ok = sum(r["outcome"] == "success" for r in index)
    print(f"{len(index)} rounds ({n_ok} success, {len(index) - n_ok} fail) -> {base / 'videos_single'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
