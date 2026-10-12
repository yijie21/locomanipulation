"""TODO 21.3.4: one left/right comparison video per object -- ours (IK planner) left, SIMPLE's CuRobo planner right.

For every object whose run has both rounds (run_batch.sh): the sim's round videos (outputs/runs/gnb_obj_XXX/videos,
round_00 = ik, round_01 = curobo) side by side at their full length (the shorter one holds its last frame), each
labelled with the planner and the round's outcome, written as a full-quality source under videos_full/ and then as
<= 500 KB previews under videos/ by locomani.pipeline.video_preview (docs/rendering.md#video-export). A round that
ghost_overlay.py drew the planned grasp into (ghost_rounds/obj_XXX/round_NN.mp4) is used in place of the sim's video and
says so under its outcome; a round without a chosen grasp has no ghost and says that instead. Also writes
summary.json / summary.md: per object and planner, the outcome, the failure, the grasp tier and planning time. With
--tag _fixed (run_batch.sh's tag): the runs gnb_obj_XXX_fixed, everything under <root>/fixed/, and each object's shared
grasp (grasp_planned.py "Fixed grasp": its rank, whether both planners reached it) in the summary.

    PYTHONPATH=src python3 scripts/experiments/graspnet_bottle_20261008/make_videos.py outputs/graspnet_bottle_20261008 [--tag _fixed]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PLANNERS = (
    ("round_00_ik", "ours: IK planner", "round_00.mp4"),
    ("round_01_curobo", "SIMPLE: CuRobo planner", "round_01.mp4"),
)


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return float(out.stdout.strip())


def label(text: str) -> str:
    return text.replace("\\", "").replace(":", "\\:").replace("'", "").replace("%", "\\%")


def outcome(rec: dict | None, manifest_row: dict | None) -> tuple[bool, str]:
    if rec is not None:
        return bool(rec.get("ok")), rec.get("why") or ""
    if manifest_row is not None:
        return bool(manifest_row.get("ok")), manifest_row.get("why") or ""
    return False, "no record"


def compose(oid: int, run: Path, ghosts: Path, rows: dict, out_full: Path) -> Path | None:
    vids = [ghosts / v if (ghosts / v).exists() else run / "videos" / v for _, _, v in PLANNERS]
    if not all(v.exists() for v in vids):
        return None
    durs = [duration(v) for v in vids]
    total = max(durs)
    parts = []
    for k, ((key, name, _), d) in enumerate(zip(PLANNERS, durs)):
        ok, why = rows[key]
        status = "LIFTED" if ok else f"FAILED - {why[:60]}"
        colour = "0x30d040" if ok else "0xff5050"
        shared = (run / "grasp_planned" / "target.json").exists()
        ghost = (
            (
                "magenta ghost = the shared graspnet grasp (zoom top right)"
                if shared
                else "magenta ghost = the graspnet grasp it chose (zoom top right)"
            )
            if vids[k].parent == ghosts
            else "no grasp chosen, no ghost"
        )
        parts.append(
            f"[{k}:v]scale=960:720,fps=12.5,tpad=stop_mode=clone:stop_duration={total - d + 0.1:.2f},"
            f"drawbox=x=0:y=0:w=960:h=114:color=black@0.6:t=fill,"
            f"drawtext=fontfile={FONT}:text='{label(f'object {oid:03d}   {name}')}':x=18:y=12:fontsize=30:fontcolor=white,"
            f"drawtext=fontfile={FONT}:text='{label(status)}':x=18:y=50:fontsize=24:fontcolor={colour},"
            f"drawtext=fontfile={FONT}:text='{label(ghost)}':x=18:y=82:fontsize=20:fontcolor=0xff40c0[v{k}]"
        )
    graph = ";".join(parts) + f";[v0][v1]hstack=inputs=2,trim=duration={total:.2f}[out]"
    out_full.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(vids[0]), "-i", str(vids[1]),
            "-filter_complex", graph, "-map", "[out]", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_full),
        ],
        check=True,
    )  # fmt: skip
    return out_full


def main(root: Path, tag: str = "") -> int:
    objects = json.loads((root / "objects.json").read_text())["objects"]
    base = root / tag.strip("_") if tag else root
    summary, made = [], []
    for o in objects:
        oid = o["id"]
        run = REPO / "outputs" / "runs" / f"gnb_obj_{oid:03d}{tag}"
        manifest = run / "skills" / "manifest.jsonl"
        if not manifest.exists():
            continue
        mrows = [json.loads(line) for line in manifest.read_text().splitlines()]
        rows, entry = {}, dict(id=oid, height=o["height"], width=o["width"], mass=o["mass"])
        for k, (key, _, _) in enumerate(PLANNERS):
            rec_path = run / "grasp_planned" / key / "record.json"
            rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
            rows[key] = outcome(rec, mrows[k] if k < len(mrows) else None)
            plan = (rec or {}).get("plan") or {}
            entry[key.split("_", 2)[2]] = dict(
                ok=rows[key][0],
                why=rows[key][1],
                tier=((rec or {}).get("grasp") or {}).get("tier"),
                candidates=(rec or {}).get("candidates"),
                plan_seconds=plan.get("plan_seconds"),
                rise_m=(rec or {}).get("rise_m"),
            )
        target = run / "grasp_planned" / "target.json"
        if target.exists():
            t = json.loads(target.read_text())
            entry["shared"] = {k: t[k] for k in ("index", "common", "candidates", "both", "either")}
        summary.append(entry)
        if len(mrows) < 2:
            continue
        full = compose(
            oid,
            run,
            base / "ghost_rounds" / f"obj_{oid:03d}",
            rows,
            base / "videos_full" / f"obj_{oid:03d}_ik_vs_curobo.mp4",
        )
        if full is None:
            continue
        subprocess.run(
            [sys.executable, "-m", "locomani.pipeline.video_preview", str(full), str(base / "videos" / full.stem)],
            check=True, cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        )  # fmt: skip
        made.append(oid)
        print(f"object {oid:03d}: ik {rows['round_00_ik'][0]} curobo {rows['round_01_curobo'][0]}", flush=True)
    (base / "summary.json").write_text(json.dumps(summary, indent=1))
    n = len(summary)
    ik = sum(1 for s in summary if s["ik"]["ok"])
    cu = sum(1 for s in summary if s["curobo"]["ok"])
    fixed = any("shared" in s for s in summary)
    lines = [
        f"# graspnet + bottle expert: IK planner vs SIMPLE's CuRobo planner ({n} objects)",
        "",
        f"lifted: ours (IK) {ik}/{n}, CuRobo {cu}/{n}",
        "",
    ]
    if fixed:
        common = [s for s in summary if (s.get("shared") or {}).get("common")]
        lines += [
            f"both rounds aim at one shared grasp; both planners reach it on {len(common)}/{n} objects, on which "
            f"lifted: ours {sum(s['ik']['ok'] for s in common)}, CuRobo {sum(s['curobo']['ok'] for s in common)}",
            "",
            "| object | h cm | w cm | kg | shared rank | both reach | IK | IK failure | CuRobo | CuRobo failure |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    else:
        lines += [
            "| object | h cm | w cm | kg | IK | IK failure | CuRobo | CuRobo failure |",
            "|---|---|---|---|---|---|---|---|",
        ]
    for s in summary:
        sh = s.get("shared") or {}
        shared_cols = f"{sh.get('index')} | {'yes' if sh.get('common') else 'no'} | " if fixed else ""
        lines.append(
            f"| {s['id']:03d} | {100 * s['height']:.1f} | {100 * s['width']:.1f} | {s['mass']:.2f} | {shared_cols}"
            f"{'yes' if s['ik']['ok'] else 'no'} | {s['ik']['why'][:60]} | {'yes' if s['curobo']['ok'] else 'no'} | "
            f"{s['curobo']['why'][:60]} |"
        )
    (base / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"{len(made)} videos; lifted ik {ik}/{n} curobo {cu}/{n}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--tag", default="", help="run-name suffix (run_batch.sh's tag); outputs go to <root>/<tag>/")
    args = ap.parse_args()
    raise SystemExit(main(args.root, args.tag))
