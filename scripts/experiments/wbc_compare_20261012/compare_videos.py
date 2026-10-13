"""wbc_compare step 5 (TODO 24.5): side-by-side SONIC | ScaleBFM videos per (object, round), and a four-quadrant reel.

Left: the SONIC round's recording (outputs/runs/gnb_obj_XXX<tag>/videos/round_RR.mp4, the lockstep sim's
third-person view with the wrist / ego insets and the magenta ghost at the commanded grasp). Right: the ScaleBFM
round (outputs/wbc_compare_20261012/bfm_grasp/<arm>/obj_XXX/round_RR/round_RR.mp4, the runtime's third-person view
with the same ghost). Both scaled to HEIGHT, the title line naming the object, the controller, the outcome and the
program's reason; <= 500 KB previews next to the full files. The reel takes one round per quadrant: both lifted,
SONIC only, ScaleBFM only, neither (the first of each in object order), 2x speed.

    python3 scripts/experiments/wbc_compare_20261012/compare_videos.py --sonic _wbc --arm paired \
        --out outputs/wbc_compare_20261012/videos/paired [--objects 40,12] [--reel]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BFM_ROOT = REPO / "outputs" / "wbc_compare_20261012" / "bfm_grasp"
HEIGHT = 480
QUAD_W = 1280  # a side-by-side clip at HEIGHT: two 4:3 views
BAND = 28  # the reel's quadrant title band above each clip (the clips carry their own two titles)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *map(str, args)], check=True)


def label(text: str) -> str:
    safe = text.replace("\\", "").replace("'", "").replace(":", "\\:").replace("%", "")
    font = f"fontfile={FONT}:" if Path(FONT).exists() else ""
    return f"drawtext={font}text='{safe}':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5"


def outcome(rec: dict | None) -> str:
    if rec is None:
        return "no round"
    return "LIFTED" if rec.get("ok") else "FAIL " + (rec.get("why") or "")[:60]


def side_by_side(sonic_mp4: Path, bfm_mp4: Path, left: str, right: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fc = (
        f"[0:v]scale=-2:{HEIGHT},{label(left)}[l];[1:v]scale=-2:{HEIGHT},{label(right)}[r];"
        f"[l][r]hstack=inputs=2:shortest=0[v]"
    )
    ff(
        "-i",
        sonic_mp4,
        "-i",
        bfm_mp4,
        "-filter_complex",
        fc,
        "-map",
        "[v]",
        "-r",
        "25",
        "-c:v",
        "libx264",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        out,
    )
    ff(
        "-i",
        out,
        "-vf",
        "scale=640:-2",
        "-c:v",
        "libx264",
        "-crf",
        "36",
        "-an",
        out.with_name(out.stem + "_preview.mp4"),
    )


def reel(clips: list[tuple[Path, str]], out: Path) -> None:
    """Four clips in quadrants (padded with black when fewer), 2x speed."""
    while len(clips) < 4:
        clips.append((None, "empty"))
    inputs, parts = [], []
    for i, (path, text) in enumerate(clips):
        if path is None:
            inputs += ["-f", "lavfi", "-t", "1", "-i", f"color=c=black:s={QUAD_W}x{HEIGHT + BAND}"]
        else:
            inputs += ["-i", str(path)]
        # vstack needs equal widths: every quadrant is padded to QUAD_W (a side-by-side clip is 2 x 640 wide)
        parts.append(
            f"[{i}:v]setpts=0.5*PTS,scale=-2:{HEIGHT},pad={QUAD_W}:{HEIGHT + BAND}:(ow-iw)/2:{BAND},{label(text)}[q{i}]"
        )
    fc = ";".join(parts) + ";[q0][q1]hstack[t];[q2][q3]hstack[b];[t][b]vstack[v]"
    ff(
        *inputs,
        "-filter_complex",
        fc,
        "-map",
        "[v]",
        "-r",
        "25",
        "-c:v",
        "libx264",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        out,
    )
    ff(
        "-i",
        out,
        "-vf",
        "scale=800:-2",
        "-c:v",
        "libx264",
        "-crf",
        "38",
        "-an",
        out.with_name(out.stem + "_preview.mp4"),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sonic", default="_wbc")
    ap.add_argument("--arm", default="paired")
    ap.add_argument("--objects", default="all")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--reel", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="keep side-by-side clips already rendered")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    objects = json.loads((REPO / "outputs" / "graspnet_bottle_20261008" / "objects.json").read_text())["objects"]
    ids = [int(o["id"]) for o in objects] if args.objects == "all" else [int(v) for v in args.objects.split(",")]
    quadrants: dict[tuple, tuple[Path, str]] = {}
    made = 0
    for oid in ids:
        for rnd in range(args.rounds):
            sdir = REPO / "outputs" / "runs" / f"gnb_obj_{oid:03d}{args.sonic}"
            s_mp4 = sdir / "videos" / f"round_{rnd:02d}.mp4"
            s_rec = sdir / "grasp_planned" / f"round_{rnd:02d}_ik" / "record.json"
            bdir = BFM_ROOT / args.arm / f"obj_{oid:03d}" / f"round_{rnd:02d}"
            b_mp4, b_rec = bdir / f"round_{rnd:02d}.mp4", bdir / "record.json"
            if not (s_mp4.exists() and b_mp4.exists()):
                continue
            s = json.loads(s_rec.read_text()) if s_rec.exists() else None
            b = json.loads(b_rec.read_text()) if b_rec.exists() else None
            left = f"obj {oid:03d} r{rnd} SONIC: {outcome(s)}"
            right = f"obj {oid:03d} r{rnd} ScaleBFM ({args.arm}): {outcome(b)}"
            out = args.out / f"obj_{oid:03d}_round_{rnd:02d}.mp4"
            if not (args.skip_existing and out.exists()):
                side_by_side(s_mp4, b_mp4, left, right, out)
            made += 1
            key = (bool(s and s.get("ok")), bool(b and b.get("ok")))
            quadrants.setdefault(key, (out, f"obj {oid:03d} r{rnd}"))
            print(f"{out.name}: {left} | {right}", flush=True)
    if args.reel and quadrants:
        order = [(True, True), (True, False), (False, True), (False, False)]
        names = ("both lifted", "SONIC only", "ScaleBFM only", "neither")
        clips = [
            (quadrants[k][0], f"{n}: {quadrants[k][1]}") if k in quadrants else (None, n) for k, n in zip(order, names)
        ]
        reel(clips, args.out / "reel.mp4")
        print(f"reel: {args.out / 'reel.mp4'}")
    print(f"{made} side-by-side videos in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
