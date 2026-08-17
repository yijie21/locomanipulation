"""Stitch one batch's per-episode videos into a single labelled reel."""
import glob, json, os, subprocess, sys

VID = "outputs/videos"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
tag = sys.argv[1] if len(sys.argv) > 1 else "nav"
out = sys.argv[2] if len(sys.argv) > 2 else f"outputs/videos/{tag}_reel.mp4"

batch = json.load(open(sorted(glob.glob("outputs/logs/batch_*.json"))[-1]))
why = {r["index"]: ("OK" if r.get("ok") else f"{r.get('stage')}: {r.get('why','')}")
       for r in batch["results"]}
dur = {r["index"]: r["duration_s"] for r in batch["results"]}

files = sorted(glob.glob(f"{VID}/{tag}_[0-9][0-9][0-9].mp4"))
parts, filt = [], []
for i, f in enumerate(files):
    idx = int(os.path.basename(f).split("_")[1].split(".")[0])
    # ffmpeg's drawtext parser treats these as syntax
    txt = why.get(idx, "").replace(":", "\\:").replace("'", "").replace("—", "-")
    # Cap each clip at its episode duration. The last recording of a batch
    # runs on until something stops it, so it is minutes of an idle robot.
    parts += ["-t", f"{dur.get(idx, 60) + 3:.1f}", "-i", f]
    # Both captions at the top. The failure reason used to sit along the
    # bottom edge, which is exactly where the table, the object and the
    # close-up inset are.
    filt.append(
        f"[{i}:v]drawtext=fontfile={FONT}:text='ep {idx:03d}':"
        f"x=12:y=8:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6,"
        f"drawtext=fontfile={FONT}:text='{txt[:78]}':"
        f"x=12:y=44:fontsize=20:fontcolor=yellow:box=1:boxcolor=black@0.6:boxborderw=6"
        f"[v{i}]")

chain = ";".join(filt) + ";" + "".join(f"[v{i}]" for i in range(len(files)))
chain += f"concat=n={len(files)}:v=1:a=0[out]"
cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *parts,
       "-filter_complex", chain, "-map", "[out]",
       "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
       "-pix_fmt", "yuv420p", out]
subprocess.run(cmd, check=True)
print(f"{out}  ({len(files)} episodes, {os.path.getsize(out) / 1e6:.1f} MB)")
