"""One comparison video per filmed episode: the methods' world-camera clips in a grid (2026-11-19).

    python -I grid.py <npz dir> <clips dir (isaac_film.py's out)> <out dir> [--only A,B] [--suffix _world]
        [--methods direct,inverse,model,model_pose] [--cols 2] [--cell 480,270] [--font 0] [--bar 46]

The report's clips are two methods side by side: --methods direct,model --cols 2 --cell 800,450 --font 24 --bar 54
(1600 x 504). The user ruled lookup and servo out on 2026-11-19 (servo corrects on the true hand position all the way,
which is not one command), then model_pose and inverse off the page. Each cell carries the method's name and the
episode's outcome within 45 s (meta success_45 of the film's npz); the title bar the column, the task and the episode.
Shorter clips hold their last frame; every clip holds 2 s more at the end.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("clips")
ap.add_argument("out")
ap.add_argument("--only", default="")
ap.add_argument("--suffix", default="_world")
ap.add_argument("--cell", default="480,270")
ap.add_argument("--crf", default="23")
ap.add_argument("--methods", default="direct,inverse,model,model_pose")
ap.add_argument("--cols", type=int, default=2)
ap.add_argument("--font", type=int, default=0, help="cell label size in px; 0 = 17 px per 640 px of cell width")
ap.add_argument("--bar", type=int, default=46, help="title bar height in px; its text is 0.48 of it")
a = ap.parse_args()
CW, CH = map(int, a.cell.split(","))
BAR = a.bar
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
os.makedirs(a.out, exist_ok=True)

LABEL = {"direct": "直接发目标位置", "lookup": "查表修正", "inverse": "逆模型", "model": "自身模型（我们的）",
         "model_pose": "自身模型，连朝向一起求", "servo": "闭环反馈"}  # fmt: skip
METHODS = [(m, LABEL[m]) for m in a.methods.split(",")]
NC = a.cols
NR = -(-len(METHODS) // NC)
FS = a.font or round(17 * CW / 640)
COL = {"truth": "目标取自仿真真值", "perception": "目标来自机器人自己的相机"}
TASK = {"XMovePickTeleop": "XMovePick，走过去站着抓", "XMoveBendPickTeleop": "XMoveBendPick，走过去弯腰抓",
        "BendPickMP": "BendPick，原地弯腰抓", "TabletopGraspMP": "TabletopGrasp，桌面抓取"}  # fmt: skip
CLIPS = [("A", "truth", "XMoveBendPickTeleop", 0), ("B", "truth", "XMoveBendPickTeleop", 6),
         ("C", "perception", "BendPickMP", 1), ("D", "truth", "XMovePickTeleop", 4),
         ("E", "perception", "TabletopGraspMP", 0), ("F", "truth", "BendPickMP", 0)]  # fmt: skip
GOOD, BAD, OURS = "0x1f7a3d", "0xb3261e", "0xf2b01e"


def esc(s):
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace(",", "\\,").replace("%", "\\%")


for tag, col, task, ep in CLIPS:
    if a.only and tag not in a.only.split(","):
        continue
    ins, metas = [], []
    for m, _ in METHODS:
        stem = f"{task}_L0_e{ep}_{col}_{m}"
        z = np.load(f"{a.npz}/{stem}.npz", allow_pickle=False)
        metas.append((json.loads(str(z["meta"])), len(z["t"])))
        ins.append(f"{a.clips}/{stem}{a.suffix}.mp4")
        assert os.path.exists(ins[-1]), ins[-1]
    n = max(k for _, k in metas)
    total = n / 25.0 + 2.0
    # one brightness lift per episode (SIMPLE's lights are randomised, some scenes are dim): the mean of a middle
    # frame of the first clip brought to TARGET by a gamma between 1 and 1.8, the same for the six cells
    import cv2

    cap = cv2.VideoCapture(ins[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, metas[0][1] // 2)
    ok0, fr = cap.read()
    cap.release()
    mean = float(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean()) / 255.0 if ok0 else 0.4
    gamma = float(np.clip(np.log(max(mean, 1e-3)) / np.log(0.40), 1.0, 1.8))
    parts = []
    for i, ((m, label), (meta, k)) in enumerate(zip(METHODS, metas)):
        ok = bool(meta["success_45"])
        pad = total - k / 25.0
        f = f"[{i}:v]scale={CW}:{CH},eq=gamma={gamma:.2f},tpad=stop_mode=clone:stop_duration={pad:.2f},"
        f += f"drawtext=fontfile={FONT}:text='{esc(label)}':fontsize={FS}:fontcolor=white:x=8:y=8:box=1:boxcolor=black@0.62:boxborderw=5,"
        f += (f"drawtext=fontfile={FONT}:text='{'抓起来了' if ok else '没抓起来'}':fontsize={FS}:fontcolor=white:x=w-tw-8:y=8:"
              f"box=1:boxcolor={GOOD if ok else BAD}@0.95:boxborderw=5")  # fmt: skip
        if m == "model":
            f += f",drawbox=x=0:y=0:w=iw:h=ih:color={OURS}:t=3"
        else:
            f += ",drawbox=x=0:y=0:w=iw:h=ih:color=black@0.55:t=1"
        parts.append(f + f"[c{i}]")
    lay = "|".join(f"{(i % NC) * CW}_{(i // NC) * CH}" for i in range(len(METHODS)))
    title = f"片段 {tag}　{COL[col]}　{TASK[task]}　第 {ep} 局"
    parts.append("".join(f"[c{i}]" for i in range(len(METHODS))) + f"xstack=inputs={len(METHODS)}:layout={lay}[g]")
    parts.append(f"[g]pad={NC * CW}:{NR * CH + BAR}:0:{BAR}:color=0x16191d,"
                 f"drawtext=fontfile={FONT}:text='{esc(title)}':fontsize={round(BAR * 0.48)}:fontcolor=white:x=14:y=({BAR}-th)/2,"
                 f"trim=duration={total:.2f},format=yuv420p[v]")  # fmt: skip
    out = f"{a.out}/clip_{tag}.mp4"
    cmd = ["ffmpeg", "-loglevel", "error", "-y"]
    for p in ins:
        cmd += ["-i", p]
    cmd += ["-filter_complex", ";".join(parts), "-map", "[v]", "-r", "25", "-c:v", "libx264", "-preset", "slow", "-crf",
            a.crf, "-movflags", "+faststart", out]  # fmt: skip
    r = subprocess.run(cmd)
    print(tag, out, "rc", r.returncode, round(os.path.getsize(out) / 1e6, 2) if os.path.exists(out) else None, "MB",
          f"{total:.1f} s", "mean", round(mean, 3), "gamma", round(gamma, 2),
          [bool(m["success_45"]) for m, _ in metas], flush=True)  # fmt: skip
    if r.returncode:
        sys.exit(1)
    # the poster: the grid's last frame (every cell's outcome on screen)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-sseof", "-0.2", "-i", out, "-frames:v", "1", "-q:v", "4",
                    f"{a.out}/poster_{tag}.jpg"])  # fmt: skip
