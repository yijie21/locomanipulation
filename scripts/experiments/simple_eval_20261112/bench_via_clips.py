"""selfmodel_bench_20261123: which episodes are filmed, and the three arms of each side by side.

    python3 -I bench_via_clips.py pick <bench_via.sh out dir> <tag> <stage dir>     # choose, copy the films under arm names
    python3 -I bench_via_clips.py grid <stage dir> <isaac_film.py out dir> <out dir>   # after isaac_film.py drew them

The choice, fixed before any result of the comparison was read (2026-11-23): on TabletopGraspMP, the task where paths exist,
the lowest-numbered episode of each kind among those where arms B and C both ran the planned path --
  c_only  C lifts the target within 45 s and B does not
  b_only  B does and C does not
  both    both do
  neither neither does
A kind with no episode is left out. `pick` copies the three arms' films (A straight reach, B the path as it is, C the path
moved by the self model) as <tag>_<kind>_<NAME>_<arm>.npz and writes <stage>/<tag>_picks.json with each arm's outcome.
`grid`: the three world-camera clips of an episode in one row, each cell with the arm's name and outcome, shorter clips
holding their last frame, 2 s more at the end -> <out>/<tag>_<kind>.mp4 and a poster _end.jpg.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

TASK = "TabletopGraspMP"
ARMS = {"A": "直线伸手", "B": "沿规划路线，原样发", "C": "沿规划路线，先挪命令"}
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
CW, CH, BAR, FPS = 640, 360, 44, 25


def stage(r: dict, name: str) -> dict:
    return next((s for s in r["stages"] if s["stage"] == name), {})


def routed(r: dict) -> bool:
    a = r.get("avoid")
    return isinstance(a, dict) and a.get("goal_cm", 9) <= 1.0 and bool(stage(r, "pregrasp"))


def pick(D: Path, tag: str, S: Path) -> None:
    S.mkdir(parents=True, exist_ok=True)
    R = {arm: {json.loads(x)["ep"]: json.loads(x) for x in open(D / "rows" / f"{arm}_{TASK}.jsonl")} for arm in ARMS}
    eps = sorted(e for e in R["B"] if e in R["C"] and e in R["A"] and routed(R["B"][e]) and routed(R["C"][e]))
    kinds = {"c_only": lambda b, c: c and not b, "b_only": lambda b, c: b and not c, "both": lambda b, c: b and c,
             "neither": lambda b, c: not b and not c}  # fmt: skip
    picks = {}
    for kind, f in kinds.items():
        e = next((e for e in eps if f(bool(R["B"][e]["success_45"]), bool(R["C"][e]["success_45"]))), None)
        if e is None:
            continue
        name = f"{TASK}_L{R['B'][e]['level']}_e{R['B'][e]['index']}"
        info = dict(ep=e, name=name, arms={})
        for arm in ARMS:
            src = D / f"films_{arm}" / f"{name}_perception_direct.npz"
            shutil.copy(src, S / f"{tag}_{kind}_{name}_{arm}.npz")
            r = R[arm][e]
            info["arms"][arm] = dict(success_45=bool(r["success_45"]), t_success=r["t_success"], pre_cm=stage(r, "pregrasp").get("err_cm"),
                                     entry_cm=stage(r, "entry").get("err_cm"),
                                     arm_hit=any(h[0] == "pregrasp" and h[1].startswith(("right_shoulder", "right_elbow", "right_wrist", "right_hand")) for h in r["hits"]))  # fmt: skip
        picks[kind] = info
    json.dump(
        dict(tag=tag, routed=len(eps), picks=picks), open(S / f"{tag}_picks.json", "w"), indent=1, ensure_ascii=False
    )
    print(tag, "routed episodes", len(eps), {k: v["ep"] for k, v in picks.items()})


def frames(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={CW}:{CH}", "-r", str(FPS), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout  # fmt: skip
    return np.frombuffer(raw, np.uint8).reshape(-1, CH, CW, 3)


def grid(S: Path, clips: Path, out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    out.mkdir(parents=True, exist_ok=True)
    f_t, f_c = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 19)
    for pf in sorted(S.glob("*_picks.json")):
        P = json.load(open(pf))
        ctrl = {"scalebfm": "ScaleBFM", "sonic": "SONIC"}.get(P["tag"], P["tag"])
        for kind, info in P["picks"].items():
            vids = {arm: frames(clips / f"{P['tag']}_{kind}_{info['name']}_{arm}_world.mp4") for arm in ARMS}
            n = max(len(v) for v in vids.values()) + 2 * FPS
            W = 3 * CW
            title = f"{ctrl} · SIMPLE 桌面抓取 · 第 {info['ep']} 局 · 目标来自机器人自己的相机"
            ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{BAR + CH}", "-r", str(FPS),
                                   "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags",
                                   "+faststart", str(out / f"{P['tag']}_{kind}.mp4")], stdin=subprocess.PIPE)  # fmt: skip
            for k in range(n):
                img = Image.new("RGB", (W, BAR + CH), (248, 249, 250))
                dr = ImageDraw.Draw(img)
                dr.text((14, 9), title, font=f_t, fill=(33, 37, 41))
                for j, arm in enumerate(ARMS):
                    v = vids[arm]
                    img.paste(Image.fromarray(v[min(k, len(v) - 1)]), (j * CW, BAR))
                    a = info["arms"][arm]
                    done = k >= len(v) - 1
                    lab = ARMS[arm] + ("" if not done else "：抓起来了" if a["success_45"] else "：没抓起来")
                    col = (28, 126, 214) if arm == "C" else (232, 89, 12) if arm == "B" else (73, 80, 87)
                    dr.rectangle([j * CW + 8, BAR + 8, j * CW + 22 + f_c.getlength(lab), BAR + 38], fill=col)
                    dr.text((j * CW + 15, BAR + 11), lab, font=f_c, fill=(255, 255, 255))
                    if j:
                        dr.line([j * CW, BAR, j * CW, BAR + CH], fill=(255, 255, 255), width=3)
                ff.stdin.write(np.asarray(img).tobytes())
                if k == n - 1:
                    img.save(out / f"{P['tag']}_{kind}_end.jpg", quality=88)
            ff.stdin.close()
            ff.wait()
            print(P["tag"], kind, info["name"], n, "frames", (out / f"{P['tag']}_{kind}.mp4").stat().st_size, "bytes")


if __name__ == "__main__":
    if sys.argv[1] == "pick":
        pick(Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
    else:
        grid(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
