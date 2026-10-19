"""A report video rendered in Unreal (user 2026-10-19: is the light real enough?). One step video of the progress
report (docs/handoff/scene-rsi-report-20261019.md): pick_single's (visualize.py made the MuJoCo ones, `ps_NN`) or
report_media.py's STEPS. Each round is replayed into the adt_apartment_mars level through UnrealView and filmed by the
level's third-person rig along the camera track the MuJoCo video took (report_media.track), from the same point:
the rig renders 1280 x 960 at the scene's fovy 45, so the frame is MuJoCo's view cut to 4:3 at the sides. (A 16:10
crop needs the camera pulled back 1.2x, which put it behind the wall the track's clearance check had kept it off.)

Rounds are filmed one after the other in URLab's "stream" capture (Lumen and TSR keep their history from the frame
before, as a game renders; the first frame of a round is a "fresh" capture). Filming the two rounds of a step in
alternation would make every frame a jump. Frames go to <out>/<name>/{before,after}/NNNNN.jpg, then the step video
<out>/<name>.mp4 (card, caption strip, before | after at 800 x 600 each) and its preview, plus stills at full size
beside MuJoCo's picture of the same state (<out>/<name>/compare_*.jpg).

No ghost grippers: the traces carry the commanded wrist positions but no attitude, and an Unreal ghost is a whole
gripper mesh. The rig and ghosts stay parked except for the rig while filming.

    bash scripts/start_unreal.sh --gpu 1
    PYTHONPATH=src:$PYTHONPATH MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 "$BRIDGE_PY" \\
        scripts/experiments/scene_rsi_20261016/report_unreal.py ps_00 [--out outputs/scene_rsi_20261016/report/media_unreal]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import report_media as RM  # noqa: E402
import visualize as V  # noqa: E402

from locomani.sim.unreal import export_mjcf  # noqa: E402
from locomani.sim.unreal.view import UnrealView  # noqa: E402

dv = V.dv
LEVEL = "/Game/Levels/adt_apartment_mars"
SCENE_XML = REPO / "outputs" / "scene_discovery" / "adt_apartment_mars" / "scene_locomani.xml"
OUT = REPO / "outputs" / "scene_rsi_20261016" / "report" / "media_unreal"
PICK = Path("/workspace/code/rsi-scene/pick-a/outputs/scene_rsi_20261016/pick_single/diag")
FILM = (1280, 960)  # the level's source.json `film`
PANEL = (800, 600)
# pick_single's four steps, as visualize.py made them: seed and accepted versions, start, card
PS = {
    "ps_00": ("incumbent_b897126e9dde", "incumbent_7ec46c87bf5c", "canister_pickup_s100053",
              ["拿起 · 第 0 轮：罐子是被手掌推开的",
               "夹爪底座在爪口中心后 4.07 cm → 预抓取站远、到位判据提前、爪口规则放宽",
               "大铁罐 s100053 · 左：种子 · 右：第 0 轮采纳的版本 · Unreal 渲染"]),
    "ps_01": ("incumbent_7ec46c87bf5c", "incumbent_e8ce52c71938", "canister_pickup_s100045",
              ["拿起 · 第 1 轮：导航图上没有地上的凳子",
               "补上地上物体、四条边都试、站位侧移 → 工程师写接近规划",
               "大铁罐 s100045 · 左：第 0 轮后 · 右：第 1 轮采纳的版本 · Unreal 渲染"]),
    "ps_02": ("incumbent_e8ce52c71938", "incumbent_acd864174766", "tv_remote_pickup_s100005",
              ["拿起 · 第 2 轮：遥控器侧着抓进不去",
               "超时放宽到 20 s 一局不变 → 是被挡住不是太慢 → 工程师写从上往下捏",
               "遥控器 s100005 · 左：第 1 轮后 · 右：第 2 轮采纳的版本 · Unreal 渲染"]),
    "ps_03": ("incumbent_acd864174766", "incumbent_0109bde905a8", "jam_pickup_s100030",
              ["拿起 · 第 3 轮：右手的站位全被吧台凳挡住",
               "镜像站位改用左手 → 工程师写换手回退",
               "果酱 s100030 · 左：第 2 轮后 · 右：第 3 轮采纳的版本 · Unreal 渲染"]),
}  # fmt: skip


def pose(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """A MuJoCo free camera as the rig's pose: pos, wxyz quat, -Z forward, +Y up (third_person_shot's convention)."""
    az, el = math.radians(cam["azimuth"]), math.radians(cam["elevation"])
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    pos = np.asarray(cam["lookat"], float) - cam["distance"] * fwd
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.stack([right, up, -fwd], axis=1).flatten())
    return pos, quat


def film_round(view: UnrealView, sc: V.Scene, r, p: dict, speed: int, out: Path, log=print) -> list[tuple[Path, float]]:
    """Every frame of one round, in stream capture; returns (jpg, sim time) per frame."""
    out.mkdir(parents=True, exist_ok=True)
    rig = view.film_adr[export_mjcf.RIG]
    frames, t0 = [], time.time()
    for k, (i, cam) in enumerate(RM.track(sc, r, p, speed)):
        pos, quat = pose(cam)
        view.data.qpos[rig : rig + 3], view.data.qpos[rig + 3 : rig + 7] = pos, quat
        view.push(sc.d.qpos.copy())
        reply = view._push({view.film_camera: "fresh" if k == 0 else "stream"})
        blk = reply["cameras"][view.film_camera]
        px = np.frombuffer(blk["data"], np.uint8).reshape(int(blk["height"]), int(blk["width"]), 4)[..., [2, 1, 0]]
        img = Image.fromarray(np.ascontiguousarray(px))
        path = out / f"{k:05d}.jpg"
        img.save(path, quality=92)
        frames.append((path, float(r.z["t"][i])))
        if k % 50 == 0:
            log(f"  {out.name} frame {k} ({(time.time() - t0) / (k + 1):.2f} s/frame)")
    view.data.qpos[rig : rig + 7] = view.client.model.qpos0[rig : rig + 7]
    return frames


def from_disk(frames: list[tuple[Path, float]]):
    for path, t in frames:
        yield np.asarray(Image.open(path).convert("RGB").resize(PANEL, Image.LANCZOS)), t


def compare(sc: V.Scene, r, p: dict, speed: int, frames, out: Path, picks=(0.35, 0.7, 1.0)) -> None:
    """Unreal over MuJoCo at full size, a few moments of the round."""
    track = list(RM.track(sc, r, p, speed))
    for f in picks:
        k = min(len(track), len(frames)) - 1 if f >= 1 else int(f * (len(frames) - 1))
        i, cam = track[k]
        sc.set(r, i)
        mj = sc.render(cam, FILM)
        ue = np.asarray(Image.open(frames[k][0]).convert("RGB"))
        img = np.vstack([V.label(ue, "Unreal", V.FG, 26), V.label(mj, "MuJoCo", V.FG, 26)])
        Image.fromarray(img).save(out / f"compare_{out.name}_{int(f * 100):03d}.jpg", quality=90)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=[*PS, *RM.STEPS])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    sc = V.Scene()
    if a.name in PS:
        before, after, label, lines = PS[a.name]
        b, aft = RM.Rd(PICK / before / label), RM.Rd(PICK / after / label)
        target = b.z[b.key][0, :3]
        az = V.pick_azimuth(sc, [b, aft], target, V.grasp_view_azimuth(aft), 1.3, -26)
        ps = [dict(lift=None, obj0=target, az1=az, dest=None, az2=None)] * 2
        speed, names = V.SPEED, ("旧版本", "新版本")
        caps = [[(names[0], V.FG), b.outcome_zh()], [(names[1], V.FG), aft.outcome_zh()]]
    else:
        fam, before, after, label, lines, fail, speed = RM.STEPS[a.name]
        if before is None:
            raise SystemExit(f"{a.name}: the seed has no trace to film (its left panel is a held still)")
        b, aft = RM.Rd(RM.DIAG[fam] / before / label), RM.Rd(RM.DIAG[fam] / after / label)
        dest = aft.z[aft.key][-1, :3] if aft.rec["ok"] else None
        ps = RM.plan(sc, [b, aft], dest)
        caps = [[("旧版本", V.FG), RM.outcome(b, fail)], [("新版本", V.FG), RM.outcome(aft, None)]]
        lines = [*lines[:-1], lines[-1] + " · Unreal 渲染"]
    view = UnrealView(str(SCENE_XML), LEVEL, repeats=8, streams=())
    try:
        view.film_setup(*FILM, 45.0)
        work = a.out / a.name
        shot = [
            film_round(view, sc, r, p, speed, work / side)
            for r, p, side in ((b, ps[0], "before"), (aft, ps[1], "after"))
        ]
    finally:
        view.close()
    RM.compose(sc, a.out / f"{a.name}.mp4", [from_disk(f) for f in shot], caps, lines, speed, PANEL, note="")
    compare(sc, aft, ps[1], speed, shot[1], work)
    print(f"wrote {a.out / a.name}.mp4", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
