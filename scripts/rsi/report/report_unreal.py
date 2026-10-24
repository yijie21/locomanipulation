"""The progress report's videos rendered in Unreal (user 2026-10-19: first one video -- is the light real enough? --
then all of them in place of the MuJoCo ones). Every video of the report (docs/handoff/scene-rsi-report-20261019.md):
pick_single's four steps (visualize.py made the MuJoCo ones, `ps_NN`), report_media.py's STEPS and its SOLOS (tasks no
family trained). Each round is replayed into the adt_apartment_mars level through UnrealView and filmed by the level's
third-person rig along the camera track the MuJoCo video took (report_media.track), from the same point: the rig
renders 1280 x 960 at the scene's fovy 45, so the frame is MuJoCo's view cut to 4:3 at the sides. (A 16:10 crop needs
the camera pulled back 1.2x, which put it behind the wall the track's clearance check had kept it off.) A one-start
solo is one 4:3 panel too, where the MuJoCo one was 16:10.

Rounds are filmed one after the other in URLab's "stream" capture (Lumen and TSR keep their history from the frame
before, as a game renders; the first frame of a round is a "fresh" capture). Filming the two rounds of a step in
alternation would make every frame a jump. A step whose old version had no skill (its seed refused in planning) shows
the start's first frame held, one fresh capture, as report_media does. Frames go to <out>/<name>/<panel>/NNNNN.jpg,
then the video <out>/<name>.mp4 (card, caption strip, panels at 800 x 600 each), its <= 500 KB preview
(<out>/previews/) and poster, plus for a step stills at full size beside MuJoCo's picture of the same state
(<out>/<name>/compare_*.jpg).

No ghost grippers: the traces carry the commanded wrist positions but no attitude, and an Unreal ghost is a whole
gripper mesh. The rig and ghosts stay parked except for the rig while filming. The editor's guard kills it when the
container is saturated (scripts/unreal/editor_guard.sh), so nothing CPU-heavy runs beside it.

Any other run: `--media <evolution_media.py --out dir>` films that directory's videos instead (steps.json, tasks.json,
tasks_final.json: the same starts, versions and cards as its MuJoCo videos; user 2026-10-23, the pick-repro run).
    bash scripts/start_unreal.sh --gpu 1
    PYTHONPATH=src:$PYTHONPATH MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 "$BRIDGE_PY" \\
        scripts/rsi/report/report_unreal.py ps_00 [on_02 u_wolf_cart ... | all] \\
        [--out outputs/scene_rsi_20261016/report/media_unreal]
    ... report_unreal.py all --media outputs/rsi_repro/visual --out outputs/rsi_repro/visual_unreal
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # scripts/rsi/report/ since TODO 36.8
sys.path[:0] = [str(HERE), str(REPO / "src")]

import evolution_media as EM  # noqa: E402
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


def held_still(view: UnrealView, sc: V.Scene, r, cam: dict, out: Path, text: str) -> np.ndarray:
    """The start's first frame, one fresh capture, labelled (a step whose old version had no skill)."""
    out.mkdir(parents=True, exist_ok=True)
    rig = view.film_adr[export_mjcf.RIG]
    sc.set(r, 0)
    pos, quat = pose(cam)
    view.data.qpos[rig : rig + 3], view.data.qpos[rig + 3 : rig + 7] = pos, quat
    view.push(sc.d.qpos.copy())
    blk = view._push({view.film_camera: "fresh"})["cameras"][view.film_camera]
    px = np.frombuffer(blk["data"], np.uint8).reshape(int(blk["height"]), int(blk["width"]), 4)[..., [2, 1, 0]]
    view.data.qpos[rig : rig + 7] = view.client.model.qpos0[rig : rig + 7]
    img = Image.fromarray(np.ascontiguousarray(px))
    img.save(out / "00000.jpg", quality=92)
    return RM.held(np.asarray(img.convert("RGB").resize(PANEL, Image.LANCZOS)), text)


def media_index(media: Path, names: list[str]) -> dict[str, dict]:
    """The rows of an evolution_media.py output directory to film, by video name (`all` = every row)."""
    rows = []
    for f in ("steps.json", "tasks.json", "tasks_final.json"):
        if (media / f).is_file():
            rows += json.loads((media / f).read_text())
    by = {Path(r["video"]).stem: r for r in rows}
    if not by:
        raise SystemExit(f"{media} has no steps.json / tasks.json / tasks_final.json: run evolution_media.py first")
    picked = list(by) if names == ["all"] else names
    bad = [n for n in picked if n not in by]
    if bad:
        raise SystemExit(f"not in {media}: {bad}; known: {sorted(by)}")
    return {n: by[n] for n in picked}


def media_job(sc: V.Scene, row: dict) -> dict:
    """A job from an evolution_media row: its panels' rounds on the row's start, cameras as report_media plans them,
    captions as evolution_media writes them, the card with " · Unreal 渲染" appended. A panel marked `held` (the old
    version had no trace) is the start's first frame, held, as the STEPS branch of `job` does."""
    label = row["label"]
    live = [p for p in row["panels"] if not p.get("held")]
    rounds = [RM.Rd(Path(p["diag"]) / label) for p in live]
    ps = RM.plan(sc, rounds, None)
    panels = [(f"panel{k}", r, p) for k, (r, p) in enumerate(zip(rounds, ps))]
    caps = [[(p["name"], V.FG), EM.outcome(r)] for p, r in zip(live, rounds)]
    lines = [*row["lines"][:-1], row["lines"][-1] + " · Unreal 渲染"]
    out = dict(panels=panels, caps=caps, lines=lines, speed=EM.SPEED, compare=rounds[-1])
    if len(live) < len(row["panels"]):
        aft, p0 = rounds[-1], ps[-1]
        r0 = aft.z["robot"][0, :3]
        lk = V.follow_lookat(r0, p0["obj0"])
        sc.set(aft, 0)
        cam = dict(lookat=lk, distance=min(V.follow_distance(r0, p0["obj0"], 1.3), sc.clear_distance(lk, p0["az1"], -26)),
                   azimuth=p0["az1"], elevation=-26)  # fmt: skip
        out["panels"] = [("before", aft, cam), *panels]
        out["caps"] = [[(row["panels"][0]["name"], V.FG), ("失败", (255, 110, 100))], *caps]
        out["held"] = RM.NO_SKILL
    return out


def job(sc: V.Scene, name: str) -> dict:
    """What one video films: panels (a round and its camera plan, or a held still), captions, card, speed."""
    if name in PS:
        before, after, label, lines = PS[name]
        b, aft = RM.Rd(PICK / before / label), RM.Rd(PICK / after / label)
        target = b.z[b.key][0, :3]
        az = V.pick_azimuth(sc, [b, aft], target, V.grasp_view_azimuth(aft), 1.3, -26)
        p = dict(lift=None, obj0=target, az1=az, dest=None, az2=None)
        caps = [
            [("旧版本", V.FG), RM.flagged(b, b.outcome_zh())],
            [("新版本", V.FG), RM.flagged(aft, aft.outcome_zh())],
        ]
        return dict(panels=[("before", b, p), ("after", aft, p)], caps=caps, lines=lines, speed=V.SPEED, compare=aft)
    if name in RM.STEPS:
        fam, before, after, label, lines, fail, speed = RM.STEPS[name]
        aft = RM.Rd(RM.DIAG[fam] / after / label)
        b = RM.Rd(RM.DIAG[fam] / before / label) if before else None
        dest = aft.z[aft.key][-1, :3] if aft.rec["ok"] else None
        ps = RM.plan(sc, [x for x in (b, aft) if x is not None], dest)
        lines = [*lines[:-1], lines[-1] + " · Unreal 渲染"]
        caps = [[("旧版本", V.FG), RM.outcome(b, fail) if b else (fail, (255, 110, 100))],
                [("新版本", V.FG), RM.outcome(aft, None)]]  # fmt: skip
        if b is None:  # the seed refused: the start's first frame, held, from the camera report_media holds it at
            p0, r0 = ps[0], aft.z["robot"][0, :3]
            lk = V.follow_lookat(r0, p0["obj0"])
            sc.set(aft, 0)
            cam = dict(lookat=lk, distance=min(V.follow_distance(r0, p0["obj0"], 1.3), sc.clear_distance(lk, p0["az1"], -26)),
                       azimuth=p0["az1"], elevation=-26)  # fmt: skip
            return dict(panels=[("before", aft, cam), ("after", aft, ps[0])], caps=caps, lines=lines, speed=speed,
                        compare=aft, held=RM.NO_SKILL)  # fmt: skip
        return dict(
            panels=[("before", b, ps[0]), ("after", aft, ps[1])], caps=caps, lines=lines, speed=speed, compare=aft
        )
    sample, labels, lines, speed = RM.SOLOS[name]
    panels, caps = [], []
    for k, lab in enumerate(labels):
        r = RM.Rd(RM.UNTRAINED / sample / "rounds" / lab)
        dest = r.z[r.key][-1, :3] if r.rec["ok"] else None
        panels.append((f"start{k}", r, RM.plan(sc, [r], dest)[0]))
        why = RM.FAIL.get(r.label) or V.why_zh(r.rec.get("plan_why") or r.rec.get("why") or "")
        caps.append([(f"起点 s{r.label.rsplit('_s', 1)[1]}", V.FG), RM.outcome(r, "失败：" + why)])
    meta = lines[1] if len(lines) > 1 else ""
    return dict(panels=panels, caps=caps, lines=[lines[0], meta, "用现在的技能目录跑的一局 · Unreal 渲染"], speed=speed,
                compare=None)  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help=f"videos, or `all`: {', '.join([*PS, *RM.STEPS, *RM.SOLOS])}")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--rounds", default="", help="read the rounds from report/<name>/ (report_rerun.py), e.g. rounds_20261020"
    )
    ap.add_argument("--media", type=Path, help="film an evolution_media.py --out directory's videos instead (any run)")
    a = ap.parse_args(argv)
    global PICK
    RM.use_rounds(a.rounds)
    if a.rounds:
        new = RM.REPO / "outputs" / "scene_rsi_20261016" / "report" / a.rounds / "pick"
        RM.ORIG[new], PICK = PICK, new
    media = media_index(a.media, a.names) if a.media else None
    names = list(media) if media else [*PS, *RM.STEPS, *RM.SOLOS] if a.names == ["all"] else a.names
    bad = [] if media else [n for n in names if n not in PS and n not in RM.STEPS and n not in RM.SOLOS]
    if bad:
        raise SystemExit(f"unknown videos: {bad}")
    sc = V.Scene()
    view = UnrealView(str(SCENE_XML), LEVEL, repeats=8, streams=())
    try:
        view.film_setup(*FILM, 45.0)
        for name in names:
            t0 = time.time()
            j = media_job(sc, media[name]) if media else job(sc, name)
            work = a.out / name
            panels = []
            for side, r, p in j["panels"]:
                if side == "before" and j.get("held"):
                    panels.append(held_still(view, sc, r, p, work / side, j["held"]))
                else:
                    panels.append(film_round(view, sc, r, p, j["speed"], work / side))
            gens = [p if isinstance(p, np.ndarray) else from_disk(p) for p in panels]
            RM.compose(sc, a.out / f"{name}.mp4", gens, j["caps"], j["lines"], j["speed"], PANEL, note="")
            if j["compare"] is not None:
                compare(sc, j["compare"], j["panels"][-1][2], j["speed"], panels[-1], work)
            print(f"wrote {a.out / name}.mp4 ({time.time() - t0:.0f} s)", flush=True)
    finally:
        view.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
