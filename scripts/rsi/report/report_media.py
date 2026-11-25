"""The scene RSI progress report's videos (user 2026-10-19; docs/handoff/scene-rsi-report-20261019.md). Two kinds:

* `step` -- one start before and after an accepted change, side by side, for every accepted change of the families
  after pick_single (visualize.py made pick_single's four, outputs/scene_rsi_20261016/visual/step_NN.mp4). The two
  rounds are the incumbent's and the candidate's diagnostic rounds of that start (`<sandbox>/<loop>/diag/`). A change
  that wrote a family's first skill has no before round with a trace (the seed refused in planning), so its left panel
  is the start's first frame, held.
* `solo` -- a task no family trained, from the calibration starts re-run with --npz
  (outputs/scene_rsi_20261016/report/untrained/<sample>/rounds/), one panel; `pair` shows two starts of one task.

The rounds are replayed on the full ADT scene (visualize.Scene). The camera follows the robot and the object until
the lift, then the robot and the object's final resting place. Translucent magenta balls are the commanded wrist
positions (the ghost-gripper rule). Each video gets a <= 500 KB preview (previews/) and a poster JPG (its last frame).

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/rsi/report/report_media.py [--only on_02,u_wolf_cart] [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # scripts/rsi/report/ since TODO 36.8
sys.path[:0] = [str(HERE)]

import visualize as V  # noqa: E402

dv = V.dv
OUT = REPO / "outputs" / "scene_rsi_20261016" / "report" / "media"
RSI = Path("/workspace/code/rsi-scene")
UNTRAINED = REPO / "outputs" / "scene_rsi_20261016" / "report" / "untrained"
DIAG = {
    "on": RSI / "r1-on-a/outputs/scene_rsi_20261016/r1_on_single/diag",
    "held": RSI / "r1-held-a/outputs/scene_rsi_20261016/r1_held_single/diag",
    "in": RSI / "r1-in-a/outputs/scene_rsi_20261016/r1_in_single/diag",
    "near": RSI / "r1-near-a/outputs/scene_rsi_20261016/r1_held_near_single/diag",
    "slide": RSI / "r1-slide-a/outputs/scene_rsi_20261016/r1_slide_then_grasp/diag",
    "on2": RSI / "r2-on-a/outputs/scene_rsi_20261016/r2_on_single/diag",
}
NO_SKILL = "旧版本没有这个技能：规划直接拒绝，机器人原地不动"
# name: (family, before dir or None, after dir, start, before outcome, speed)
STEPS = {
    "on_02": ("on", None, "incumbent_49e184847690", "tv_remote_to_tv_stand_s110000", NO_SKILL, 3),
    "on_05": ("on", "incumbent_49e184847690", "candidate_9da5a5e404f4", "hand_soap_to_bathroom_sink_s110020",
              "失败：预抓位置差 22 cm，放弃", 3),
    "held_01": ("held", "incumbent_9da5a5e404f4", "incumbent_f1ad0545bdcc", "pick_candle_coffee_table_s120005",
                "失败：伸手超时，离抓取点还差 9.6 cm", 2),
    "held_04": ("held", "incumbent_f1ad0545bdcc", "candidate_498b72b75889", "pick_car_key_s120017",
                "失败：伸手超时，离抓取点还差 14 cm", 2),
    "in_00": ("in", None, "incumbent_af37bb8d307b", "coffee_can_in_trash_s130040", NO_SKILL, 3),
    "in_03": ("in", "incumbent_af37bb8d307b", "candidate_56bde6992e0a", "apple_in_black_bowl_s130001",
              "失败：规划拒绝（搬运 67 s，夹持余量不够）", 3),
    "near_00": ("near", None, "incumbent_9ff30cad950b", "bring_canister_to_island_s140036", NO_SKILL, 2),
    "near_01": ("near", "incumbent_9ff30cad950b", "candidate_00a754a08cc4", "bring_remote_to_sofa_s140005",
                "失败：沙发旁找不到能站的地方", 2),
    "slide_01": ("slide", "incumbent_bb14d5bf66d8", "incumbent_fba81e5762ec", "pick_thick_novel_s150000",
                 "失败：伸手时把书推开 5 cm", 2),
    "on2_00": ("on2", "incumbent_fba81e5762ec", "incumbent_470aff016d4c", "canister_to_island_s1110014",
               "失败：伸手时把铁罐推开 2 cm", 3),
    "on2_04": ("on2", "incumbent_470aff016d4c", "candidate_d99d433524f3", "big_white_vase_to_tv_stand_s1110007",
               "失败：规划拒绝（电视柜上找不到空位）", 3),
}  # fmt: skip
# name: (sample, start(s), speed)
SOLOS = {
    "u_wolf_cart": ("cal_r4", ["b1_put_DisplayWolf_on_UtilityCart__2_s8200041"], 3),
    "u_canister_round": ("cal_r4", ["b1_canister_to_round_table_s8200021"], 3),
    "u_soap_pick": ("cal_r2", ["b1_pick_HandSoap_1_s8000108"], 2),
    "u_candle_pick": ("cal_r3", ["b2_pick_CandlePattern02_1_s8100096"], 2),
    "u_vase_trash": ("cal_r4", ["b1_put_WhiteVaseBig_1_in_WhiteTrashBin_s8200071"], 3),
    "u_jam_stool": ("cal_r2", ["b1_put_Jam_Anon_on_BlackBarStool_B__0_s8000085"], 3),
    "u_remote_console": ("cal_r2", ["b1_put_TvRemote_1_on_ConsoleTable__1_s8000021",
                                    "b1_put_TvRemote_1_on_ConsoleTable__1_s8000057"], 4),
}  # fmt: skip


FAIL = {"b1_put_TvRemote_1_on_ConsoleTable__1_s8000057": "举起时没夹正（偏 2.7 cm），最后遥控器掉到了地上"}
ORIG: dict[Path, Path] = {}  # use_rounds: a re-run tree -> where the report first found those rounds


class Rd(V.Round):
    """A round from any directory (`base` = <dir>/<label>, no suffix)."""

    def __init__(self, base: Path):
        self.base, self.label, self.version = base, base.name, base.parent.name
        self.rec = json.loads(base.with_suffix(".json").read_text())
        self.z = dict(np.load(base.with_suffix(".npz")))
        self.key = next(k for k in self.z if k.startswith("obj_"))
        self.n = len(self.z["t"])

    def lift(self) -> int | None:
        obj = self.z[self.key][:, 2]
        up = np.nonzero(obj > obj[0] + 0.03)[0]
        return int(up[0]) if len(up) else None

    def side(self) -> str:
        i = self.lift() or self.n - 1
        d = [np.linalg.norm(self.z["wrists"][i, k] - self.z[self.key][i, :3]) for k in range(2)]
        return ("left", "right")[int(np.argmin(d))]


def wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def facing(frm, to) -> float:
    return math.degrees(math.atan2(to[1] - frm[1], to[0] - frm[0]))


def choose_az(sc: V.Scene, samples: list, target, base: float, dist=1.3, el=-26.0) -> float:
    """The azimuth near `base` from which the robot and `target` are least often hidden over the sampled ticks."""
    best = None
    for off in (0, 15, -15, 30, -30, 50, -50, 75, -75, 110, -110, 145, -145, 180):
        az, miss = base + off, 0
        for r, ticks in samples:
            for i in ticks:
                sc.set(r, i)
                p = r.z["robot"][i, :3]
                look = V.follow_lookat(p, target)
                cam = dict(lookat=look, distance=V.follow_distance(p, target, dist), azimuth=az, elevation=el)
                room = sc.clear_distance(look, az, el)
                if room < 0.85 * cam["distance"]:
                    cam["distance"] = room
                    miss += 1
                c = V.cam_pos(cam)
                miss += sc.blocked(c, p + [0, 0, 0.25]) + sc.blocked(c, np.asarray(target))
        if best is None or (miss, abs(off)) < best[0]:
            best = ((miss, abs(off)), az)
    return best[1]


def plan(sc: V.Scene, rounds: list[Rd], dest) -> list[dict]:
    """Per round: the pick target and azimuth, and (after the lift) the destination and its azimuth."""
    obj0 = rounds[-1].z[rounds[-1].key][0, :3]
    out = []
    for r in rounds:
        lift = r.lift()
        sign = 80 if r.side() == "right" else -80
        i_l = lift if lift is not None else r.n - 1
        t1 = range(max(0, i_l - 400), i_l, 40)
        az1 = choose_az(sc, [(r, t1)], obj0, facing(r.z["robot"][i_l, :2], obj0) + sign)
        az2 = None
        if lift is not None and dest is not None:
            t2 = range(lift + (r.n - lift) // 2, r.n, 60)
            az2 = choose_az(sc, [(r, t2)], dest, facing(r.z["robot"][-1, :2], dest) + sign)
        out.append(dict(lift=lift, obj0=obj0, az1=az1, dest=dest, az2=az2))
    return out


def frames(sc: V.Scene, r: Rd, p: dict, speed: int, size=V.PANEL, dist=1.3, el=-26.0):
    """One round at `speed` x real time (25 fps): the camera eases from the pick to the destination after the lift."""
    for i, cam in track(sc, r, p, speed, dist, el):
        ghosts = [r.z["ref_wrists"][i, k] for k in range(2)]
        yield sc.render(cam, size, ghosts=ghosts), float(r.z["t"][i])


def track(sc: V.Scene, r: Rd, p: dict, speed: int, dist=1.3, el=-26.0):
    """(tick, MuJoCo free camera) per frame of `frames`, with the scene set to that tick (report_unreal.py films
    the same track)."""
    look = dist_s = az = None
    for i in range(0, r.n, 2 * speed):
        sc.set(r, i)
        pel = r.z["robot"][i, :3]
        later = p["lift"] is not None and p["dest"] is not None and i >= p["lift"]
        target = p["dest"] if later else p["obj0"]
        az_t = p["az2"] if later else p["az1"]
        lk, ds = V.follow_lookat(pel, target), V.follow_distance(pel, target, dist)
        ds = min(ds, sc.clear_distance(lk, az_t if az is None else az, el))
        look = lk if look is None else look + 0.06 * (lk - look)
        dist_s = ds if dist_s is None else dist_s + 0.06 * (ds - dist_s)
        az = az_t if az is None else az + 0.04 * wrap(az_t - az)
        yield i, dict(lookat=look, distance=dist_s, azimuth=az, elevation=el)


def held(img: np.ndarray, text: str) -> np.ndarray:
    return V.label(img.copy(), text, (255, 210, 120), 20, (12, img.shape[0] - 40))


def outcome(r: Rd, fail: str | None) -> tuple[str, tuple]:
    """A round's caption. `fail` is the hand-written reason of the round as first filmed; a round re-run in a later
    scene (use_rounds) keeps it only when it failed the same way, and a success that became a failure says so."""
    if r.rec["ok"]:
        return "成功", (90, 220, 120)
    old, why = then(r), r.rec.get("plan_why") or r.rec.get("why") or ""
    if old is not None and (old["ok"] or (old.get("plan_why") or old.get("why") or "") != why):
        fail = "失败：" + V.why_zh(why)
    return flagged(r, (fail or "失败", (255, 110, 100)))


def then(r: Rd) -> dict | None:
    """The record of `r` as the report first filmed it, when `r` is a re-run (use_rounds); else None."""
    for new, old in ORIG.items():
        if r.base.is_relative_to(new):
            f = (old / r.base.relative_to(new)).with_suffix(".json")
            return json.loads(f.read_text()) if f.exists() else None
    return None


def flagged(r: Rd, cap: tuple[str, tuple]) -> tuple[str, tuple]:
    """`cap` of a re-run round that succeeded when first filmed and fails now: the walls and the furniture's upper
    parts collide since 2026-10-19 (TODO 27.13), which every such round of 2026-10-20 traced to."""
    old = then(r)
    return ("补碰撞后" + cap[0], cap[1]) if old is not None and old["ok"] and not r.rec["ok"] else cap


def compose(sc, out: Path, panels: list, caps: list, speed: int, size=V.PANEL) -> Path:
    """`panels`: per panel a frame generator or a held image, under their caption strip. The video opens on the first
    frame (no title card since user 2026-11-25) and holds the last one 2 s."""
    strip = np.hstack([dv.caption_strip(c, size[0]) for c in caps])
    W, H = size[0] * len(panels), size[1] + strip.shape[0]
    w = dv.Writer(out, W, H)
    last = [p if isinstance(p, np.ndarray) else None for p in panels]
    frame = None
    while True:
        live = False
        for k, p in enumerate(panels):
            if isinstance(p, np.ndarray):
                continue
            try:
                last[k] = next(p)
                live = True
            except StopIteration:
                pass
        if not live:
            break
        tiles = [x if isinstance(x, np.ndarray) else x[0] for x in last]
        t = max(x[1] for x in last if not isinstance(x, np.ndarray))
        frame = V.label(np.vstack([strip, np.hstack(tiles)]), f"t = {t:5.1f} s（{speed}× 速度）", V.DIM, 18,
                        (W - 250, strip.shape[0] + 8))  # fmt: skip
        w.write(frame)
    w.write(frame, int(2 * dv.FPS))
    w.close()
    Image.fromarray(frame).save(out.with_suffix(".jpg"), quality=85)
    dv.preview(out, out.parent / "previews" / out.stem)
    return out


def step(sc: V.Scene, name: str, out_dir: Path) -> Path:
    fam, before, after, label, fail, speed = STEPS[name]
    a = Rd(DIAG[fam] / after / label)
    b = Rd(DIAG[fam] / before / label) if before else None
    dest = a.z[a.key][-1, :3] if a.rec["ok"] else None
    rounds = [x for x in (b, a) if x is not None]
    ps = plan(sc, rounds, dest)
    if b is None:
        sc.set(a, 0)
        p0 = ps[0]
        lk = V.follow_lookat(a.z["robot"][0, :3], p0["obj0"])
        cam = dict(lookat=lk, distance=min(V.follow_distance(a.z["robot"][0, :3], p0["obj0"], 1.3),
                                           sc.clear_distance(lk, p0["az1"], -26)), azimuth=p0["az1"], elevation=-26)  # fmt: skip
        left = held(sc.render(cam, V.PANEL, ghosts=[a.z["ref_wrists"][0, k] for k in range(2)]), NO_SKILL)
        panels, caps = [left, frames(sc, a, ps[0], speed)], [[("旧版本", V.FG), (fail, (255, 110, 100))],
                                                             [("新版本", V.FG), outcome(a, None)]]  # fmt: skip
    else:
        panels = [frames(sc, b, ps[0], speed), frames(sc, a, ps[1], speed)]
        caps = [[("旧版本", V.FG), outcome(b, fail)], [("新版本", V.FG), outcome(a, None)]]
    return compose(sc, out_dir / f"{name}.mp4", panels, caps, speed)


def solo(sc: V.Scene, name: str, out_dir: Path) -> Path:
    sample, labels, speed = SOLOS[name]
    rs = [Rd(UNTRAINED / sample / "rounds" / lab) for lab in labels]
    size = V.PANEL if len(rs) > 1 else (960, 600)
    panels, caps = [], []
    for r in rs:
        dest = r.z[r.key][-1, :3] if r.rec["ok"] else None
        panels.append(frames(sc, r, plan(sc, [r], dest)[0], speed, size))
        why = FAIL.get(r.label) or V.why_zh(r.rec.get("plan_why") or r.rec.get("why") or "")
        caps.append([(f"起点 s{r.label.rsplit('_s', 1)[1]}", V.FG), outcome(r, "失败：" + why)])
    return compose(sc, out_dir / f"{name}.mp4", panels, caps, speed, size)


def use_rounds(name: str | None) -> None:
    """Read the videos' rounds from `report/<name>/` (report_rerun.py's tree: `diag/<step family>/`,
    `untrained/`) instead of the loops' own directories."""
    global UNTRAINED
    if not name:
        return
    base = REPO / "outputs" / "scene_rsi_20261016" / "report" / name
    for k in DIAG:
        ORIG[base / "diag" / k] = DIAG[k]
        DIAG[k] = base / "diag" / k
    ORIG[base / "untrained"] = UNTRAINED
    UNTRAINED = base / "untrained"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", default="")
    ap.add_argument(
        "--rounds", default="", help="read the rounds from report/<name>/ (report_rerun.py), e.g. rounds_20261020"
    )
    a = ap.parse_args(argv)
    use_rounds(a.rounds)
    names = [n for n in [*STEPS, *SOLOS] if not a.only or n in a.only.split(",")]
    sc = V.Scene()
    for n in names:
        (step if n in STEPS else solo)(sc, n, a.out)
        print("done", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
