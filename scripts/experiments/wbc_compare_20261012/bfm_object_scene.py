"""wbc_compare step 1 (2026-10-12): the 65 GraspNet object scenes under the ScaleBFM runtime, SONIC line's finger law.

The SONIC-vs-ScaleBFM comparison on the 65-object grasp line (docs/handoff/wbc-compare-20261012.md) swaps only the
whole-body controller; everything the hand program touches must be the same on both. The object scenes
(outputs/graspnet_bottle_20261008/scenes/obj_XXX, make_scenes.py: object settled on the 0.62 m table, pose rebuilt at
rest) carry the same robot, Dex1 grippers, pads and actuators as the ScaleBFM bottle scene (diffed 2026-10-12: the
<actuator>, <option> and robot body blocks are identical), so `R.Sim` loads them as they are through `scene_dir` +
`object_name="water_bottle"`. What differed between the two lines and is pinned here:

- finger PD: the ScaleBFM line's yplus200 setting (kp 1.0 / kd 0.025, `sim.grip_clip` 0.6 per round) vs the SONIC
  line's deploy hand command (kp 1.5 / kd 0.1, ctrl clamp 20 -- grasp_cache.py KP / KD / FORCE_MAX, transcribed from
  base_sim.compute_hand_torques). The comparison runs the SONIC line's law on both controllers.
- pads: bottle_visual.scene.make_sim hardens the finger pads at load (solref 0.002, solimp 0.995 / 0.9999); the SONIC
  lockstep sim runs the XML as built (friction 1.2, solref 0.004, solimp 0.99 / 0.999). The comparison runs the XML;
  `verify_hand` refuses a sim whose hand is not the built one.
- jaw: open 0.0245, closed -0.02 (agent_server._hand_poses; the ScaleBFM line closes to -0.023).

`check` is the adapter's acceptance before any grasp program is ported: on every scene it teleports the robot to the
SONIC line's stance (grasp_planned STAND_FORWARD / STAND_RIGHT along the object's edge inward normal -- the bottle
line's face-on heading that run_grasp spawns with), settles R.SETTLE_S under mode 2 with the hands at the side, then
ramps the right jaw closed in the air over CLOSE_S, and reports pelvis / yaw delivery, fall, object displacement and
tilt, pad error against the hands-at-side target, jaw closed time and the finger force ceiling.

    CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa PYTHONPATH=src third_party/Isaac-GR00T/.venv/bin/python \
        scripts/experiments/wbc_compare_20261012/bfm_object_scene.py check --objects all \
        --out outputs/wbc_compare_20261012/scene_check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402

GRASPNET_ROOT = REPO / "outputs" / "graspnet_bottle_20261008"
GRASP_CACHE_PY = REPO / "scripts" / "experiments" / "graspnet_bottle_20261008" / "grasp_cache.py"
OBJECT_NAME = "water_bottle"  # make_scenes.py KEY: every object scene's scene_meta entry

# SONIC line's finger law and jaw (grasp_cache.py KP / KD / FORCE_MAX / JAW_OPEN / JAW_CLOSED / CLOSE_S; the numbers
# are re-read from that file by `assert_matches_grasp_cache` so the two copies cannot drift apart silently).
KP, KD, FORCE_MAX = 1.5, 0.1, 20.0
JAW_OPEN, JAW_CLOSED = 0.0245, -0.02
CLOSE_S = 1.0
# SONIC line's stance (grasp_planned.py STAND_FORWARD / STAND_RIGHT, rule "own_heading").
STAND_FORWARD, STAND_RIGHT = 0.38, 0.12
# the hand as built by build_sim_scene.py / make_dex1_model.py (both scene families, checked 2026-10-12)
HAND_AS_BUILT = dict(
    jaw_range=(-0.02, 0.0245),
    frcrange=20.0,
    gear=250.0,
    ctrlrange=20.0,
    pad_friction=1.2,
    pad_solref=0.004,
    pad_condim=4,
)


def finger_config(scene_dir) -> ScaleBfmBackendConfig:
    return ScaleBfmBackendConfig(
        scene_dir=str(scene_dir), object_name=OBJECT_NAME, grip_kp=KP, grip_kd=KD, grip_clip=FORCE_MAX
    )


def objects() -> dict[int, dict]:
    rows = json.loads((GRASPNET_ROOT / "objects.json").read_text())["objects"]
    return {int(r["id"]): r for r in rows}


def scene_dir(obj_id: int) -> Path:
    return REPO / objects()[obj_id]["scene"]


def assert_matches_grasp_cache() -> None:
    """The constants above are copies of grasp_cache.py's (importing it would pull grasp_planned and the SONIC stack)."""
    text = GRASP_CACHE_PY.read_text()
    law = re.search(r"^KP, KD, FORCE_MAX, GEAR = ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)", text, re.M)
    jaw = re.search(r"^JAW_OPEN, JAW_CLOSED = ([\d.]+), (-?[\d.]+)", text, re.M)
    close = re.search(r"^CLOSE_S, CONTACT_TIMEOUT_S, LIFT_M, LIFT_S, HOLD_S = ([\d.]+),", text, re.M)
    assert law and jaw and close, "grasp_cache.py no longer declares its finger law on one line"
    got = tuple(float(v) for v in law.groups()[:3]) + tuple(float(v) for v in jaw.groups()) + (float(close.group(1)),)
    want = (KP, KD, FORCE_MAX, JAW_OPEN, JAW_CLOSED, CLOSE_S)
    assert got == want, f"finger law drifted: grasp_cache.py {got} vs here {want}"


def verify_hand(sim: R.Sim) -> dict:
    """Refuse a sim whose Dex1 is not the built one (joint range, force limit, gear, pads); return what was read."""
    m = sim.m
    gj = [int(m.actuator_trnid[a, 0]) for a in sim.gact]
    read = dict(
        jaw_range=[tuple(float(v) for v in m.jnt_range[j]) for j in gj],
        frcrange=[float(m.jnt_actfrcrange[j, 1]) for j in gj],
        gear=[float(m.actuator_gear[a, 0]) for a in sim.gact],
        ctrlrange=[float(m.actuator_ctrlrange[a, 1]) for a in sim.gact],
    )
    pads = [
        g
        for g in range(m.ngeom)
        if "dex1_finger" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
        and m.geom_contype[g]
    ]
    read.update(
        pads=len(pads),
        pad_friction=[float(m.geom_friction[g, 0]) for g in pads],
        pad_solref=[float(m.geom_solref[g, 0]) for g in pads],
        pad_condim=[int(m.geom_condim[g]) for g in pads],
    )
    want = HAND_AS_BUILT
    problems = []
    if any(tuple(np.round(r, 6)) != want["jaw_range"] for r in read["jaw_range"]):
        problems.append(f"jaw range {read['jaw_range']}")
    for key in ("frcrange", "gear", "ctrlrange", "pad_friction", "pad_solref", "pad_condim"):
        if any(abs(v - want[key]) > 1e-9 for v in read[key]):
            problems.append(f"{key} {read[key]}")
    if read["pads"] != 4:
        problems.append(f"{read['pads']} pad geoms")
    if problems:
        raise RuntimeError("hand is not the built Dex1: " + "; ".join(problems))
    return read


def make_object_sim(meta: dict, obj_id: int, video: bool = False) -> R.Sim:
    sim = R.Sim(meta, video=video, backend=finger_config(scene_dir(obj_id)))
    verify_hand(sim)
    return sim


def stance(sim: R.Sim):
    """grasp_planned's stance: STAND_FORWARD behind and STAND_RIGHT beside the object along the heading, the heading
    being the inward normal of the table edge the object hugs (bottle_visual.geometry.sample_scene's edge pick)."""
    surf = sim.smeta["surfaces"]["A"]
    cx, cy = surf["xy"]
    hx, hy = surf["flat_half_extent"]
    nom = sim.crate0[:2]
    dists = {
        (0, -1): nom[0] - (cx - hx),
        (0, 1): (cx + hx) - nom[0],
        (1, -1): nom[1] - (cy - hy),
        (1, 1): (cy + hy) - nom[1],
    }
    ax, sgn = min(dists, key=dists.get)
    f = np.zeros(2)
    f[ax] = -sgn
    right = np.array([f[1], -f[0]])
    xy = nom - STAND_FORWARD * f - STAND_RIGHT * right
    return xy, float(np.arctan2(f[1], f[0])), f"{'xy'[ax]}{'+' if sgn > 0 else '-'}"


def tilt_deg(q0, q1) -> float:
    return float(np.degrees(2 * np.arccos(min(1.0, abs(float(np.asarray(q0) @ np.asarray(q1)))))))


def check_one(
    pol: R.Policy, obj_id: int, settle_s: float = R.SETTLE_S, close_s: float = CLOSE_S, hold_s: float = 0.5
) -> dict:
    t0 = time.time()
    sim = make_object_sim(pol.meta, obj_id)
    xy, yaw, edge = stance(sim)
    sim.reset(xy, yaw, 0.0)
    refs = R.Refs(pol, sim)
    pol.reset(sim.state())
    ref = dict(refs.pelvis(xy, yaw), **refs.hands_at_side(np.array([*xy, R.PELVIS_Z]), yaw))
    pad_goal = {
        sd: ref[f"{sd}_wrist_yaw_link"][0] + R.mat(ref[f"{sd}_wrist_yaw_link"][1]) @ R.OFF for sd in ("left", "right")
    }
    obj_p0, obj_q0 = sim.crate()
    n_settle, n_close = int(round(settle_s / R.DT)), int(round((close_s + hold_s) / R.DT))
    row = dict(
        id=obj_id,
        edge=edge,
        stance_xy=[round(float(v), 3) for v in xy],
        stance_yaw_deg=round(float(np.degrees(yaw)), 1),
    )
    closed_at = None
    force_max = 0.0
    fell_at = None
    for k in range(n_settle + n_close):
        t = k * R.DT
        if k >= n_settle:
            el = t - settle_s
            sim.grip_target[2:] = JAW_OPEN + min(1.0, el / close_s) * (JAW_CLOSED - JAW_OPEN)
        sim.step(pol(sim.state(), refs.futures(lambda _: ref, t)))
        force_max = max(
            force_max, float(np.abs(sim.d.qfrc_actuator[sim.gv]).max())
        )  # joint force after the +-20 N clamp
        if k >= n_settle and closed_at is None and np.abs(sim.d.qpos[sim.gq[2:]] - JAW_CLOSED).max() < 1e-3:
            closed_at = round(float(t - settle_s), 2)
        if sim.fallen():
            fell_at = round(float(t), 2)
            break
        if k == n_settle - 1:
            pp, pq = sim.body("pelvis")
            op, oq = sim.crate()
            row.update(
                pelvis_err_cm=round(100 * float(np.linalg.norm(pp[:2] - xy)), 2),
                yaw_err_deg=round(float(np.degrees(R.wrap(R.yaw_of(pq) - yaw))), 2),
                pelvis_z=round(float(pp[2]), 3),
                pad_err_cm={
                    sd: round(100 * float(np.linalg.norm(sim.pad(sd) - pad_goal[sd])), 2) for sd in ("left", "right")
                },
                object_moved_cm=round(100 * float(np.linalg.norm(op - obj_p0)), 2),
                object_tilt_deg=round(tilt_deg(obj_q0, oq), 2),
            )
    row.update(
        fell_at_s=fell_at,
        jaw_closed_at_s=closed_at,
        jaw_q_final=[round(float(v), 4) for v in sim.d.qpos[sim.gq[2:]]],
        finger_force_max_n=round(force_max, 2),
        seconds=round(time.time() - t0, 1),
    )
    return row


def summarize(rows: list[dict]) -> str:
    def col(key, sub=None):
        vals = [r[key] if sub is None else r[key][sub] for r in rows if key in r and r[key] is not None]
        return (float(np.median(vals)), float(np.max(vals))) if vals else (float("nan"), float("nan"))

    lines = [
        "# ScaleBFM runtime on the 65 object scenes, SONIC finger law (wbc_compare step 1)",
        "",
        f"- scenes {len(rows)}; fell {sum(r['fell_at_s'] is not None for r in rows)}; jaw never closed {sum(r['jaw_closed_at_s'] is None for r in rows)}",
        "- median / max after the {:.1f} s settle: pelvis xy err {:.1f} / {:.1f} cm, yaw err {:.1f} / {:.1f} deg, right pad err {:.1f} / {:.1f} cm, object moved {:.1f} / {:.1f} cm, object tilt {:.1f} / {:.1f} deg".format(
            R.SETTLE_S,
            *col("pelvis_err_cm"),
            *[abs(v) for v in col("yaw_err_deg")],
            *col("pad_err_cm", "right"),
            *col("object_moved_cm"),
            *col("object_tilt_deg"),
        ),
        "- jaw closed at median / max {:.2f} / {:.2f} s after the ramp start (ramp {:.1f} s); finger force max {:.1f} N (joint limit 20)".format(
            *col("jaw_closed_at_s"), CLOSE_S, col("finger_force_max_n")[1]
        ),
        "",
        "| id | edge | pelvis err cm | yaw err deg | pad R err cm | obj moved cm | obj tilt deg | fell s | jaw closed s | jaw q final | F max N | s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']:03d} | {r['edge']} | {r.get('pelvis_err_cm', '-')} | {r.get('yaw_err_deg', '-')} | {r.get('pad_err_cm', {}).get('right', '-')} | "
            f"{r.get('object_moved_cm', '-')} | {r.get('object_tilt_deg', '-')} | {r['fell_at_s'] if r['fell_at_s'] is not None else '-'} | "
            f"{r['jaw_closed_at_s'] if r['jaw_closed_at_s'] is not None else '-'} | {r['jaw_q_final']} | {r['finger_force_max_n']} | {r['seconds']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="stand, settle and close the jaw in the air on each object scene; write a table")
    c.add_argument("--objects", default="all", help="'all' or comma-separated ids")
    c.add_argument("--device", default="cpu")
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    assert_matches_grasp_cache()
    ids = sorted(objects()) if args.objects == "all" else [int(v) for v in args.objects.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    pol = R.Policy(device=args.device, backend=ScaleBfmBackendConfig())
    rows = []
    for obj_id in ids:
        row = check_one(pol, obj_id)
        rows.append(row)
        print(
            f"obj {obj_id:03d} {row['edge']} pelvis {row.get('pelvis_err_cm', '-')} cm yaw {row.get('yaw_err_deg', '-')} deg "
            f"padR {row.get('pad_err_cm', {}).get('right', '-')} cm obj {row.get('object_moved_cm', '-')} cm "
            f"fell {row['fell_at_s']} jaw {row['jaw_closed_at_s']} s F {row['finger_force_max_n']} N ({row['seconds']} s)",
            flush=True,
        )
        (args.out / "scene_check.json").write_text(json.dumps(rows, indent=1))
    (args.out / "scene_check.md").write_text(summarize(rows))
    print(f"wrote {args.out / 'scene_check.md'}")


if __name__ == "__main__":
    main()
