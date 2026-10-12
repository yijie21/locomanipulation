"""Which change made the curobo rounds enter with a straighter right arm (route plan section 6, step 1 of the entry
investigation, 2026-10-09).

For rounds of a curobo collect run: the grasp loop's SupportedMotion is replayed offline from the state where the loop
started (the measured state at its first `reach` sample; the reference = the measured joints with the route's commanded end
for the waist and right arm), along the round's own recorded right-hand targets, lean and measured tools, until the
entry starts -- twice: with the left arm held by its joints as the curobo mode does (`loop_prior(..., True)` plus the
full blend) and with the lifted-ready settings (left hand held at its world point, weight 0.35, no blend). Prints the
commanded right elbow, shoulder pitch and waist at the entry start for both, and what the robot actually had there.

    PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
        scripts/experiments/curobo_route_20261009/entry_replay.py outputs/runs/<curobo run> [...]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from locomani.primitives.arm_ik import ArmIK  # noqa: E402
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402
from locomani.primitives.supported_motion import SupportedMotion  # noqa: E402
from locomani.tasks.bottle.ready_grasp_policy import OFFSET  # noqa: E402
from locomani.tasks.bottle.simple_policy import ROUTE_JOINTS, grasp_rotation, loop_prior  # noqa: E402
from locomani.tasks.bottle.whole_body_config import WholeBodyConfig  # noqa: E402

XML = str(REPO / "outputs/scene/sim_lowcom/scene_locomani.xml")
MODEL = mujoco.MjModel.from_xml_path(XML)
DATA = mujoco.MjData(MODEL)
JADR = [MODEL.jnt_qposadr[MODEL.joint(n).id] for n in MUJOCO_JOINT_NAMES]
PALMS = [MODEL.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")]
WATCH = {"R elbow": 25, "R sh pitch": 22, "waist yaw": 12, "waist roll": 13, "waist pitch": 14}


def tools_of(q):
    DATA.qpos[:] = q
    mujoco.mj_kinematics(MODEL, DATA)
    rots = np.array([DATA.xmat[b].reshape(3, 3) for b in PALMS])
    return np.array([DATA.xpos[b] + r @ OFFSET for b, r in zip(PALMS, rots)]), rots


def replay(ik, cfg, z, k0, k1, reference, samples, direction, hang):
    q0 = z["qpos"][k0]
    state = dict(pelvis=dict(pos=q0[:3].tolist(), quat=q0[3:7].tolist()), joints=q0[JADR].tolist())
    tools0, rots0 = tools_of(q0)
    posture, weights, hands = loop_prior(reference, reference.copy(), np.full(29, 0.04), hang)
    motion = SupportedMotion(
        ik, state, cfg, reference, q0[3:7].copy(), tool_offset=OFFSET, hand_weights=hands,
        posture_reference=posture, posture_weights=weights,
    )  # fmt: skip
    if hang:
        motion.blend_posture(1.0)
    heading = motion.body.heading
    targets = tools0.copy()
    rotations = np.array([rots0[0], grasp_rotation(direction)])
    times = np.array([s["t"] for s in samples])
    t0 = samples[0]["t"]
    pose = None
    for k in range(k0, k1):
        t = t0 + (k - k0) * 0.02
        i = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
        w = float(np.clip((t - times[i - 1]) / max(times[i] - times[i - 1], 1e-9), 0, 1))
        targets[1] = (1 - w) * np.array(samples[i - 1]["target"]) + w * np.array(samples[i]["target"])
        lean = (1 - w) * samples[i - 1]["lean"] + w * samples[i]["lean"]
        measured, mrots = tools_of(z["qpos"][k])
        torso = heading @ Rotation.from_euler("y", lean).as_matrix()
        pose = motion.step(targets, rotations, measured, 0.02, torso_rotation=torso, measured_rotations=mrots)
    return pose.joints


def main() -> int:
    ik, cfg = ArmIK(XML), WholeBodyConfig()
    rows = []
    for run in sys.argv[1:]:
        traces = [np.load(f) for f in sorted(glob.glob(f"{run}/collect/state_trace/*.npz"))]
        for manifest in sorted(glob.glob(f"{run}/collect/manifest_*.jsonl")):
            for line in Path(manifest).read_text().splitlines():
                r = json.loads(line)
                s = r["execution"]["metrics"].get("simple", {})
                phases = {p["phase"]: p["seq"] for p in s.get("phases", [])}
                route = s.get("route") or {}
                # the loop's first phase is entered without a phases row: take its first sample's seq
                reach_seq = next((p["seq"] for p in s.get("samples", []) if p["phase"] == "reach"), None)
                if reach_seq is None or "entry" not in phases or not route.get("path"):
                    continue
                z = next((t for t in traces if t["seq"][0] <= reach_seq <= t["seq"][-1]), None)
                if z is None:
                    continue
                k0 = int(np.searchsorted(z["seq"], reach_seq))
                k1 = int(np.searchsorted(z["seq"], phases["entry"]))
                samples = [p for p in s["samples"] if p["phase"] in ("reach", "settle", "entry")]
                if len(samples) < 3 or k1 <= k0:
                    continue
                reference = z["qpos"][k0][JADR].copy()
                reference[list(ROUTE_JOINTS)] = route["path"][-1]
                plan = s["plan"].get("replanned") or s["plan"]
                direction = np.asarray(plan["direction"], dtype=float)
                hang = replay(ik, cfg, z, k0, k1, reference, samples, direction, True)
                lifted = replay(ik, cfg, z, k0, k1, reference, samples, direction, False)
                actual = z["qpos"][k1][JADR]
                sag = min((p["tool"][2] - p["target"][2] for p in s["samples"] if p["phase"] == "entry"), default=0.0)
                rows.append(
                    dict(
                        case=f"{Path(run).name[-14:]} {r['case_id']}",
                        sag_cm=100 * sag,
                        **{f"hang {k}": float(hang[j]) for k, j in WATCH.items()},
                        **{f"lifted {k}": float(lifted[j]) for k, j in WATCH.items()},
                        **{f"actual {k}": float(actual[j]) for k, j in WATCH.items()},
                    )
                )
    out = REPO / "outputs/curobo_route_20261009/entry_replay.json"
    out.write_text(json.dumps(rows, indent=1))
    for k in WATCH:
        h = np.array([r[f"hang {k}"] for r in rows])
        lf = np.array([r[f"lifted {k}"] for r in rows])
        a = np.array([r[f"actual {k}"] for r in rows])
        print(
            f"{k:12s} at the entry start, median over {len(rows)} rounds: commanded with the left arm held by joints "
            f"{np.median(h):+.2f}, with the lifted-ready settings {np.median(lf):+.2f}, measured {np.median(a):+.2f}"
        )
    sagged = [r for r in rows if r["sag_cm"] < -3]
    print(f"rounds that sagged more than 3 cm in the entry: {len(sagged)}")
    for r in sagged:
        print(
            f"  {r['case']}: sag {r['sag_cm']:.1f} cm | R elbow commanded hang {r['hang R elbow']:+.2f} / lifted settings "
            f"{r['lifted R elbow']:+.2f} / measured {r['actual R elbow']:+.2f} | waist yaw {r['hang waist yaw']:+.2f} / "
            f"{r['lifted waist yaw']:+.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
