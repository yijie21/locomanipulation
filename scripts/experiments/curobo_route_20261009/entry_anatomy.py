"""Why the right fingers come closer to the table during the entry under arm_route curobo (route plan section 6).

For every round of the given collect runs that reached the entry, over its entry window (entry -> close, or the
trace's end) from the state trace: the right Dex1 fingers' closest approach to the table top, and at that frame the
right tool's height above the grip it aims at, the palm's pitch (its approach axis below level) and roll (its jaw
axis off level), the finger tips' depth below the tool, and the waist joints. Also the same quantities at the start
of the entry. Prints per-run medians and the rounds closest to the table.

    PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
        scripts/experiments/curobo_route_20261009/entry_anatomy.py NAME=outputs/runs/<run> ...
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "src"), str(Path(__file__).resolve().parent)]

import table_clearance as tc  # noqa: E402

from locomani.tasks.bottle.ready_grasp_policy import OFFSET  # noqa: E402

MODEL = mujoco.MjModel.from_xml_path(str(REPO / "outputs/scene/sim_lowcom/scene_locomani.xml"))
DATA = mujoco.MjData(MODEL)
TABLE = next(
    g for g in range(MODEL.ngeom) if "dining_table" in MODEL.body(MODEL.geom_bodyid[g]).name and MODEL.geom_type[g] == 6
)
FINGERS = [
    (g, tc.geom_samples(MODEL, g))
    for g in range(MODEL.ngeom)
    if MODEL.body(MODEL.geom_bodyid[g]).name.startswith("right_dex1_finger")
    and (MODEL.geom_contype[g] or MODEL.geom_conaffinity[g])
]
WRIST = MODEL.body("right_wrist_yaw_link").id
WAIST = [MODEL.jnt_qposadr[MODEL.joint(n).id] for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")]


def measure(q, grip):
    DATA.qpos[:] = q
    mujoco.mj_kinematics(MODEL, DATA)
    clear, lowest = np.inf, np.inf
    for g, local in FINGERS:
        pts = local @ DATA.geom_xmat[g].reshape(3, 3).T + DATA.geom_xpos[g]
        clear = min(
            clear,
            float(
                tc.box_signed_distance(
                    pts, DATA.geom_xpos[TABLE], DATA.geom_xmat[TABLE].reshape(3, 3), MODEL.geom_size[TABLE]
                ).min()
            ),
        )
        lowest = min(lowest, float(pts[:, 2].min()))
    r = DATA.xmat[WRIST].reshape(3, 3)
    tool = DATA.xpos[WRIST] + r @ OFFSET
    return dict(
        clear=clear,
        tool_above_grip=float(tool[2] - grip[2]),
        palm_pitch_deg=float(np.degrees(np.arcsin(-r[2, 0]))),  # approach axis below level: positive
        palm_roll_deg=float(np.degrees(np.arcsin(r[2, 1]))),  # jaw axis off level
        tips_below_tool=float(tool[2] - lowest),
        waist=[float(q[a]) for a in WAIST],
    )


def rounds(run: Path):
    traces = [np.load(f) for f in sorted(glob.glob(str(run / "collect/state_trace/*.npz")))]
    for manifest in sorted(glob.glob(str(run / "collect/manifest_*.jsonl"))):
        for line in Path(manifest).read_text().splitlines():
            r = json.loads(line)
            s = r["execution"]["metrics"].get("simple", {})
            phases = {p["phase"]: p["seq"] for p in s.get("phases", [])}
            if "entry" not in phases or "plan" not in s:
                continue
            z = next((t for t in traces if t["seq"][0] <= phases["entry"] <= t["seq"][-1]), None)
            if z is None:
                continue
            k0 = int(np.searchsorted(z["seq"], phases["entry"]))
            k1 = int(np.searchsorted(z["seq"], phases.get("close", z["seq"][-1])))
            grip = np.asarray((s["plan"].get("replanned") or s["plan"])["grip"])
            frames = [measure(z["qpos"][k], grip) for k in range(k0, max(k0 + 1, k1))]
            closest = min(frames, key=lambda f: f["clear"])
            yield dict(case=r["case_id"], ok=r["ok"], why=r.get("why", ""), start=frames[0], closest=closest)


def main() -> int:
    runs = dict(a.split("=", 1) for a in sys.argv[1:])
    table = {}
    for name, path in runs.items():
        rows = list(rounds(REPO / path))
        table[name] = {r["case"]: r for r in rows}
        c = [r["closest"] for r in rows]
        st = [r["start"] for r in rows]

        def med(key, src):
            return float(np.median([x[key] for x in src]))

        print(
            f"{name}: {len(rows)} entries | closest {100 * med('clear', c):.1f} cm (p10 "
            f"{100 * np.percentile([x['clear'] for x in c], 10):.1f}) | at closest: tool above grip "
            f"{100 * med('tool_above_grip', c):+.1f} cm, palm pitch {med('palm_pitch_deg', c):+.1f} deg, roll "
            f"{med('palm_roll_deg', c):+.1f} deg, tips below tool {100 * med('tips_below_tool', c):.1f} cm | at entry "
            f"start: tool above grip {100 * med('tool_above_grip', st):+.1f} cm, palm pitch "
            f"{med('palm_pitch_deg', st):+.1f}, waist yaw/roll/pitch "
            f"{np.round(np.median([x['waist'] for x in st], axis=0), 2).tolist()}"
        )
    out = REPO / "outputs/curobo_route_20261009/entry_anatomy.json"
    out.write_text(json.dumps(table, indent=1))
    worst = sorted(
        ((name, r) for name, rows in table.items() for r in rows.values()), key=lambda x: x[1]["closest"]["clear"]
    )[:8]
    for name, r in worst:
        c = r["closest"]
        print(
            f"  {name} {r['case']}: {100 * c['clear']:.1f} cm | tool above grip {100 * c['tool_above_grip']:+.1f} cm, "
            f"pitch {c['palm_pitch_deg']:+.1f}, roll {c['palm_roll_deg']:+.1f}, tips below tool "
            f"{100 * c['tips_below_tool']:.1f} cm, waist {np.round(c['waist'], 2).tolist()} | {r['why'][:45]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
