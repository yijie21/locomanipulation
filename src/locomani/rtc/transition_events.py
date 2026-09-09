"""Measure stage events in 20 real expert episodes and saved VLA traces.

Geometric eligibility is explicitly a proxy, not a full-body reachability
certificate. The crossed live experiments provide the execution check.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from locomani.sim.isaac.resolve_scene import _pid_for_seed  # noqa: E402


def sustained(t, mask, seconds=0.2):
    start = None
    for i, good in enumerate(mask):
        if not good:
            start = None
        elif start is None:
            start = i
        elif t[i] - t[start] >= seconds:
            return float(t[start])
    return None


def first(t, mask):
    found = np.flatnonzero(mask)
    return float(t[found[0]]) if len(found) else None


def measure(t, pelvis, obj, grip, wrist=None, cmd_squat=None):
    R = Rotation.from_quat(pelvis[:, 3:7], scalar_first=True).as_matrix()
    rel = np.einsum("nji,nj->ni", R, obj[:, :3] - pelvis[:, :3])
    az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    # Legacy full-pose depth/cone proxy, retained for historical comparison.
    # The repaired collector checks yaw-only depth and records its actual
    # decision in manifest.transition. This proxy excludes velocity,
    # collision and dynamic WBC tracking; it does not mean "can grasp".
    geometry = (rel[:, 0] >= 0.18) & (rel[:, 0] <= 0.27) & (abs(az) <= 36.5)
    z0 = np.median(pelvis[: max(1, min(10, len(pelvis))), 2])
    close = first(t, grip < 0.005)
    result = dict(
        near45=first(t, np.linalg.norm(rel[:, :2], axis=1) < 0.45),
        geometry_proxy=sustained(t, geometry),
        squat_command=first(t, cmd_squat) if cmd_squat is not None else None,
        actual_drop6cm=sustained(t, pelvis[:, 2] < z0 - 0.06),
        close=close,
        lift6cm=sustained(t, obj[:, 2] > obj[0, 2] + 0.06),
        end_azimuth=float(az[-1]),
        min_pelvis_z=float(pelvis[:, 2].min()),
        end_pelvis_z=float(pelvis[-1, 2]),
        duration=float(t[-1]),
    )
    if wrist is not None:
        result["min_wrist_object_distance"] = float(np.linalg.norm(wrist[:, :3] - rel, axis=1).min())
        # Dex1 pad centre, expressed in the same root frame as object pose.
        # This is geometric proximity, not a contact/force closure test.
        wrist_rotation = Rotation.from_quat(wrist[:, 3:7], scalar_first=True)
        centre = wrist[:, :3] + wrist_rotation.apply(np.tile([0.1515, 0, 0], (len(t), 1)))
        result["grip_centre_near4cm"] = sustained(t, np.linalg.norm(centre - rel, axis=1) < 0.04, 0.1)
    return result, rel


def experts(out):
    base = ROOT / "outputs/logs/collect_cover400"
    pids = _pid_for_seed(str(base))
    info = json.loads((ROOT / "datasets/g1_locomani_cover400_train/meta/info.json").read_text())
    rows = []
    for batch, session in enumerate(info["script_config"]["sessions"]):
        seed = str(5100 + batch)
        manifest = [json.loads(s) for s in (base / f"manifest_{seed}.jsonl").read_text().splitlines()]
        produced = [r for r in manifest if r.get("produced")]
        ep, entry = next((i, r) for i, r in enumerate(produced) if r["ok"])
        sp = base / "pose_sidecar" / f"{pids[seed]}_round_{entry['round']:03d}_attempt_{entry['attempt']}.jsonl"
        sidecar = {r["seq"]: r for r in map(json.loads, sp.read_text().splitlines()) if r.get("seq") is not None}
        session = Path(session)
        seq = json.loads((session / f"priv_seq/episode_{ep:06d}.json").read_text())["priv_seq"]
        data = pq.read_table(session / f"data/chunk-000/episode_{ep:06d}.parquet").to_pydict()
        valid = np.array([i for i, s in enumerate(seq) if s in sidecar])
        if len(valid) / len(seq) < 0.95:
            raise ValueError(f"Insufficient exact seq alignment: {seed}")
        matched = [sidecar[seq[i]] for i in valid]
        pel = np.array([r["pelvis"] for r in matched])
        obj = np.array([r["objects"]["water_bottle"] for r in matched])
        root = np.array(data["observation.root_orientation"])[valid]
        quat_error = np.minimum(
            np.linalg.norm(root - pel[:, 3:], axis=1), np.linalg.norm(root + pel[:, 3:], axis=1)
        ).mean()
        if quat_error > 0.02:
            raise ValueError(f"Wrong sidecar pair {seed}: quaternion error {quat_error}")
        t = np.array(data["timestamp"])[valid]
        t = t - t[0]
        grip = np.array(data["teleop.right_hand_joints"])[valid, 0]
        squat = np.array(data["teleop.planner_mode"])[valid] == 4
        wrist = np.array(data["observation.eef_state"])[valid, 7:14]
        m, rel = measure(t, pel, obj, grip, wrist=wrist, cmd_squat=squat)
        row = dict(
            source="expert",
            episode=f"{seed}/{ep}",
            success=True,
            exact_seq_fraction=len(valid) / len(seq),
            quaternion_error=quat_error,
            **m,
        )
        rows.append(row)
        np.savez_compressed(
            out / f"expert_{seed}.npz",
            t=t,
            pelvis=pel,
            obj=obj,
            state=np.array(data["observation.state"])[valid],
            token=np.array(data["action.motion_token"])[valid],
            wbc=np.array(data["action.wbc"])[valid],
            grip=grip,
            mode=np.array(data["teleop.planner_mode"])[valid],
            relative=rel,
            session=str(session),
            source_episode=ep,
            sidecar=str(sp),
            scene=str(base / f"scene_{seed}/scene_locomani.xml"),
        )
    return rows


def historical(arm):
    rows = []
    for f in sorted((ROOT / "outputs/eval" / arm / "traces").glob("round_*.json")):
        rec = json.loads(f.read_text())
        tr = rec.get("trace") or []
        if not tr or any(r.get("pelvis_z") is None or r.get("obj_z") is None for r in tr):
            continue
        # Historical traces do not retain full orientation: explicitly use
        # yaw-only reconstruction. This geometry is not exactly the live rule.
        pel = np.array(
            [
                [*r["pelvis_xy"], r["pelvis_z"], np.cos(r["pelvis_yaw"] / 2), 0, 0, np.sin(r["pelvis_yaw"] / 2)]
                for r in tr
            ]
        )
        obj = np.array([[*r["obj_xy"], r["obj_z"]] for r in tr])
        m, _ = measure(np.array([r["t"] for r in tr]), pel, obj, np.array([r["grip"] for r in tr]))
        rows.append(dict(source=arm, episode=f.stem, success=rec["ok"], geometry_frame="yaw_only", **m))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = experts(out)
    for arm in ["cover400_tok100", "base100"]:
        rows += historical(arm)
    (out / "events.json").write_text(json.dumps(rows, indent=2))
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (out / "events.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    for source in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == source]
        print(
            source,
            "n=",
            len(rs),
            {
                k: sum(r.get(k) is not None for r in rs)
                for k in ["geometry_proxy", "actual_drop6cm", "close", "lift6cm"]
            },
        )


if __name__ == "__main__":
    main()
