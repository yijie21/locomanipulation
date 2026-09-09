"""Decode captured plans at their actual WBC observation, with calibration.

Future tokens are counterfactual one-step commands at the SAME observation,
not a physics rollout. First reconstruct actual commands and check against
the next debug frame. This uses recorded velocities, not finite differences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation
from sonic_decoder import ACTION_SCALE, DEFAULT_ANGLES, SonicDecoder
from sonic_decoder import MUJOCO_TO_ISAACLAB as ORDER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--replan", type=float, default=0.4)
    ap.add_argument(
        "--onnx", default="/workspace/deps/gear_sonic_deploy_ours_lockstep_nvdef/policy/release/model_decoder.onnx"
    )
    args = ap.parse_args()
    p = Path(args.capture)
    records = [json.loads(l) for l in (p / "debug.jsonl").open()]
    states = [r["state"] for r in records]
    wall = np.array([r["wall"] for r in records])
    idx = np.array([r["index"] for r in states])
    array = lambda k: np.array([r[k] for r in states])
    q = array("body_q")
    a = array("last_action")
    tokens = array("token_state")
    gravity = (
        Rotation.from_quat(array("base_quat"), scalar_first=True)
        .inv()
        .apply(np.tile([0.0, 0.0, -1.0], (len(states), 1)))
    )
    pre = {
        "q": (q - DEFAULT_ANGLES)[:, ORDER],
        "dq": array("body_dq")[:, ORDER],
        "a": ((a - DEFAULT_ANGLES) / ACTION_SCALE)[:, ORDER],
        "omega": array("base_ang_vel"),
        "g": gravity,
    }
    decoder = SonicDecoder.__new__(SonicDecoder)
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 1
    opt.inter_op_num_threads = 1
    decoder.sess = ort.InferenceSession(args.onnx, sess_options=opt, providers=["CPUExecutionProvider"])
    decoder.iname = decoder.sess.get_inputs()[0].name
    valid = np.array([i for i in range(12, len(states) - 4, 13) if np.all(np.diff(idx[i - 10 : i + 4]) == 1)])
    if not len(valid):
        raise RuntimeError("No contiguous WBC history for calibration")
    predicted = decoder.joint_targets(np.array([decoder.obs_at(pre, i, tokens[i]) for i in valid]))
    calibration = {str(k): float(np.degrees(abs(predicted - a[valid + k])).mean()) for k in [-1, 0, 1, 2]}
    if calibration["1"] > 0.1:
        raise RuntimeError(f"Decoder calibration failed: {calibration}")
    priv = [json.loads(l) for l in (p / "privileged.jsonl").open()]
    pw = np.array([r["wall"] for r in priv])
    result = []
    matrix = []
    last = -1e9
    run = -1
    missing_history = 0
    for f in sorted((p / "chunks").glob("*.npz")):
        with np.load(f) as c:
            ts = float(c["started"])
            ended = float(c["ended"])
            if ts - last > 5:
                run += 1
                t0 = ts
            last = ts
            i = int(np.searchsorted(wall, ts) - 1)
            if i < 10 or not np.all(np.diff(idx[i - 9 : i + 1]) == 1):
                missing_history += 1
                continue
            commands = decoder.joint_targets(np.array([decoder.obs_at(pre, i, t) for t in c["token"]]))
            knees = commands[:, [3, 9]].mean(axis=1)
            current = a[i, [3, 9]].mean()
            rise = np.flatnonzero(knees > current + 0.25)
            st = priv[max(0, np.searchsorted(pw, ts) - 1)]["state"]
            pel = np.array(st["pelvis"]["pos"])
            obj = np.array(st["objects"]["water_bottle"]["pos"])
            g = c["grip"]
            span = min(len(knees), int(round(args.replan * 25)))
            result.append(
                dict(
                    run=run,
                    t=ts - t0,
                    wall=ts,
                    latency=ended - ts,
                    distance=float(np.linalg.norm(obj[:2] - pel[:2])),
                    pelvis_z=float(pel[2]),
                    knee_current=float(current),
                    knee_future_max=float(knees.max()),
                    knee_prefix_max=float(knees[:span].max()),
                    first_knee_increase=int(rise[0]) if len(rise) else None,
                    grip_min=float(g.min()),
                    grip_prefix_min=float(g[:span].min()),
                    file=f.name,
                )
            )
            matrix.append(knees)
    summary = {
        "calibration_mean_deg": calibration,
        "calibration_samples": len(valid),
        "missing_chunk_history": missing_history,
        "runs": [],
    }
    for r in range(run + 1):
        selected = [
            x for x in result if x["run"] == r and x["t"] > 15 and x["distance"] < 0.45 and x["pelvis_z"] > 0.73
        ]
        stats = (
            {
                k: np.percentile([x[k] for x in selected], [10, 50, 90]).tolist()
                for k in ["knee_current", "knee_future_max", "knee_prefix_max", "latency"]
            }
            if selected
            else {}
        )
        first = [x["first_knee_increase"] for x in selected if x["first_knee_increase"] is not None]
        summary["runs"].append(
            dict(
                run=r,
                upright_near_after15s=len(selected),
                with_future_knee_increase=len(first),
                first_increase_p10_p50_p90=np.percentile(first, [10, 50, 90]).tolist() if first else [],
                any_close=sum(x["grip_min"] < 0.005 for x in selected),
                percentiles=stats,
            )
        )
    (p / "intent_summary.json").write_text(json.dumps(summary, indent=2))
    (p / "intent_rows.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        p / "intent_matrix.npz",
        knee=matrix,
        wall=[x["wall"] for x in result],
        run=[x["run"] for x in result],
        t=[x["t"] for x in result],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
