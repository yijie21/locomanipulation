"""Paired fixed-observation diagnostic; never sends robot actions.

Reports one-step decoder proxies under recorded history, NOT closed-loop
success. Near-table observations are from the known non-toppled stall run.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from locomani import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-snap", action="store_true", help="diagnose raw outputs; default matches deployed FSQ snap")
    # The ACTIVE checkpoint (v1.1 since 2026-09-11). A capture recorded under
    # another one must pass --onnx for ITS decoder: the token is checkpoint-private.
    ap.add_argument("--onnx", default=paths.sonic_policy_file("model_decoder.onnx"))
    args = ap.parse_args()
    if args.out.exists():
        ap.error("refusing to overwrite probe")
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "configs"))
    import gr00t_sonic_2cam
    import onnxruntime as ort
    import torch
    from scipy.spatial.transform import Rotation

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.utils import parse_observation_gr00t
    from gr00t.eval.open_loop_eval import parse_action_gr00t
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from locomani.backends.sonic.sonic_decoder import ACTION_SCALE, DEFAULT_ANGLES, SonicDecoder
    from locomani.backends.sonic.sonic_decoder import MUJOCO_TO_ISAACLAB as ORDER
    from locomani.backends.sonic.sonic_token import snap

    tag = EmbodimentTag.UNITREE_G1_SONIC
    cfg = MODALITY_CONFIGS[tag.value]
    capture = args.capture
    events = json.loads((capture / "intent_rows.json").read_text())
    topple = json.loads((capture / "topple_audit.json").read_text())
    if topple[2]["topple_s"] is not None:
        raise ValueError("selected stall run toppled")
    near = [r for r in events if r["run"] == 2 and r["t"] > 15 and r["distance"] < 0.45 and r["pelvis_z"] > 0.72]
    walk = [r for r in events if r["distance"] > 0.8 and r["t"] < 10]
    selections = []
    for name, rs, n in [("stall", near, 30), ("walking_control", walk, 20)]:
        selections += [(name, rs[i]) for i in np.unique(np.linspace(0, len(rs) - 1, min(n, len(rs))).astype(int))]
    debug = [json.loads(s) for s in (capture / "debug.jsonl").open()]
    wall = np.array([r["wall"] for r in debug])
    states = [r["state"] for r in debug]
    ar = lambda k: np.array([r[k] for r in states])
    idx = ar("index")
    a = ar("last_action")
    tokens = ar("token_state")
    gravity = (
        Rotation.from_quat(ar("base_quat"), scalar_first=True).inv().apply(np.tile([0.0, 0.0, -1.0], (len(states), 1)))
    )
    pre = dict(
        q=(ar("body_q") - DEFAULT_ANGLES)[:, ORDER],
        dq=ar("body_dq")[:, ORDER],
        a=((a - DEFAULT_ANGLES) / ACTION_SCALE)[:, ORDER],
        omega=ar("base_ang_vel"),
        g=gravity,
    )
    dec = SonicDecoder.__new__(SonicDecoder)
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 1
    opt.inter_op_num_threads = 1
    dec.sess = ort.InferenceSession(args.onnx, sess_options=opt, providers=["CPUExecutionProvider"])
    dec.iname = dec.sess.get_inputs()[0].name
    valid = [i for i in range(12, len(states) - 4, 13) if np.all(np.diff(idx[i - 10 : i + 4]) == 1)]
    pred = dec.joint_targets(np.array([dec.obs_at(pre, i, tokens[i]) for i in valid]))
    calibration = float(np.degrees(abs(pred - a[np.array(valid) + 1])).mean())
    if not calibration < 0.1:
        raise ValueError("decoder calibration failed")
    results = []
    for model in args.models:
        policy = Gr00tPolicy(model_path=model, embodiment_tag=tag, device="cuda")
        policy.model.action_head.num_inference_timesteps = 4
        rows = []
        for group, r in selections:
            with np.load(capture / "chunks" / r["file"]) as z:
                ts = float(z["started"])
                i = int(np.searchsorted(wall, ts) - 1)
                if i < 10 or not np.all(np.diff(idx[i - 9 : i + 1]) == 1):
                    raise ValueError("missing history")
                obs = {
                    k: (z[k].item() if z[k].ndim == 0 else z[k])
                    for k in z.files
                    if k.startswith(("state.", "video.", "annotation."))
                }
            torch.manual_seed(10000 + int(Path(r["file"]).stem))
            torch.cuda.manual_seed_all(10000 + int(Path(r["file"]).stem))
            action, _ = policy.get_action(parse_observation_gr00t(obs, cfg))
            action = parse_action_gr00t(action)
            tok = np.asarray(action["action.motion_token"])
            if not args.no_snap:
                tok = snap(tok)
            grip = np.asarray(action["action.right_gripper"]).reshape(-1)
            q = dec.joint_targets(np.array([dec.obs_at(pre, i, t) for t in tok]))
            knee = q[:, [3, 9]].mean(axis=1)
            rise = np.flatnonzero(knee > a[i, [3, 9]].mean() + 0.25)
            rows.append(
                dict(
                    group=group,
                    file=r["file"],
                    first_knee=int(rise[0]) if len(rise) else None,
                    prefix_knee_delta=float(knee[:10].max() - a[i, [3, 9]].mean()),
                    prefix_knee_mean_delta=float(knee[:10].mean() - a[i, [3, 9]].mean()),
                    knee_delta=(knee - a[i, [3, 9]].mean()).tolist(),
                    close_prefix=bool(np.any(grip[:10] < 0.005)),
                )
            )
        summary = {
            g: dict(
                n=sum(r["group"] == g for r in rows),
                knee_prefix=sum(r["group"] == g and r["first_knee"] is not None and r["first_knee"] < 10 for r in rows),
                knee_any=sum(r["group"] == g and r["first_knee"] is not None for r in rows),
                close_prefix=sum(r["group"] == g and r["close_prefix"] for r in rows),
            )
            for g in ["stall", "walking_control"]
        }
        results.append(dict(model=model, summary=summary, rows=rows))
        print(json.dumps(results[-1]["summary"]), flush=True)
        del policy
        torch.cuda.empty_cache()
    args.out.write_text(
        json.dumps(
            dict(
                calibration_deg=calibration,
                snap=not args.no_snap,
                results=results,
                scope="fixed recorded observations, counterfactual decoder proxy, no actions sent",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
