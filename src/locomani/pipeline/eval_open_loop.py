"""Open-loop evaluation for the SONIC VLA, in units that can be read.

    python src/locomani/pipeline/eval_open_loop.py \
        --model-path outputs/train/bottle200/checkpoint-5000 \
        --dataset  datasets/g1_locomani_bottle200_val \
        --out      outputs/eval/open_loop/checkpoint-5000.json

WHAT IS REPORTED, AND WHY EACH ONE

Plain MSE on this action space is close to meaningless, for two independent
reasons, so four numbers are reported instead of one:

  nMSE, token and gripper SEPARATELY
      nMSE = MSE / var(gt) = 1 - R^2, so 1.0 is "no better than predicting the
      mean" regardless of scale. Split because the action is 64 token dims and
      ONE gripper dim: pooled, the dim that decides whether the bottle is
      actually grasped contributes 1/65 of the number and vanishes.

  snapped hit rate
      The token is FSQ-quantised -- 32 grid points per dim, and this dataset
      uses about 2-5 of them per dim (sonic_token.py). In a space that
      discrete, "how often is the codeword right" is readable and an MSE is
      not. Reported per-dim and as all-64-correct.

  joint-command delta, in degrees
      Token space is a learned latent; Euclidean distance in it has no
      physical unit. Pushing prediction and ground truth through SONIC's own
      decoder with the observation held fixed converts the error into degrees
      of commanded joint motion (sonic_decoder.py). This is the headline
      number: it is comparable with the 0.2-0.9 deg that token replay already
      reproduces.

  gripper close-frame error
      Grasp success depends on WHEN the jaw closes, and no MSE shows that. The
      first frame the predicted gripper crosses its own midpoint, against the
      same for ground truth.

EVERY NUMBER COMES WITH A PERSISTENCE BASELINE

30.6% of frames carry a token identical to the previous frame; the data is
strongly autocorrelated. A model that learned nothing but "hold the last
action" scores well on all of the above. The baseline here is exactly that:
within each execution block, emit the ground-truth action from the block's
first frame. A metric that does not beat it has measured autocorrelation.

FOUR EXECUTION HORIZONS

Each metric is computed at execution_horizon in {1, 8, 16, 40}: how many steps
of a predicted 40-step chunk are executed before re-planning from a fresh
observation. horizon=1 is single-step accuracy, horizon=40 trusts the whole
1.33 s chunk. The growth curve is the direct evidence for how far the chunk
can be trusted -- and closed-loop has to pick this number anyway, so it is
cheaper to measure it here than to guess it there.

A NOTE ON THE DEGREE METRIC'S ACCURACY

sonic_decoder rebuilds the deploy's 994-dim observation, whose history is 10
frames at the 50 Hz CONTROL rate. This dataset is resampled to 30 Hz, so the
history spacing is 33 ms rather than 20 ms. The comparison still holds -- both
decoder calls use the SAME reconstructed observation, so the spacing error
enters both terms and cancels -- but the operating point shifts. Rather than
assert that is harmless, the script measures it: `degrees_noise_floor` in the
output is the mean error between the decoder's output on ground-truth tokens
and the joint command actually recorded. Differences smaller than that floor
are not meaningful.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

TOKEN_DIM = 64
HORIZONS = (1, 8, 16, 40)


# ---------------------------------------------------------------- rollout ----


def rollout(policy, loader, traj_id, embodiment_tag, horizon, parse_action):
    """Open-loop roll: re-plan every `horizon` steps from a ground-truth obs.

    Mirrors gr00t/eval/open_loop_eval.py's loop, but returns the arrays instead
    of plotting them.
    """
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t

    traj = loader[traj_id]
    T = len(traj)
    action_keys = loader.modality_configs["action"].modality_keys
    obs_cfgs = deepcopy(loader.modality_configs)
    obs_cfgs.pop("action")

    pred = []
    for t0 in range(0, T, horizon):
        dp = extract_step_data(traj, t0, obs_cfgs, embodiment_tag)
        obs = {f"state.{k}": v for k, v in dp.states.items()}
        obs.update({f"video.{k}": np.array(v) for k, v in dp.images.items()})
        for lk in loader.modality_configs["language"].modality_keys:
            obs[lk] = dp.text
        chunk, _ = policy.get_action(parse_observation_gr00t(obs, loader.modality_configs))
        chunk = parse_action(chunk)
        for j in range(horizon):
            pred.append(np.concatenate([np.atleast_1d(np.atleast_1d(chunk[f"action.{k}"])[j]) for k in action_keys]))
    pred = np.asarray(pred)[:T]

    gt = np.concatenate([np.vstack([a for a in traj[f"action.{k}"]]) for k in action_keys], axis=-1)[:T]
    return gt, pred, traj


def persistence(gt, horizon):
    """The do-nothing reference: hold the block's first ground-truth action.

    Deliberately generous -- it is handed the true action at each re-plan point,
    which the model is not. If a model cannot beat a baseline that is given the
    answer once per block and then freezes, it has not learned the dynamics,
    only that the signal barely moves.
    """
    idx = (np.arange(len(gt)) // horizon) * horizon
    return gt[idx]


# ---------------------------------------------------------------- metrics ----


def nmse(gt, pred):
    """MSE / var(gt), averaged over dims. 1.0 == predicting the mean."""
    var = gt.var(axis=0)
    ok = var > 1e-12  # constant dims carry no information
    if not ok.any():
        return float("nan")
    return float((((pred - gt) ** 2).mean(axis=0)[ok] / var[ok]).mean())


def close_frame(g, lo=None, hi=None):
    """First frame the gripper crosses the midpoint of its own range."""
    lo = np.percentile(g, 5) if lo is None else lo
    hi = np.percentile(g, 95) if hi is None else hi
    if hi - lo < 1e-6:
        return None
    below = np.flatnonzero(g < (lo + hi) / 2)
    return int(below[0]) if len(below) else None


def score(gt, pred, dec, pre, tag):
    """All four metric families for one (gt, pred) pair."""
    from locomani.backends.sonic.sonic_token import snap, token_index

    # Two action layouts share this scorer. SONIC: [64 token dims | gripper],
    # judged on the 1/16-quantised grid (snap + hit rate). Anything else (the
    # 19-d ScaleBFM `bfm` action: 18 continuous body dims | gripper) has no
    # grid, so the body block gets a plain nMSE and the token metrics are
    # omitted; the gripper and close-frame metrics below are layout-agnostic.
    sonic = gt.shape[1] == TOKEN_DIM + 1
    body = TOKEN_DIM if sonic else gt.shape[1] - 1
    g_tok, g_grip = gt[:, :body], gt[:, body]
    p_tok, p_grip = pred[:, :body], pred[:, body]
    out = {
        "nmse_gripper": nmse(g_grip[:, None], p_grip[:, None]),
        "gripper_mae": float(np.abs(p_grip - g_grip).mean()),
    }
    if sonic:
        p_snap = snap(p_tok)
        hit = token_index(p_tok) == token_index(g_tok)
        out.update(
            {
                "nmse_token": nmse(g_tok, p_snap),
                "nmse_token_unsnapped": nmse(g_tok, p_tok),
                "hit_rate_per_dim": float(hit.mean()),
                "hit_rate_all64": float(hit.all(axis=1).mean()),
            }
        )
    else:
        p_snap = p_tok
        out["nmse_body"] = nmse(g_tok, p_tok)
        out["body_mae"] = float(np.abs(p_tok - g_tok).mean())

    # --- MAGNITUDE SHRINKAGE ------------------------------------------------
    # Under-integrated flow matching does not just add error, it pulls the
    # sample toward the prior mean -- an output that is systematically SMALLER
    # than the target. nMSE cannot see that (it is symmetric in the sign of
    # the residual); a std ratio can. Ratio < 1 = the model is hedging.
    #
    # This is the offline judge for the closed-loop signature measured on
    # walk200's traces: in every `reach` failure the commanded gripper sat at
    # +0.0168, inside a band the training data occupies 3.3% of the time,
    # while the two real modes are >0.020 (53.3%) and <0.005 (43.5%). If the
    # ratio is < 1 at 4 denoising steps and climbs toward 1 with more steps,
    # the standing attractor is a sampler artefact and --denoising-steps is
    # the lever. If it is already ~1 at 4, it is not, and the lever is
    # elsewhere.
    eps = 1e-12
    r_tok = p_tok.std(axis=0) / (g_tok.std(axis=0) + eps)
    out["std_ratio_token"] = float(np.median(r_tok))
    out["std_ratio_token_mean"] = float(np.mean(r_tok))
    out["std_ratio_gripper"] = float(p_grip.std() / (g_grip.std() + eps))
    # the dead band itself, both sides, in the dataset's own units
    G_OPEN, G_SHUT = 0.020, 0.005
    for tagn, v in (("pred", p_grip), ("gt", g_grip)):
        out[f"grip_open_{tagn}"] = float((v > G_OPEN).mean())
        out[f"grip_shut_{tagn}"] = float((v < G_SHUT).mean())
        out[f"grip_deadband_{tagn}"] = float(((v >= G_SHUT) & (v <= G_OPEN)).mean())

    # close-frame timing, both sides judged against the GROUND TRUTH range so a
    # prediction that never opens/closes properly cannot flatter itself with a
    # midpoint of its own.
    lo, hi = np.percentile(g_grip, 5), np.percentile(g_grip, 95)
    cg, cp = close_frame(g_grip, lo, hi), close_frame(p_grip, lo, hi)
    out["close_frame_gt"] = cg
    out["close_frame_pred"] = cp
    out["close_frame_err"] = None if (cg is None or cp is None) else int(cp - cg)

    if sonic and dec is not None and pre is not None:
        n = min(len(g_tok), len(pre["q"]))
        fr = np.arange(n)
        d = dec.joint_delta_deg(pre, fr, g_tok[:n], p_snap[:n])
        out["deg_mean"] = float(np.abs(d).mean())
        out["deg_p95"] = float(np.percentile(np.abs(d), 95))
        out["deg_max"] = float(np.abs(d).max())
    return out


def aggregate(rows):
    """Mean over trajectories, ignoring the entries that are None."""
    keys = {k for r in rows for k in r}
    agg = {}
    for k in sorted(keys):
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals or isinstance(vals[0], str):
            continue
        agg[k] = float(np.mean(vals))
        if k == "close_frame_err":
            agg["close_frame_err_abs"] = float(np.mean(np.abs(vals)))
    return agg


# ------------------------------------------------------------------- main ----


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--modality-config", default=None, help="defaults to configs/gr00t_sonic_2cam.py next to the repo")
    ap.add_argument("--embodiment-tag", default="UNITREE_G1_SONIC")
    ap.add_argument("--traj-ids", type=int, nargs="*", default=None, help="default: every episode in the dataset")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--no-decoder", action="store_true", help="skip the degrees metric (no onnxruntime / no deploy tree)"
    )
    ap.add_argument(
        "--wandb-run",
        default=None,
        help="wandb run NAME to append these metrics to, so the val curve lands on the same axes as the train loss",
    )
    ap.add_argument("--wandb-project", default="locomani-bottle")
    ap.add_argument(
        "--step",
        type=int,
        default=None,
        help="training step these metrics belong to; inferred from a checkpoint-N path when not given",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]
    # src/gr00t is the vendored, COMMITTED copy (src/gr00t/UPSTREAM.md);
    # the third_party clone is reference only and must not shadow it
    sys.path.insert(0, str(repo / "src"))
    cfg_path = Path(args.modality_config or repo / "configs/gr00t_sonic_2cam.py")
    sys.path.insert(0, str(cfg_path.parent))
    __import__(cfg_path.stem)  # rewrites MODALITY_CONFIGS in place

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag

    # parse_action_gr00t lives in the eval module itself, not data.utils
    from gr00t.eval.open_loop_eval import parse_action_gr00t
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    tag = EmbodimentTag[args.embodiment_tag.upper()]
    cfg = MODALITY_CONFIGS[tag.value]
    loader = LeRobotEpisodeLoader(dataset_path=args.dataset, modality_configs=cfg)
    policy = Gr00tPolicy(model_path=args.model_path, embodiment_tag=tag, device="cuda")
    policy.model.action_head.num_inference_timesteps = args.denoising_steps

    traj_ids = args.traj_ids if args.traj_ids is not None else list(range(len(loader)))
    print(f"model    {args.model_path}")
    print(f"dataset  {args.dataset}  ({len(loader)} episodes, evaluating {len(traj_ids)})")
    print(f"horizons {args.horizons}   denoising steps {args.denoising_steps}")

    dec = None
    if not args.no_decoder:
        try:
            from locomani.backends.sonic.sonic_decoder import SonicDecoder

            dec = SonicDecoder(
                str(repo / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx")
            )
        except Exception as e:  # noqa: BLE001
            print(f"  (degrees metric off: {e})")

    results = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "denoising_steps": args.denoising_steps,
        "horizons": {},
    }
    floors = []

    for h in args.horizons:
        model_rows, base_rows = [], []
        for tid in traj_ids:
            gt, pred, traj = rollout(policy, loader, tid, tag, h, parse_action_gr00t)

            pre = None
            if dec is not None:
                import pandas as pd  # noqa: F401  (traj is a DataFrame)

                pre, floor = _decoder_context(dec, loader, tid, gt)
                if floor is not None:
                    floors.append(floor)

            model_rows.append(score(gt, pred, dec, pre, "model"))
            base_rows.append(score(gt, persistence(gt, h), dec, pre, "persistence"))

        results["horizons"][str(h)] = {
            "model": aggregate(model_rows),
            "persistence": aggregate(base_rows),
            "n_traj": len(traj_ids),
        }
        m, b = results["horizons"][str(h)]["model"], results["horizons"][str(h)]["persistence"]
        print(f"\nhorizon {h:2d}   {'model':>12s} {'persistence':>12s}")
        for k in (
            "nmse_token",
            "nmse_token_unsnapped",
            "nmse_body",
            "body_mae",
            "nmse_gripper",
            "hit_rate_per_dim",
            "hit_rate_all64",
            "deg_mean",
            "close_frame_err_abs",
            "std_ratio_token",
            "std_ratio_gripper",
            "grip_deadband_pred",
            "grip_deadband_gt",
        ):
            if k in m:
                print(f"  {k:22s} {m[k]:12.4f} {b.get(k, float('nan')):12.4f}")

    if floors:
        results["degrees_noise_floor"] = float(np.mean(floors))
        print(
            f"\ndegrees noise floor: {results['degrees_noise_floor']:.3f} deg "
            f"(decoder on GT tokens vs the recorded command; differences "
            f"below this are not meaningful)"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")

    if args.wandb_run:
        _log_wandb(args, results)
    return results


def _decoder_context(dec, loader, tid, gt):
    """Build the decoder's per-episode context, and measure its noise floor.

    The floor is the mean error between decoding the GROUND-TRUTH token and the
    joint command actually recorded (action.wbc). It is not a constant: this
    dataset is 30 Hz while the decoder's history is defined at the 50 Hz control
    rate, so the reconstruction is imperfect and the size of that imperfection
    is exactly what a reader needs in order to know which differences count.
    """
    import numpy as np
    import pyarrow.parquet as pq

    from locomani.backends.sonic.sonic_decoder import BODY_IDX, SonicDecoder

    info = json.loads((Path(loader.dataset_path) / "meta/info.json").read_text())
    ep = loader.episode_ids[tid] if hasattr(loader, "episode_ids") else tid
    p = Path(loader.dataset_path) / "data/chunk-000" / f"episode_{ep:06d}.parquet"
    if not p.exists():
        return None, None
    if not {"action.wbc", "action.motion_token"} <= set(pq.read_schema(p).names):
        return None, None  # not a SONIC dataset (e.g. ScaleBFM)
    t = pq.read_table(
        p,
        columns=[
            "observation.state",
            "action.wbc",
            "observation.root_orientation",
            "observation.projected_gravity",
            "action.motion_token",
        ],
    )
    S = np.stack(t["observation.state"].to_pylist())
    A = np.stack(t["action.wbc"].to_pylist())
    Q = np.stack(t["observation.root_orientation"].to_pylist())
    G = np.stack(t["observation.projected_gravity"].to_pylist())
    TOK = np.stack(t["action.motion_token"].to_pylist())
    pre = SonicDecoder.prepare(S, A, Q, G, hz=float(info.get("fps", 30)))

    n = min(len(TOK), 200)
    fr = np.arange(0, n)
    obs = np.stack([SonicDecoder.obs_at(pre, int(i), TOK[i]) for i in fr])
    floor = float(np.degrees(np.abs(dec.joint_targets(obs) - A[fr][:, BODY_IDX]).mean()))
    return pre, floor


def _log_wandb(args, results):
    """Append the val metrics to the training run, at the right step.

    Same run, not a new one: "choose the checkpoint from the validation curve"
    is only workable if that curve sits on the same x-axis as the train loss.
    """
    import re

    import wandb

    step = args.step
    if step is None:
        m = re.search(r"checkpoint-(\d+)", args.model_path)
        step = int(m.group(1)) if m else 0
    flat = {}
    for h, d in results["horizons"].items():
        for who, metrics in (("model", d["model"]), ("persist", d["persistence"])):
            for k, v in metrics.items():
                flat[f"val/h{h}/{who}/{k}"] = v
    if "degrees_noise_floor" in results:
        flat["val/degrees_noise_floor"] = results["degrees_noise_floor"]
    api_run = None
    try:
        api = wandb.Api()
        for r in api.runs(args.wandb_project, filters={"display_name": args.wandb_run}):
            api_run = r.id
            break
    except Exception as e:  # noqa: BLE001
        print(f"  (wandb lookup failed: {e})")
    run = wandb.init(project=args.wandb_project, name=args.wandb_run, id=api_run, resume="allow")
    run.log(flat, step=step)
    run.finish()
    print(f"logged {len(flat)} metrics to wandb {args.wandb_project}/{args.wandb_run} at step {step}")


if __name__ == "__main__":
    main()
