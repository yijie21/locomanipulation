#!/usr/bin/env python3
"""Is the right-wrist image what triggers the reach-down?

Ledger bottle.md 18.3 localised the 2/100 to the arm never leaving its approach
pose, and found that of all 13 observation keys only `video.right_wrist` flips
the gripper decision. 18.4 asks the follow-on question: does that image also
flip the ARM, or only the gripper?

Method, entirely offline (serve_policy on :5590, no sim, no stack):
  * take live at-the-table observations from the obs_dump run
  * query the policy as-is, and again with only `video.right_wrist` replaced by
    a training frame in which the bottle sits between the jaws
  * nearest-neighbour BOTH predicted tokens against the training token bank, and
    read off the matched frames' wrist height and gripper state

If the substituted token's neighbours move from the approach pose to the grasp
pose, the wrist image is the trigger for the whole reach -- and the policy is in
a closed-loop absorbing state: it needs to reach down to see the bottle in that
camera, and needs to see the bottle in that camera to reach down.

Usage (Isaac-GR00T venv, server already listening):
  python3 scripts/experiments/visible200_reach_trigger_20260911.py --donor DONOR.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from locomani.backends.sonic.sonic_token import FSQ_STEP, snap  # noqa: E402
from locomani.backends.sonic.vla_obs import PolicyLink  # noqa: E402
from locomani.pipeline.replay_obs import get_chunk, load_dump  # noqa: E402

CLOSE_THRESHOLD = 0.005


def token_bank(dataset: Path, episodes: int):
    """(tokens, right-wrist z in the pelvis frame, expert gripper) for training frames."""
    import pyarrow.parquet as pq

    tok, z, grip = [], [], []
    for f in sorted(glob.glob(str(dataset / "data/chunk-000/*.parquet")))[:episodes]:
        t = pq.read_table(f, columns=["action.motion_token", "observation.eef_state", "teleop.right_hand_joints"])
        tok.append(np.stack(t["action.motion_token"].to_pylist()))
        z.append(np.stack(t["observation.eef_state"].to_pylist())[:, 9])
        grip.append(np.stack(t["teleop.right_hand_joints"].to_pylist())[:, 0])
    T = np.concatenate(tok).astype(np.float32)
    return T / np.linalg.norm(T, axis=1, keepdims=True), T, np.concatenate(z), np.concatenate(grip)


def match(Tn: np.ndarray, vec: np.ndarray) -> int:
    v = vec.astype(np.float32)
    return int(np.argmax(Tn @ (v / np.linalg.norm(v))))


def to_dataset_scale(tok: np.ndarray) -> np.ndarray:
    """The dump stores the raw predicted token; the dataset stores it FSQ-snapped."""
    tok = np.asarray(tok, dtype=np.float64)
    return snap(tok) * FSQ_STEP if np.abs(tok).max() > 1.5 else tok


def describe(Tn, Z, G, tok) -> dict:
    tok = to_dataset_scale(tok)
    out = {}
    for idx, label in ((0, "now"), (len(tok) - 1, "ahead")):
        j = match(Tn, tok[idx])
        out[label] = {"wrist_z": float(Z[j]), "expert_closing": bool(G[j] < CLOSE_THRESHOLD)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=str(REPO / "outputs/experiments/visible200_obsdump_20260911/obs"))
    ap.add_argument("--donor", required=True, help="npz holding a training-frame observation (replay_obs --save-donor)")
    ap.add_argument("--dataset", default=str(REPO / "outputs/experiments/visible200_20260910/train_dataset"))
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--bases", default="obs_00130,obs_00249,obs_00468,obs_00700")
    ap.add_argument("--port", type=int, default=5590)
    ap.add_argument("--json", dest="out_json")
    args = ap.parse_args(argv)

    Tn, _, Z, G = token_bank(Path(args.dataset), args.episodes)
    print(f"training token bank: {len(Tn)} frames from {args.episodes} episodes")
    print(f"  expert wrist z: approach ~{np.median(Z):.3f} m, grasp ~{np.percentile(Z[G < CLOSE_THRESHOLD], 10):.3f} m")

    donor, _ = load_dump(Path(args.donor))
    link = PolicyLink(port=args.port)
    if not link.ping():
        raise SystemExit(f"no policy server on :{args.port}")

    rows = []
    for name in args.bases.split(","):
        obs, _ = load_dump(Path(args.dumps) / f"{name}.npz")
        swapped = dict(obs, **{"video.right_wrist": donor["video.right_wrist"]})
        for label, o in (("live", obs), ("wrist<-train", swapped)):
            tok, grip = get_chunk(link, o, do_snap=False)
            nn = describe(Tn, Z, G, tok)
            row = dict(
                base=name,
                variant=label,
                grip_min=float(grip.min()),
                closes=bool(grip.min() < CLOSE_THRESHOLD),
                nn_wrist_z_now=nn["now"]["wrist_z"],
                nn_wrist_z_ahead=nn["ahead"]["wrist_z"],
                nn_closing_ahead=nn["ahead"]["expert_closing"],
            )
            rows.append(row)
            print(
                f"  {name} {label:<13} grip_min={row['grip_min']:+.4f} close={'YES' if row['closes'] else 'no ':<3}"
                f"  nearest training pose: wrist_z now {row['nn_wrist_z_now']:.3f} -> ahead {row['nn_wrist_z_ahead']:.3f}"
                f"  (that frame closing: {row['nn_closing_ahead']})"
            )

    print()
    for label in ("live", "wrist<-train"):
        g = [r for r in rows if r["variant"] == label]
        print(
            f"  {label:<13} closes {sum(r['closes'] for r in g)}/{len(g)}   "
            f"median nearest wrist_z ahead {np.median([r['nn_wrist_z_ahead'] for r in g]):.3f} m"
        )
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rows, indent=1) + "\n")
        print(f"-> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
