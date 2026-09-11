#!/usr/bin/env python3
"""Two controls on ledger bottle.md §18.4's claim that the wrist camera triggers the descent.

§18.4 showed that substituting `video.right_wrist` into a stuck live observation
flips the predicted token from rising to descending. Two things were NOT shown,
and this script shows them.

(a) IS IT THE BOTTLE, or just an in-distribution wrist image?
    The donors used in §18.4 differed from the live frame in several ways at once
    (bottle framed between the jaws, a different arm pose, a different episode).
    Two controls separate them:
      * approach-pose donors (close-150/-120/-90) have essentially the LIVE camera
        framing and NO bottle in view at all -- if the trigger is the bottle they
        must not fire;
      * an ABLATED donor: the close-60 frame with the bottle inpainted out and
        everything else byte-identical. This isolates the bottle pixels from the
        camera pose completely.

(b) IS IT ONLY THE WRIST IMAGE?
    §18.4's per-key sweep measured the GRIPPER for all 13 keys but the TOKEN only
    for the wrist image. This re-runs the whole sweep measuring the descent.

Both read out the same way: nearest-neighbour the predicted token against a
training token bank keyed on wrist height, and compare `ahead` against `now`.

Usage (Isaac-GR00T venv, serve_policy listening on :5590):
  python3 scripts/experiments/visible200_trigger_controls_20260911.py --donor-dir DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from locomani.backends.sonic.vla_obs import PolicyLink  # noqa: E402
from locomani.pipeline.replay_obs import STATE_KEYS, VIDEO_KEYS, get_chunk, load_dump  # noqa: E402

_spec = importlib.util.spec_from_file_location("rt", REPO / "scripts/experiments/visible200_reach_trigger_20260911.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)

CLOSE_THRESHOLD = 0.005
REPEATS = 8  # the action head is a diffusion model; one sample cannot separate a 0.005 effect
ALL_KEYS = (*STATE_KEYS, *VIDEO_KEYS, "annotation.human.task_description")


def verdict(link, Tn, Z, G, obs, repeats: int = REPEATS) -> dict:
    """Mean and sd of (ahead - now) implied wrist height, over `repeats` samples.

    A SINGLE sample cannot support a claim here. Measured 2026-09-11: the
    sampling sd of this statistic ranges from 0.002 to 0.017 depending on the
    observation, while the single-shot differences that ledger §18.4 read as
    "the wrist camera triggers the descent" were 0.005-0.007. That claim was
    under-powered and did not survive this control.
    """
    deltas, grips = [], []
    for _ in range(repeats):
        tok, grip = get_chunk(link, obs, do_snap=False)
        nn = rt.describe(Tn, Z, G, tok)
        deltas.append(nn["ahead"]["wrist_z"] - nn["now"]["wrist_z"])
        grips.append(float(grip.min()))
    d, g = np.array(deltas), np.array(grips)
    return dict(
        delta_mean=float(d.mean()),
        delta_sd=float(d.std()),
        grip_mean=float(g.mean()),
        grip_sd=float(g.std()),
        closes=bool(g.mean() < CLOSE_THRESHOLD),
        n=repeats,
    )


def significant(v: dict, baseline: dict) -> bool:
    """A shift counts only if it clears 3 pooled sd of the two measurements."""
    pooled = float(np.hypot(v["delta_sd"], baseline["delta_sd"])) or 1e-9
    return (baseline["delta_mean"] - v["delta_mean"]) > 3 * pooled


def show(label: str, v: dict, baseline: dict | None = None) -> None:
    tag = ""
    if baseline is not None:
        tag = "DESCENDS vs live" if significant(v, baseline) else "no significant shift"
    print(
        f"  {label:<34} d(wrist_z) {v['delta_mean']:+.4f} +-{v['delta_sd']:.4f}  {tag:<20}"
        f"  grip {v['grip_mean']:+.4f} {'CLOSE' if v['closes'] else ''}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor-dir", required=True, help="directory holding donor_f*.npz")
    ap.add_argument("--dumps", default=str(REPO / "outputs/experiments/visible200_obsdump_20260911/obs"))
    ap.add_argument("--bases", default="obs_00130,obs_00468,obs_00700")
    ap.add_argument("--dataset", default=str(REPO / "outputs/experiments/visible200_20260910/train_dataset"))
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--port", type=int, default=5590)
    ap.add_argument("--json", dest="out_json")
    args = ap.parse_args(argv)

    Tn, _, Z, G = rt.token_bank(Path(args.dataset), args.episodes)
    link = PolicyLink(port=args.port)
    if not link.ping():
        raise SystemExit(f"no policy server on :{args.port}")
    donors = Path(args.donor_dir)
    bases = [(n, load_dump(Path(args.dumps) / f"{n}.npz")[0]) for n in args.bases.split(",")]
    out = {"a": [], "b": []}

    # ---- (a) which wrist images fire the descent? -------------------------
    cases = [
        ("f121 approach-150, NO bottle", "donor_f121.npz"),
        ("f151 approach-120, NO bottle", "donor_f151.npz"),
        ("f181 approach -90, NO bottle", "donor_f181.npz"),
        ("f211 wind-up -60, bottle", "donor_f211.npz"),
        ("f211 wind-up -60, BOTTLE REMOVED", "donor_f211_nobottle.npz"),
        ("f226 descent -45, bottle", "donor_f226.npz"),
    ]
    print("(a) donor wrist image -> does the model command the descent?")
    print("    base = a stuck live at-the-table observation; ONLY video.right_wrist is replaced\n")
    for name, base in bases:
        print(f"  --- base {name} ---")
        live = verdict(link, Tn, Z, G, base)
        show("live (no substitution)", live)
        for label, fn in cases:
            path = donors / fn
            if not path.exists():
                print(f"  {label:<34} (missing {fn})")
                continue
            donor, _ = load_dump(path)
            swapped = dict(base, **{"video.right_wrist": donor["video.right_wrist"]})
            v = verdict(link, Tn, Z, G, swapped)
            show(label, v, live)
            out["a"].append(dict(base=name, donor=label, significant=significant(v, live), **v))
        print()

    # ---- (b) the full per-key sweep, measured on the TOKEN -----------------
    print("(b) per-key substitution, measured on the descent (not just the gripper)")
    print("    donor = f226 (a frame the model definitely acts on)\n")
    donor, _ = load_dump(donors / "donor_f226.npz")
    for name, base in bases[:2]:
        print(f"  --- base {name} ---")
        live = verdict(link, Tn, Z, G, base)
        show("live (no substitution)", live)
        for key in ALL_KEYS:
            swapped = dict(base, **{key: donor[key]})
            v = verdict(link, Tn, Z, G, swapped)
            show(key, v, live)
            out["b"].append(dict(base=name, key=key, significant=significant(v, live), **v))
        print()

    fired = {r["donor"] for r in out["a"] if r["significant"]}
    print(f"(a) donors that fired the descent on at least one base: {sorted(fired) or 'none'}")
    keys = {r["key"] for r in out["b"] if r["significant"]}
    print(f"(b) keys that fired the descent on at least one base:   {sorted(keys) or 'none'}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=1) + "\n")
        print(f"-> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
