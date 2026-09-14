"""Inspect a recorded bottle run's action contract and timing without altering data.

Run with the data-collection environment. Receipts measure publisher/receiver
clocks; they cannot certify exposure-time alignment or VLA learnability.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from locomani.backends.sonic.sonic_token import expand_hands, snap, split_action
from locomani.schema import SONIC_BOTTLE


def runs(mask, dt):
    """Durations of contiguous true intervals on a uniform recorded timeline."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return ((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) * dt).tolist()


def action_contract(token, left, right):
    token, left, right = map(np.asarray, (token, left, right))
    if token.ndim != 2 or token.shape[1] != 64 or left.shape != (len(token), 7) or right.shape != left.shape:
        raise ValueError("recording must contain token[64] and both hand[7] fields")
    if not all(np.isfinite(a).all() for a in (token, left, right)) or not len(token):
        raise ValueError("action fields must be nonempty and finite")
    selected = np.concatenate([token, right[:, :1]], axis=1)
    decoded_token, gripper = split_action(selected)
    hands = np.asarray([expand_hands(g) for g in gripper])
    token_error = float(np.max(abs(token - decoded_token)))
    hand_error = float(max(np.max(abs(left - hands[:, 0])), np.max(abs(right - hands[:, 1]))))
    return dict(
        model_action_width=SONIC_BOTTLE.dimension("action"),
        model_state_width=SONIC_BOTTLE.dimension("state"),
        cameras=list(SONIC_BOTTLE.video_keys),
        token_grid_error=float(np.max(abs(token - snap(token)))),
        token_wire_error=token_error,
        hand_wire_error=hand_error,
        passed=token_error < 1e-7 and hand_error < 1e-7,
    )


def consumed_hand_contract(left, right, receipts):
    """Check supervision against commands consumed by the deployment, row for row."""
    if len(receipts) != len(left) or not len(left):
        return dict(verified=False, reason="missing or unmatched source receipts")
    expected = [r.get("consumed_hand_action", {}) for r in receipts]
    if any(set(r) != {"left", "right"} for r in expected):
        return dict(verified=False, reason="recording predates consumed-hand receipts")
    errors = []
    for side, values in (("left", left), ("right", right)):
        source = np.asarray([r[side] for r in expected])
        if source.shape != np.asarray(values).shape or not np.isfinite(source).all():
            return dict(verified=False, reason="invalid consumed-hand receipts")
        errors.append(float(np.max(abs(source - values))))
    return dict(verified=max(errors) < 1e-7, max_error=max(errors))


def episode_metrics(table, row, receipts):
    def array(key):
        return np.asarray(table[key].to_pylist())

    token, left, right = map(array, ("action.motion_token", "teleop.left_hand_joints", "teleop.right_hand_joints"))
    result = action_contract(token, left, right)
    result["consumed_hand_labels"] = consumed_hand_contract(left, right, receipts)
    timestamps = array("timestamp")
    dt = float(np.median(np.diff(timestamps)))
    state, command = array("observation.state"), array("action.wbc")
    body = np.r_[0:22, 29:36]
    result.update(
        frames=len(table),
        recording_hz=1 / dt,
        mode_frames=dict(Counter(map(int, array("teleop.stream_mode")))),
        repeated_token_longest_s=max(runs(np.all(token[1:] == token[:-1], axis=1), dt), default=0),
        measured_joint_step_p99_rad=float(np.quantile(abs(np.diff(state[:, body], axis=0)), 0.99)),
        wbc_joint_step_p99_rad=float(np.quantile(abs(np.diff(command[:, body], axis=0)), 0.99)),
    )
    # A displacement window distinguishes stationary intervals from adjacent
    # repeated receipt rows. This is a diagnostic threshold, not a success gate.
    window = max(1, round(0.2 / dt))
    velocity = np.max(abs(state[window:, body] - state[:-window, body]), axis=1) / (window * dt)
    grip_velocity = abs(right[window:, 0] - right[:-window, 0]) / (window * dt)
    quiet = (velocity < 0.08) & (grip_velocity < 0.001)
    tail = row.get("execution", {}).get("metrics", {}).get("training_tail", {}).get("end_seq")
    usable = receipts[: len(table)]
    if len(usable) == len(table):
        if tail is not None:
            keep = np.asarray([r.get("priv_seq") is not None and r["priv_seq"] <= tail for r in usable])
            quiet &= keep[window:]
        ages = [
            r["row_build_wall"] - r["image_publish_wall"][camera]
            for r in usable
            for camera in SONIC_BOTTLE.video_keys
            if camera in r["image_publish_wall"]
        ]
        result["image_publish_age_p99_s"] = float(np.quantile(ages, 0.99)) if ages else None
    result["receipt_rows_match"] = len(receipts) == len(table)
    result["longest_quiet_before_verification_s"] = max(runs(quiet, dt), default=0)
    events = row.get("execution", {}).get("events", [])
    result["control_stage_seconds"] = [
        dict(stage=a["stage"], seconds=b["elapsed_s"] - a["elapsed_s"]) for a, b in zip(events, events[1:])
    ]
    motion = row.get("execution", {}).get("metrics", {}).get("whole_body", {})
    samples = motion.get("samples", [])
    if samples:
        result["supported_reference"] = dict(
            samples=len(samples),
            max_joint_speed=float(np.max(abs(np.asarray([s["velocity"] for s in samples])))),
            max_foot_error_m=float(np.max([s["foot_error"] for s in samples])),
            max_com_shift_m=float(np.max(np.linalg.norm([s["com_shift"] for s in samples], axis=1))),
            capture=motion.get("capture"),
            loaded_upright=motion.get("loaded_upright"),
        )
    return result


def audit(run):
    import pyarrow.parquet as pq

    from locomani.pipeline.convert_to_hf_format import check_manifest, load_manifest

    run = Path(run)
    inputs = json.loads((run / "collect/inputs.json").read_text())
    if len(inputs["sessions"]) != len(inputs["manifests"]):
        raise ValueError("session/manifest pairing is incomplete")
    episodes = []
    for batch, (session, manifest) in enumerate(zip(inputs["sessions"], inputs["manifests"])):
        session = Path(session)
        rows = load_manifest(manifest)
        files = sorted((session / "data").glob("chunk-*/episode_*.parquet"))
        indices = [int(p.stem.split("_")[-1]) for p in files]
        discarded = {
            int(p.stem.split("_")[-1])
            for p in (session / "priv_seq").glob("episode_*.json")
            if json.loads(p.read_text()).get("discarded")
        }
        check_manifest(rows, indices, discarded, session.name)
        receipt_file = run / f"collect/frame_audit_{batch:02d}.jsonl"
        receipts = {}
        for line in receipt_file.read_text().splitlines():
            receipt = json.loads(line)
            receipts.setdefault(receipt["episode_index"], []).append(receipt)
        for row, index, file in zip(rows, indices, files):
            measurements = episode_metrics(pq.read_table(file), row, receipts.get(index, []))
            episodes.append(dict(case_id=row.get("case_id"), ok=row["ok"], why=row["why"], **measurements))
    return dict(
        run=str(run.resolve()),
        action_contract_passed=bool(episodes)
        and all(e["passed"] and e["receipt_rows_match"] and e["consumed_hand_labels"]["verified"] for e in episodes),
        episodes=episodes,
        limitations="Receipt clocks and kinematic diagnostics; not exact exposure alignment, dynamic replay or a VLA learning result.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.run)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"action contract: {result['action_contract_passed']}; episodes: {len(result['episodes'])}; {args.out}")
    return 0 if result["action_contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
