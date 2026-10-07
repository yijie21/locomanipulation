"""Ask the policy, offline, what it would have done -- no sim, no stack, no ports.

    .venv (Isaac-GR00T) $ python src/locomani/pipeline/replay_obs.py --dump <dir>
    .venv (Isaac-GR00T) $ python src/locomani/pipeline/replay_obs.py --dataset <ds> --near-close

The closed-loop symptom this exists for is a PREDICTION symptom: on 94 of 99
rounds the visible200 checkpoint never commanded a gripper close, while the same
checkpoint's open loop on its own val closes within 2.3 frames (ledger bottle.md
section 18 / 18.2). A prediction symptom does not need physics to reproduce, so
the feedback loop for it should not cost a 3-hour eval on a live SONIC stack.

Two sources, deliberately the two sides of the train/deploy seam:

  --dump DIR     the npz files `vla_obs.PolicyLink.get_chunk` writes when
                 `OBS_DUMP` is set (`--set backend.obs_dump=DIR` on
                 eval_closed_loop.py). Each holds the FULL observation the
                 policy saw live AND the tok/grip it answered, so replaying it
                 both re-asks the question and checks that inference is
                 reproducible.

  --dataset DS   observations rebuilt from a LeRobot episode in the EVAL's obs
                 format -- the same keys `ObsBuilder.build()` produces, filled
                 from the parquet and the recorded mp4 instead of from the live
                 ZMQ streams. `--near-close` picks the frames around the
                 expert's own close, which is where the model MUST close.

The two together are the bisection: if the model closes on dataset-built
observations and not on live ones, the defect is in what the closed loop puts
into those keys, not in the model. Substituting one key at a time between the
two localises which key (`--substitute`).

Exit code is the verdict: 0 when the close expectation held, 1 when it did not,
so this runs unattended as a pass/fail loop.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from locomani.backends.sonic.sonic_token import GRIPPER_MAX, GRIPPER_MIN, snap  # noqa: E402
from locomani.backends.sonic.vla_obs import PolicyLink  # noqa: E402

CLOSE_THRESHOLD = 0.005  # vla_round.py: a frame below this counts as a commanded close

# The eval's observation keys. Kept here as one list so a change to
# ObsBuilder.build() that this tool does not follow fails loudly on a key diff
# rather than silently replaying a different observation than the eval sends.
STATE_KEYS = (
    "state.left_leg",
    "state.right_leg",
    "state.waist",
    "state.left_arm",
    "state.right_arm",
    "state.left_hand",
    "state.right_hand",
    "state.projected_gravity",
)
VIDEO_KEYS = ("video.ego_view", "video.right_wrist")
# Where each state group lives in the dataset's 43-wide observation.state. NOT
# the same order as the 29-wide SONIC `body` vector the live builder slices --
# the dataset interleaves the hands (left_arm, left_hand, right_arm, right_hand)
# while `body` keeps the arms adjacent. Both produce the same NAMED groups,
# which is all the model sees; see meta/modality_gr00t_sonic.json.
DATASET_STATE_SLICES = {
    "state.left_leg": (0, 6),
    "state.right_leg": (6, 12),
    "state.waist": (12, 15),
    "state.left_arm": (15, 22),
    "state.left_hand": (22, 29),
    "state.right_arm": (29, 36),
    "state.right_hand": (36, 43),
}


def load_dump(path: Path) -> tuple[dict, dict]:
    """(observation, what the policy answered live) from one obs_dump npz."""
    z = np.load(path, allow_pickle=False)
    recorded = {"tok": z["tok"], "grip": np.asarray(z["grip"]).reshape(-1)}
    obs = {}
    for k in z.files:
        if k in ("wall", "tok", "grip"):
            continue
        v = z[k]
        obs[k] = str(v) if v.dtype.kind in "US" else v
    return obs, recorded


def decode_frame(video_path: Path, frame_index: int) -> np.ndarray:
    """One RGB frame out of a dataset mp4, as the eval hands images over."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        got, bgr = cap.read()
        if not got:
            raise SystemExit(f"could not decode frame {frame_index} of {video_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def dataset_observations(ds: Path, episode: int, frames: list[int], task: str) -> list[tuple[str, dict, dict | None]]:
    """Eval-format observations rebuilt from a LeRobot episode.

    The point is that these carry the SAME keys the live builder produces, so
    the model cannot tell the two apart by shape -- only by content.
    """
    import pyarrow.parquet as pq

    pqs = (
        sorted(glob.glob(str(ds / "data/chunk-000" / f"*episode_{episode:06d}.parquet")))
        or sorted(glob.glob(str(ds / "data/chunk-000/*.parquet")))[episode : episode + 1]
    )
    if not pqs:
        raise SystemExit(f"no parquet for episode {episode} under {ds}")
    table = pq.read_table(pqs[0])
    state = np.stack(table["observation.state"].to_pylist())
    grav = np.stack(table["observation.projected_gravity"].to_pylist())
    expert_grip = np.stack(table["teleop.right_hand_joints"].to_pylist())[:, 0]

    stem = Path(pqs[0]).stem
    videos = {k: ds / f"videos/chunk-000/observation.images.{k.split('.')[-1]}/{stem}.mp4" for k in VIDEO_KEYS}

    out = []
    f32 = lambda a: np.asarray(a, dtype=np.float32).reshape(1, -1)  # noqa: E731
    for i in frames:
        obs = {k: f32(state[i, a:b]) for k, (a, b) in DATASET_STATE_SLICES.items()}
        obs["state.projected_gravity"] = f32(grav[i])
        obs["annotation.human.task_description"] = task
        for key, path in videos.items():
            obs[key] = decode_frame(path, i)[None].astype(np.uint8)
        out.append((f"ep{episode:03d}:f{i:04d}", obs, {"expert_grip": float(expert_grip[i])}))
    return out


def near_close_frames(ds: Path, episode: int, radius: int) -> list[int]:
    """Frames around the expert's own close -- where the model MUST close."""
    import pyarrow.parquet as pq

    pqs = sorted(glob.glob(str(ds / "data/chunk-000/*.parquet")))
    table = pq.read_table(pqs[episode], columns=["teleop.right_hand_joints"])
    g = np.stack(table["teleop.right_hand_joints"].to_pylist())[:, 0]
    closing = np.flatnonzero(g < CLOSE_THRESHOLD)
    if not len(closing):
        raise SystemExit(f"episode {episode} never closes -- nothing to expect")
    c = int(closing[0])
    return [i for i in range(c - radius, c + radius + 1) if 0 <= i < len(g)]


def check_keys(obs: dict) -> None:
    missing = [k for k in (*STATE_KEYS, *VIDEO_KEYS, "annotation.human.task_description") if k not in obs]
    extra = [k for k in obs if k not in (*STATE_KEYS, *VIDEO_KEYS, "annotation.human.task_description")]
    if missing or extra:
        raise SystemExit(f"observation keys do not match the eval's: missing={missing} extra={extra}")


def substitute(base: dict, donor: dict, keys: list[str]) -> dict:
    out = dict(base)
    for k in keys:
        if k not in donor:
            raise SystemExit(f"--substitute {k}: donor has no such key")
        out[k] = donor[k]
    return out


def collect_items(args) -> list[tuple[str, dict, dict]]:
    """(name, observation, extras) triples from whichever source was asked for."""
    if args.dump:
        dump = Path(args.dump)
        paths = [dump] if dump.is_file() else sorted(dump.glob("obs_*.npz"))
        if not paths:
            raise SystemExit(f"no obs_*.npz under {args.dump}")
        if args.stride > 1:
            paths = paths[:: args.stride]
        items = []
        for path in paths:
            obs, rec = load_dump(path)
            items.append((path.name, obs, {"recorded_grip_min": float(rec["grip"].min()), "_rec": rec}))
    else:
        ds = Path(args.dataset)
        if args.frames:
            frames = [int(x) for x in args.frames.split(",")]
        elif args.near_close:
            frames = near_close_frames(ds, args.episode, args.radius)
        else:
            raise SystemExit("--dataset needs --near-close or --frames")
        items = dataset_observations(ds, args.episode, frames, args.task)
    items = items[: args.limit] if args.limit else items
    if args.save_donor:
        name, obs, _ = items[0]
        Path(args.save_donor).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_donor,
            wall=0.0,
            tok=np.zeros((40, 64), np.float32),
            grip=np.zeros((40, 1), np.float32),
            **{k: (np.array(v) if isinstance(v, str) else v) for k, v in obs.items()},
        )
        print(f"donor written from {name} -> {args.save_donor}")
    return items


def get_chunk(link, obs: dict, do_snap: bool) -> tuple[np.ndarray, np.ndarray]:
    """(token[40,64], gripper[40]) -- `PolicyLink.get_chunk` without its imports.

    The real method reaches into `vla_round` for OBS_DUMP/SNAP, and `vla_round`
    imports gear_sonic, which only exists on a deploy box. Calling the wire
    endpoint here keeps this tool runnable in the Isaac-GR00T venv (the one that
    can also host the server) and makes snapping an explicit flag rather than a
    module global -- snapping only touches the TOKEN, never the gripper, so the
    close verdict is the same either way.
    """
    out = link._call("get_action_raw", {"obs": obs})
    tok = np.asarray(out["action.motion_token"], dtype=np.float64)
    grip = np.asarray(out["action.right_gripper"], dtype=np.float64).reshape(-1)
    return (snap(tok) if do_snap else tok), grip


def ask(link, name: str, obs: dict, extra: dict, sub_keys: list[str], donor: dict | None, do_snap: bool = True) -> dict:
    """One observation -> one verdict row, printed as it goes."""
    check_keys(obs)
    _, grip = get_chunk(link, substitute(obs, donor, sub_keys) if sub_keys else obs, do_snap)
    gmin = float(np.min(grip))
    did_close = gmin < CLOSE_THRESHOLD
    row = dict(
        source=name,
        grip_min=gmin,
        grip_first=float(grip[0]),
        predicted_close=bool(did_close),
        first_close_step=int(np.argmax(grip < CLOSE_THRESHOLD)) if did_close else None,
        chunk_len=int(len(grip)),
    )
    note = ""
    if "recorded_grip_min" in extra:
        # The action head is a diffusion model, so a replay is close but not
        # bit-identical to the recorded answer. What matters is whether the
        # difference could change the verdict, not whether it is zero.
        row["recorded_grip_min"] = extra["recorded_grip_min"]
        delta = float(np.max(np.abs(grip - np.asarray(extra["_rec"]["grip"]).reshape(-1))))
        row["replay_delta"] = delta
        row["verdict_stable"] = bool(delta < CLOSE_THRESHOLD)
        note = f"  live={extra['recorded_grip_min']:+.4f} delta={delta:.4f}"
        if not row["verdict_stable"]:
            note += "  <-- larger than the close threshold; the replay may not answer the same question"
    if "expert_grip" in extra:
        row["expert_grip"] = extra["expert_grip"]
        note = f"  expert={extra['expert_grip']:+.4f}"
    print(f"  {name:<28} grip_min={gmin:+.4f}  close={'YES' if did_close else 'no ':<3}{note}")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dump", help="directory of obs_dump npz files")
    src.add_argument("--dataset", help="LeRobot dataset to rebuild eval-format observations from")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--near-close", action="store_true", help="use the frames around the expert's own close")
    ap.add_argument("--radius", type=int, default=4, help="frames either side of the close (--near-close)")
    ap.add_argument("--frames", help="explicit frame indices, e.g. 100,120,140")
    ap.add_argument("--limit", type=int, default=0, help="at most N observations (0 = all)")
    ap.add_argument("--stride", type=int, default=1, help="--dump: take every Nth file")
    ap.add_argument("--save-donor", help="write the first built observation as an npz, to use as a --donor-dump")
    ap.add_argument("--task", default="pick up the water bottle")
    ap.add_argument("--substitute", default="", help="comma-separated keys to take from --donor-dump instead")
    ap.add_argument("--donor-dump", help="npz whose keys --substitute pulls in")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5590)
    ap.add_argument("--expect-close", action="store_true", help="exit 1 unless every observation predicts a close")
    ap.add_argument("--no-snap", action="store_true", help="skip FSQ snapping of the token (gripper is unaffected)")
    ap.add_argument("--json", dest="out_json")
    args = ap.parse_args(argv)

    items = collect_items(args)
    sub_keys = [k for k in args.substitute.split(",") if k]
    donor = None
    if sub_keys:
        if not args.donor_dump:
            raise SystemExit("--substitute needs --donor-dump")
        donor, _ = load_dump(Path(args.donor_dump))

    link = PolicyLink(host=args.host, port=args.port)
    if not link.ping():
        raise SystemExit(f"no policy server on {args.host}:{args.port} -- start serve_policy.py (Isaac-GR00T venv)")

    print(f"replaying {len(items)} observation(s) against {args.host}:{args.port}")
    if sub_keys:
        print(f"substituting {sub_keys} from {args.donor_dump}")
    print(f"close threshold < {CLOSE_THRESHOLD}   GRIPPER_MIN {GRIPPER_MIN:+.4f}  GRIPPER_MAX {GRIPPER_MAX:+.4f}")
    print()

    rows = [ask(link, n, o, e or {}, sub_keys, donor, not args.no_snap) for n, o, e in items]
    closed = sum(r["predicted_close"] for r in rows)

    print()
    print(f"VERDICT: {closed}/{len(rows)} observations predicted a close")
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1, default=str) + "\n")
        print(f"per-observation records -> {args.out_json}")
    if args.expect_close and closed != len(rows):
        print(f"RED: expected every observation to predict a close; {len(rows) - closed} did not")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
