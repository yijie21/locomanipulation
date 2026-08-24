#!/usr/bin/env python3
"""Audit a converted LeRobot v2.1 dataset before anyone spends GPU hours on it.

Three passes, cheapest first, each answering a different question:

  structure   do the files agree with each other and with meta/?
  grasp       do the numbers describe a robot picking the object up?
  alignment   is row t the same instant as frame t?

The third is the one a viewer cannot answer. Scrubbing a video next to a plot
looks fine at any offset under a second, so alignment gets MEASURED here: the
wrist cameras are bolted to the arm, so per-frame image change is a proxy for
arm speed, and the arm's speed is also in the table as |d(observation.state)/dt|.
Cross-correlate the two and the lag at the peak is the misalignment, in frames.

Run it with the venv the converter runs in (it needs pyarrow, cv2, ffprobe):

    third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python \
        src/teleop_free/tools/check_dataset.py datasets/g1_locomani_pilot_can

Exits non-zero if any check fails, so it drops straight into CI or a pre-train
gate. --quick skips the video passes (structure only, a couple of seconds).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

try:
    import cv2
except ImportError:                                   # structure pass still runs
    cv2 = None

# The eight state groups and three action groups Isaac-GR00T's unitree_g1_sonic
# config selects, mirrored from gr00t/configs/data/embodiment_configs.py. It
# picks BY NAME and a name it cannot find is only a warning, so a rename here
# is a silent 64-dim action with both hands dropped -- hence checking the names
# rather than trusting them.
SONIC_STATE = ["left_leg", "right_leg", "waist", "left_arm", "right_arm",
               "left_hand", "right_hand", "projected_gravity"]
SONIC_ACTION = ["motion_token", "left_hand_joints", "right_hand_joints"]
SONIC_STATE_DIM, SONIC_ACTION_DIM = 46, 78

ARM_STATE = slice(15, 43)     # both arms + hands: the joints a wrist cam sees
MAX_LAG = 15                  # frames searched either side of zero


class Report:
    def __init__(self):
        self.passed = self.failed = 0

    def check(self, ok, msg):
        print(("  OK   " if ok else "  FAIL ") + msg)
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        return ok

    def note(self, msg):
        print("       " + msg)


def load_meta(root: Path):
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text())
    eps = [json.loads(l) for l in (meta / "episodes.jsonl").open()]
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in (meta / "tasks.jsonl").open()}
    stats = [json.loads(l) for l in (meta / "episodes_stats.jsonl").open()]
    return info, eps, tasks, stats


def ep_paths(root: Path, info: dict, ep: int):
    chunk = ep // info["chunks_size"]
    data = root / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
    vids = {k.split(".")[-1]: root / info["video_path"].format(
                episode_chunk=chunk, video_key=k, episode_index=ep)
            for k, v in info["features"].items() if v["dtype"] == "video"}
    return data, vids


def pass_structure(root, info, eps, tasks, stats, r, n_videos):
    """Do the files agree with each other?"""
    print("\n[1/3] structure")
    vid_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    fps = info["fps"]

    r.check(info["codebase_version"] == "v2.1",
            f"codebase_version = {info['codebase_version']}")
    r.check(len(eps) == info["total_episodes"],
            f"episodes.jsonl {len(eps)} == info.total_episodes {info['total_episodes']}")
    r.check(len(stats) == len(eps),
            f"episodes_stats.jsonl {len(stats)} == episodes {len(eps)}")
    r.check(len(tasks) == info["total_tasks"],
            f"tasks {len(tasks)} == info.total_tasks {info['total_tasks']}")
    r.check(len(vid_keys) * len(eps) == info["total_videos"],
            f"{len(vid_keys)} streams x {len(eps)} episodes == "
            f"info.total_videos {info['total_videos']}")

    idx = [e["episode_index"] for e in eps]
    r.check(idx == list(range(len(idx))), f"episode_index contiguous 0..{len(idx)-1}")

    total, gindex = 0, 0
    bad_len, bad_book, bad_num, missing, bad_width = [], [], [], [], []
    for e in eps:
        i = e["episode_index"]
        p, _ = ep_paths(root, info, i)
        if not p.exists():
            missing.append(p.name)
            continue
        t = pq.read_table(p)
        n = t.num_rows
        total += n
        if n != e["length"]:
            bad_len.append((i, e["length"], n))

        cols = {c: t[c].to_numpy(zero_copy_only=False) for c in
                ("index", "frame_index", "episode_index", "timestamp", "task_index")
                if c in t.column_names}
        if "frame_index" in cols and not np.array_equal(cols["frame_index"], np.arange(n)):
            bad_book.append((i, "frame_index != 0..n-1"))
        if "episode_index" in cols and not (cols["episode_index"] == i).all():
            bad_book.append((i, "episode_index column != meta"))
        if "index" in cols and not np.array_equal(
                cols["index"], np.arange(gindex, gindex + n)):
            bad_book.append((i, f"global index not continuing from {gindex}"))
        if "timestamp" in cols and not np.allclose(
                cols["timestamp"].astype(float), np.arange(n) / fps, atol=1e-4):
            bad_book.append((i, "timestamp != frame_index / fps"))
        if "task_index" in cols and int(cols["task_index"][0]) not in tasks:
            bad_book.append((i, f"task_index {cols['task_index'][0]} not in tasks.jsonl"))
        gindex += n

        for c in t.column_names:
            first = t[c].to_pylist()[0]
            if not isinstance(first, (list, tuple)):
                continue
            decl = info["features"].get(c, {}).get("shape")
            if decl and decl[0] != len(first):
                bad_width.append((i, c, decl[0], len(first)))
            arr = np.asarray(t[c].to_pylist(), dtype=float)
            if arr.dtype.kind == "f" and not np.isfinite(arr).all():
                bad_num.append((i, c, int((~np.isfinite(arr)).sum())))

    r.check(not missing, f"all {len(eps)} parquet files present"
            + (f" -- missing {missing[:3]}" if missing else ""))
    r.check(not bad_len, "parquet rows == episodes.jsonl length"
            + (f" -- {bad_len[:5]}" if bad_len else ""))
    r.check(not bad_book, "index/frame_index/episode_index/timestamp/task_index consistent"
            + (f" -- {bad_book[:5]}" if bad_book else ""))
    r.check(not bad_width, "column widths == info.json shapes"
            + (f" -- {bad_width[:5]}" if bad_width else ""))
    r.check(not bad_num, "no NaN/Inf in any float column"
            + (f" -- {bad_num[:5]}" if bad_num else ""))
    r.check(total == info["total_frames"],
            f"summed frames {total} == info.total_frames {total}")

    # A video shorter than its table is the classic silent truncation: training
    # reads the last frame over and over for the tail of the episode.
    if n_videos:
        step = max(1, len(eps) // n_videos)
        bad_vid, ok_vid = [], 0
        for e in eps[::step][:n_videos]:
            i = e["episode_index"]
            _, vids = ep_paths(root, info, i)
            for name, vp in vids.items():
                if not vp.exists():
                    bad_vid.append((i, name, "missing"))
                    continue
                out = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-count_frames", "-show_entries",
                     "stream=nb_read_frames,r_frame_rate", "-of", "json", str(vp)],
                    capture_output=True, text=True)
                s = json.loads(out.stdout)["streams"][0]
                nf, rate = int(s["nb_read_frames"]), s["r_frame_rate"]
                if nf != e["length"]:
                    bad_vid.append((i, name, f"{nf} frames vs {e['length']} rows"))
                elif rate != f"{fps}/1":
                    bad_vid.append((i, name, f"fps {rate} != {fps}"))
                else:
                    ok_vid += 1
        r.check(not bad_vid,
                f"sampled {ok_vid} videos: frame count == rows, fps == {fps}"
                + (f" -- {bad_vid[:5]}" if bad_vid else ""))

    r.note(f"{len(eps)} episodes, {total} frames, {total/fps/60:.1f} min @ {fps} fps")
    r.note(f"tasks: {list(tasks.values())}")


def pass_modality(root, r):
    """Does the official SONIC config actually resolve against this dataset?"""
    print("\n[2/3] modality (Isaac-GR00T unitree_g1_sonic)")
    mod = json.loads((root / "meta/modality.json").read_text())
    _, ep0 = None, None
    info = json.loads((root / "meta/info.json").read_text())
    t = pq.read_table(ep_paths(root, info, 0)[0])
    defaults = {"state": "observation.state", "action": "action.wbc"}

    for grp, want, want_dim in (("state", SONIC_STATE, SONIC_STATE_DIM),
                                ("action", SONIC_ACTION, SONIC_ACTION_DIM)):
        missing = [k for k in want if k not in mod.get(grp, {})]
        r.check(not missing,
                f"{grp}: all {len(want)} groups the config selects are present"
                + (f" -- MISSING {missing} (GR00T only WARNS about these, "
                   f"then trains without them)" if missing else ""))
        total, oor = 0, []
        for k in want:
            e = mod.get(grp, {}).get(k)
            if not e:
                continue
            total += e["end"] - e["start"]
            col = e.get("original_key", defaults[grp])
            if col in t.column_names and e["end"] > len(t[col].to_pylist()[0]):
                oor.append((k, col, e["end"], len(t[col].to_pylist()[0])))
        r.check(not oor, f"{grp}: every slice is in range of its source column"
                + (f" -- {oor}" if oor else ""))
        r.check(total == want_dim, f"{grp} totals {total} (expected {want_dim})")

    have = set(mod.get("video", {}))
    r.check("ego_view" in have, f"video streams present: {sorted(have)}")
    if have - {"ego_view"}:
        r.note("the built-in config reads ego_view only -- pass "
               "configs/gr00t_sonic_3cam.py to train on the wrist streams too")


def pass_grasp(root, info, eps, r):
    """Do the numbers describe a grasp, in the right causal order?

    Anchoring on the object's own height does NOT work: can_pose is expressed in
    the robot's frame, so the robot squatting reads as the can rising, and a
    naive "did it go up" test reports 0/N on data that is fine. Anchor on the
    gripper-close frame instead.
    """
    print("\n[3/3] grasp causality")
    at_hand, rose_after, held, no_event = 0, 0, 0, 0
    rises, dists = [], []
    for e in eps:
        p, _ = ep_paths(root, info, e["episode_index"])
        t = pq.read_table(p, columns=["observation.can_pose", "observation.eef_state",
                                      "teleop.right_hand_joints"])
        can = np.stack(t["observation.can_pose"].to_pylist())[:, :3]
        wrist = np.stack(t["observation.eef_state"].to_pylist())[:, 7:10]
        grip = np.stack(t["teleop.right_hand_joints"].to_pylist())[:, 0]
        thr = (np.percentile(grip, 95) + np.percentile(grip, 5)) / 2
        closed = grip < thr
        if not closed.any():
            no_event += 1
            continue
        fc = int(np.argmax(closed))
        d = np.linalg.norm(can - wrist, axis=1)
        rise = float(can[fc:, 2].max() - can[fc, 2])
        at_hand += d[fc] < 0.30
        rose_after += rise > 0.03
        held += float(np.std(d[-30:])) < 0.02
        rises.append(rise)
        dists.append(float(d[-30:].mean()))

    n = len(eps) - no_event
    r.check(no_event == 0, f"every episode has a gripper-close event "
                           f"({n}/{len(eps)})")
    r.check(at_hand == n, f"wrist within 30 cm of the object at close ({at_hand}/{n})")
    r.check(rose_after == n, f"object rose >3 cm AFTER the close ({rose_after}/{n})")
    r.check(held == n, f"object-to-wrist distance holds steady at the end "
                       f"({held}/{n}) -- rigid attachment, not a knock-over")
    if rises:
        r.note(f"rise after close: median {np.median(rises)*100:.1f} cm, "
               f"min {min(rises)*100:.1f} cm")
        r.note(f"final object-to-wrist offset: median {np.median(dists)*100:.1f} cm")


def video_motion(path: Path, n: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    prev, sig = None, []
    while len(sig) < n:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        g = g.astype(np.float32)
        sig.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    return np.asarray(sig)


def _unit(x):
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-9)


def _profile(a, b, max_lag=MAX_LAG):
    """Correlation of a against b at each lag; positive lag = a trails b."""
    m = min(len(a), len(b))
    a, b = _unit(a[:m]), _unit(b[:m])
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b[:m - lag] if lag else b) if lag >= 0 else (a[:m + lag], b[-lag:])
        out[lag] = float(np.dot(_unit(x), _unit(y)))
    return out


def pass_alignment(root, info, eps, r, n_eps):
    """Measure the video/table lag, and the action/state lag."""
    print("\n[4/4] alignment")
    if cv2 is None:
        r.note("cv2 not importable, skipping")
        return
    step = max(1, len(eps) // n_eps)
    picks = [e["episode_index"] for e in eps[::step][:n_eps]]

    lags, weak = [], []
    act_lags = []
    for ep in picks:
        p, vids = ep_paths(root, info, ep)
        t = pq.read_table(p, columns=["observation.state", "action.motion_token"])
        st = np.stack(t["observation.state"].to_pylist())
        body = np.linalg.norm(np.diff(st[:, ARM_STATE], axis=0, prepend=st[:1, ARM_STATE]), axis=1)

        for name, vp in vids.items():
            if "wrist" not in name or not vp.exists():
                continue           # the head camera sees the scene, not the arm
            prof = _profile(video_motion(vp, t.num_rows), body)
            peak = max(prof, key=prof.get)
            # A "peak" that beats lag 0 by less than this is the period-3 ripple
            # the 50->30 Hz resample leaves behind (5:3 repeats every 3 frames),
            # not a real delay.
            lags.append(0 if prof[peak] - prof[0] < 0.02 else peak)
            if prof[peak] < 0.25:
                weak.append((ep, name, round(prof[peak], 2)))

        tok = np.stack(t["action.motion_token"].to_pylist())
        # body first: _profile's positive lag means the FIRST argument trails
        # the second, and what we want to read out is how far the body trails
        # the token.
        prof = _profile(np.linalg.norm(np.diff(st, axis=0), axis=1),
                        np.linalg.norm(np.diff(tok, axis=0), axis=1), max_lag=10)
        act_lags.append(max(prof, key=prof.get))

    uniq, counts = np.unique(lags, return_counts=True)
    r.check(set(lags) == {0},
            f"video vs table: {len(lags)} wrist-camera measurements, "
            f"lag distribution {dict(zip(uniq.tolist(), counts.tolist()))}")
    if weak:
        r.note(f"weak correlation (lag unreliable) on {weak}")

    med = float(np.median(act_lags))
    r.check(med > 0,
            f"action vs state: body trails the token by a median {med:+.0f} frames "
            f"({med/info['fps']*1000:.0f} ms)")
    r.note("a token is a command and the body is its consequence, so the body "
           "must TRAIL. Zero or negative means the columns are swapped in time.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--videos", type=int, default=10,
                    help="episodes to ffprobe for frame-count agreement (0 = skip)")
    ap.add_argument("--align-episodes", type=int, default=12,
                    help="episodes to measure the video/table lag on")
    ap.add_argument("--quick", action="store_true",
                    help="structure and modality only -- no video decode")
    args = ap.parse_args()

    root = Path(args.dataset)
    if not (root / "meta/info.json").exists():
        raise SystemExit(f"{root} does not look like a LeRobot dataset "
                         f"(no meta/info.json)")
    info, eps, tasks, stats = load_meta(root)
    r = Report()
    print(f"auditing {root}")

    pass_structure(root, info, eps, tasks, stats, r, 0 if args.quick else args.videos)
    pass_modality(root, r)
    pass_grasp(root, info, eps, r)
    if not args.quick:
        pass_alignment(root, info, eps, r, args.align_episodes)

    print(f"\n{r.passed} passed, {r.failed} failed")
    sys.exit(1 if r.failed else 0)


if __name__ == "__main__":
    main()
