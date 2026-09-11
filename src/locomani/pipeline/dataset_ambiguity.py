"""Does this dataset ask the policy to do two different things from the same picture?

A single-frame VLA can only represent a MEMORYLESS function of the current
observation. Any state the expert script carries in its own head -- "have I
started the wind-up yet", "how long have I been walking" -- is unlearnable by
construction, and shows up in the data as pairs of frames that look the same but
whose futures diverge.

That is exactly how visible200 failed (ledger bottle.md §18.4-18.7): the expert's
wind-up out of the approach pose fires on a scripted frame index, not on anything
visible, so the policy sees near-identical approach frames labelled both "hold"
and "start the wind-up", learns the majority answer (hold), and freezes there for
85 s in closed loop.

This scores a dataset for that defect BEFORE anyone spends a day collecting on a
new expert design. For every sampled frame it finds the visually nearest frames
from OTHER episodes and asks how much their futures disagree:

    ambiguity = divergence(futures of visual near-neighbours)
                ------------------------------------------------
                divergence(futures of random pairs)

0 means the picture determines what happens next. 1 means the picture tells you
nothing the random baseline did not. The score is reported overall and per phase,
because a design can be fine everywhere except the one transition that matters.

Usage:
  python3 -m locomani.pipeline.dataset_ambiguity DATASET [--episodes 40] [--stride 3]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

CLOSE_THRESHOLD = 0.005  # teleop.right_hand_joints[0] below this = the expert is closing
HORIZON = 40  # the action chunk the policy predicts, in frames
THUMB = (32, 24)  # per-camera thumbnail (w, h) for the visual distance
SAME_EPISODE_GUARD = 50  # frames; neighbours nearer in time than this are excluded


def episode_frames(pq_path: str, video_dirs: dict[str, str], stride: int):
    """Per-frame observation thumbnails, future summaries and phase labels."""
    import cv2
    import pyarrow.parquet as pq

    t = pq.read_table(
        pq_path,
        columns=["action.motion_token", "observation.eef_state", "teleop.right_hand_joints"],
    )
    tok = np.stack(t["action.motion_token"].to_pylist()).astype(np.float32)
    wrist_z = np.stack(t["observation.eef_state"].to_pylist())[:, 9].astype(np.float32)
    grip = np.stack(t["teleop.right_hand_joints"].to_pylist())[:, 0].astype(np.float32)
    closing = np.flatnonzero(grip < CLOSE_THRESHOLD)
    close_at = int(closing[0]) if len(closing) else len(tok)

    stem = Path(pq_path).stem
    thumbs = {}
    for cam, d in video_dirs.items():
        cap = cv2.VideoCapture(str(Path(d) / f"{stem}.mp4"))
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(cv2.resize(bgr, THUMB), cv2.COLOR_BGR2GRAY))
        cap.release()
        thumbs[cam] = np.stack(frames).astype(np.float32) / 255.0 if frames else None
    if any(v is None for v in thumbs.values()):
        return None

    n = min(len(tok), *(len(v) for v in thumbs.values()))
    idx = np.arange(0, n - HORIZON, stride)
    if not len(idx):
        return None
    return dict(
        thumb=np.concatenate([thumbs[c][idx].reshape(len(idx), -1) for c in sorted(thumbs)], axis=1),
        chunk=np.stack([tok[i : i + HORIZON].ravel() for i in idx]),
        dz=np.array([wrist_z[i + HORIZON - 1] - wrist_z[i] for i in idx], dtype=np.float32),
        phase=np.array([i - close_at for i in idx], dtype=np.int32),
        frame=idx,
    )


def load(dataset: Path, episodes: int, stride: int):
    pqs = sorted(glob.glob(str(dataset / "data/chunk-000/*.parquet")))[:episodes]
    vdirs = {
        "ego": dataset / "videos/chunk-000/observation.images.ego_view",
        "wrist": dataset / "videos/chunk-000/observation.images.right_wrist",
    }
    out, owner = [], []
    for e, p in enumerate(pqs):
        rec = episode_frames(p, {k: str(v) for k, v in vdirs.items()}, stride)
        if rec is None:
            continue
        out.append(rec)
        owner.append(np.full(len(rec["frame"]), e, dtype=np.int32))
    if not out:
        raise SystemExit(f"no usable episodes under {dataset}")
    joined = {k: np.concatenate([r[k] for r in out]) for k in out[0]}
    joined["episode"] = np.concatenate(owner)
    return joined


def whiten(x: np.ndarray, dims: int = 64) -> np.ndarray:
    """Centre, project onto the leading principal directions, and unit-norm."""
    x = x - x.mean(0, keepdims=True)
    dims = min(dims, x.shape[0], x.shape[1])
    _, _, vt = np.linalg.svd(x[:: max(1, len(x) // 2000)], full_matrices=False)
    z = x @ vt[:dims].T
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)


def ambiguity(data: dict, k: int, rng: np.random.Generator) -> dict:
    """Future divergence among visual near-neighbours, against a random baseline."""
    V = whiten(data["thumb"])
    chunk = data["chunk"] / (np.linalg.norm(data["chunk"], axis=1, keepdims=True) + 1e-9)
    ep, frame = data["episode"], data["frame"]
    n = len(V)

    # forbid a frame's own episode when it is temporally close: those pairs are
    # trivially similar in both picture and future and would mask the defect
    near_chunk, near_dz, vis_d = np.zeros(n), np.zeros(n), np.zeros(n)
    for a in range(0, n, 512):
        b = min(a + 512, n)
        sim = V[a:b] @ V.T
        same = (ep[a:b, None] == ep[None, :]) & (np.abs(frame[a:b, None] - frame[None, :]) < SAME_EPISODE_GUARD)
        sim[same] = -np.inf
        sim[np.arange(b - a), np.arange(a, b)] = -np.inf
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        rows = np.arange(b - a)[:, None]
        vis_d[a:b] = 1.0 - sim[rows, top].mean(1)
        near_chunk[a:b] = np.linalg.norm(chunk[a:b, None, :] - chunk[top], axis=2).mean(1)
        near_dz[a:b] = np.abs(data["dz"][a:b, None] - data["dz"][top]).mean(1)

    j = rng.integers(0, n, size=(n, k))
    rand_chunk = np.linalg.norm(chunk[:, None, :] - chunk[j], axis=2).mean(1)
    rand_dz = np.abs(data["dz"][:, None] - data["dz"][j]).mean(1)
    return dict(
        near_chunk=near_chunk,
        near_dz=near_dz,
        rand_chunk=float(rand_chunk.mean()),
        rand_dz=float(rand_dz.mean()),
        vis_d=vis_d,
    )


PHASES = [
    ("far approach", -(10**9), -120),
    ("late approach", -120, -70),
    ("WIND-UP", -70, -45),
    ("descent", -45, -10),
    ("close + lift", -10, 10**9),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--k", type=int, default=5, help="visual neighbours per frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", dest="out_json")
    args = ap.parse_args(argv)

    data = load(Path(args.dataset), args.episodes, args.stride)
    print(f"dataset  : {args.dataset}")
    print(f"frames   : {len(data['frame'])} from {data['episode'].max() + 1} episodes (stride {args.stride})")
    res = ambiguity(data, args.k, np.random.default_rng(args.seed))
    print(f"baseline : random-pair future divergence  chunk {res['rand_chunk']:.4f}  wrist dz {res['rand_dz']:.4f} m")
    print()
    print(f"  {'phase':<16}{'n':>6}{'visual dist':>13}{'chunk amb.':>12}{'wrist-dz amb.':>15}{'dz spread':>12}")
    rows = []
    for label, lo, hi in PHASES:
        m = (data["phase"] >= lo) & (data["phase"] < hi)
        if m.sum() < 10:
            continue
        a_chunk = float(res["near_chunk"][m].mean() / res["rand_chunk"])
        a_dz = float(res["near_dz"][m].mean() / res["rand_dz"])
        rows.append(dict(phase=label, n=int(m.sum()), visual=float(res["vis_d"][m].mean()), chunk=a_chunk, dz=a_dz))
        print(
            f"  {label:<16}{m.sum():>6}{res['vis_d'][m].mean():>13.4f}{a_chunk:>12.3f}{a_dz:>15.3f}"
            f"{res['near_dz'][m].mean():>11.4f}m"
        )
    overall = float(res["near_chunk"].mean() / res["rand_chunk"])
    print()
    print(f"  OVERALL chunk ambiguity {overall:.3f}   (0 = the picture decides, 1 = it tells you nothing)")
    worst = max(rows, key=lambda r: r["dz"])
    print(f"  WORST PHASE by wrist-dz ambiguity: {worst['phase']} at {worst['dz']:.3f}")
    print()
    print("  Read it as: within a phase, frames that LOOK the same have futures that")
    print("  disagree this much relative to two frames picked at random. A phase near")
    print("  1.0 is one the policy cannot learn from a single frame.")
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps({"dataset": args.dataset, "overall_chunk": overall, "phases": rows}, indent=1) + "\n"
        )
        print(f"\n-> {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
