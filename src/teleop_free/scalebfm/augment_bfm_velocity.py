"""Add the 7 velocity columns (bfm_action.py) to a recorded ScaleBFM LeRobot dataset, in place,
and optionally trim the idle prefix every episode starts with.

    python augment_bfm_velocity.py <dataset> [--trim-idle 5] [--dry]

  * action.bfm_vel (7, float32) is computed from action.bfm + the recorded root pose per row
    (observation.root_position / root_orientation), so no re-recording is needed.
  * --trim-idle K drops the rows before (first row whose pelvis target is >3 cm away) - K.
    record_bottle.py writes from tick 0, so every episode carries the 1.5 s settle plus the
    ramp: 46 rows (15% of the frames) in which the frames are identical and the label is
    "stand still", the same absorbing-state hazard as the SONIC creep datasets. Videos are
    re-cut with ffmpeg (frame-accurate: -ss before -i on a transcode), parquet rows
    renumbered (timestamp, frame_index, index), meta rewritten.
  * meta/info.json features, meta/modality.json (action.bfm_vel) and meta/episodes_stats.jsonl
    are updated. Re-run gr00t.data.stats with configs/gr00t_bfm_vel_2cam.py afterwards; this
    script does not touch meta/stats.json / relative_stats.json.
Idempotent: a dataset that already has action.bfm_vel and no idle prefix is left alone.
"""
import argparse, json, os, subprocess, sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfm_action import velocity_columns  # noqa: E402

VEL_KEY = "action.bfm_vel"
MOVE_M = 0.03


def feat_stats(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [int(a.shape[0])]}


def cut_video(path, start_s, n_frames, fps):
    tmp = path + ".cut.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start_s:.6f}", "-i", path, "-frames:v", str(n_frames),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-r", str(fps), "-an", tmp], check=True)
    os.replace(tmp, path)


def n_video_frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("dataset")
    ap.add_argument("--trim-idle", type=int, default=None, metavar="K", help="keep K rows before the first pelvis motion; omit = no trim")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    ds = a.dataset
    info = json.load(open(os.path.join(ds, "meta/info.json")))
    fps = info["fps"]
    video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    episodes = [json.loads(l) for l in open(os.path.join(ds, "meta/episodes.jsonl"))]
    ep_stats = [json.loads(l) for l in open(os.path.join(ds, "meta/episodes_stats.jsonl"))]
    assert len(episodes) == len(ep_stats) == info["total_episodes"]

    gidx = 0; total = 0; trimmed = []; vel_added = 0
    for e, es in zip(episodes, ep_stats):
        ei = e["episode_index"]
        pth = os.path.join(ds, f"data/chunk-000/episode_{ei:06d}.parquet")
        t = pq.read_table(pth)
        act = np.stack(t["action.bfm"].to_pylist()).astype(np.float64)
        n0 = len(act)
        rp = np.stack(t["observation.root_position"].to_pylist()); rq = np.stack(t["observation.root_orientation"].to_pylist())
        # --- trim ---
        start = 0
        if a.trim_idle is not None:
            mv = np.where(np.linalg.norm(act[:, 0:2], axis=1) > MOVE_M)[0]
            first = int(mv[0]) if len(mv) else 0
            start = max(0, first - a.trim_idle)
        if start > 0:
            t = t.slice(start)
            act = act[start:]; rp = rp[start:]; rq = rq[start:]
        n = len(act)
        # --- velocity (recomputed on the kept rows; identical to computing then slicing except at the cut edge) ---
        vel = velocity_columns(act, rp, rq, fps=fps)
        cols = {name: t[name] for name in t.column_names if name not in (VEL_KEY, "timestamp", "frame_index", "index")}
        cols[VEL_KEY] = pa.array([list(map(float, v)) for v in vel], type=pa.list_(pa.float32(), 7))
        cols["timestamp"] = pa.array((np.arange(n) / fps).astype(np.float32))
        cols["frame_index"] = pa.array(np.arange(n, dtype=np.int64))
        cols["index"] = pa.array(np.arange(gidx, gidx + n, dtype=np.int64))
        order = [c for c in t.column_names if c in cols] + ([VEL_KEY] if VEL_KEY not in t.column_names else [])
        new_t = pa.table({c: cols[c] for c in order})
        if not a.dry:
            pq.write_table(new_t, pth)
            if start > 0:
                for vk in video_keys:
                    vp = os.path.join(ds, f"videos/chunk-000/{vk}/episode_{ei:06d}.mp4")
                    cut_video(vp, start / fps, n, fps)
                    nf = n_video_frames(vp)
                    if nf != n:
                        raise RuntimeError(f"{vp}: {nf} frames after the cut, parquet has {n}")
        # --- per-episode stats ---
        st = es["stats"]
        st[VEL_KEY] = feat_stats(vel)
        if start > 0:
            for name in new_t.column_names:
                if name in st and name not in video_keys:
                    arr = np.stack(new_t[name].to_pylist()) if pa.types.is_list(new_t[name].type) or pa.types.is_fixed_size_list(new_t[name].type) else np.asarray(new_t[name].to_pylist())
                    st[name] = feat_stats(arr)
        e["length"] = n
        gidx += n; total += n; trimmed.append(n0 - n); vel_added += int(VEL_KEY not in t.column_names)
        print(f"episode {ei:3d}: {n0} -> {n} rows (cut {n0 - n}), |v_pel| max {np.linalg.norm(vel[:, :3], axis=1).max():.2f} m/s", flush=True)

    info["features"][VEL_KEY] = {"dtype": "float32", "shape": [7], "names": None}
    info["total_frames"] = total
    info.setdefault("script_config", {})["augment"] = dict(source="teleop_free/scalebfm/augment_bfm_velocity.py", trim_idle=a.trim_idle,
                                                           rows_cut=int(sum(trimmed)))
    mod = json.load(open(os.path.join(ds, "meta/modality.json")))
    mod["action"]["bfm_vel"] = {"start": 0, "end": 7, "original_key": VEL_KEY}
    if a.dry:
        print("dry run: nothing written"); return
    json.dump(info, open(os.path.join(ds, "meta/info.json"), "w"), indent=4)
    json.dump(mod, open(os.path.join(ds, "meta/modality.json"), "w"), indent=4)
    with open(os.path.join(ds, "meta/episodes.jsonl"), "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(ds, "meta/episodes_stats.jsonl"), "w") as f:
        for es in ep_stats:
            f.write(json.dumps(es) + "\n")
    print(f"done: {len(episodes)} episodes, {total} frames, {sum(trimmed)} rows cut, {VEL_KEY} added to {vel_added}")


if __name__ == "__main__":
    main()
