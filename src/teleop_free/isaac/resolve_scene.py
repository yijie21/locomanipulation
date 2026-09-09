"""Which scene, and which sidecar files, does this exporter session belong to?

Nothing records the link. collect.sh runs one gym process per batch, each batch
fills its own exporter session and writes its own manifest, and the converter
pairs them POSITIONALLY. For a re-render that is not good enough: the wrong
batch gives a correct trajectory under a table at the wrong height (0.620-0.646
m across the cover400 seeds) and nothing errors.

Worse, the pose sidecar's `seq` RESETS per gym process and the ranges overlap
almost completely across batches (~1900-9200 everywhere), so a glob spanning
several batches lets the last-sorted pid win every collision and the re-render
draws pelvis and object poses from a different batch than its joints. A naive
`pose_sidecar/p*_round_*.jsonl` glob does exactly that.

**The fingerprint is the pelvis quaternion.** The parquet carries
`observation.root_orientation` per frame and the sidecar carries the pelvis
pose; joined on priv_seq they agree to ~8e-4 for the batch that produced the
session and ~4e-2 for the next best -- a 50x margin. Everything cheaper was
tried and does NOT work, so do not "simplify" this back:

  * object position -- the spawn RNG is not tied to the scene seed, so EVERY
    batch draws the same sequence of object positions. Matched to 0.02 mm on
    several batches at once, in order. Carries no batch identity at all.
  * priv_seq coverage -- the seq ranges overlap, so many batches contain 100%
    of a session's seq values while holding entirely different data.
  * object resting height -- seeds sit ~1.3 mm apart in table top and a single
    episode's settle noise is ~2 mm, so it cannot separate neighbours.
  * file mtime -- works here (the gym log lands within seconds of the batch)
    but is filesystem metadata, not data; kept only to order the candidates.

    "$RENDER_PY" -m teleop_free.isaac.resolve_scene --session <dir>

Going forward the honest fix is for collection to record the session path in
the batch's provenance.json; this module is what recovers the link for data
already on disk.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .. import paths


def _sidecar_groups(collect_dir: str) -> dict:
    """{pid: [files]} -- one gym process per batch, so one pid per batch."""
    out: dict[str, list[str]] = {}
    for f in glob.glob(os.path.join(collect_dir, "pose_sidecar", "p*_round_*.jsonl")):
        out.setdefault(os.path.basename(f).split("_round")[0], []).append(f)
    return out


def _pid_for_seed(collect_dir: str) -> dict:
    """Seed -> pid, by which gym log sits next to that group's last write."""
    groups = _sidecar_groups(collect_dir)
    gym = {os.path.basename(g)[4:-4]: os.path.getmtime(g) for g in glob.glob(os.path.join(collect_dir, "gym_*.log"))}
    if not groups or not gym:
        return {}
    seed_of = {}
    for pid, files in groups.items():
        last = max(os.path.getmtime(f) for f in files)
        seed, _ = min(((s, abs(t - last)) for s, t in gym.items()), key=lambda r: r[1])
        seed_of[seed] = pid
    return seed_of


def _fingerprint(session: str, episode: int, files: list) -> tuple:
    """-> (mean pelvis-quaternion error, seq hit rate) for one batch's files."""
    import pandas as pd

    pq = os.path.join(session, "data", "chunk-000", f"episode_{episode:06d}.parquet")
    sq = os.path.join(session, "priv_seq", f"episode_{episode:06d}.json")
    if not (os.path.exists(pq) and os.path.exists(sq)):
        return 9.9, 0.0
    ro = np.stack(
        pd.read_parquet(pq, columns=["observation.root_orientation"])["observation.root_orientation"].to_numpy()
    )
    priv = json.load(open(sq))["priv_seq"]
    by = {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r.get("seq") is not None:
                by[r["seq"]] = r
    errs = []
    for i, s in enumerate(priv):
        r = by.get(s)
        if r is None or i >= len(ro):
            continue
        q = np.asarray(r["pelvis"][3:7], dtype=float)
        # q and -q are the same rotation
        errs.append(min(float(np.linalg.norm(q - ro[i])), float(np.linalg.norm(q + ro[i]))))
    if not errs:
        return 9.9, 0.0
    return float(np.mean(errs)), len(errs) / max(1, len(priv))


QUAT_MATCH_TOL = 5e-3  # right batch ~8e-4, next best ~4e-2
QUAT_MARGIN = 5.0  # winner must beat the runner-up by this factor


def resolve(session: str, collect_dir: str = "", episode: int = 0) -> dict:
    collect_dir = collect_dir or os.path.join(paths.LOGS_DIR, "collect")
    seed_of = _pid_for_seed(collect_dir)
    if not seed_of:
        raise SystemExit(f"no sidecar groups or gym logs under {collect_dir}")
    groups = _sidecar_groups(collect_dir)

    scored = []
    for seed, pid in sorted(seed_of.items()):
        err, hit = _fingerprint(session, episode, groups[pid])
        scored.append((err, seed, pid, hit))
    scored.sort()
    best = scored[0]
    runner = scored[1] if len(scored) > 1 else None
    if best[0] > QUAT_MATCH_TOL:
        raise SystemExit(
            f"no batch matches {session}: best is seed {best[1]} at pelvis-quat "
            f"error {best[0]:.5f}, tolerance {QUAT_MATCH_TOL}"
        )
    if runner is not None and runner[0] < best[0] * QUAT_MARGIN:
        raise SystemExit(
            f"ambiguous: seed {best[1]} at {best[0]:.5f} and seed {runner[1]} "
            f"at {runner[0]:.5f} are within {QUAT_MARGIN}x"
        )

    err, seed, pid, hit = best
    sidecar_glob = os.path.join(collect_dir, "pose_sidecar", f"{pid}_round_*.jsonl")
    scene_dir = os.path.join(collect_dir, f"scene_{seed}")
    meta_p = os.path.join(scene_dir, "scene_meta.json")
    man_p = os.path.join(collect_dir, f"manifest_{seed}.jsonl")
    table_manifest = table_meta = None
    if os.path.exists(man_p):
        for line in open(man_p):
            t = json.loads(line).get("table_top")
            if t is not None:
                table_manifest = float(t)
                break
    if os.path.exists(meta_p):
        table_meta = float(json.load(open(meta_p))["surfaces"]["A"]["top"])
    if table_manifest is not None and table_meta is not None and abs(table_manifest - table_meta) > 1e-3:
        raise SystemExit(
            f"seed {seed}: manifest table_top {table_manifest} disagrees with "
            f"scene_meta {table_meta} -- the snapshot is not this batch's"
        )
    return {
        "session": session,
        "seed": seed,
        "pid": pid,
        "snapshot_dir": scene_dir,
        "scene_xml": os.path.join(scene_dir, "scene_locomani.xml"),
        "scene_meta": meta_p,
        "sidecar_glob": sidecar_glob,
        "table_top": table_meta if table_meta is not None else table_manifest,
        "quat_err": err,
        "seq_hit_rate": hit,
        "runner_up": None if runner is None else {"seed": runner[1], "quat_err": runner[0]},
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--collect-dir", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    r = resolve(args.session, args.collect_dir)
    if args.json:
        print(json.dumps(r, indent=2))
        return
    for k in ("seed", "pid", "table_top", "quat_err", "seq_hit_rate"):
        print(f"  {k:18s} {r[k]}")
    print(f"  runner_up          {r['runner_up']}")
    print(f"  scene_xml          {r['scene_xml']}")
    print(f"  sidecar_glob       {r['sidecar_glob']}")


if __name__ == "__main__":
    main()
