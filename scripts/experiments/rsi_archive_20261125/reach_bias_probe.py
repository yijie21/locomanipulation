"""TODO 33.1.4 (2) (user 2026-10-25: probe free-space reaches first): does the controller miss a reach target in a way
that can be predicted -- and so pre-compensated -- when nothing is in the way?

The task records cannot tell (scripts/rsi/reach_vectors.py: out of start, no model of the error vector beats leaving
it alone; their large misses sit beside furniture). This runs single-arm reaches on open floor (walk_mars, the
capability probe's `Probe`, as the A800's dense envelope did) and keeps the miss as a vector:

  grid   GRID cells (forward X x outward Y of the hand's own side x height Z x standing / pelvis 0.68 m x wrist level /
         tilt45 / down x hand): reset at the origin, hold, the posture, `reach` to the target with the attitude's
         rotation, hold; recorded the jaw centre's miss (delivered - target, world = the start pelvis frame), its norm
         before and after the hold, the attitude error, a fall.
  check  a distance-weighted k-NN of the miss vector over (x, y, z, attitude, pelvis, hand), leaving out one forward
         slice at a time: the residual |e - e_hat| against |e|. A bias worth compensating leaves a residual clearly
         under |e|.
  comp   COMP cells, reached again with the target moved by -e_hat (e_hat from the grid without that cell): the miss
         with and without the compensation, per cell -- the closed-loop answer.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    nohup setsid $PY scripts/rsi/reach_bias_probe.py run grid --procs 12 > /workspace/tmp/reach_bias.log 2>&1 &
    $PY scripts/rsi/reach_bias_probe.py check
    nohup setsid $PY scripts/rsi/reach_bias_probe.py run comp --procs 12 >> /workspace/tmp/reach_bias.log 2>&1 &
    $PY scripts/rsi/reach_bias_probe.py report
    $PY scripts/rsi/reach_bias_probe.py table      # the compensation table body.bias reads

2026-10-26: the miss is predictable on open floor (left-out forward slices: residual 1.89 cm against 2.74), and
moving the target by -e_hat halves it in closed loop (240 cells: 3.07 -> 1.57 cm, misses >= 3 cm 67 -> 32, no fall;
wrist down 6.63 -> 2.62 cm, tilt45 1.43 -> 1.03, level 1.14 -> 1.07).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "scripts" / "rsi"), str(REPO / "scripts" / "rsi" / "scene"), str(REPO / "src")]

SIM_PY = REPO / "third_party" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python"
OUT = REPO / "outputs" / "scene_rsi_20261016" / "body" / "reach_bias_20261025"
XS = (0.30, 0.40, 0.50, 0.60, 0.70)
YS = (-0.05, 0.05, 0.15, 0.25)  # + outward of the hand's own side
ZS = {
    None: (0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25),
    0.68: (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05),
}
ATTS = ("level", "tilt45", "down")
COMP = 240
K = 12
ATT_CODE = {"level": 0.0, "down": 1.0, "tilt45": 2.0}


def _num(v: float, scale: int = 100, width: int = 3) -> str:
    return f"{int(round(v * scale)):0{width}d}"


def grid_cells() -> list[dict]:
    out = []
    for side in ("right", "left"):
        sign = -1.0 if side == "right" else 1.0
        for att in ATTS:
            for pel, zs in ZS.items():
                tag = "stand" if pel is None else f"p{_num(pel)}"
                for y in YS:
                    for z in zs:
                        for x in XS:
                            out.append(dict(id=f"{side}_{att}_{tag}_y{_num(y + 0.1)}_z{_num(z)}_x{_num(x, width=2)}",
                                            side=side, att=att, pelvis=pel, y_out=y, y=round(sign * y, 2), z=z, x=x))  # fmt: skip
    return out


def features(c: dict) -> list[float]:
    return [c["x"], c["y_out"], c["z"], ATT_CODE[c["att"]], 0.78 if c["pelvis"] is None else c["pelvis"],
            1.0 if c["side"] == "right" else 0.0]  # fmt: skip


def reach_cell(pr, c: dict, offset=(0.0, 0.0, 0.0)) -> dict:
    """One reach on open floor; the target moved by `offset` (world) for a compensated reach, the miss always against the
    cell's own target."""
    m, R = pr.m, pr.R
    m.reset((0.0, 0.0), 0.0)
    rows: list = []
    pr.step(rows, "hold", m.hold(1.0))
    if c["pelvis"] is not None and not pr.step(rows, "posture", m.posture(pelvis_z=c["pelvis"])):
        return dict(ok=False, fell=True, steps=rows)
    target = np.array([c["x"], c["y"], c["z"]])
    rot = {"level": np.eye(3), "down": R.rot_y(np.pi / 2), "tilt45": R.rot_y(np.pi / 4)}[c["att"]]
    out = m.reach(c["side"], target + np.asarray(offset, float), rot, pelvis_z=c["pelvis"])
    jaw0, _ = m.jaw_centre(c["side"])
    alive = pr.step(rows, "reach", out)
    if alive:
        pr.step(rows, "hold", m.hold(1.5))
    jaw, Rj = m.jaw_centre(c["side"])
    from locomani.backends.scalebfm.motor import rotvec

    return dict(ok=bool(out.ok and m.fell_at is None), fell=m.fell_at is not None,
                miss=[round(float(v), 4) for v in (jaw0 - target)], miss_after_hold=[round(float(v), 4) for v in (jaw - target)],
                miss_cm=round(100 * float(np.linalg.norm(jaw0 - target)), 2),
                miss_after_hold_cm=round(100 * float(np.linalg.norm(jaw - target)), 2),
                attitude_err_deg=round(math.degrees(float(np.linalg.norm(rotvec(rot @ Rj.T)))), 1),
                offset=[round(float(v), 4) for v in offset], steps=rows)  # fmt: skip


def load(kind: str) -> dict[str, dict]:
    d = OUT / kind
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.exists() else {}


def _knn(Xtr, Etr, x, mu, sd, k=K):
    dist = np.sqrt((((Xtr - x) / sd) ** 2).sum(1))
    idx = np.argpartition(dist, min(k, len(dist) - 1))[:k]
    w = 1.0 / (dist[idx] + 0.05)
    return (w[:, None] * Etr[idx]).sum(0) / w.sum()


def table() -> tuple[np.ndarray, np.ndarray, list[str]]:
    recs = load("grid")
    rows = [(cid, r) for cid, r in recs.items() if not r.get("fell") and r.get("miss_after_hold") is not None]
    X = np.array([features(r["cell"]) for _, r in rows])
    E = np.array([r["miss_after_hold"] for _, r in rows])
    return X, E, [cid for cid, _ in rows]


def check() -> dict:
    X, E, ids = table()
    mu, sd = X.mean(0), X.std(0) + 1e-9
    P = np.zeros_like(E)
    for x in XS:  # leave one forward slice out
        te = np.isclose(X[:, 0], x)
        for i in np.where(te)[0]:
            P[i] = _knn(X[~te], E[~te], X[i], mu, sd)
    err, res = np.linalg.norm(E, axis=1), np.linalg.norm(E - P, axis=1)
    miss = err >= 0.03
    out = dict(cells=len(ids), mean_miss_cm=round(100 * float(err.mean()), 2), miss_rate=round(float(miss.mean()), 3),
               residual_cm=round(100 * float(res.mean()), 2), better=round(float((res < err).mean()), 3),
               missed_residual_cm=round(100 * float(res[miss].mean()), 2) if miss.any() else None,
               missed_still=round(float((res[miss] >= 0.03).mean()), 3) if miss.any() else None,
               bias_mean_cm=[round(100 * float(v), 2) for v in E.mean(0)])  # fmt: skip
    (OUT / "check.json").write_text(json.dumps(out, indent=1))
    return out


def comp_cells() -> list[dict]:
    """COMP grid cells spread evenly (every n-th by id), each with e_hat from the grid without the cell itself."""
    X, E, ids = table()
    mu, sd = X.mean(0), X.std(0) + 1e-9
    by = {c["id"]: c for c in grid_cells()}
    pick = ids[:: max(1, len(ids) // COMP)][:COMP]
    out = []
    for cid in pick:
        i = ids.index(cid)
        keep = np.arange(len(ids)) != i
        e_hat = _knn(X[keep], E[keep], X[i], mu, sd)
        out.append(dict(by[cid], id=f"comp_{cid}", grid_id=cid, offset=[float(v) for v in -e_hat]))
    return out


def worker(kind: str, ids: list[str]) -> None:
    from capability_probe import Probe

    cells = {
        c["id"]: c for c in (grid_cells() if kind == "grid" else json.loads((OUT / "comp_cells.json").read_text()))
    }
    out = OUT / kind
    out.mkdir(parents=True, exist_ok=True)
    pr = Probe("envelope")
    for i in ids:
        f = out / f"{i}.json"
        if f.exists():
            continue
        t0 = time.time()
        try:
            rec = reach_cell(pr, cells[i], cells[i].get("offset", (0.0, 0.0, 0.0)))
        except Exception as e:  # noqa: BLE001 -- a cell that raises is recorded, the shard goes on
            rec = dict(ok=False, fell=False, error=f"{type(e).__name__}: {e}")
        rec.update(cell=cells[i], wall_s=round(time.time() - t0, 1))
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, default=float))
        tmp.rename(f)


def run(kind: str, procs: int, gpus: str, shard: int = 30) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if kind == "comp":
        (OUT / "comp_cells.json").write_text(json.dumps(comp_cells()))
        ids = [c["id"] for c in json.loads((OUT / "comp_cells.json").read_text())]
    else:
        ids = [c["id"] for c in grid_cells()]
    todo = [i for i in ids if not (OUT / kind / f"{i}.json").exists()]
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", NUMPY_MADVISE_HUGEPAGE="0")
    queue = [[str(SIM_PY), str(Path(__file__).resolve()), "worker", kind, "--ids", ",".join(todo[k : k + shard])]
             for k in range(0, len(todo), shard)]  # fmt: skip
    devices, running = gpus.split(","), []
    print(f"{kind}: {len(todo)} cells in {len(queue)} shards", flush=True)
    with (OUT / "run.log").open("a") as fh:
        while queue or running:
            while queue and len(running) < procs:
                dev = min(devices, key=lambda x: sum(g == x for _, g in running))
                running.append((subprocess.Popen(queue.pop(0), cwd=REPO, env={**env, "CUDA_VISIBLE_DEVICES": dev},
                                                 stdout=fh, stderr=subprocess.STDOUT), dev))  # fmt: skip
            time.sleep(1.0)
            running = [(p, g) for p, g in running if p.poll() is None]
    print(f"{kind}: done", flush=True)


TABLE = REPO / "outputs" / "scene_rsi_20261016" / "body" / "reach_bias_knn.npz"


def write_table(path: Path = TABLE) -> dict:
    """The grid's miss vectors as the body model's compensation table (locomani.tasks.scene.body.bias): features as
    `features`, the vector in the start's pelvis heading frame (x ahead, y left, z up) after the hold."""
    X, E, ids = table()
    mu, sd = X.mean(0), X.std(0) + 1e-9
    np.savez(path, X=((X - mu) / sd).astype(np.float32), E=E.astype(np.float32), mu=mu, sd=sd, k=np.array(K),
             features=np.array(["x", "y_out", "z", "att", "pelvis", "hand"]))  # fmt: skip
    return dict(cells=len(ids), out=str(path))


def report() -> dict:
    grid, comp = load("grid"), load("comp")
    pairs = [
        (grid[r["cell"]["grid_id"]], r) for r in comp.values() if r["cell"]["grid_id"] in grid and not r.get("fell")
    ]
    before = np.array([p[0]["miss_after_hold_cm"] for p in pairs])
    after = np.array([p[1]["miss_after_hold_cm"] for p in pairs])
    out = dict(check=check(), comp=dict(cells=len(pairs), miss_cm_before=round(float(before.mean()), 2),
                                        miss_cm_after=round(float(after.mean()), 2),
                                        better=round(float((after < before).mean()), 3),
                                        missed_before=int((before >= 3).sum()), missed_after=int((after >= 3).sum()),
                                        fell=sum(bool(r.get("fell")) for r in comp.values())))  # fmt: skip
    (OUT / "report.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("kind", choices=("grid", "comp"))
    r.add_argument("--procs", type=int, default=12)
    r.add_argument("--gpus", default="0,1")
    w = sub.add_parser("worker")
    w.add_argument("kind", choices=("grid", "comp"))
    w.add_argument("--ids", required=True)
    sub.add_parser("check")
    sub.add_parser("report")
    sub.add_parser("table")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.kind, a.procs, a.gpus)
    elif a.cmd == "worker":
        worker(a.kind, a.ids.split(","))
    elif a.cmd == "table":
        print(json.dumps(write_table()))
    else:
        print(json.dumps(check() if a.cmd == "check" else report(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
