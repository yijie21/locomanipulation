"""selfmodel_freespace_20261126: the target grid, the per-episode seeds, the wrist attitudes, the IK labels and the
target-coverage labels (offline, before any episode). 2026-11-26.

    cd $G; PYTHONPATH=$G:$W/src $PY fs_grid.py ik --out $O/grid --procs 8              # -> ik.json
    cd $G; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 PYTHONPATH=$G:$W/src \
        $PY fs_grid.py cover --out $O/grid --backend scalebfm                          # -> cover_<backend>.json
    $PY fs_grid.py show --out $O/grid                                                  # -> grid.json + the tables

Grid (the entry's "目标网格"): heights Z, distances R from the pelvis axis, directions TH (deg from straight ahead,
positive = left = across the body for the right hand), three postures. Pelvis heading frame at the command: x ahead,
y left, z world height. Cell id cid = ((posture * 6 + iz) * 3 + ir) * 4 + ith (0..215); loc = cid % 72 (the spatial
cell, the same for the three postures).

Seeds (both from numpy SeedSequence word lists whose third and fourth words are nonzero, so no integer seed of another
round -- one word, the rest zero -- and no two-word list (hardgrasp's [ep, 4747]) can give the same stream):
  reset_rng(cid, ep)  default_rng([cid, ep, 20261126, 1]): the env's generator before the reset (robosuite's robot
                      init noise, EmptyEnv's "default" 0.02 rad Gaussian on every joint -- data3's start perturbation).
  jitter(loc, ep)     default_rng([loc, ep, 20261126, 2]).uniform(-0.025, 0.025, 3): the target's offset in the cell's
                      +-2.5 cm cube. A function of the spatial cell and the episode only: every group, every
                      controller, every benchmark and the three postures get the same targets.

Wrist (every group and controller): yaw = the direction TH, roll 0, pitch the first of PITCHES whose plan of the
nominal target has IK error <= 2 cm, else the one with the least error. Planned with lml_selfmodel.Planner (protocol 2's
whole-body IK, what LocoManiLab's reaches use for all three controllers) from the posture's standard pose:
  stand        Planner.start_pose() (robot_fk default pose, pelvis 0.8 m), no pelvis height asked;
  standsquat   the same pose, pelvis height PZ_SQ asked in the reach;
  squat        the standard squat: start_pose's squat_plan (both hands lowered with the pelvis to PZ_SQ, as fs_run's
               squat), pelvis height PZ_SQ asked in the reach.
IK label: the chosen attitude's error, <= 2 cm reachable, 2-3 cm edge, > 3 cm out of reach. LocoManiLab's three
controllers start from the same reference pose and plan with the same planner, so their labels are one set.

Target coverage (per controller): D's training split (train3b's D spec: data3, data3b, the two mid files; data.splits'
train at frac 1.0 with D's REACHWM flags), right-hand segments (data/ holds one-hand reaches only), pelvis class as the
posture's (stand: no pelvis height asked; both squat postures: asked 0.55-0.65 m), command target within COVER_M of the
nominal target. The label counts data3 + data3b segments (the reaches themselves); the count with the mid starts (mid
starts are later rows of the same reaches) is reported beside it. >= COVER_N -> "有", else "稀".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(EXP / "selfmodel_data3_20261125"), str(EXP / "locomanilab_20261117"), str(EXP / "hardgrasp_20261125")]

Z = (0.10, 0.35, 0.60, 0.85, 1.10, 1.35)
R = (0.30, 0.50, 0.70)
TH = (30.0, 0.0, -30.0, -60.0)
POSTURES = ("stand", "standsquat", "squat")  # 站立伸手 / 站着出发、蹲下伸手 / 蹲着出发
PZ_SQ = 0.60
PITCHES = (0.0, 45.0, 90.0)
JIT_M = 0.025
IK_OK, IK_EDGE = 0.02, 0.03
COVER_M, COVER_N = 0.075, 10
SQ_CLASS = (0.55, 0.65)
TAG = 20261126
N_LOC = len(Z) * len(R) * len(TH)
ROOT = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs")


def cell(cid: int) -> dict:
    ith, rest = cid % 4, cid // 4
    ir, rest = rest % 3, rest // 3
    iz, ip = rest % 6, rest // 6
    th = TH[ith]
    r, z = R[ir], Z[iz]
    nom = np.array([r * math.cos(math.radians(th)), r * math.sin(math.radians(th)), z])
    return dict(cid=cid, loc=cid % N_LOC, posture=POSTURES[ip], ip=ip, iz=iz, ir=ir, ith=ith, z=z, r=r, th=th,
                pz=None if ip == 0 else PZ_SQ, nominal=nom)  # fmt: skip


def cells() -> list[dict]:
    return [cell(c) for c in range(len(POSTURES) * N_LOC)]


def reset_rng(cid: int, ep: int):
    return np.random.default_rng([cid, ep, TAG, 1])


def jitter(loc: int, ep: int) -> np.ndarray:
    return np.random.default_rng([loc, ep, TAG, 2]).uniform(-JIT_M, JIT_M, 3)


def goal(c: dict, ep: int) -> np.ndarray:
    return np.asarray(c["nominal"], float) + jitter(c["loc"], ep)


def ik_label(err_m: float) -> str:
    return "reach" if err_m <= IK_OK else "edge" if err_m <= IK_EDGE else "out"


def support(c: dict, table: dict) -> str:
    """The entry's sampling-support mark of the jitter cube: C (inside CORE), W (inside WIDE), - (outside WIDE); + x
    when the nominal target is outside the reach table's box for the posture's pelvis class."""
    import d3_collect as D3
    import lml_selfmodel as L

    lo, hi = np.asarray(c["nominal"]) - JIT_M, np.asarray(c["nominal"]) + JIT_M
    if (lo >= L.CORE[0]).all() and (hi <= L.CORE[1]).all():
        s = "C"
    elif (lo >= L.WIDE[0]).all() and (hi <= L.WIDE[1]).all():
        s = "W"
    else:
        s = "-"
    row = table["rows"]["right"][D3.pz_key(c["pz"])]
    return s + ("" if D3.in_table(row, np.asarray(c["nominal"])) else "x")


# -- the standard squat ------------------------------------------------------------------------------------------------
def squat_plan(pl, pose, pz: float = PZ_SQ):
    """Both hands lowered with the pelvis: d3_collect.reach_both (both hands weight 1) from `pose` to each hand point
    and attitude at `pose` moved down by (pelvis z - pz), the pelvis height pz asked. Returns reach_both's output."""
    import d3_collect as D3

    pos, quat, q = pose
    cur = pl.fk_world(pos, quat, q)
    dz = np.array([0.0, 0.0, float(pos[2]) - pz])
    tw = {s: cur[s][0] - dz for s in ("left", "right")}
    rw = {s: cur[s][1] for s in ("left", "right")}
    return D3.reach_both(pl, (np.asarray(pos, float), np.asarray(quat, float), np.asarray(q, float)), tw, rw, pz)


# -- IK labels -----------------------------------------------------------------------------------------------------------
_PL = None
_POSE = {}


def _pose(posture: str):
    global _PL
    import lml_selfmodel as L

    if _PL is None:
        _PL = L.Planner()
    if posture not in _POSE:
        p0 = _PL.start_pose()
        if posture == "squat":
            poses, _, errs, _ = squat_plan(_PL, p0)
            p = poses[-1]
            _POSE[posture] = ((np.asarray(p[0], float), np.asarray(p[1], float), np.asarray(p[2], float)), errs)
        else:
            _POSE[posture] = (p0, None)
    return _PL, _POSE[posture][0]


def _ik_job(cid: int) -> dict:
    import lml_selfmodel as L

    c = cell(cid)
    pl, pose = _pose(c["posture"])
    p, quat = np.asarray(pose[0], float), np.asarray(pose[1], float)
    yaw = L.yaw_of(L._qmat(quat))
    Rh = L.rot_z(yaw)
    nom = np.asarray(c["nominal"])
    tw = np.r_[p[:2] + (Rh @ np.r_[nom[:2], 0.0])[:2], nom[2]]
    errs = {}
    t0 = time.time()
    for pt in PITCHES:
        rw = L.rot_z(yaw + math.radians(c["th"])) @ L.rot_y(math.radians(pt))
        try:
            _, _, e, _ = pl.reach((p, quat, np.asarray(pose[2], float)), "right", tw, rw, c["pz"])
            errs[pt] = float(e)
        except (np.linalg.LinAlgError, ValueError):
            errs[pt] = math.inf
    ok = [pt for pt in PITCHES if errs[pt] <= IK_OK]
    pick = ok[0] if ok else min(PITCHES, key=lambda v: errs[v])
    return dict(cid=cid, pitch_deg=pick, ik_err_cm=100 * errs[pick], ik_label=ik_label(errs[pick]),
                ik_err_by_pitch_cm={f"{k:g}": (100 * v if math.isfinite(v) else None) for k, v in errs.items()},
                plan_s=time.time() - t0)  # fmt: skip


def ik(out: Path, procs: int) -> None:
    from multiprocessing import Pool

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out.mkdir(parents=True, exist_ok=True)
    pl, sq = _pose("squat")
    sq_errs = _POSE["squat"][1]
    t0 = time.time()
    with Pool(procs) as P:
        res = sorted(P.map(_ik_job, range(len(POSTURES) * N_LOC), chunksize=1), key=lambda r: r["cid"])
    doc = dict(made_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), pitches=PITCHES, ik_ok_cm=100 * IK_OK, ik_edge_cm=100 * IK_EDGE,
               squat_pose=dict(pelvis=np.r_[sq[0], sq[1]].tolist(), q=np.asarray(sq[2]).tolist(), hand_err_cm={k: 100 * v for k, v in sq_errs.items()}),
               cells=res, wall_s=time.time() - t0)  # fmt: skip
    (out / "ik.json").write_text(json.dumps(doc, indent=1))
    print(f"ik: {len(res)} cells in {time.time() - t0:.0f} s; standard squat pelvis {np.round(sq[0], 3).tolist()}, hand errors "
          f"{ {k: round(100 * v, 2) for k, v in sq_errs.items()} } cm")  # fmt: skip


# -- target coverage -----------------------------------------------------------------------------------------------------
def d_spec(backend: str) -> str:
    T = ROOT / "selfmodel_train3b_20261125" / backend / "data"
    return ",".join(str(p) for p in (ROOT / "selfmodel_data3_20261125" / backend / "data" / "all.npz",
                                      ROOT / "selfmodel_data3b_20261125" / backend / "data" / "all.npz",
                                      T / "mid_data3.npz", T / "mid_data3b.npz"))  # fmt: skip


def cover(out: Path, backend: str) -> None:
    for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX"):
        assert os.environ.get(k) == "1", f"{k}=1 is D's switch"
    for k in ("REACHWM_X0_REF", "REACHWM_FDVEL"):
        assert os.environ.get(k, "0") == "0", f"{k} must be off (D)"
    sys.path[:0] = [str(EXP / "reachwm_20261030"), str(EXP / "selfwm_20261029")]
    import data as DT

    t0 = time.time()
    spec = d_spec(backend)
    D = DT.load(spec)
    tr = DT.splits(D, 1.0)["train"]
    files = D["files"]
    n_f = [len(np.load(p)["n_steps"]) for p in files]
    origin = np.repeat(np.arange(len(files)), n_f)
    assert len(origin) == len(D["n_steps"]), "per-file counts do not add up"
    real = origin < 2  # data3, data3b
    side, pz, tg = np.asarray(D["side"]), np.asarray(D["pelvis_z"]).astype(float), np.asarray(D["target"]).astype(float)
    m_tr = np.zeros(len(side), bool)
    m_tr[tr] = True
    base = m_tr & (side > 0)
    cls = dict(stand=base & (pz < 0), squat=base & (pz >= SQ_CLASS[0]) & (pz <= SQ_CLASS[1]))
    res = []
    for c in cells():
        m = cls["stand" if c["posture"] == "stand" else "squat"]
        near = np.linalg.norm(tg - np.asarray(c["nominal"])[None], axis=1) <= COVER_M
        n_real, n_all = int((m & near & real).sum()), int((m & near).sum())
        res.append(dict(cid=c["cid"], n_cover=n_real, n_cover_with_mid=n_all, cover="有" if n_real >= COVER_N else "稀"))
    doc = dict(backend=backend, spec=spec, files=files, n_per_file=n_f, n_train=int(len(tr)),
               n_train_right=dict(stand=int((cls["stand"] & real).sum()), squat=int((cls["squat"] & real).sum())),
               cover_m=COVER_M, cover_n=COVER_N, cells=res, wall_s=time.time() - t0)  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / f"cover_{backend}.json").write_text(json.dumps(doc, indent=1))
    print(f"cover {backend}: {len(tr)} training segments ({doc['n_train_right']} right-hand, data3 + data3b); "
          f"有 {sum(r['cover'] == '有' for r in res)} / {len(res)} cells; {time.time() - t0:.0f} s")  # fmt: skip


# -- the merged labels ---------------------------------------------------------------------------------------------------
def show(out: Path) -> None:
    import d3_collect as D3

    table = json.loads(D3.TABLE.read_text())
    ikd = json.loads((out / "ik.json").read_text())
    cov = {be: json.loads((out / f"cover_{be}.json").read_text()) for be in ("scalebfm", "heft", "sonic") if (out / f"cover_{be}.json").exists()}
    rows = []
    for c, k in zip(cells(), ikd["cells"]):
        assert k["cid"] == c["cid"]
        r = dict(cid=c["cid"], loc=c["loc"], posture=c["posture"], z=c["z"], r=c["r"], th=c["th"], nominal=np.round(c["nominal"], 6).tolist(),
                 support=support(c, table), pitch_deg=k["pitch_deg"], ik_err_cm=k["ik_err_cm"], ik_label=k["ik_label"],
                 ik_err_by_pitch_cm=k["ik_err_by_pitch_cm"], cover={})  # fmt: skip
        for be, d in cov.items():
            cc = d["cells"][c["cid"]]
            assert cc["cid"] == c["cid"]
            r["cover"][be] = dict(n=cc["n_cover"], n_with_mid=cc["n_cover_with_mid"], label=cc["cover"])
        rows.append(r)
    doc = dict(made_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), table=str(D3.TABLE), ik=ikd["made_utc"], squat_pose=ikd["squat_pose"],
               cover={be: dict(spec=d["spec"], n_train=d["n_train"], n_train_right=d["n_train_right"]) for be, d in cov.items()}, cells=rows)  # fmt: skip
    (out / "grid.json").write_text(json.dumps(doc, indent=1))
    lab = {"reach": "R", "edge": "e", "out": "o"}
    for ip, ps in enumerate(POSTURES):
        print(f"\n== {ps}: support / IK label (R reachable, e edge, o out) / pitch / IK cm" + "".join(f" / {be} cover n" for be in cov))
        print("z \\ r,th   " + " | ".join(f"{r:.2f} m: " + " ".join(f"{t:+.0f}" for t in TH) for r in R))
        for iz, z in enumerate(Z):
            parts = []
            for ir in range(3):
                cs = [rows[((ip * 6 + iz) * 3 + ir) * 4 + ith] for ith in range(4)]
                parts.append(" ".join(f"{x['support']}{lab[x['ik_label']]}{int(x['pitch_deg'])}:{x['ik_err_cm']:.1f}" +
                                      "".join(f"/{x['cover'][be]['n']}" for be in cov) for x in cs))  # fmt: skip
            print(f"{z:.2f}   " + " | ".join(parts))
        sel = [x for x in rows if x["posture"] == ps]
        print(f"   IK: reachable {sum(x['ik_label'] == 'reach' for x in sel)}, edge {sum(x['ik_label'] == 'edge' for x in sel)}, "
              f"out {sum(x['ik_label'] == 'out' for x in sel)}; pitch " + str({p: sum(x['pitch_deg'] == p for x in sel) for p in PITCHES}) +
              "".join(f"; {be} 有 {sum(x['cover'][be]['label'] == '有' for x in sel)}" for be in cov))  # fmt: skip
    print(f"\nall: IK reachable {sum(x['ik_label'] == 'reach' for x in rows)}, edge {sum(x['ik_label'] == 'edge' for x in rows)}, out "
          f"{sum(x['ik_label'] == 'out' for x in rows)} of {len(rows)}" +
          "".join(f"; {be} 有 {sum(x['cover'][be]['label'] == '有' for x in rows)}, reachable and 有 "
                  f"{sum(x['cover'][be]['label'] == '有' and x['ik_label'] == 'reach' for x in rows)}" for be in cov))  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["ik", "cover", "show"])
    ap.add_argument("--out", type=Path, default=ROOT / "selfmodel_freespace_20261126" / "grid")
    ap.add_argument("--backend", default="scalebfm")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.phase == "ik":
        ik(a.out, a.procs)
    elif a.phase == "cover":
        cover(a.out, a.backend)
    else:
        show(a.out)
