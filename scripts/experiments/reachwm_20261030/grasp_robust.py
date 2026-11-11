"""2026-11-10: how well each cached grasp tolerates the pose ScaleBFM really reaches.

grasp_cache.py validates a grasp with the hand placed exactly at it. In the room the jaw arrives about 1 cm off and pitched a few
degrees up (more the further down the wrist is commanded), and a third of the failures are grasps that close and then lose the
object in the lift. Given the jaw pose a run really reached, the hand-only validator reproduces those: on the development
layouts it lifts 109 of 111 held grasps and loses 9 of 14 dropped ones. So each held grasp is validated again at K poses drawn
from the reached-pose errors of the development runs (ERR file: position and rotation error in the tool frame, with the
commanded pitch), from the samples of its own pitch band, for both finger assignments (room_grasp.py's flip: the error is in
the frame the robot uses, and a flip mirrors it in the cache's frame). No shake: the room does not shake.

Writes <scene>/grasp_robust.npz: rob (n, 2) = share of the K poses that close and lift, columns no flip / flip; -1 where the grasp
was not held in the cache.

--paths <entry_paths.json> (the second version, same day): the end pose alone prefers level grasps, whose entry sweeps the
object more often (development: drops 15 -> 12, knocked away 12 -> 17). So the hand also replays the entry: the jaw's
measured path of a development run in the tool frame (from the pre-grasp to the grip, stops removed, PATH_SPEED), hand open,
then close, lift, hold. A pose fails when the object has moved more than KNOCK_M before the close, or is not lifted. Writes
grasp_robust2.npz: rob (n, 2) and knock (n, 2), the share of poses that knocked. Only grasps whose rob in grasp_robust.npz
is at least --min-rob are run (the others keep their first value, knock -1).

    GRASP_CACHE_ROOT=$I/simready GRASP_PAIR_FAR=1 PYTHONPATH=src $PY grasp_robust.py --ids 3,4,5 --err <err_samples.json> [--k 6]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graspnet_bottle_20261008"))
import grasp_cache as GC  # noqa: E402

GC.SHAKE_S = 0.0  # the shake phase ends at once
BANDS = ((-90.0, 10.0), (10.0, 40.0), (40.0, 180.0))
FLIP = np.diag([1.0, -1.0, -1.0])


def expm(v):
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * K @ K


def draws(err_file: str, k: int):
    S = json.load(open(err_file))
    rng = np.random.default_rng(0)
    out = []
    for lo, hi in BANDS:
        L = [s for s in S if lo <= s["pitch"] < hi]
        idx = rng.choice(len(L), size=min(k, len(L)), replace=False)
        out.append([(np.asarray(L[i]["ep"], float), expm(np.asarray(L[i]["er"], float))) for i in idx])
    return out


PATH_SPEED, KNOCK_M = 0.16, 0.02


def path_draws(path_file: str, k: int):
    S = json.load(open(path_file))
    rng = np.random.default_rng(0)
    out = []
    for lo, hi in BANDS:
        L = [s for s in S if lo <= s["pitch"] < hi]
        idx = rng.choice(len(L), size=min(k, len(L)), replace=False)
        out.append([(np.asarray(L[i]["pos"], float), [expm(np.asarray(v, float)) for v in L[i]["rot"]]) for i in idx])
    return out


def run_path(hm, tool, U, pos, rots) -> dict:
    """The jaw along a measured entry (tool frame of the grasp `tool`, `U`), then grasp_cache.run's close, lift and hold."""
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp

    m, d = hm.m, hm.d
    mujoco.mj_resetData(m, d)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
    T = max(arc[-1] / PATH_SPEED, 0.2)
    keep = np.r_[True, np.diff(arc) > 1e-9]
    arc_k = arc[keep]
    sl = Slerp(arc_k, Rotation.from_matrix(np.array(rots)[keep])) if keep.sum() > 1 else None

    def pose(u):
        a = min(1.0, u) * arc[-1]
        p = np.array([np.interp(a, arc, pos[:, j]) for j in range(3)])
        Re = sl(a).as_matrix() if sl is not None else rots[0]
        R_w = hm.rest_R @ U @ Re
        tool_w = hm.rest_pos + hm.rest_R @ (tool + U @ p)
        return tool_w - R_w @ GC.OFFSET, Rotation.from_matrix(R_w).as_quat()[[3, 0, 1, 2]]

    wrist, quat = pose(0.0)
    d.mocap_pos[0], d.mocap_quat[0] = wrist, quat
    d.qpos[hm.wrist_adr : hm.wrist_adr + 3], d.qpos[hm.wrist_adr + 3 : hm.wrist_adr + 7] = wrist, quat
    d.qpos[hm.obj_adr : hm.obj_adr + 3], d.qpos[hm.obj_adr + 3 : hm.obj_adr + 7] = hm.rest_pos, hm.rest_quat
    for a in hm.f_qadr:
        d.qpos[a] = GC.JAW_OPEN
    mujoco.mj_forward(m, d)
    if hm._contacts()[3] < -0.003:
        return dict(ok=False, knock=True)
    z0, t, phase, began, target = float(d.xpos[hm.obj][2]), 0.0, "entry", 0.0, GC.JAW_OPEN
    wrist0 = wrist
    for _ in range(int((T + GC.CLOSE_S + GC.CONTACT_TIMEOUT_S + GC.LIFT_S + GC.HOLD_S + 1.0) / GC.DT) + 10):
        el = t - began
        if phase == "entry":
            d.mocap_pos[0], d.mocap_quat[0] = pose(el / T)
            if el >= T + 0.2:
                if np.linalg.norm(d.xpos[hm.obj] - hm.rest_pos) > KNOCK_M:
                    return dict(ok=False, knock=True)
                wrist0, phase, began = d.mocap_pos[0].copy(), "close", t
        elif phase == "close":
            target = GC.JAW_OPEN + min(1.0, el / GC.CLOSE_S) * (GC.JAW_CLOSED - GC.JAW_OPEN)
            if el >= GC.CLOSE_S:
                f1, f2, _, _ = hm._contacts()
                if f1 and f2:
                    phase, began = "lift", t
                elif el >= GC.CLOSE_S + GC.CONTACT_TIMEOUT_S:
                    return dict(ok=False, knock=False)
        elif phase == "lift":
            d.mocap_pos[0] = wrist0 + [0.0, 0.0, GC.LIFT_M * GC.minjerk(el / GC.LIFT_S)]
            if el >= GC.LIFT_S:
                phase, began = "hold", t
        elif el >= GC.HOLD_S:
            f1, f2, _, _ = hm._contacts()
            return dict(ok=bool(f1 and f2) and float(d.xpos[hm.obj][2]) - z0 >= GC.RISE_MIN, knock=False)
        for qa, va, act in zip(hm.f_qadr, hm.f_vadr, hm.acts):
            d.ctrl[act] = float(
                np.clip(GC.KP * (target - d.qpos[qa]) - GC.KD * d.qvel[va], -GC.FORCE_MAX, GC.FORCE_MAX)
            )
        mujoco.mj_step(m, d)
        t += GC.DT
    return dict(ok=False, knock=False)


def main_paths(a):
    D = path_draws(a.paths, a.k)
    for oid in (int(x) for x in a.ids.split(",")):
        scene = GC.ROOT / "scenes" / f"obj_{oid:03d}"
        z = np.load(scene / "grasp_cache.npz", allow_pickle=True)
        rob1 = np.load(scene / "grasp_robust.npz")["rob"]
        hm = GC.HandModel(str(scene / "scene_locomani.xml"))
        rob, knock = rob1.copy(), -np.ones_like(rob1)
        for i in np.flatnonzero(z["held"]):
            for f in (0, 1):
                if rob1[i, f] < a.min_rob:
                    continue
                U = z["R"][i] @ (FLIP if f else np.eye(3))
                pitch = np.degrees(np.arcsin(np.clip(-(hm.rest_R @ U)[2, 0], -1, 1)))
                band = next(b for b, (lo, hi) in enumerate(BANDS) if lo <= pitch < hi)
                rs = [run_path(hm, z["tool"][i], U, pos, rots) for pos, rots in D[band]]
                rob[i, f], knock[i, f] = np.mean([r["ok"] for r in rs]), np.mean([r["knock"] for r in rs])
        np.savez_compressed(scene / "grasp_robust2.npz", rob=rob, knock=knock, k=a.k)
        m_ = knock >= 0
        print(
            f"obj {oid:03d}: run {int(m_.sum())} rob2 mean {rob[m_].mean():.2f} (rob1 {rob1[m_].mean():.2f}) knock mean {knock[m_].mean():.2f} share >= 0.75: {(rob[m_] >= 0.75).mean():.2f}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--err", default="")
    ap.add_argument("--paths", default="")
    ap.add_argument("--min-rob", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args()
    if a.paths:
        return main_paths(a)
    D = draws(a.err, a.k)
    for oid in (int(x) for x in a.ids.split(",")):
        scene = GC.ROOT / "scenes" / f"obj_{oid:03d}"
        z = np.load(scene / "grasp_cache.npz", allow_pickle=True)
        hm = GC.HandModel(str(scene / "scene_locomani.xml"))
        n = len(z["held"])
        rob = -np.ones((n, 2))
        for i in np.flatnonzero(z["held"]):
            for f in (0, 1):
                U = z["R"][i] @ (FLIP if f else np.eye(3))
                pitch = np.degrees(np.arcsin(np.clip(-(hm.rest_R @ U)[2, 0], -1, 1)))
                band = next(b for b, (lo, hi) in enumerate(BANDS) if lo <= pitch < hi)
                ok = 0
                for ep, Re in D[band]:
                    r = hm.run(z["tool"][i] + U @ ep, U @ Re)
                    ok += (
                        bool(r.get("closed"))
                        and r.get("rise_m", 0.0) >= GC.RISE_MIN
                        and bool(r.get("touching_after_hold"))
                    )
                rob[i, f] = ok / len(D[band])
        np.savez_compressed(scene / "grasp_robust.npz", rob=rob, k=a.k)
        h = rob[z["held"].astype(bool)]
        print(
            f"obj {oid:03d}: held {len(h)} robust mean {h.mean():.2f} share >= 0.8: {(h >= 0.8).mean():.2f} share <= 0.34: {(h <= 0.34).mean():.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
