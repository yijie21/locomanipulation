"""selfmodel_data3_20261125: one collector for ScaleBFM, HEFT and SONIC in LocoManiLab's benchmark host (2026-11-25).

    PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
    G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend; cd $G
    export PYTHONPATH=$G:$W/src SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1
    $PY d3_collect.py gen   --procs 60                       # IK probe -> d3_reach_table.json (fixed, next to this file)
    $PY d3_collect.py drive --backend scalebfm --out DIR --resets 30000 --shards 60 --shard K [--no-walk]
    $PY d3_collect.py merge --out DIR
    $PY d3_collect.py check --out DIR

The host is lml_host's (humanoid_benchmark robosuite G1FloatingBody, G1_BRIDGE, EmptyEnv, Dex3 hands): HostDriver with
backend scalebfm (humanoid_transformer_m) or heft (heft/pmg), d3_sonic.SonicHost for sonic. Every backend is driven by
the same calls -- reset, start_frames, push, step, sample (and its m / d / pl / to_mj / c) -- and nothing else: the
reaches are planned here (our whole-body IK on robot_fk, as lml_selfmodel.OnlineDriver._plan, protocol 2: placed and
solved from the measured state at the command row, the reference spliced after the current frame and shifted in x/y onto
the current reference root), not by Host.plan, so one and two-hand reaches go through one code path for all three.

One reset (the entry's "做法" plus revision 0; draws from default_rng(seed + i), in this order whatever the backend):
  walk      a third of the resets walk first: 0.3-1.5 m in a uniformly random direction, the heading turned 0-90 deg
            (either way), as hg_walk.walk_frames (turn toward the goal, walk straight, turn to the goal heading; a gait
            reference with footsteps, the upper body in the default pose). The goal is placed by the SIMULATOR pelvis
            at the walk's start (the data need not walk accurately, so no leg odometry); the walk starts at 0.7 s.
  reaches   two, each commanded only once the robot has settled; the hand(s) drawn per reach: right 40 %, left 40 %,
            both at once 20 % (each hand's target drawn independently, one pelvis height).
  settle    pelvis and both hand points (13 cm ahead of wrist_yaw_link) all slower than 2 cm/s for 0.5 s on end.
            Speed: displacement over the last 0.1 s (5 policy steps), evaluated every step, so the 0.5 s are 25
            consecutive steps (data.py's `settled` uses the same 0.1 s differences on the acting jaw at 10 Hz).
            The wait opens when the robot is told its reference has stopped: the reach's (or walk's) last reference
            frame plus the buffer delay (Host: buf.delay_s, ScaleBFM 0.14 s, HEFT 0.16 s; SonicHost.ref_delay_s 0.94 s; the
            `lag` phase measures each controller's real lag on single-joint ramps: 0.16 / 0.16 / 0.96 s); the 0.5 s window
            must lie after that; at most 3 s. Before the first in-place reach the wait opens at 0.2 s, so the earliest
            command is at 0.7 s, where hostdata commanded it. The reach that ends a wait is commanded on that row, and
            that row is the previous reach's end label (its segment's last row, on its 10 Hz grid). A wait that runs
            out: the reach it was for is skipped and logged ("unsettled"); the previous reach keeps its row at 3 s as
            its label, flagged settled_end = 0. A first reach skipped this way gives the second its own 3 s.
  command   pelvis height: standing 45 % (no height asked; gen_reach), else uniform 0.52-0.76 m. Target in the pelvis
            heading frame (x ahead, y left, world z), drawn in the hand's reach table for the nearest probed pelvis
            height (d3_reach_table.json, written once by `gen`): z uniform over the heights the IK probe reached within
            0.05-1.5 m, then x / y uniform in the box the probe reached at the nearest probed height (WIDE); 70 % of
            the draws (gen_reach's CORE share) in that box cut to gen_reach's CORE x / y (y mirrored for the left hand)
            and z 0.15-1.35. Wrist: R = Rz(heading + yaw) Ry(pitch) Rx(roll), yaw +-90, pitch -30..90, roll +-90 deg,
            uniform; recorded as rot6 (the first two columns of R in the heading frame).
  ends      a fall (pelvis below 0.45 m), a lost reach (pelvis displacement off the reference's by 0.5 m since the
            anchor: the reach's command, or the walk's end while waiting after a walk) or an IK exception ends the reset
            (lml_host's rules; the reference's displacement is taken the backend's reference delay behind, as d3_sonic
            does: SONIC's robot is 0.94 s behind the frames pushed, which during a walk alone is ~0.4 m). Revision 1:
            no lost during the walk itself (a fall still ends it); its largest deviation is logged (walk_dev_cm).

Written (gen_reach / segments2's layout, so reachwm_20261030/data.py and train.py read it unchanged; S is hostdata's
65-D state: joints and joint velocities in ScaleBFM order, gravity in the pelvis frame, gyro, simulator pelvis height):
  DIR/data/       one-hand segments: w<shard>_<chunk>.npz -> all.npz. walked = 1 on every segment of a walk-first
                  reset (gen_reach's meaning: the reach starts after a walk).
                  per row (10 Hz, PRE_S before the command to the end label), as hostdata: S (65), W (pelvis xyz, yaw,
                  right hand point, left hand point, torso point; world), J (acting wrist's R columns 0-1, world), C (the
                  reference hand in the pelvis frame, time since the command); added: HQ (2, 4) both wrists' attitude
                  (wrist_yaw_link, quaternion wxyz, world; [right, left]), KP (2, 3, 3) both hands' hand point (the data's
                  jaw: 13 cm ahead of wrist_yaw_link), thumb tip, index tip (world; tips: the distal link's collision
                  mesh vertex furthest along the link).
                  per segment, as hostdata: k0, side, target (heading xy, world z), dyaw, pitch, rot6, pelvis_z (-1
                  standing), start (pelvis xyz, yaw at the command), walked, chain (reaches done before in the reset),
                  homed, fell, lost, ok, jaw_err_cm (last row's hand point to the target), att_err_deg, ik_err_cm,
                  reach_s, reset, ref_s, n_sidesteps, ref_q, ref_pel, sidestep, wrist_dev; added: hands (1 right, -1
                  left, 2 both), roll, core (CORE or WIDE draw), slot (0 / 1: which of the reset's two draws),
                  settled_end (the label row is settled), settle_s (label row after the reference end), pre_wait_s
                  (settle wait before the command, from the wait's opening), v_end_cms (the settle speed's max over the
                  label's last 0.5 s), ref_delay_s (the backend's reference delay: the robot executes a pushed frame
                  that much later; the settle wait and so the end label are counted from the reference end plus it),
                  end_pos (2, 3) / end_quat (2, 4) / end_kp (2, 3, 3) both hands at the label in the
                  heading frame (hand point; wrist attitude wxyz; hand point, thumb tip, index tip), walk_cmd (forward,
                  left, turn asked), walk_real (done, measured at the first command, start heading frame), walk_dev_cm
                  (largest walk deviation), cmd_target (the drawn target), target_o / rot6_o / ik_err_o_cm (the other
                  hand's command in a two-hand reach, else 0), hand_q (the acting hand's 7 finger joints at the
                  command), and the via-model fields with no via points (n_via 0, via_p / via_r6 / via_gen zeros,
                  via_speed 0.15, via_route -1) plus n_ref and RT (per IK waypoint: its time after the command).
  DIR/data/arm/   -> DIR/data/arm.npz (via_model.build's arm.npz): X (rows, 24, 3) the acting arm's 24 collision spheres
                  (cuRobo's G1 + Dex3 spheres, arm_fk.ArmFK's set and mirroring) at every recorded row, from the
                  simulator's bodies; XR (waypoints, 24, 3) the same spheres on each IK waypoint (the plan in measured
                  coordinates, the fingers as at the command); RQ / RP the waypoints' joints (MuJoCo order) and pelvis
                  (robot_fk convention); all in the segment's heading frame at the command.
  DIR/data_both/  two-hand reaches in the same layout, one segment per hand (side = the hand whose labels it holds),
                  hands = 2; kept out of data/ so the one-hand end model does not train on them.
  DIR/data/w*_log.json  one entry per reset: walk asked / done, each slot's fate, fall / lost / why, wall seconds.
Splits: data.py's (crc32 of the reset id, buckets 0-7 train, 8 val, 9 test), reset ids from 20000000, seeds from
73000000 (hostdata: 41000000 / 4000000; hostgap 43000000 / 6000000; hardgrasp 7000000+ / 9000000+ episode ids).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP / "locomanilab_20261117"))
sys.path.insert(0, str(EXP / "hardgrasp_20261125"))
import lml_selfmodel as L  # noqa: E402

SEED, ID0, RESETS, CHUNK = 73_000_000, 20_000_000, 30000, 50
FPS, TICKS = L.FPS, L.TICKS
PRE_F = int(round(L.PRE_S * FPS))  # 15: three rows before the command (S[k0-2..k0])
K0 = PRE_F // TICKS
CHAIN = 2
WALK_P, WALK_M, WALK_TURN_DEG = 1 / 3, (0.3, 1.5), 90.0
WALK_F = int(round(L.START_HOLD_S * FPS))  # 35
SETTLE_V, SETTLE_F, SETTLE_MAX_F = 0.02, int(round(0.5 * FPS)), int(round(3.0 * FPS))
FIRST_OPEN_F = WALK_F - SETTLE_F  # 10
MODES = (("right", 0.4), ("left", 0.4), ("both", 0.2))
STAND_P, PZ_RANGE, CORE_P = 0.45, (0.52, 0.76), 0.7
DYAW_DEG, PITCH_DEG, ROLL_DEG = (-90.0, 90.0), (-30.0, 90.0), (-90.0, 90.0)
Z_RANGE, CORE_Z = (0.05, 1.5), (0.15, 1.35)
HOLD_S = 4.5  # reference hold appended after each splice (> buffer delay + 3 s wait)
TABLE = HERE / "d3_reach_table.json"
# the probe: pelvis heights (None = standing), heights of the hand, the x / y grid (right hand; the left mirrored),
# the wrist attitudes tried in turn until one reaches (yaw 0; pitch 0 = ahead, 90 = down), the reach criterion
PROBE_PZ = (None, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76)
PROBE_Z = tuple(round(0.05 + 0.1 * k, 2) for k in range(15)) + (1.5,)
PROBE_X = tuple(round(0.05 + 0.1 * k, 2) for k in range(12))  # 0.05..1.15 (gen_reach WIDE starts at 0.05)
PROBE_Y = tuple(round(-0.95 + 0.1 * k, 2) for k in range(13)) + (0.35,)  # outward -0.95..0.25, across to 0.35 (WIDE)
PROBE_PITCH = (0.0, 90.0)
PROBE_OK_M = 0.02
# cuRobo's G1 + Dex3 collision spheres (SIMPLE data/robots/g1/curobo/spheres/g1_29dof.yml, right side; the order and
# mirroring of selfmodel_via_20261120/arm_fk.py): link, centre in the link frame, radius
SPHERES = (
    ("shoulder_pitch_link", (-0.0, -0.038, -0.016), 0.04007),
    ("shoulder_roll_link", (0.001, -0.006, -0.08), 0.0326), ("shoulder_roll_link", (0.002, -0.011, -0.065), 0.02923),
    ("shoulder_roll_link", (-0.008, -0.002, -0.061), 0.02518),
    ("shoulder_yaw_link", (0.015, 0.003, -0.084), 0.03284), ("shoulder_yaw_link", (-0.006, 0.001, -0.019), 0.03284),
    ("elbow_link", (0.071, -0.002, -0.011), 0.03819),
    ("wrist_pitch_link", (0.006, 0.001, 0.001), 0.03112), ("wrist_pitch_link", (0.049, -0.002, -0.001), 0.03112),
    ("hand_palm_link", (0.055, 0.007, 0.0), 0.01), ("hand_palm_link", (0.055, 0.007, 0.025), 0.01),
    ("hand_palm_link", (0.055, 0.007, -0.025), 0.01),
    ("hand_thumb_0_link", (-0.001, 0.02, 0.0), 0.01),
    ("hand_thumb_1_link", (-0.0012, 0.025, 0.0), 0.01), ("hand_thumb_1_link", (-0.0012, 0.045, 0.0), 0.01),
    ("hand_thumb_2_link", (-0.0015, 0.025, 0.0), 0.01),
    ("hand_middle_0_link", (0.0, 0.001, 0.0), 0.01), ("hand_middle_0_link", (0.025, 0.0, 0.0), 0.01),
    ("hand_middle_1_link", (0.0, -0.0015, 0.0), 0.01), ("hand_middle_1_link", (0.03, -0.002, 0.0), 0.01),
    ("hand_index_0_link", (0.0, 0.001, 0.0), 0.01), ("hand_index_0_link", (0.025, 0.0, 0.0), 0.01),
    ("hand_index_1_link", (0.0, -0.0015, 0.0), 0.01), ("hand_index_1_link", (0.03, -0.002, 0.0), 0.01),
)  # fmt: skip
PALM_OFF = {"right": np.array([0.0415, -0.003, 0.0]), "left": np.array([0.0415, 0.003, 0.0])}  # URDF hand_palm_joint
FINGERS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
SIDES = ("right", "left")


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def pz_key(pz) -> str:
    """The probed pelvis height nearest a command's (None = standing)."""
    if pz is None:
        return "stand"
    return f"{min(PROBE_PZ[1:], key=lambda v: abs(v - pz)):.2f}"


# -- the reach table (gen) ------------------------------------------------------------------------------------------
_PL = None


def _probe(job):
    """One (hand, pelvis height, hand height) slice of the grid: the best IK hand error of the wrist attitudes tried."""
    global _PL
    if _PL is None:
        _PL = L.Planner()
    side, pz, z, i = job
    pose = _PL.start_pose()
    sy = 1.0 if side == "right" else -1.0
    err = np.full(len(PROBE_Y), np.inf)
    t0 = time.time()
    x = PROBE_X[i]
    if True:
        for j, y in enumerate(PROBE_Y):
            for p in PROBE_PITCH:  # the next attitude only while none has reached
                try:
                    _, _, e, _ = _PL.reach(pose, side, np.array([x, sy * y, z]), L.rot_y(math.radians(p)), pz)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                err[j] = min(err[j], e)
                if err[j] < PROBE_OK_M:
                    break
    return side, pz, z, i, err, time.time() - t0


def gen(procs: int, out: Path) -> None:
    from multiprocessing import Pool

    jobs = [(s, pz, z, i) for s in SIDES for pz in PROBE_PZ for z in PROBE_Z for i in range(len(PROBE_X))]
    t0 = time.time()
    part = []
    with Pool(procs) as pool:
        for r in pool.imap_unordered(_probe, jobs, chunksize=1):
            part.append(r)
            if len(part) % 100 == 0:
                print(f"probe {len(part)}/{len(jobs)} jobs, {time.time() - t0:.0f} s", flush=True)
    grid = {}
    for side, pz, z, i, e, dt in part:
        g = grid.setdefault((side, pz, z), [np.full((len(PROBE_X), len(PROBE_Y)), np.inf), 0.0])
        g[0][i], g[1] = e, g[1] + dt
    res = [(side, pz, z, g[0], g[1]) for (side, pz, z), g in grid.items()]
    table = dict(meta=dict(x=PROBE_X, y=PROBE_Y, z=PROBE_Z, pitch_deg=PROBE_PITCH, ok_m=PROBE_OK_M, pz=[pz_key(p) for p in PROBE_PZ],
                           note="y is the hand's outward-signed coordinate: right hand y as is, left hand y = -(listed); "
                                "box = [x_lo, x_hi, y_lo, y_hi] in the pelvis heading frame (true sign), null = nothing reached",
                           seconds=round(time.time() - t0, 1)), rows={})  # fmt: skip
    raw = {}
    for side, pz, z, err, _ in res:
        ok = err < PROBE_OK_M
        row = table["rows"].setdefault(side, {}).setdefault(pz_key(pz), dict(z=[], box=[], n_ok=[]))
        row["z"].append(z)
        row["n_ok"].append(int(ok.sum()))
        if ok.any():
            xs = np.array(PROBE_X)[ok.any(1)]
            ys = np.array(PROBE_Y)[ok.any(0)] * (1.0 if side == "right" else -1.0)
            row["box"].append([float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())])
        else:
            row["box"].append(None)
        raw[f"{side}_{pz_key(pz)}_{z:.2f}"] = err
    for rows in table["rows"].values():  # imap_unordered: put each row's heights in order
        for r in rows.values():
            o = sorted(range(len(r["z"])), key=lambda k: r["z"][k])
            for key in ("z", "box", "n_ok"):
                r[key] = [r[key][k] for k in o]
    out.write_text(json.dumps(table, indent=1))
    np.savez_compressed(out.with_suffix(".raw.npz"), **raw)
    print(f"reach table -> {out} ({len(res)} slices, {len(jobs)} jobs, {time.time() - t0:.0f} s, worst slice {max(r[4] for r in res):.0f} cpu s)")
    show_table(out)


def show_table(path: Path = TABLE) -> None:
    T = json.loads(Path(path).read_text())
    for side, rows in T["rows"].items():
        for key, r in rows.items():
            zs = [z for z, b in zip(r["z"], r["box"]) if b is not None]
            print(f"{side:5s} pelvis {key:5s}: hand height {min(zs):.2f}-{max(zs):.2f} m reached; " + "; ".join(
                f"z {z:.2f} x {b[0]:.2f}..{b[1]:.2f} y {b[2]:+.2f}..{b[3]:+.2f}" for z, b in zip(r["z"], r["box"]) if b is not None and z in (0.05, 0.45, 0.85, 1.25, 1.5)))  # fmt: skip


# -- commands -------------------------------------------------------------------------------------------------------
def sample_target(rng, row: dict, side: str, core: bool) -> np.ndarray:
    zs = np.array(row["z"], float)
    ok = np.array([b is not None for b in row["box"]])
    zlo, zhi = max(Z_RANGE[0], zs[ok].min()), min(Z_RANGE[1], zs[ok].max())
    if core:
        zlo, zhi = max(zlo, CORE_Z[0]), min(zhi, CORE_Z[1])
    z = float(rng.uniform(zlo, zhi))
    j = int(np.argmin(np.where(ok, np.abs(zs - z), np.inf)))
    xlo, xhi, ylo, yhi = row["box"][j]
    u = rng.uniform(0.0, 1.0, 2)  # drawn whatever the box, so CORE / WIDE do not shift the stream
    if core:
        clo, chi = L.CORE
        cy = (clo[1], chi[1]) if side == "right" else (-chi[1], -clo[1])
        a, b, c, d = max(xlo, clo[0]), min(xhi, chi[0]), max(ylo, cy[0]), min(yhi, cy[1])
        if a < b and c < d:
            xlo, xhi, ylo, yhi = a, b, c, d
    return np.array([xlo + u[0] * (xhi - xlo), ylo + u[1] * (yhi - ylo), z])


def sample_command(rng, table: dict) -> dict:
    r = rng.random()
    mode = "right" if r < MODES[0][1] else "left" if r < MODES[0][1] + MODES[1][1] else "both"
    pz = None if rng.random() < STAND_P else float(rng.uniform(*PZ_RANGE))
    hands = {}
    for s in ("right", "left") if mode == "both" else (mode,):
        core = bool(rng.random() < CORE_P)
        t = sample_target(rng, table["rows"][s][pz_key(pz)], s, core)
        hands[s] = dict(target=t, core=core, dyaw=math.radians(rng.uniform(*DYAW_DEG)),
                        pitch=math.radians(rng.uniform(*PITCH_DEG)), roll=math.radians(rng.uniform(*ROLL_DEG)))  # fmt: skip
    return dict(mode=mode, pelvis_z=pz, hands=hands)


def reset_plan(rng, table: dict, walk_ok: bool) -> dict:
    u, d, a, turn = rng.random(), rng.uniform(*WALK_M), rng.uniform(-math.pi, math.pi), rng.uniform(-1.0, 1.0)
    walk = None
    if u < WALK_P:
        walk = dict(d=float(d), dir=float(a), turn=float(math.radians(WALK_TURN_DEG) * turn), done=walk_ok)
    return dict(walk=walk, cmds=[sample_command(rng, table) for _ in range(CHAIN)])


# -- planning (our whole-body IK; one and two hands) ------------------------------------------------------------------
def reach_both(pl, pose, tw: dict, rw: dict, pz):
    """Planner.reach with both hands moving at once (weights 1 and 1): each hand on its straight line and its
    rotation-vector turn, the waypoints REACH_STEP_M apart along the longer path, the duration that path's."""
    from scipy.spatial.transform import Rotation

    from locomani.primitives.whole_body_ik import WholeBodyIK

    pos, quat, q = pose
    solver = WholeBodyIK(pl.ik, dict(joints=q, pelvis=dict(pos=pos, quat=quat)), L.IK_CFG, tool_offset=tuple(L.HAND_OFF),
                         knee_max=1.5, pelvis_forward_max=0.20, pelvis_height_target=pz, jacobian="analytic")  # fmt: skip
    if pz is not None:
        solver.hi[2] = max(solver.hi[2], pz + 0.01)
    solver.seed = np.clip(solver.initial, solver.lo, solver.hi)
    cur = pl.fk_world(pos, quat, q)
    a = {s: cur[s][0] for s in ("left", "right")}
    R0 = {s: cur[s][1] for s in ("left", "right")}
    rv = {s: Rotation.from_matrix(R0[s].T @ rw[s]).as_rotvec() for s in ("left", "right")}
    length = max(float(np.linalg.norm(np.asarray(tw[s]) - a[s])) for s in ("left", "right"))
    n = max(1, int(math.ceil(length / L.REACH_STEP_M)))
    dur = max(L.REACH_MIN_S, length / L.REACH_SPEED)
    poses, durs, errs = [pose], [], {}
    for k in range(1, n + 1):
        u = k / n
        tp = [a[s] + u * (np.asarray(tw[s]) - a[s]) for s in ("left", "right")]
        tr = [R0[s] @ Rotation.from_rotvec(u * rv[s]).as_matrix() for s in ("left", "right")]
        solver.hand_weights = np.array([1.0, 1.0])
        sol = solver.solve(np.array(tp), np.array(tr), hand_rotation_weight=L.ROT_W)
        qw = Rotation.from_matrix(np.asarray(sol.rotation)).as_quat()[[3, 0, 1, 2]]
        poses.append((np.asarray(sol.position, float), qw, np.asarray(sol.joints, float).copy()))
        durs.append(dur / n)
        errs = {"left": float(sol.hand_errors[0]), "right": float(sol.hand_errors[1])}
    return poses, durs, errs, dur


def plan_reach(h, x: dict, ref_frame: np.ndarray, cmd: dict):
    """lml_selfmodel.OnlineDriver._plan with the wrist roll and two-hand commands: blended frames to follow ref_frame,
    the last (shifted) pose and the record (world = measured coordinates)."""
    p, quat = x["root"][:3].astype(float), x["root"][3:].astype(float)
    yaw = L.yaw_of(L._qmat(quat))
    Rh = L.rot_z(yaw)
    hands = {}
    for s, c in cmd["hands"].items():
        tw = np.r_[p[:2] + (Rh @ np.r_[c["target"][:2], 0.0])[:2], c["target"][2]]
        rw = L.rot_z(yaw + c["dyaw"]) @ L.rot_y(c["pitch"]) @ rot_x(c["roll"])
        hands[s] = dict(target_w=tw, rot_w=rw, dyaw=c["dyaw"], pitch=c["pitch"], roll=c["roll"], core=c["core"],
                        cmd_target=np.asarray(c["target"], float))  # fmt: skip
    q_mj = x["q"][h.to_mj].astype(float)
    t0 = time.time()
    if len(hands) == 1:
        (s,) = hands
        poses, durs, e, dur = h.pl.reach((p, quat, q_mj), s, hands[s]["target_w"], hands[s]["rot_w"], cmd["pelvis_z"])
        errs = {s: float(e)}
    else:
        poses, durs, errs, dur = reach_both(h.pl, (p, quat, q_mj), {s: v["target_w"] for s, v in hands.items()},
                                            {s: v["rot_w"] for s, v in hands.items()}, cmd["pelvis_z"])  # fmt: skip
    plan_s = time.time() - t0
    for s in hands:
        hands[s]["ik_err"] = errs[s]
    delta = np.r_[ref_frame[:2] - p[:2], 0.0]
    ref_pose = (ref_frame[:3].astype(float), ref_frame[3:7].astype(float), ref_frame[7:].astype(float))
    shifted = [ref_pose] + [(pp + delta, qq, jj) for pp, qq, jj in poses[1:]]
    last = poses[-1]
    rec = dict(mode=cmd["mode"], hands=hands, pelvis_z=cmd["pelvis_z"], reach_s=dur, plan_s=plan_s, ref_q=last[2].copy(),
               ref_pel=np.r_[last[0], last[1]], delta=delta, RT=np.r_[0.0, np.cumsum(durs)],
               RQ=np.array([pp[2] for pp in poses]), RP=np.array([np.r_[pp[0], pp[1]] for pp in poses]))  # fmt: skip
    return L.blend_frames(shifted, durs), shifted[-1], rec


# -- the host wrapper -----------------------------------------------------------------------------------------------
class Rig:
    """Whatever the backend: the arm spheres (live and on a reference pose), the fingers, the settle speeds."""

    def __init__(self, h):
        mj, m = h.mj, h.m
        self.h, self.mj = h, mj
        bid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "robot0_" + n)  # noqa: E731
        jad = lambda n: int(m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "robot0_" + n)])  # noqa: E731
        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        self.links = [s[0] for s in SPHERES]
        self.radius = np.array([s[2] for s in SPHERES], np.float32)
        self.body, self.local = {}, {}
        for side in SIDES:
            mir = np.array([1.0, 1.0 if side == "right" else -1.0, 1.0])
            ids, loc = [], []
            for link, c, _ in SPHERES:
                b = bid(f"{side}_wrist_yaw_link" if link == "hand_palm_link" else f"{side}_{link}")
                assert b >= 0, f"{side}_{link} is not a body of the host"
                ids.append(b)
                loc.append((PALM_OFF[side] if link == "hand_palm_link" else 0.0) + np.array(c) * mir)
            self.body[side], self.local[side] = np.array(ids), np.array(loc)
        self.fing = {s: np.array([jad(f"{s}_hand_{f}_joint") for f in FINGERS]) for s in SIDES}
        self.qadr_mj = np.array([jad(n) for n in MUJOCO_JOINT_NAMES])
        fj = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "robot0_floating_base_joint")
        self.rq = int(m.jnt_qposadr[fj])
        self.pelvis = bid("pelvis")
        assert int(m.jnt_bodyid[fj]) == self.pelvis, "the free joint is not on the pelvis"
        self.scratch = mj.MjData(m)
        self.c = np.asarray(h.c, float)
        # fingertips: on the distal thumb / index link, the collision mesh's vertex furthest along the link (the
        # direction from the body origin to the mesh's centre)
        self.tip = {}
        for side in SIDES:
            for f in ("thumb_2", "index_1"):
                b = bid(f"{side}_hand_{f}_link")
                g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH]
                g = max(g, key=lambda k: int(m.geom_contype[k]))
                mid = int(m.geom_dataid[g])
                V = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
                Rg = np.zeros(9)
                mj.mju_quat2Mat(Rg, m.geom_quat[g])
                Vb = V @ Rg.reshape(3, 3).T + m.geom_pos[g]
                u = m.geom_pos[g] / np.linalg.norm(m.geom_pos[g])
                self.tip[(side, f)] = (b, Vb[int(np.argmax(Vb @ u))].copy())
        self._q = np.zeros(4)
        self.v = deque(maxlen=SETTLE_F)
        self.pts = deque(maxlen=TICKS + 1)

    def spheres(self, d, side: str) -> np.ndarray:
        b = self.body[side]
        Rm = d.xmat[b].reshape(-1, 3, 3)
        return d.xpos[b] + np.einsum("kij,kj->ki", Rm, self.local[side])

    def ref_spheres(self, side: str, root7: np.ndarray, q_mj: np.ndarray, fingers: np.ndarray) -> np.ndarray:
        """The side's spheres with the pelvis at root7 (robot_fk convention: Host.pelvis_pose's point) and the joints."""
        d = self.scratch
        R = L._qmat(root7[3:7])
        d.qpos[:] = self.h.d.qpos
        d.qpos[self.rq : self.rq + 3] = np.asarray(root7[:3], float) - R @ self.c
        d.qpos[self.rq + 3 : self.rq + 7] = root7[3:7]
        d.qpos[self.qadr_mj] = q_mj
        d.qpos[self.fing[side]] = fingers
        self.mj.mj_kinematics(self.h.m, d)
        return self.spheres(d, side)

    def check_root(self) -> float:
        """|pelvis_pose() - (free joint position + R c)|: ref_spheres' root convention, on the live state."""
        d = self.h.d
        p, R = self.h.pelvis_pose()
        return float(np.linalg.norm(p - (d.qpos[self.rq : self.rq + 3] + R @ self.c)))

    def quat(self, R: np.ndarray) -> np.ndarray:
        self.mj.mju_mat2Quat(self._q, np.asarray(R, float).reshape(9))
        return self._q.copy()

    def keypoints(self, d, x: dict, side: str) -> np.ndarray:
        """(3, 3): the hand point (13 cm ahead of wrist_yaw_link, the data's jaw), the thumb tip, the index tip; world."""
        pw, Rw = x["wr"][side]
        out = [pw + Rw @ L.HAND_OFF]
        for f in ("thumb_2", "index_1"):
            b, o = self.tip[(side, f)]
            out.append(d.xpos[b] + d.xmat[b].reshape(3, 3) @ o)
        return np.array(out)

    def record(self, x: dict) -> float:
        """Add the row's spheres, fingers, both hands' attitudes (wrist_yaw_link quaternion wxyz) and keypoints to x;
        returns the settle speed (max over pelvis and both hand points of the displacement over the last 0.1 s, m/s;
        inf until there is one)."""
        d = self.h.d
        x["sph"] = {s: self.spheres(d, s).astype(np.float32) for s in SIDES}
        x["fing"] = {s: d.qpos[self.fing[s]].astype(np.float32) for s in SIDES}
        x["hq"] = np.array([self.quat(x["wr"][s][1]) for s in SIDES], np.float32)
        x["kp"] = np.array([self.keypoints(d, x, s) for s in SIDES], np.float32)
        pts = np.stack([x["root"][:3]] + [pw + Rw @ L.HAND_OFF for pw, Rw in (x["wr"]["right"], x["wr"]["left"])])
        self.pts.append(pts)
        v = float(np.linalg.norm(pts - self.pts[0], axis=1).max() / (TICKS * L.DT)) if len(self.pts) > TICKS else np.inf
        self.v.append(v)
        x["v"] = v
        return v

    def quiet(self) -> bool:
        return len(self.v) == SETTLE_F and max(self.v) < SETTLE_V

    def clear(self) -> None:
        self.v.clear()
        self.pts.clear()


def make_host(backend: str, seed: int):
    os.environ.setdefault("SIM2REAL_ORT_NUM_THREADS", "1")
    if backend == "sonic":
        from d3_sonic import SonicHost

        return SonicHost(env_name="EmptyEnv", seed=seed)
    import lml_host as H

    return H.HostDriver(env_name="EmptyEnv", backend=backend, seed=seed)


def ref_delay_f(h) -> int:
    if hasattr(h, "ref_delay_f"):
        return int(h.ref_delay_f)
    return int(round(float(h.buf.delay_s) * FPS))


def run_reset(h, rig: Rig, plan: dict) -> tuple[dict, dict, np.ndarray, list[dict]]:
    """One reset (see the module docstring): rows {frame: sample + spheres}, info, the reference frames, the reaches."""
    import hg_walk as HW

    t_wall = time.time()
    h.reset()
    rig.clear()
    frames = h.start_frames()
    frames = np.concatenate([frames, np.repeat(frames[-1:], int(HOLD_S * FPS), 0)])
    rows, reaches, slots = {}, [], []
    fell = lost = None
    why, f = "", 0
    walk = plan["walk"] if plan["walk"] and plan["walk"]["done"] else None
    stage = "walk" if walk else "wait"  # walk | walking | wait (for the next reach) | label (of the last reach)
    k = 0  # next command slot
    open_f = FIRST_OPEN_F  # the frame the current wait opened at
    anchor, cur_delta, active = None, np.zeros(3), None
    winfo = dict(asked=None, real=None, dev_cm=0.0, plan=None)
    x_walk0 = None
    delay = None
    plan_s = 0.0
    while True:
        h.push(frames[min(f, len(frames) - 1)])
        if getattr(h, "buf", 0) is None:
            h.buf = h._find_buffer()
        h.step()
        if delay is None:
            delay = ref_delay_f(h)
        if f == 1 and getattr(getattr(h, "comp", None), "_settling", False):
            raise RuntimeError("the bridge is still settling: our reference is not being tracked")
        x = rows[f] = h.sample()
        rig.record(x)
        if float(x["root"][2]) < L.FALL_Z:
            fell, why = f, "height"
            break
        if anchor is not None:  # the reference displacement taken the reference delay behind, where the robot is
            fr = frames[min(max(f - delay, anchor[2]), len(frames) - 1)]
            dev = float(np.linalg.norm((x["root"][:2] - anchor[0]) - (fr[:2] - anchor[1])))
            if stage == "walking":  # revision 1: no lost during the walk (falls still end it); its deviation is logged
                winfo["dev_cm"] = max(winfo["dev_cm"], 100 * dev)
            elif dev > L.LOST_M:
                fell, lost, why = f, f, "lost"
                break
        act = None  # what to do on this row
        if stage == "walk" and f == WALK_F:
            act = "walk"
        elif stage == "walking" and f >= open_f:
            stage = "wait"
            # revision 1: from the walk's end, lost is the change of (robot - reference) since this row (each later
            # reach re-anchors at its command, as in place, where the walk-end offset is 0)
            anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
        if stage in ("wait", "label") and act is None:
            on_grid = active is None or (f - active["f_cmd"]) % TICKS == 0
            if f >= open_f + SETTLE_F and on_grid and rig.quiet():
                act = "settled"
            elif f >= open_f + SETTLE_MAX_F and on_grid:
                act = "timeout"
        if act == "walk":
            p, yaw = x["root"][:2].astype(float), L.yaw_of(L._qmat(x["root"][3:]))
            g_xy = p + walk["d"] * np.array([math.cos(yaw + walk["dir"]), math.sin(yaw + walk["dir"])])
            info = {}
            t0 = time.time()
            wf = HW.walk_frames(h, x, frames[f], g_xy, yaw + walk["turn"], info=info)
            info["plan_s"] = time.time() - t0
            plan_s += info["plan_s"]
            nh = int(round(HW.HOLD_END_S * FPS))
            frames = np.concatenate([frames[: f + 1], wf, np.repeat(wf[-1:], int(HOLD_S * FPS), 0)])
            anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
            cur_delta = np.r_[frames[f, :2] - x["root"][:2], 0.0]
            x_walk0 = x
            winfo.update(asked=[walk["d"] * math.cos(walk["dir"]), walk["d"] * math.sin(walk["dir"]), walk["turn"]], plan=info)
            open_f = f + len(wf) - nh + delay  # the stance's first frame (+ the buffer delay)
            stage = "walking"
        elif act in ("settled", "timeout"):
            if active is not None:  # the previous reach's end label is this row
                active.update(f_label=f, settled_end=act == "settled", settle_s=(f - active["f_end"]) / FPS)
                active = None
            if act == "timeout":
                if k < CHAIN:
                    slots.append(dict(slot=k, mode=plan["cmds"][k]["mode"], status="unsettled", wait_s=(f - open_f) / FPS))
                    k += 1
                open_f = f  # a skipped first reach gives the second its own wait (and the reference its hold)
                frames = np.concatenate([frames[: f + 1], np.repeat(frames[f : f + 1], int(HOLD_S * FPS), 0)])
            if k >= CHAIN:
                break
            if act == "settled":
                cmd = plan["cmds"][k]
                if x_walk0 is not None and winfo["real"] is None:
                    p0, y0 = x_walk0["root"][:2].astype(float), L.yaw_of(L._qmat(x_walk0["root"][3:]))
                    dp = L.rot_z(y0).T @ np.r_[x["root"][:2] - p0, 0.0]
                    winfo["real"] = [float(dp[0]), float(dp[1]), wrap(L.yaw_of(L._qmat(x["root"][3:])) - y0)]
                try:
                    blend, last, rec = plan_reach(h, x, frames[f], cmd)
                except (np.linalg.LinAlgError, ValueError) as e:
                    why = f"ik {type(e).__name__}: {e}"
                    break
                plan_s += rec["plan_s"]
                rec.update(slot=k, chain=len(reaches), f_cmd=f, f_end=f + len(blend), pre_wait_s=(f - open_f) / FPS,
                           d_prev=cur_delta.copy(), f_label=None, settled_end=False, settle_s=-1.0)  # fmt: skip
                slots.append(dict(slot=k, mode=cmd["mode"], status="done", wait_s=(f - open_f) / FPS))
                frames = np.concatenate([frames[: f + 1], blend, L.hold(last, HOLD_S)])
                anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
                cur_delta = rec["delta"]
                reaches.append(rec)
                active = rec
                k += 1
                open_f = rec["f_end"] + delay
                stage = "label"
        if f >= len(frames) - 1:
            why = "motion ended"
            break
        f += 1
    while k < CHAIN:
        slots.append(dict(slot=k, mode=plan["cmds"][k]["mode"], status="not reached"))
        k += 1
    info = dict(fell=fell, lost=lost, why=why, steps=len(rows), walked=walk is not None, walk=winfo, slots=slots,
                planned=len(reaches), wall_s=time.time() - t_wall, plan_s=plan_s, delay_f=delay)  # fmt: skip
    return rows, info, frames, reaches


# -- segments -------------------------------------------------------------------------------------------------------
def segments(rows: dict, info: dict, frames: np.ndarray, reaches: list[dict], reset: int, pl, rig: Rig) -> list[dict]:
    """segments2 per hand of every reach: rows every TICKS steps from PRE_S before the command to the end label; a
    reach the reset ended inside (fall / lost) is kept to the last row before that (fell = 1)."""
    out = []
    fell, lost = info["fell"], info["lost"]
    w = info["walk"]
    walk_cmd = np.array(w["asked"] if w["asked"] else [0, 0, 0], np.float32)
    walk_real = np.array(w["real"] if w["real"] else [0, 0, 0], np.float32)
    for i, rc in enumerate(reaches):
        f_cmd = rc["f_cmd"]
        if rc["f_label"] is not None:
            end, seg_fell = rc["f_label"], False
        elif fell is not None:
            end, seg_fell = fell, True
        else:
            break  # ended otherwise (IK, motion ended): no label
        want = list(range(f_cmd - PRE_F, end + 1, TICKS))
        if f_cmd not in want or any(f not in rows for f in want) or len(want) < want.index(f_cmd) + 2:
            break
        d_prev, d_this = np.asarray(rc["d_prev"]), np.asarray(rc["delta"])
        both = rc["mode"] == "both"
        for side, hc in rc["hands"].items():
            S, Wr, Jr, Cr, X, HQ, KP = [], [], [], [], [], [], []
            for f in want:
                x = rows[f]
                p, Rp = x["root"][:3], L._qmat(x["root"][3:])
                hands = {s: (pw + Rw @ L.HAND_OFF, Rw) for s, (pw, Rw) in x["wr"].items()}
                rf = frames[min(f, len(frames) - 1)]
                ref_hand = pl.fk_world(rf[:3], rf[3:7], rf[7:])[side][0] - (d_this if f > f_cmd else d_prev)
                S.append(np.r_[x["q"], x["dq"], Rp.T @ np.array([0.0, 0.0, -1.0]), x["w"], p[2]].astype(np.float32))
                Wr.append(np.r_[p, L.yaw_of(Rp), hands["right"][0], hands["left"][0],
                                x["torso"][0] + x["torso"][1] @ np.array([0, 0, 0.35])].astype(np.float32))  # fmt: skip
                Rj = hands[side][1]
                Jr.append(np.r_[Rj[:, 0], Rj[:, 1]].astype(np.float32))
                Cr.append(np.r_[Rp.T @ (ref_hand - p), -1.0 if f < f_cmd else (f - f_cmd) / FPS].astype(np.float32))
                X.append(x["sph"][side])
                HQ.append(x["hq"])
                KP.append(x["kp"])
            k0 = want.index(f_cmd)
            p0, yaw0 = Wr[k0][:3].astype(float), float(Wr[k0][3])
            Rh = L.rot_z(yaw0)
            head = lambda v: np.c_[(v[..., :2].reshape(-1, 2) - p0[:2]) @ Rh[:2, :2], v[..., 2].reshape(-1)].reshape(v.shape)  # noqa: E731
            tw, rw = np.asarray(hc["target_w"]), np.asarray(hc["rot_w"])
            tg = Rh.T @ np.r_[tw[:2] - p0[:2], 0.0]
            rot_h = Rh.T @ rw
            pw, Rw = rows[want[-1]]["wr"][side]
            land = pw + Rw @ L.HAND_OFF
            att = float(np.degrees(np.arccos(np.clip((np.trace(rw.T @ Rw) - 1) / 2, -1, 1))))
            jaw_err = 100 * float(np.linalg.norm(land - tw))
            fing = rows[f_cmd]["fing"][side]
            XR = np.array([rig.ref_spheres(side, rp, rq, fing) for rp, rq in zip(rc["RP"], rc["RQ"])])
            o = [s for s in rc["hands"] if s != side]
            if o:
                ho = rc["hands"][o[0]]
                tgo = Rh.T @ np.r_[np.asarray(ho["target_w"])[:2] - p0[:2], 0.0]
                Ro = Rh.T @ np.asarray(ho["rot_w"])
                target_o, rot6_o, ik_o = np.r_[tgo[:2], ho["target_w"][2]], np.r_[Ro[:, 0], Ro[:, 1]], 100 * ho["ik_err"]
            else:
                target_o, rot6_o, ik_o = np.zeros(3), np.zeros(6), 0.0
            # both hands at the end label, heading frame: hand point, attitude (quaternion wxyz), the three keypoints
            xe = rows[want[-1]]
            end_pos = np.array([head(np.asarray(xe["wr"][s][0] + xe["wr"][s][1] @ L.HAND_OFF, float)[None])[0] for s in SIDES])
            end_quat = np.array([rig.quat(Rh.T @ xe["wr"][s][1]) for s in SIDES])
            end_kp = head(np.asarray(xe["kp"], float))
            v_end = max(rows[f]["v"] for f in range(max(want[-1] - SETTLE_F + 1, min(rows)), want[-1] + 1))
            out.append(dict(
                S=np.array(S), W=np.array(Wr), J=np.array(Jr), C=np.array(Cr), k0=k0, side=1 if side == "right" else -1,
                target=np.r_[tg[:2], tw[2]].astype(np.float32), dyaw=hc["dyaw"], pitch=hc["pitch"],
                rot6=np.r_[rot_h[:, 0], rot_h[:, 1]].astype(np.float32),
                pelvis_z=-1.0 if rc["pelvis_z"] is None else rc["pelvis_z"], start=np.r_[p0, yaw0].astype(np.float32),
                walked=bool(info["walked"]), chain=rc["chain"], homed=False, fell=seg_fell, lost=seg_fell and lost is not None,
                ok=jaw_err < 2.0, jaw_err_cm=jaw_err, att_err_deg=att, ik_err_cm=100 * hc["ik_err"], reach_s=rc["reach_s"],
                reset=reset, ref_s=rc["reach_s"], n_sidesteps=0, ref_q=np.asarray(rc["ref_q"], np.float32),
                ref_pel=np.asarray(rc["ref_pel"], np.float32), sidestep=np.zeros(2, np.float32), wrist_dev=np.zeros(3, np.float32),
                # this experiment's
                hands=2 if both else (1 if side == "right" else -1), roll=hc["roll"], core=bool(hc["core"]), slot=rc["slot"],
                settled_end=bool(rc["settled_end"]), settle_s=float(rc["settle_s"]), pre_wait_s=float(rc["pre_wait_s"]),
                v_end_cms=100 * float(v_end), ref_delay_s=info["delay_f"] / FPS, walk_cmd=walk_cmd, walk_real=walk_real, walk_dev_cm=float(w["dev_cm"]),
                HQ=np.array(HQ, np.float32), KP=np.array(KP, np.float32), end_pos=end_pos.astype(np.float32),
                end_quat=end_quat.astype(np.float32), end_kp=end_kp.astype(np.float32),
                cmd_target=np.asarray(hc["cmd_target"], np.float32), target_o=np.asarray(target_o, np.float32),
                rot6_o=np.asarray(rot6_o, np.float32), ik_err_o_cm=float(ik_o), hand_q=np.asarray(fing, np.float32),
                n_via=0, via_p=np.zeros((16, 3), np.float32), via_r6=np.zeros((16, 6), np.float32), via_speed=L.REACH_SPEED,
                via_gen=np.zeros(16, np.float32), via_route=-1, n_ref=len(rc["RT"]), RT=np.asarray(rc["RT"], np.float32),
                # arm.npz
                X=head(np.array(X, float)).astype(np.float32), XR=head(XR).astype(np.float32),
                RQ=np.asarray(rc["RQ"], np.float32), RP=np.asarray(rc["RP"], np.float32),
            ))  # fmt: skip
        if seg_fell:
            break
    return out


ROW_KEYS = ("S", "W", "J", "C", "HQ", "KP")
REF_KEYS = ("RT",)
ARM_ROW, ARM_REF = ("X",), ("XR", "RQ", "RP")
SCALARS = L.SEG_SCALARS2 + ("hands", "roll", "core", "slot", "settled_end", "settle_s", "pre_wait_s", "v_end_cms", "ref_delay_s",
                            "walk_dev_cm", "ik_err_o_cm", "n_via", "via_speed", "via_route", "n_ref")  # fmt: skip
VECTORS = L.SEG_VECTORS + ("end_pos", "end_quat", "end_kp", "walk_cmd", "walk_real", "cmd_target", "target_o", "rot6_o", "hand_q", "via_p", "via_r6", "via_gen")


def pack(segs: list[dict], scalars: tuple = SCALARS) -> tuple[dict, dict]:
    z = {k: np.concatenate([s[k] for s in segs]) for k in ROW_KEYS + REF_KEYS}
    z["n_steps"] = np.array([len(s["S"]) for s in segs])
    for k in scalars:
        z[k] = np.array([s[k] for s in segs])
    for k in VECTORS:
        z[k] = np.stack([s[k] for s in segs])
    a = {k: np.concatenate([s[k] for s in segs]) for k in ARM_ROW + ARM_REF}
    a["n_steps"], a["n_ref"] = z["n_steps"], z["n_ref"]
    return z, a


def save_chunk(segs: list[dict], d: Path, name: str, rig: Rig, scalars: tuple = SCALARS) -> None:
    (d / "arm").mkdir(parents=True, exist_ok=True)
    if not segs:
        return
    z, a = pack(segs, scalars)
    a.update(radius=rig.radius, link=np.array(rig.links))
    for path, arr in ((d / f"{name}.npz", z), (d / "arm" / f"{name}.npz", a)):
        tmp = path.with_name(f".{path.stem}.tmp.npz")
        np.savez_compressed(tmp, **arr)
        os.replace(tmp, path)


LAG_MOVES = ((22, -0.5), (25, -0.4), (23, -0.3), (15, -0.5), (18, -0.4), (16, 0.3))  # (joint in MuJoCo order, rad)


def lag(a) -> None:
    """The controller's real tracking lag, measured on clean moves: from the stance, one arm joint (right / left
    shoulder pitch, elbow, shoulder roll) follows a 1 s min-jerk ramp, then holds 3 s; the lag is the time the robot's
    joint crosses half its own displacement minus the time the reference crosses half the ramp (and the same at 90 %)."""
    if a.backend == "sonic":
        import d3_sonic  # noqa: F401
    h = make_host(a.backend, a.seed + 977)
    out = []
    for j, amp in LAG_MOVES:
        h.reset()
        fr = h.start_frames()
        f0, n = len(fr) + 10, FPS
        fr = np.concatenate([fr, np.repeat(fr[-1:], 10 + n + 3 * FPS, 0)])
        s_ = np.arange(n + 1) / n
        fr[f0 : f0 + n + 1, 7 + j] += amp * (10 * s_**3 - 15 * s_**4 + 6 * s_**5)
        fr[f0 + n + 1 :, 7 + j] += amp
        q = []
        for f in range(len(fr)):
            h.push(fr[f])
            if getattr(h, "buf", 0) is None:
                h.buf = h._find_buffer()
            h.step()
            q.append(float(h.d.qpos[h.qadr_mj[j]]))  # the frames' (MuJoCo) joint order; sample()["q"] is ScaleBFM's
        q = np.array(q)
        da = q[-1] - q[f0 - 1]
        cross = lambda v, lvl: int(np.argmax((v[f0 - 1 :] - v[f0 - 1]) / (v[-1] - v[f0 - 1]) >= lvl))  # noqa: E731
        out.append((j, cross(q, 0.5) - cross(fr[:, 7 + j], 0.5), cross(q, 0.9) - cross(fr[:, 7 + j], 0.9), da / amp))
    print(f"d3 lag {a.backend}: configured reference delay {ref_delay_f(h)} frames ({ref_delay_f(h) / FPS:.2f} s); per move "
          f"(joint, lag at 50 % / 90 % in frames, reached share of the ramp): {out}; median lag at 50 % "
          f"{np.median([o[1] for o in out]):.0f} frames ({np.median([o[1] for o in out]) / FPS:.2f} s)")  # fmt: skip


# == selfmodel_data3b_20261125: the supplement (--b; off by default, data3's path above is untouched) =================
# Settled raised-hand starts. Per reset (draws from default_rng(seed + i); the walk's four draws first, as data3): a third
# walk first (data3's walk), then four reach slots, each commanded once settled (data3's settle, lost and delay rules),
# each with data3's hand draw (right 40 % / left 40 % / both 20 %) and pelvis draw (standing 45 %, else 0.52-0.76 m), a pair
# sharing both. Kinds:
#   detour  (a third of the resets, slot 0) the hand(s) to a data3 target and attitude through 1-3 via points pushed up and
#           outward (away from the body's midline); only gives the next reach a different arm posture -> data_detour/.
#   pre     a grasp point drawn as data3 (reach table, CORE / WIDE) and an attitude drawn, per pair, pair_mild_p of the
#           time from the mild ranges (revision 0: |yaw| <= 60, pitch -10..45, |roll| <= 45 deg), else from data3's; the
#           hand point sent
#           pre_back back from it against the palm normal, the attitude already the grasp's; at pre_speed (data3's
#           0.15 m/s). The palm normal: the right wrist's +y (simple_bench / hg_run), the left wrist's -y (robot.xml: the
#           left fingers close towards -y, the right towards +y, both wrist frames parallel to the pelvis at q = 0).
#           Redrawn (up to `tries`) until the pre-grasp point lies in the reach table too.
#   entry   (right after its pre) into the grasp point, attitude unchanged, at entry_speed; the reference's path starts
#           entry_cont of the way from the measured hand point to the pre command (simple_bench ENTRY_CONT = 0.5). The IK
#           is solved on the measured posture as every other reach here (simple_bench builds the entry's on the reference).
#   rel     a target d from the current hand point (log-uniform in rel_d, a uniform direction, redrawn until it lies in
#           the reach table: the nearest probed height's box, heights within the probed ones), the wrist turned rel_small
#           of the time by 0-rel_small_deg about a random axis from its measured attitude, else drawn as data3; rel_speed.
# The non-detour slots: at each free position a pair with probability pair_q[slots after the detour] if two slots are
# left, else a rel (pair_q makes half the non-detour reaches pair reaches on average).
# Speed sets the reference's duration (max(REACH_MIN_S, path length / speed)), as Planner.reach does with 0.15.
# Added fields: kind (1 pre, 2 entry, 3 rel, 4 detour; data3 has none = its independent reaches), pair_att (pre and entry:
# 1 the mild attitude ranges, 2 data3's; else 0), pre_back_cm (pre and
# entry, else 0), d_start_cm (the measured hand point at the command to the target), att_change_deg (the measured wrist
# attitude at the command to the commanded one); via_speed = the reach's speed; a detour's via points in n_via / via_p /
# via_r6 (the attitude interpolated there) / via_gen (1) / via_route (1 = up and out), heading frame at the command.
SEED_B, ID0_B, RESETS_B = 77_000_000, 30_000_000, 15000
KIND = {"pre": 1, "entry": 2, "rel": 3, "detour": 4}
KIND_N = {v: k for k, v in KIND.items()}
DEFAULT_B = dict(
    n_slots=4, detour_p=1 / 3, detour_vias=(1, 3), detour_t=(0.2, 0.8), detour_up_m=(0.05, 0.25), detour_out_m=(0.0, 0.20),
    detour_speed=0.15, pair_q={"4": 0.43, "3": 0.5, "2": 0.5}, pre_back_m=(0.05, 0.12), pre_speed=0.15,
    pair_mild_p=2 / 3, mild_dyaw_deg=(-60.0, 60.0), mild_pitch_deg=(-10.0, 45.0), mild_roll_deg=(-45.0, 45.0),
    entry_speed=(0.05, 0.12), entry_cont=0.5, rel_d_m=(0.03, 0.60), rel_small=0.5, rel_small_deg=(0.0, 25.0),
    rel_speed=(0.05, 0.20), tries=100,
)  # fmt: skip  (revision 0, fa8fed07: rel_small_deg 0-15 -> 0-25; the pairs' mild attitudes)
SCALARS_B = SCALARS + ("kind", "pair_att", "pre_back_cm", "d_start_cm", "att_change_deg")


def b_config(path) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_B))
    if path:
        extra = json.loads(Path(path).read_text())
        unknown = set(extra) - set(cfg)
        assert not unknown, f"unknown data3b settings: {sorted(unknown)}"
        cfg.update(extra)
    return cfg


def euler_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """R = Rz(yaw) Ry(pitch) Rx(roll) -> (yaw, pitch, roll)."""
    return (math.atan2(R[1, 0], R[0, 0]), math.asin(float(np.clip(-R[2, 0], -1, 1))), math.atan2(R[2, 1], R[2, 2]))


def in_table(row: dict, t: np.ndarray) -> bool:
    zs = np.array(row["z"], float)
    ok = np.array([b is not None for b in row["box"]])
    if not ok.any() or not (max(Z_RANGE[0], zs[ok].min()) <= t[2] <= min(Z_RANGE[1], zs[ok].max())):
        return False
    xlo, xhi, ylo, yhi = row["box"][int(np.argmin(np.where(ok, np.abs(zs - t[2]), np.inf)))]
    return bool(xlo <= t[0] <= xhi and ylo <= t[1] <= yhi)


def attitude_h(rng, rngs=(DYAW_DEG, PITCH_DEG, ROLL_DEG)) -> dict:
    return dict(dyaw=math.radians(rng.uniform(*rngs[0])), pitch=math.radians(rng.uniform(*rngs[1])),
                roll=math.radians(rng.uniform(*rngs[2])))  # fmt: skip


def R_of(a: dict) -> np.ndarray:
    return L.rot_z(a["dyaw"]) @ L.rot_y(a["pitch"]) @ rot_x(a["roll"])


def palm_n(R: np.ndarray, side: str) -> np.ndarray:
    """The palm normal (towards what the hand grasps) of a wrist attitude: right +y, left -y."""
    return R[:, 1] if side == "right" else -R[:, 1]


def reset_plan_b(rng, table: dict, walk_ok: bool, cfg: dict) -> dict:
    u, d, a, turn = rng.random(), rng.uniform(*WALK_M), rng.uniform(-math.pi, math.pi), rng.uniform(-1.0, 1.0)
    walk = None
    if u < WALK_P:
        walk = dict(d=float(d), dir=float(a), turn=float(math.radians(WALK_TURN_DEG) * turn), done=walk_ok)

    def hands_pz():
        r = rng.random()
        mode = "right" if r < MODES[0][1] else "left" if r < MODES[0][1] + MODES[1][1] else "both"
        pz = None if rng.random() < STAND_P else float(rng.uniform(*PZ_RANGE))
        return mode, pz, (("right", "left") if mode == "both" else (mode,))

    slots, n = [], int(cfg["n_slots"])
    if rng.random() < cfg["detour_p"]:
        mode, pz, sides = hands_pz()
        hs = {}
        for s in sides:
            core = bool(rng.random() < CORE_P)
            t = sample_target(rng, table["rows"][s][pz_key(pz)], s, core)
            nv = int(rng.integers(cfg["detour_vias"][0], cfg["detour_vias"][1] + 1))
            vias = [(float(tk), float(rng.uniform(*cfg["detour_up_m"])), float(rng.uniform(*cfg["detour_out_m"])))
                    for tk in np.sort(rng.uniform(*cfg["detour_t"], nv))]  # fmt: skip
            hs[s] = dict(target=t, core=core, att=attitude_h(rng), vias=vias)
        slots.append(dict(kind="detour", mode=mode, pelvis_z=pz, speed=float(cfg["detour_speed"]), hands=hs))
    q = float(cfg["pair_q"][str(n - len(slots))])
    pair = 0
    while len(slots) < n:
        mode, pz, sides = hands_pz()
        if n - len(slots) >= 2 and rng.random() < q:
            mild = bool(rng.random() < cfg["pair_mild_p"])
            rngs = (cfg["mild_dyaw_deg"], cfg["mild_pitch_deg"], cfg["mild_roll_deg"]) if mild else (DYAW_DEG, PITCH_DEG, ROLL_DEG)
            hs = {}
            for s in sides:
                row = table["rows"][s][pz_key(pz)]
                for _ in range(int(cfg["tries"])):  # the pre-grasp point must lie in the reach table as well
                    core = bool(rng.random() < CORE_P)
                    g = sample_target(rng, row, s, core)
                    att, back = attitude_h(rng, rngs), float(rng.uniform(*cfg["pre_back_m"]))
                    pre = g - back * palm_n(R_of(att), s)
                    if in_table(row, pre):
                        break
                hs[s] = dict(grasp=g, core=core, att=att, back=back, pre=pre)
            es = float(rng.uniform(*cfg["entry_speed"]))
            pa = 1 if mild else 2
            slots.append(dict(kind="pre", mode=mode, pelvis_z=pz, speed=float(cfg["pre_speed"]), hands=hs, pair=pair, pair_att=pa))
            slots.append(dict(kind="entry", mode=mode, pelvis_z=pz, speed=es, hands=hs, pair=pair, pair_att=pa))
            pair += 1
        else:
            hs = {}
            for s in sides:
                small = bool(rng.random() < cfg["rel_small"])
                ax = rng.normal(size=3)
                hs[s] = dict(seed=int(rng.integers(1 << 31)), small=small, axis=ax / np.linalg.norm(ax),
                             theta=math.radians(rng.uniform(*cfg["rel_small_deg"])), att=attitude_h(rng),
                             d=float(math.exp(rng.uniform(math.log(cfg["rel_d_m"][0]), math.log(cfg["rel_d_m"][1])))))  # fmt: skip
            slots.append(dict(kind="rel", mode=mode, pelvis_z=pz, speed=float(rng.uniform(*cfg["rel_speed"])), hands=hs))
    return dict(walk=walk, slots=slots)


def command_b(x: dict, it: dict, table: dict, cfg: dict, stat: dict) -> dict:
    """The slot's command in world coordinates, placed by the measured state at the command row."""
    from scipy.spatial.transform import Rotation

    p, yaw = x["root"][:3].astype(float), L.yaw_of(L._qmat(x["root"][3:]))
    Rh = L.rot_z(yaw)
    to_w = lambda th: np.r_[p[:2] + (Rh @ np.r_[th[:2], 0.0])[:2], th[2]]  # noqa: E731
    meas = {s: (pw + Rw @ L.HAND_OFF, Rw) for s, (pw, Rw) in x["wr"].items()}
    hands = {}
    for s, hs in it["hands"].items():
        hc = dict(start=None, vias=[], back=0.0, core=bool(hs.get("core", False)))
        if it["kind"] in ("pre", "entry"):
            hs.setdefault("world", {})
            if it["kind"] == "pre":  # the grasp is placed once, by the pre's command row; the entry keeps it in the world
                Rg = Rh @ R_of(hs["att"])
                hs["world"] = dict(g=to_w(hs["grasp"]), R=Rg, pre=to_w(hs["grasp"]) - hs["back"] * palm_n(Rg, s))
                hc.update(target_w=hs["world"]["pre"], rot_w=Rg, back=hs["back"])
            else:
                W = hs["world"]
                j = meas[s][0]
                hc.update(target_w=W["g"], rot_w=W["R"], back=hs["back"], start=j + cfg["entry_cont"] * (W["pre"] - j))
        elif it["kind"] == "rel":
            rng = np.random.default_rng(hs["seed"])
            row = table["rows"][s][pz_key(it["pelvis_z"])]
            hh = np.r_[(Rh.T @ np.r_[meas[s][0][:2] - p[:2], 0.0])[:2], meas[s][0][2]]
            for _ in range(int(cfg["tries"])):
                v = rng.normal(size=3)
                t = hh + hs["d"] * v / np.linalg.norm(v)
                if in_table(row, t):
                    break
            else:
                stat["rel_fallback"] = stat.get("rel_fallback", 0) + 1
                t = sample_target(rng, row, s, False)
            Rw = Rotation.from_rotvec(hs["axis"] * hs["theta"]).as_matrix() @ meas[s][1] if hs["small"] else Rh @ R_of(hs["att"])
            hc.update(target_w=to_w(t), rot_w=Rw)
        else:  # detour
            tw = to_w(hs["target"])
            a = meas[s][0]
            out = Rh @ np.array([0.0, -1.0 if s == "right" else 1.0, 0.0])
            vias = [a + tk * (tw - a) + np.array([0.0, 0.0, up]) + o * out for tk, up, o in hs["vias"]]
            for v in vias:
                v[2] = min(v[2], Z_RANGE[1] + 0.1)
            hc.update(target_w=tw, rot_w=Rh @ R_of(hs["att"]), vias=vias)
        hands[s] = hc
    return dict(kind=it["kind"], mode=it["mode"], pelvis_z=it["pelvis_z"], speed=it["speed"], hands=hands,
                pair_att=it.get("pair_att", 0))  # fmt: skip


def path_reach(pl, pose, spec: dict, pz, speed: float):
    """Planner.reach / reach_both generalised: each moving hand along its polyline (start, via points, target; the start
    the hand's own point unless given), waypoints REACH_STEP_M apart along the longest path, the attitude turned by its
    rotation vector in proportion to the path done, the duration max(REACH_MIN_S, longest path / speed); a hand not in
    spec held where it is at weight 0.3 (Planner.reach), two moving hands at 1 and 1 (reach_both)."""
    from scipy.spatial.transform import Rotation

    from locomani.primitives.whole_body_ik import WholeBodyIK

    pos, quat, q = pose
    solver = WholeBodyIK(pl.ik, dict(joints=q, pelvis=dict(pos=pos, quat=quat)), L.IK_CFG, tool_offset=tuple(L.HAND_OFF),
                         knee_max=1.5, pelvis_forward_max=0.20, pelvis_height_target=pz, jacobian="analytic")  # fmt: skip
    if pz is not None:
        solver.hi[2] = max(solver.hi[2], pz + 0.01)
    solver.seed = np.clip(solver.initial, solver.lo, solver.hi)
    cur = pl.fk_world(pos, quat, q)
    poly, rv = {}, {}
    for s, sp in spec.items():
        pts = [cur[s][0] if sp["start"] is None else np.asarray(sp["start"], float)] + [np.asarray(v, float) for v in sp["vias"]]
        pts.append(np.asarray(sp["target"], float))
        cum = np.r_[0.0, np.cumsum([np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)])]
        poly[s] = (np.array(pts), cum)
        rv[s] = Rotation.from_matrix(cur[s][1].T @ sp["rot"]).as_rotvec()
    length = max(c[-1] for _, c in poly.values())
    n = max(1, int(math.ceil(length / L.REACH_STEP_M)))
    dur = max(L.REACH_MIN_S, length / speed)
    w = np.array([1.0 if s in spec else 0.3 for s in ("left", "right")])

    def at(s, u):
        pts, cum = poly[s]
        d = u * cum[-1]
        i = int(np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(pts) - 2))
        t = (d - cum[i]) / max(cum[i + 1] - cum[i], 1e-12)
        return pts[i] + min(max(t, 0.0), 1.0) * (pts[i + 1] - pts[i])

    poses, durs, errs = [pose], [], {}
    for k in range(1, n + 1):
        u = k / n
        tp = [at(s, u) if s in spec else cur[s][0] for s in ("left", "right")]
        tr = [cur[s][1] @ Rotation.from_rotvec(u * rv[s]).as_matrix() if s in spec else cur[s][1] for s in ("left", "right")]
        solver.hand_weights = w
        sol = solver.solve(np.array(tp), np.array(tr), hand_rotation_weight=L.ROT_W)
        qw = Rotation.from_matrix(np.asarray(sol.rotation)).as_quat()[[3, 0, 1, 2]]
        poses.append((np.asarray(sol.position, float), qw, np.asarray(sol.joints, float).copy()))
        durs.append(dur / n)
        errs = {"left": float(sol.hand_errors[0]), "right": float(sol.hand_errors[1])}
    via = {s: [(cur[s][1] @ Rotation.from_rotvec(float(cum[i] / cum[-1]) * rv[s]).as_matrix(), pts[i])
               for i in range(1, len(pts) - 1)] for s, (pts, cum) in poly.items()}  # fmt: skip
    return poses, durs, errs, dur, via


def plan_reach_b(h, x: dict, ref_frame: np.ndarray, cmd: dict):
    """plan_reach for command_b's commands (any kind): the same splice, shift and record, plus the kind's fields."""
    p, quat = x["root"][:3].astype(float), x["root"][3:].astype(float)
    yaw = L.yaw_of(L._qmat(quat))
    Rh = L.rot_z(yaw)
    q_mj = x["q"][h.to_mj].astype(float)
    spec = {s: dict(start=c["start"], vias=c["vias"], target=c["target_w"], rot=c["rot_w"]) for s, c in cmd["hands"].items()}
    t0 = time.time()
    poses, durs, errs, dur, via = path_reach(h.pl, (p, quat, q_mj), spec, cmd["pelvis_z"], cmd["speed"])
    plan_s = time.time() - t0
    hands = {}
    for s, c in cmd["hands"].items():
        tw, rw = np.asarray(c["target_w"], float), np.asarray(c["rot_w"], float)
        dyaw, pitch, roll = euler_zyx(Rh.T @ rw)
        pw, Rw = x["wr"][s]
        hp = pw + Rw @ L.HAND_OFF
        hands[s] = dict(target_w=tw, rot_w=rw, dyaw=dyaw, pitch=pitch, roll=roll, core=c["core"], ik_err=errs[s],
                        cmd_target=np.r_[(Rh.T @ np.r_[tw[:2] - p[:2], 0.0])[:2], tw[2]],
                        d_start_cm=100 * float(np.linalg.norm(tw - hp)), pre_back_cm=100 * float(c["back"]),
                        att_change_deg=float(np.degrees(np.arccos(np.clip((np.trace(Rw.T @ rw) - 1) / 2, -1, 1)))),
                        via=via[s])  # fmt: skip
    delta = np.r_[ref_frame[:2] - p[:2], 0.0]
    ref_pose = (ref_frame[:3].astype(float), ref_frame[3:7].astype(float), ref_frame[7:].astype(float))
    shifted = [ref_pose] + [(pp + delta, qq, jj) for pp, qq, jj in poses[1:]]
    last = poses[-1]
    rec = dict(mode=cmd["mode"], kind=cmd["kind"], speed=cmd["speed"], pair_att=cmd["pair_att"], hands=hands, pelvis_z=cmd["pelvis_z"], reach_s=dur,
               plan_s=plan_s, ref_q=last[2].copy(), ref_pel=np.r_[last[0], last[1]], delta=delta, RT=np.r_[0.0, np.cumsum(durs)],
               RQ=np.array([pp[2] for pp in poses]), RP=np.array([np.r_[pp[0], pp[1]] for pp in poses]))  # fmt: skip
    return L.blend_frames(shifted, durs), shifted[-1], rec


def run_reset_b(h, rig: Rig, plan: dict, table: dict, cfg: dict) -> tuple[dict, dict, np.ndarray, list[dict]]:
    """run_reset with data3b's slots (same walk, settle, lost, delay and splice rules). A wait that runs out skips the
    slot it was for; a skipped pre takes its entry with it ("pair skipped")."""
    import hg_walk as HW

    t_wall = time.time()
    h.reset()
    rig.clear()
    frames = h.start_frames()
    frames = np.concatenate([frames, np.repeat(frames[-1:], int(HOLD_S * FPS), 0)])
    rows, reaches, slots, stat = {}, [], [], {}
    items = plan["slots"]
    fell = lost = None
    why, f = "", 0
    walk = plan["walk"] if plan["walk"] and plan["walk"]["done"] else None
    stage = "walk" if walk else "wait"
    k = 0
    open_f = FIRST_OPEN_F
    anchor, cur_delta, active = None, np.zeros(3), None
    winfo = dict(asked=None, real=None, dev_cm=0.0, plan=None)
    x_walk0 = None
    delay = None
    plan_s = 0.0
    while True:
        h.push(frames[min(f, len(frames) - 1)])
        if getattr(h, "buf", 0) is None:
            h.buf = h._find_buffer()
        h.step()
        if delay is None:
            delay = ref_delay_f(h)
        if f == 1 and getattr(getattr(h, "comp", None), "_settling", False):
            raise RuntimeError("the bridge is still settling: our reference is not being tracked")
        x = rows[f] = h.sample()
        rig.record(x)
        if float(x["root"][2]) < L.FALL_Z:
            fell, why = f, "height"
            break
        if anchor is not None:
            fr = frames[min(max(f - delay, anchor[2]), len(frames) - 1)]
            dev = float(np.linalg.norm((x["root"][:2] - anchor[0]) - (fr[:2] - anchor[1])))
            if stage == "walking":
                winfo["dev_cm"] = max(winfo["dev_cm"], 100 * dev)
            elif dev > L.LOST_M:
                fell, lost, why = f, f, "lost"
                break
        act = None
        if stage == "walk" and f == WALK_F:
            act = "walk"
        elif stage == "walking" and f >= open_f:
            stage = "wait"
            anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
        if stage in ("wait", "label") and act is None:
            on_grid = active is None or (f - active["f_cmd"]) % TICKS == 0
            if f >= open_f + SETTLE_F and on_grid and rig.quiet():
                act = "settled"
            elif f >= open_f + SETTLE_MAX_F and on_grid:
                act = "timeout"
        if act == "walk":
            p, yaw = x["root"][:2].astype(float), L.yaw_of(L._qmat(x["root"][3:]))
            g_xy = p + walk["d"] * np.array([math.cos(yaw + walk["dir"]), math.sin(yaw + walk["dir"])])
            info = {}
            t0 = time.time()
            wf = HW.walk_frames(h, x, frames[f], g_xy, yaw + walk["turn"], info=info)
            info["plan_s"] = time.time() - t0
            plan_s += info["plan_s"]
            nh = int(round(HW.HOLD_END_S * FPS))
            frames = np.concatenate([frames[: f + 1], wf, np.repeat(wf[-1:], int(HOLD_S * FPS), 0)])
            anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
            cur_delta = np.r_[frames[f, :2] - x["root"][:2], 0.0]
            x_walk0 = x
            winfo.update(asked=[walk["d"] * math.cos(walk["dir"]), walk["d"] * math.sin(walk["dir"]), walk["turn"]], plan=info)
            open_f = f + len(wf) - nh + delay
            stage = "walking"
        elif act in ("settled", "timeout"):
            if active is not None:
                active.update(f_label=f, settled_end=act == "settled", settle_s=(f - active["f_end"]) / FPS)
                active = None
            if act == "timeout":
                if k < len(items):
                    slots.append(dict(slot=k, kind=items[k]["kind"], mode=items[k]["mode"], status="unsettled", wait_s=(f - open_f) / FPS))
                    k += 1
                    if items[k - 1]["kind"] == "pre" and k < len(items):
                        slots.append(dict(slot=k, kind="entry", mode=items[k]["mode"], status="pair skipped"))
                        k += 1
                open_f = f
                frames = np.concatenate([frames[: f + 1], np.repeat(frames[f : f + 1], int(HOLD_S * FPS), 0)])
            if k >= len(items):
                break
            if act == "settled":
                it = items[k]
                if x_walk0 is not None and winfo["real"] is None:
                    p0, y0 = x_walk0["root"][:2].astype(float), L.yaw_of(L._qmat(x_walk0["root"][3:]))
                    dp = L.rot_z(y0).T @ np.r_[x["root"][:2] - p0, 0.0]
                    winfo["real"] = [float(dp[0]), float(dp[1]), wrap(L.yaw_of(L._qmat(x["root"][3:])) - y0)]
                try:
                    blend, last, rec = plan_reach_b(h, x, frames[f], command_b(x, it, table, cfg, stat))
                except (np.linalg.LinAlgError, ValueError) as e:
                    why = f"ik {type(e).__name__}: {e}"
                    break
                plan_s += rec["plan_s"]
                rec.update(slot=k, chain=len(reaches), f_cmd=f, f_end=f + len(blend), pre_wait_s=(f - open_f) / FPS,
                           d_prev=cur_delta.copy(), f_label=None, settled_end=False, settle_s=-1.0)  # fmt: skip
                slots.append(dict(slot=k, kind=it["kind"], mode=it["mode"], status="done", wait_s=(f - open_f) / FPS))
                frames = np.concatenate([frames[: f + 1], blend, L.hold(last, HOLD_S)])
                anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
                cur_delta = rec["delta"]
                reaches.append(rec)
                active = rec
                k += 1
                open_f = rec["f_end"] + delay
                stage = "label"
        if f >= len(frames) - 1:
            why = "motion ended"
            break
        f += 1
    while k < len(items):
        slots.append(dict(slot=k, kind=items[k]["kind"], mode=items[k]["mode"], status="not reached"))
        k += 1
    info = dict(fell=fell, lost=lost, why=why, steps=len(rows), walked=walk is not None, walk=winfo, slots=slots,
                planned=len(reaches), wall_s=time.time() - t_wall, plan_s=plan_s, delay_f=delay, **stat)  # fmt: skip
    return rows, info, frames, reaches


def augment_b(segs: list[dict], reaches: list[dict]) -> None:
    """segments()'s records plus data3b's fields (the reach found by chain, the hand by side)."""
    for sg in segs:
        rc = reaches[sg["chain"]]
        hc = rc["hands"]["right" if sg["side"] > 0 else "left"]
        sg.update(kind=KIND[rc["kind"]], pair_att=int(rc["pair_att"]), pre_back_cm=hc["pre_back_cm"], d_start_cm=hc["d_start_cm"],
                  att_change_deg=hc["att_change_deg"], via_speed=float(rc["speed"]))  # fmt: skip
        if hc["via"]:
            p0, Rh = sg["start"][:3].astype(float), L.rot_z(float(sg["start"][3]))
            vp, vr, vg = np.zeros((16, 3), np.float32), np.zeros((16, 6), np.float32), np.zeros(16, np.float32)
            for i, (Rv, pv) in enumerate(hc["via"][:16]):
                vp[i] = np.r_[(Rh.T @ np.r_[pv[:2] - p0[:2], 0.0])[:2], pv[2]]
                Rl = Rh.T @ Rv
                vr[i], vg[i] = np.r_[Rl[:, 0], Rl[:, 1]], 1.0
            sg.update(n_via=len(hc["via"][:16]), via_p=vp, via_r6=vr, via_gen=vg, via_route=1)


def drive_b(a, ids: list[int], n: int, h, rig: Rig, table: dict, tag: str, before: int, chunks, done) -> None:
    cfg = b_config(a.b_config)
    cp = a.out / "b_config.json"
    if cp.exists():
        assert json.loads(cp.read_text()) == cfg, f"{cp} differs from this run's data3b settings"
    else:
        cp.write_text(json.dumps(cfg, indent=1))
    dirs = {k: a.out / k for k in ("data", "data_both", "data_detour")}
    t0, nres, nseg, nfall, nlost, nskip, nslot = time.time(), 0, 0, 0, 0, 0, 0
    for c, rng_ids in enumerate(chunks):
        if done(c):
            continue
        out, log = {k: [] for k in dirs}, []
        for j in rng_ids:
            i = ids[j]
            plan = reset_plan_b(np.random.default_rng(a.seed + i), table, not a.no_walk, cfg)
            rows, info, frames, reaches = run_reset_b(h, rig, plan, table, cfg)
            s = segments(rows, info, frames, reaches, a.id0 + i, h.pl, rig)
            augment_b(s, reaches)
            for x in s:
                out["data_detour" if x["kind"] == KIND["detour"] else "data_both" if x["hands"] == 2 else "data"].append(x)
            log.append(dict(reset=a.id0 + i, n_segs=len(s), walk_asked=plan["walk"] is not None,
                            kinds=[sl["kind"] for sl in plan["slots"]], **info))  # fmt: skip
            nres, nseg = nres + 1, nseg + len(s)
            nfall += info["fell"] is not None and info["lost"] is None
            nlost += info["lost"] is not None
            nskip += sum(sl["status"] == "unsettled" for sl in info["slots"])
            nslot += sum(sl["status"] not in ("not reached", "pair skipped") for sl in info["slots"])
            if nres % 5 == 0:
                print(f"{tag}: {before + nres}/{n} resets, {nseg} segments, falls {nfall}, lost {nlost}, unsettled {nskip}/{nslot}, "
                      f"{time.time() - t0:.0f} s ({(time.time() - t0) / nres:.1f} s/reset)", flush=True)  # fmt: skip
        name = f"w{a.shard:03d}_{c:03d}"
        for k, d in dirs.items():
            save_chunk(out[k], d, name, rig, SCALARS_B)
        (dirs["data"] / f"{name}_log.json").write_text(json.dumps(log, default=_json))
    print(f"{tag} done: {before + nres} resets ({nres} this run, {nseg} segments, falls {nfall}, lost {nlost}, "
          f"unsettled {nskip}/{nslot}), {time.time() - t0:.0f} s", flush=True)  # fmt: skip


def drive(a) -> None:
    if a.backend == "sonic":  # first: with D3_SONIC_INFER=onnx-gpu it puts the onnxruntime-gpu overlay ahead of the env
        import d3_sonic  # noqa: F401
    ids = [int(v) for v in a.ids.split(",")] if a.ids else list(range(a.shard, a.resets, a.shards))  # --ids: dev only
    n = min(a.limit or len(ids), len(ids))
    od, ob = a.out / "data", a.out / "data_both"
    for d in (od, ob):
        d.mkdir(parents=True, exist_ok=True)
    chunks = [range(c, min(c + CHUNK, n)) for c in range(0, n, CHUNK)]
    done = lambda c: (od / f"w{a.shard:03d}_{c:03d}_log.json").exists()  # noqa: E731 -- written after the chunk's data
    before = sum(len(r) for c, r in enumerate(chunks) if done(c))
    tag = f"d3{'b' if a.b else ''} {a.backend} shard {a.shard}"
    print(f"{tag}: {before}/{n} resets already driven", flush=True)
    if before == n:
        return
    table = json.loads(Path(a.table).read_text())
    h = make_host(a.backend, a.seed + a.shard)
    rig = Rig(h)
    print(f"{tag}: pelvis offset c {np.round(h.c, 4).tolist()}, root convention check {rig.check_root():.2e} m, reference delay "
          f"{ref_delay_f(h) if getattr(h, 'buf', None) is not None or hasattr(h, 'ref_delay_f') else 'from the buffer'}, "
          f"inference {getattr(h, 'inference_backend', 'onnx-cpu')} {(getattr(h, 'providers', None) or ['-'])[0]}", flush=True)
    if a.b:
        return drive_b(a, ids, n, h, rig, table, tag, before, chunks, done)
    t0, nres, nseg, nfall, nlost, nwalk, nskip, nslot = time.time(), 0, 0, 0, 0, 0, 0, 0
    for c, rng_ids in enumerate(chunks):
        if done(c):
            continue
        segs, segb, log = [], [], []
        for j in rng_ids:
            i = ids[j]
            plan = reset_plan(np.random.default_rng(a.seed + i), table, not a.no_walk)
            rows, info, frames, reaches = run_reset(h, rig, plan)
            s = segments(rows, info, frames, reaches, a.id0 + i, h.pl, rig)
            segs += [x for x in s if x["hands"] != 2]
            segb += [x for x in s if x["hands"] == 2]
            log.append(dict(reset=a.id0 + i, n_segs=len(s), walk_asked=plan["walk"] is not None, **info))
            nres, nseg, nwalk = nres + 1, nseg + len(s), nwalk + bool(info["walked"])
            nfall += info["fell"] is not None and info["lost"] is None
            nlost += info["lost"] is not None
            nskip += sum(sl["status"] == "unsettled" for sl in info["slots"])
            nslot += sum(sl["status"] != "not reached" for sl in info["slots"])
            if nres % 5 == 0:
                print(f"{tag}: {before + nres}/{n} resets, {nseg} segments, walked {nwalk}, falls {nfall}, lost {nlost}, "
                      f"unsettled {nskip}/{nslot}, {time.time() - t0:.0f} s ({(time.time() - t0) / nres:.1f} s/reset)", flush=True)  # fmt: skip
        name = f"w{a.shard:03d}_{c:03d}"
        save_chunk(segs, od, name, rig)
        save_chunk(segb, ob, name, rig)
        (od / f"{name}_log.json").write_text(json.dumps(log, default=_json))
    print(f"{tag} done: {before + nres} resets ({nres} this run, {nseg} segments, walked {nwalk}, falls {nfall}, lost {nlost}, "
          f"unsettled {nskip}/{nslot}), {time.time() - t0:.0f} s", flush=True)  # fmt: skip


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def merge(out: Path) -> None:
    for sub in ("data", "data_both") + (("data_detour",) if (out / "data_detour").exists() else ()):
        d = out / sub
        names = sorted(p.name for p in d.glob("w[0-9]*_[0-9]*.npz"))
        if not names:
            print(f"{sub}: no chunks")
            continue
        for src, dst in ((d, d / "all.npz"), (d / "arm", d / "arm.npz")):
            zs = [dict(np.load(src / nm)) for nm in names]
            keep = ("radius", "link")
            allz = {k: (zs[0][k] if k in keep else np.concatenate([z[k] for z in zs])) for k in zs[0]}
            tmp = dst.with_name(f".{dst.stem}.tmp.npz")
            np.savez_compressed(tmp, **allz)
            os.replace(tmp, dst)
        z = np.load(d / "all.npz")
        ok = ~z["fell"].astype(bool)
        print(f"{sub}: {len(names)} chunks, {len(z['n_steps'])} segments, {int(z['n_steps'].sum())} rows, fell {int((~ok).sum())}, "
              f"jaw_err median {np.median(z['jaw_err_cm'][ok]):.2f} cm", flush=True)  # fmt: skip


# -- check ----------------------------------------------------------------------------------------------------------
def logs(out: Path) -> list[dict]:
    return [r for p in sorted((out / "data").glob("w[0-9]*_[0-9]*_log.json")) for r in json.loads(p.read_text())]


def check(out: Path, id0: int, resets: int) -> None:
    os.environ["REACHWM_ROOT"] = str(out)
    # the reader being checked: the repo's reachwm_20261030/data.py (D3_REACHWM: its directory, when this file runs
    # from a frozen copy outside the repo)
    sys.path.insert(0, os.environ.get("D3_REACHWM", str(EXP / "reachwm_20261030")))
    import data as DT

    bad = 0
    pl = L.Planner()
    for sub in ("data", "data_both") + (("data_detour",) if (out / "data_detour").exists() else ()):
        p = out / sub / "all.npz"
        if not p.exists():
            print(f"{sub}: no all.npz")
            continue
        z, A = dict(np.load(p)), dict(np.load(out / sub / "arm.npz"))
        n = z["n_steps"].astype(int)
        so, last = np.r_[0, np.cumsum(n)[:-1]], np.cumsum(n) - 1
        side = z["side"]
        # 1. the end error recomputed from the last row (acting jaw W, the target back in world) = jaw_err_cm, every segment
        st = z["start"].astype(float)
        c, s = np.cos(st[:, 3]), np.sin(st[:, 3])
        t = z["target"].astype(float)
        tw = np.c_[st[:, 0] + c * t[:, 0] - s * t[:, 1], st[:, 1] + s * t[:, 0] + c * t[:, 1], t[:, 2]]
        jaw = np.where(side[:, None] > 0, z["W"][last, 4:7], z["W"][last, 7:10]).astype(float)
        d1 = np.abs(100 * np.linalg.norm(jaw - tw, axis=1) - z["jaw_err_cm"])
        print(f"{sub}: {len(n)} segments; end error recomputed from the last row vs jaw_err_cm: max |d| {d1.max():.4f} cm")
        bad += int(d1.max() > 0.01)
        # 2. rows, waypoints and arm arrays line up
        nr = z["n_ref"].astype(int)
        al = (len(A["X"]) == len(z["S"]) == n.sum() and len(A["XR"]) == len(z["RT"]) == nr.sum() and (A["n_steps"] == n).all()
              and (A["n_ref"] == nr).all())  # fmt: skip
        ro = np.r_[0, np.cumsum(nr)[:-1]]
        rt0 = z["RT"][ro]
        print(f"  arrays aligned: {al}; RT starts at 0: {bool(np.all(rt0 == 0))}; k0 {sorted(set(z['k0'].tolist()))}")
        bad += int(not al)
        # 3. the plan's last waypoint (robot_fk) puts the hand at ik_err_cm from the target
        idx = np.random.default_rng(0).choice(len(n), size=min(400, len(n)), replace=False)
        e3 = []
        for i in idx:
            j = ro[i] + nr[i] - 1
            hp = pl.fk_world(A["RP"][j, :3].astype(float), A["RP"][j, 3:7].astype(float), A["RQ"][j].astype(float))
            sd = "right" if side[i] > 0 else "left"
            e3.append(abs(100 * np.linalg.norm(hp[sd][0] - tw[i]) - z["ik_err_cm"][i]))
        print(f"  ik_err_cm recomputed from the last waypoint ({len(idx)} segments): max |d| {max(e3):.4f} cm")
        bad += int(max(e3) > 0.05)
        # 4. spheres: the reference's last waypoint against the settled arm (sanity), the acting palm vs the jaw
        ok = (~z["fell"].astype(bool)) & z["settled_end"].astype(bool)
        pal = [SPHERES.index(x) for x in SPHERES if x[0] == "hand_palm_link"][0]
        dpal = np.linalg.norm(A["X"][last][:, pal] - np.c_[
            np.c_[(jaw[:, :2] - st[:, :2])[:, 0] * c + (jaw[:, :2] - st[:, :2])[:, 1] * s,
                  -(jaw[:, :2] - st[:, :2])[:, 0] * s + (jaw[:, :2] - st[:, :2])[:, 1] * c], jaw[:, 2]], axis=1)  # fmt: skip
        dref = np.linalg.norm(A["X"][last] - A["XR"][ro + nr - 1], axis=2).max(1)
        print(f"  acting palm sphere to the jaw point on the last row: median {100 * np.median(dpal):.2f} cm (spread "
              f"{100 * (np.percentile(dpal, 99) - np.percentile(dpal, 1)):.2f}); arm vs last waypoint, settled: median of the "
              f"worst sphere {100 * np.median(dref[ok]) if ok.any() else float('nan'):.2f} cm")  # fmt: skip
        # 5. data.py: derived error = jaw_err_cm on every segment, its settled flag vs ours
        D = DT.load(str(p))
        d5 = np.abs(D["err_cm"] - D["jaw_err_cm"])
        agree = float(np.mean(D["settled"][ok]))
        print(f"  data.py derived err_cm vs jaw_err_cm, all {len(d5)} segments: max |d| {d5.max():.4f} cm; data.py settled on our "
              f"settled-end segments {100 * agree:.1f} %")  # fmt: skip
        bad += int(d5.max() > 0.01)
        if sub == "data":
            sp = DT.splits(D, 1.0)
            pairs = [(x, y) for x in ("train", "val", "test_iid") for y in ("train", "val", "test_iid") if x < y]
            disj = all(len(np.intersect1d(sp[x], sp[y])) == 0 for x, y in pairs)
            rs = lambda k: set(D["reset"][sp[k]].tolist())  # noqa: E731
            shared = len(rs("train") & rs("val")) + len(rs("train") & rs("test_iid")) + len(rs("val") & rs("test_iid"))
            nested = set(DT.splits(D, 0.1)["train"]) <= set(DT.splits(D, 0.3)["train"]) <= set(sp["train"])
            r = D["reset"]
            inrange = bool(r.min() >= id0 and r.max() < id0 + resets)
            print(f"  splits {dict((k, len(v)) for k, v in sp.items())}; disjoint {disj}, resets shared {shared}, nested {nested}, "
                  f"reset ids {int(r.min())}..{int(r.max())} in [{id0}, {id0 + resets}): {inrange}")  # fmt: skip
            bad += int(not (disj and shared == 0 and nested and inrange))
        elif sub == "data_both":
            key = z["reset"].astype(np.int64) * 10 + z["slot"]
            u, cnt = np.unique(key, return_counts=True)
            pairs_ok = bool((cnt == 2).all()) and all(sorted(side[key == v].tolist()) == [-1, 1] for v in u[:200])
            print(f"  two-hand reaches: {len(u)}, each one right and one left segment: {pairs_ok}")
            bad += int(not pairs_ok)
    print("CHECK", "PASSED" if bad == 0 else f"FAILED ({bad})")


def summary(out: Path) -> None:
    """Rates and errors (the stop report): falls / lost / unsettled, in place vs walk-first, per hand mode."""
    R = logs(out)
    if not R:
        print("no logs")
        return
    wk = np.array([r["walked"] for r in R])
    fell = np.array([r["fell"] is not None and r["lost"] is None for r in R])
    lost = np.array([r["lost"] is not None for r in R])
    lw = np.array([r["why"] == "lost walk" for r in R])
    wall = np.array([r["wall_s"] for r in R])
    print(f"{len(R)} resets ({int(wk.sum())} walked first, {sum(r['walk_asked'] for r in R)} drew a walk); "
          f"wall {np.median(wall):.1f} s/reset median, {wall.mean():.1f} mean")  # fmt: skip
    for name, m in (("in place", ~wk), ("walk first", wk)):
        if m.any():
            print(f"  {name:10s}: {int(m.sum())} resets, fell {100 * fell[m].mean():.1f} %, lost {100 * lost[m].mean():.1f} % "
                  f"(during the walk {100 * lw[m].mean():.1f} %), fell+lost {100 * (fell | lost)[m].mean():.1f} %, "
                  f"wall {wall[m].mean():.1f} s/reset")  # fmt: skip
    sl = [s for r in R for s in r["slots"]]
    tried = [s for s in sl if s["status"] != "not reached"]
    for mode in ("right", "left", "both", None):
        t = [s for s in tried if mode is None or s["mode"] == mode]
        if t:
            print(f"  reaches {mode or 'all':5s}: {len(t)} tried, unsettled (skipped) {100 * np.mean([s['status'] == 'unsettled' for s in t]):.1f} %, "
                  f"wait before the command median {np.median([s['wait_s'] for s in t if s['status'] == 'done'] or [0]):.2f} s")  # fmt: skip
    ws = [r for r in R if r["walked"] and r["walk"]["real"]]
    if ws:
        asked = np.array([r["walk"]["asked"] for r in ws])
        real = np.array([r["walk"]["real"] for r in ws])
        print(f"  walks reaching a command: {len(ws)}; distance asked {np.median(np.linalg.norm(asked[:, :2], axis=1)):.2f} m, "
              f"done {np.median(np.linalg.norm(real[:, :2], axis=1)):.2f} m (median); position miss median "
              f"{100 * np.median(np.linalg.norm(asked[:, :2] - real[:, :2], axis=1)):.1f} cm, turn miss median "
              f"{np.degrees(np.median(np.abs([wrap(a - b) for a, b in zip(asked[:, 2], real[:, 2])]))):.1f} deg; walk deviation "
              f"max median {np.median([r['walk']['dev_cm'] for r in ws]):.1f} cm")  # fmt: skip
    for sub in ("data", "data_both"):
        p = out / sub / "all.npz"
        if not p.exists():
            continue
        z = np.load(p)
        ok = ~z["fell"].astype(bool)
        e = z["jaw_err_cm"]
        print(f"  {sub}: {len(e)} segments ({int((~ok).sum())} fell/lost); settled_end {100 * z['settled_end'][ok].mean():.1f} %")
        for nm, m in (("right", z["side"] > 0), ("left", z["side"] < 0), ("in place", z["walked"] == 0),
                      ("walk first", z["walked"] == 1), ("chain 0", z["chain"] == 0), ("chain 1", z["chain"] == 1)):  # fmt: skip
            m = m & ok
            if m.any():
                print(f"    {nm:10s} n {int(m.sum()):5d}: direct end error median {np.median(e[m]):.2f} cm, IK error median "
                      f"{np.median(z['ik_err_cm'][m]):.2f} cm, attitude error median {np.median(z['att_err_deg'][m]):.1f} deg")  # fmt: skip


def summary_b(out: Path) -> None:
    """data3b's stop report: falls / lost per reset and per reach kind, skipped and unsettled reaches, the direct end
    error per kind (rel by start distance), the elbow at the command after a detour vs after our IK's reaches."""
    R = logs(out)
    if not R:
        print("no logs")
        return
    wk = np.array([r["walked"] for r in R])
    fl = np.array([r["fell"] is not None for r in R])
    lost = np.array([r["lost"] is not None for r in R])
    wall = np.array([r["wall_s"] for r in R])
    print(f"{len(R)} resets ({int(wk.sum())} walked first); wall {np.median(wall):.1f} s/reset median, {wall.mean():.1f} mean; "
          f"rel targets that fell back to a table draw {sum(r.get('rel_fallback', 0) for r in R)}")  # fmt: skip
    for name, m in (("all", np.ones(len(R), bool)), ("in place", ~wk), ("walk first", wk)):
        if m.any():
            print(f"  {name:10s}: {int(m.sum())} resets, fell {100 * (fl & ~lost)[m].mean():.1f} %, lost {100 * lost[m].mean():.1f} %, "
                  f"fell+lost {100 * fl[m].mean():.1f} %")  # fmt: skip
    sl = [s for r in R for s in r["slots"]]
    st = {k: sum(s["status"] == k for s in sl) for k in ("done", "unsettled", "pair skipped", "not reached")}
    tried = st["done"] + st["unsettled"]
    print(f"  slots {len(sl)}: {st}; skipped (the wait before it ran out) {100 * st['unsettled'] / max(tried, 1):.1f} % of "
          f"{tried} waited for")  # fmt: skip
    zs = {sub: dict(np.load(out / sub / "all.npz")) for sub in ("data", "data_both", "data_detour") if (out / sub / "all.npz").exists()}
    cat = {k: np.concatenate([z[k] for z in zs.values()]) for k in ("reset", "chain", "kind", "fell", "settled_end", "hands")}
    key = cat["reset"].astype(np.int64) * 10 + cat["chain"]
    u, first = np.unique(key, return_index=True)
    inv = {k: KIND_N[int(v)] for k, v in zip(u, cat["kind"][first])}
    print("  per reach (commanded, with a label or a fall):")
    for kn, kv in KIND.items():
        m = cat["kind"][first] == kv
        if m.any():
            f_ = cat["fell"][first][m].astype(bool)
            print(f"    {kn:6s}: {int(m.sum()):5d} reaches, fell+lost during it {100 * f_.mean():.1f} %, not settled 3 s after the "
                  f"reference ended {100 * (~cat['settled_end'][first][m & ~cat['fell'][first].astype(bool)].astype(bool)).mean():.1f} %")  # fmt: skip
    z = zs["data"]
    ok = (~z["fell"].astype(bool)) & z["settled_end"].astype(bool)
    e, d0 = z["jaw_err_cm"], z["d_start_cm"]
    print(f"  data (one hand, no detour): {len(e)} segments, kept (not fell, settled) {int(ok.sum())}")
    cells = [(kn, z["kind"] == kv) for kn, kv in KIND.items() if kv != KIND["detour"]]
    cells += [("rel <15 cm", (z["kind"] == KIND["rel"]) & (d0 < 15)), ("rel 15-30", (z["kind"] == KIND["rel"]) & (d0 >= 15) & (d0 < 30)),
              ("rel >=30 cm", (z["kind"] == KIND["rel"]) & (d0 >= 30)), ("right", z["side"] > 0), ("left", z["side"] < 0)]  # fmt: skip
    for nm, m in cells:
        m = m & ok
        if m.any():
            print(f"    {nm:11s} n {int(m.sum()):5d}: direct end error median {np.median(e[m]):.2f} cm (p25 {np.percentile(e[m], 25):.2f}, "
                  f"p75 {np.percentile(e[m], 75):.2f}); start distance median {np.median(d0[m]):.1f} cm; wrist turn median "
                  f"{np.median(z['att_change_deg'][m]):.1f} deg; attitude error median {np.median(z['att_err_deg'][m]):.1f} deg; "
                  f"reference {np.median(z['reach_s'][m]):.2f} s")  # fmt: skip
    pq = lambda v: f"p10 / p50 / p90 {np.percentile(v, 10):.1f} / {np.median(v):.1f} / {np.percentile(v, 90):.1f}"  # noqa: E731
    for nm, pa in (("entry, both ranges", (1, 2)), ("entry, mild", (1,)), ("entry, data3 range", (2,))):
        m = (z["kind"] == KIND["entry"]) & np.isin(z["pair_att"], pa) & ok
        if m.any():
            d_, a_ = d0[m], z["att_change_deg"][m]
            print(f"    {nm:18s} n {int(m.sum()):4d}: d_start_cm {pq(d_)}, in 8-17 cm {100 * np.mean((d_ >= 8) & (d_ <= 17)):.0f} %; "
                  f"wrist turn deg {pq(a_)}, in 8-20 {100 * np.mean((a_ >= 8) & (a_ <= 20)):.0f} %; direct error median "
                  f"{np.median(e[m]):.2f} cm; pre_back_cm {z['pre_back_cm'][m].min():.1f}..{z['pre_back_cm'][m].max():.1f}")  # fmt: skip
    # the elbow (acting side) at the command: after a detour vs after a pre / entry / rel (our IK's straight lines)
    S0 = z["S"][np.r_[0, np.cumsum(z["n_steps"])[:-1]] + z["k0"]]
    el = np.degrees(np.where(z["side"] > 0, S0[:, L.SCALEBFM_ORDER.index("right_elbow_joint")],
                             S0[:, L.SCALEBFM_ORDER.index("left_elbow_joint")]))  # fmt: skip
    prev = np.array([inv.get(int(r) * 10 + int(c) - 1, "none" if c == 0 else "?") for r, c in zip(z["reset"], z["chain"])])
    groups = (("after a detour", prev == "detour"), ("after our IK", np.isin(prev, ("pre", "entry", "rel"))), ("from the stance", prev == "none"),
              ("all", np.ones(len(el), bool)))  # fmt: skip
    bins = np.arange(-90, 181, 5)
    hist = {}
    for nm, m in groups:
        if m.any():
            hist[nm] = np.histogram(el[m], bins)[0] / m.sum()
            print(f"    elbow at the command (deg, the acting arm's joint), {nm:15s}: n {int(m.sum()):5d}, p1 / p5 / p50 / p95 / p99 "
                  f"{np.percentile(el[m], 1):.0f} / {np.percentile(el[m], 5):.0f} / {np.median(el[m]):.0f} / {np.percentile(el[m], 95):.0f} / "
                  f"{np.percentile(el[m], 99):.0f}; in -18..20 {100 * np.mean((el[m] >= -18) & (el[m] <= 20)):.1f} % "
                  f"({int(((el[m] >= -18) & (el[m] <= 20)).sum())}), in 86..108 {100 * np.mean((el[m] >= 86) & (el[m] <= 108)):.1f} % "
                  f"({int(((el[m] >= 86) & (el[m] <= 108)).sum())})")  # fmt: skip
    sp = np.degrees(np.where(z["side"] > 0, S0[:, L.SCALEBFM_ORDER.index("right_shoulder_pitch_joint")],
                             S0[:, L.SCALEBFM_ORDER.index("left_shoulder_pitch_joint")]))  # fmt: skip
    for nm, m in (("walked, first reach", (z["walked"] == 1) & (z["chain"] == 0)), ("in place, first reach", (z["walked"] == 0) & (z["chain"] == 0)),
                  ("all", np.ones(len(sp), bool))):  # fmt: skip
        if m.any():
            print(f"    shoulder pitch at the command (deg, acting arm), {nm:21s}: n {int(m.sum()):4d}, p5 / p50 / p95 "
                  f"{np.percentile(sp[m], 5):.1f} / {np.median(sp[m]):.1f} / {np.percentile(sp[m], 95):.1f}, max {sp[m].max():.1f}; "
                  f"in 5..15 {100 * np.mean((sp[m] >= 5) & (sp[m] <= 15)):.0f} %")  # fmt: skip
    if "after a detour" in hist and "after our IK" in hist:
        print(f"    elbow overlap (5 deg histogram intersection), after a detour vs after our IK: "
              f"{100 * np.minimum(hist['after a detour'], hist['after our IK']).sum():.0f} %")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gen", "table", "drive", "merge", "check", "summary", "lag"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--backend", choices=["scalebfm", "heft", "sonic"], default="scalebfm")
    ap.add_argument("--resets", type=int, default=None, help=f"default {RESETS} ({RESETS_B} with --b)")
    ap.add_argument("--seed", type=int, default=None, help=f"default {SEED} ({SEED_B} with --b)")
    ap.add_argument("--id0", type=int, default=None, help=f"default {ID0} ({ID0_B} with --b)")
    ap.add_argument("--b", action="store_true", help="selfmodel_data3b_20261125's supplement (see the data3b block)")
    ap.add_argument("--b-config", default=None, help="--b: a JSON of DEFAULT_B keys to override")
    ap.add_argument("--shards", type=int, default=60)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-walk", action="store_true", help="in place only (the walk draws are still made, so the commands stay the same)")
    ap.add_argument("--ids", default="", help="dev: these reset indices (comma-separated) instead of the shard's")
    ap.add_argument("--table", type=Path, default=TABLE)
    ap.add_argument("--procs", type=int, default=60)
    a = ap.parse_args()
    a.resets = a.resets if a.resets is not None else (RESETS_B if a.b else RESETS)
    a.seed = a.seed if a.seed is not None else (SEED_B if a.b else SEED)
    a.id0 = a.id0 if a.id0 is not None else (ID0_B if a.b else ID0)
    if a.phase == "gen":
        gen(a.procs, a.table)
    elif a.phase == "table":
        show_table(a.table)
    elif a.phase == "drive":
        drive(a)
    elif a.phase == "merge":
        merge(a.out)
    elif a.phase == "lag":
        lag(a)
    elif a.phase == "check":
        check(a.out, a.id0, a.resets)
    else:
        summary_b(a.out) if a.b else summary(a.out)
