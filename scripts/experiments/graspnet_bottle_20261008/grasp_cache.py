"""TODO 23.2: an offline grasp cache per GraspNet object scene -- privileged geometry in, physics-validated Dex1
grasps out -- as the alternative to graspnet-baseline's perception (grasp_planned.py `candidates`).

Per scene (make_scenes.py, the object at rest on the table, body frame = the scene's rest attitude):
1. sampling (`sample`): antipodal pairs on the object's own surface (the visual manifold mesh): a surface point and
   its normal, a ray through the object along the inward normal, the exit point whose outward normal lies within
   ANTIPODAL_DEG of the closing line; the pair's distance is the jaw width (CLOSED_GAP..JAW_MAX). The closing line is
   the Dex1 palm y; N_ROLL approach directions (palm x) around it; the tool origin at the pair's midpoint, shifted
   DEPTHS along the approach. The same geometric filters as graspnet's candidates (table clearance of the open hand,
   object clearance, enough of the object between the pads over PAD_OVERLAP, the straight PREGRASP_BACK entry clear),
   but against the full surface instead of a depth image. Near-duplicates (DEDUPE_POS / DEDUPE_DEG) dropped.
2. validation (`validate`): a hand-only MuJoCo model built from the scene itself (MjSpec attach: the right wrist
   subtree -- Dex1 base, two finger slides, the pad contact parameters -- under a mocap carrier; the object body as the
   scene has it; the table top box), the expert's own jaw law (sim: tau = KP (q_des - q) - KD dq, clipped to the
   20 N motor limit, actuator gear 250, joint actuatorfrcrange +-20 N; the deploy's hand gains KP 1.5 KD 0.1,
   dex3_hands.hpp sizeCommand) at the sim's 200 Hz. For each candidate: carrier at the wrist pose, jaw open, the
   object at rest; close over CLOSE_S (the expert's close_seconds ramp), bilateral pad contact within CONTACT_TIMEOUT_S
   after it (the expert's gate); lift LIFT_M in LIFT_S (minjerk, as the expert), hold HOLD_S; then a shake the expert
   does not do (SHAKE_S of +-SHAKE_AMP at SHAKE_HZ along the world x then y) to rank the survivors. Recorded per
   candidate: closed (bilateral contact after the ramp), rise after the hold, slip (object motion in the hand frame,
   lift end -> shake end), held at the end, the finger forces. score = held ? 1 - slip / SLIP_FULL : 0.
3. cache: <scene>/grasp_cache.npz with every sampled candidate (tool, R in the body frame, width, pair, validation
   fields) and `summary.json`. grasp_planned.py reads it with `--grasp-source cache` (run_grasp.py).

    PYTHONPATH=src "$VENV_SIM_PY" scripts/experiments/graspnet_bottle_20261008/grasp_cache.py 0 [--max-validate 300]
    # all objects, 12 processes:
    bash scripts/experiments/graspnet_bottle_20261008/cache_all.sh 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grasp_planned import (  # noqa: E402
    HAND_POINTS,
    JAW_MAX,
    OBJECT_CLEAR,
    PAD_OVERLAP,
    PAD_X,
    PAD_Y,
    PAD_Z,
    PREGRASP_BACK,
    TABLE_CLEAR,
)

REPO = HERE.parents[2]
ROOT = Path(
    os.environ.get("GRASP_CACHE_ROOT", REPO / "outputs" / "graspnet_bottle_20261008")
)  # env: another object set in the same layout (2026-11-09, reachwm_20261030/simready)
PAIR_FAR = bool(int(os.environ.get("GRASP_PAIR_FAR", "0")))
BODY = "water_bottle_object"
TABLE_TOP = 0.62
OFFSET = np.array([0.1515, 0.0, 0.0])  # tool (grip centre) in the wrist frame, ready_grasp_policy.OFFSET for dex1
JAW_OPEN, JAW_CLOSED = 0.0245, -0.02  # finger slide: HAND_OPEN / HAND_CLOSED (primitives_tuning)
CLOSED_GAP = 0.006 + 0.004  # the jaw's gap fully closed, plus margin: nothing thinner is a grasp
# sampling
N_SURFACE, N_DENSE, N_ROLL, ANTIPODAL_DEG = 3000, 20000, 12, 30.0
PAIR_RADIUS = 0.003  # the exit point's distance from the closing line through the first point
DEPTHS = (0.0, 0.01, -0.01)  # tool origin shifts along the approach from the pair's midpoint
DEDUPE_POS, DEDUPE_DEG = 0.01, 15.0
SPARSE_STEP = 8  # the quick-reject subset of the hand points
SWEPT_STEPS = 6  # the straight entry sampled this often (graspnet's candidates: 11)
MIN_ENCLOSED = 20
# validation (the expert's numbers: simple_config close_seconds 1.0, contact_timeout_s 0.8, lift_seconds 2.0,
# grasp_planned LIFT_M 0.10 HOLD_S 1.5)
KP, KD, FORCE_MAX, GEAR = 1.5, 0.1, 20.0, 250.0
DT = 0.005
CLOSE_S, CONTACT_TIMEOUT_S, LIFT_M, LIFT_S, HOLD_S = 1.0, 0.8, 0.10, 2.0, 1.5
SHAKE_S, SHAKE_AMP, SHAKE_HZ = 2.0, 0.02, 1.5
SLIP_FULL = 0.02  # slip that zeroes the score
# The carrier weld: time constant >= 2 dt, impedance near 1 -- at solimp 0.95/0.99 the hand sagged 3 cm under its own
# weight and 6 cm with the object (2026-10-10), which ate most of the 10 cm lift.
WELD_SOLREF, WELD_SOLIMP = 0.01, [0.9999, 0.9999, 0.001, 0.5, 2.0]
RISE_MIN = 0.05


def minjerk(u):
    u = min(1.0, max(0.0, u))
    return u * u * u * (10 - 15 * u + 6 * u * u)


# ----------------------------------------------------------------------------------------------- geometry


def object_mesh(m: mujoco.MjModel, body: int, visual: bool):
    """(verts, faces) of the body's visual (group 1) or collision (contype) mesh geoms, body frame."""
    verts, faces = [], []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        is_visual = not (m.geom_contype[g] or m.geom_conaffinity[g])
        if is_visual != visual:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        f = m.mesh_face[m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]].astype(int)
        R = Rotation.from_quat(m.geom_quat[g][[1, 2, 3, 0]]).as_matrix()
        faces.append(f + sum(len(x) for x in verts))
        verts.append(v @ R.T + m.geom_pos[g])
    return np.vstack(verts), np.vstack(faces)


def surface_samples(verts, faces, n, rng):
    """Area-weighted points on the mesh with the face normals, oriented outward (flipped if the mesh is inward)."""
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    centre = verts.mean(0)
    if ((tri.mean(1) - centre) * normals).sum(1).mean() < 0:
        normals = -normals
    k = rng.choice(len(faces), n, p=area / area.sum())
    a, b = rng.random((2, n))
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    pts = tri[k, 0] + a[:, None] * (tri[k, 1] - tri[k, 0]) + b[:, None] * (tri[k, 2] - tri[k, 0])
    return pts, normals[k], normals


def antipodal_pairs(first, pts, nrm, tree):
    """For each of the `first` surface points, the nearest surface point across the object along its inward normal
    (within PAIR_RADIUS of that line) whose outward normal lies within ANTIPODAL_DEG of the closing line:
    (i, j, width), indices into pts."""
    out = []
    for i in first:
        p1, n1 = pts[i], nrm[i]
        near = tree.query_ball_point(p1, JAW_MAX)
        if not near:
            continue
        near = np.asarray(near)
        rel = pts[near] - p1
        along = rel @ (-n1)
        off = np.linalg.norm(rel - np.outer(along, -n1), axis=1)
        ok = (along >= CLOSED_GAP) & (off < PAIR_RADIUS)
        if not ok.any():
            continue
        if (
            PAIR_FAR
        ):  # 2026-11-09: hollow objects (cups, bottles): the farthest point with an opposed normal, not the inner wall
            anti = np.degrees(np.arccos(np.clip(nrm[near[ok]] @ (-n1), -1, 1))) <= ANTIPODAL_DEG
            if not anti.any():
                continue
            j = np.flatnonzero(anti)[np.argmax(along[ok][anti])]
            out.append((i, near[ok][j], float(along[ok][j])))
            continue
        k = near[ok][np.argmin(along[ok])]
        width = float(along[ok].min())
        n2 = nrm[k]
        if np.degrees(np.arccos(np.clip(n2 @ (-n1), -1, 1))) > ANTIPODAL_DEG:
            continue
        out.append((i, k, width))
    return out


def frames_about(y, n_roll):
    """n_roll right-handed tool frames [x y z] sharing the closing axis y, x spread around it."""
    y = y / np.linalg.norm(y)
    helper = np.array([0.0, 0.0, 1.0]) if abs(y[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(y, helper)
    u /= np.linalg.norm(u)
    v = np.cross(y, u)
    out = []
    for phi in np.linspace(0, 2 * np.pi, n_roll, endpoint=False):
        x = np.cos(phi) * u + np.sin(phi) * v
        z = np.cross(x, y)
        out.append(np.column_stack([x, y, z]))
    return out


def sample(m: mujoco.MjModel, body: int, rng, log=print):
    """Antipodal Dex1 grasps on the object's surface, body frame, after the geometric filters; list of dicts."""
    verts, faces = object_mesh(m, body, visual=True)
    dense, dense_n, _ = surface_samples(verts, faces, N_DENSE, rng)
    tree = cKDTree(dense)
    z_table = TABLE_TOP - float(m.body_pos[body][2])  # the table top in the body frame (object at rest, quat identity)
    first = rng.choice(N_DENSE, N_SURFACE, replace=False)
    pairs = antipodal_pairs(first, dense, dense_n, tree)
    centre = dense.mean(0)
    radius = float(np.linalg.norm(dense - centre, axis=1).max()) + OBJECT_CLEAR + 0.01
    pad_reach = float(np.hypot(np.hypot(max(abs(PAD_X[0]), PAD_X[1]), PAD_Y), PAD_Z)) + 0.005

    def touches(points):
        """Any point within OBJECT_CLEAR of the surface: a sparse pass first (most rejects are deep), then the full
        set, both only inside the object's bounding sphere and with the query bounded (the unbounded nearest-neighbour
        query was 90 % of the sampling time, 2026-10-10)."""
        for sub in (points[::SPARSE_STEP], points):
            close = sub[np.linalg.norm(sub - centre, axis=1) < radius]
            if len(close) and tree.query(close, k=1, distance_upper_bound=OBJECT_CLEAR)[0].min() < OBJECT_CLEAR:
                return True
        return False

    out = []
    for i, k, width in pairs:
        p1, n1, p2 = dense[i], dense_n[i], dense[k]
        mid = (p1 + p2) / 2
        for R in frames_about(p2 - p1, N_ROLL):
            approach = R[:, 0]
            for a in DEPTHS:
                tool = mid + a * approach
                hand_R = HAND_POINTS @ R.T
                hand = tool + hand_R
                if hand[:, 2].min() < z_table + TABLE_CLEAR:
                    continue
                near = tree.query_ball_point(tool, pad_reach)
                local = (dense[near] - tool) @ R
                inside = (
                    (np.abs(local[:, 1]) < PAD_Y)
                    & (np.abs(local[:, 2]) < PAD_Z)
                    & (local[:, 0] > PAD_X[0])
                    & (local[:, 0] < PAD_X[1])
                )
                if inside.sum() < MIN_ENCLOSED or np.ptp(local[inside, 0]) < PAD_OVERLAP:
                    continue
                if touches(hand):
                    continue
                swept = (
                    tool[None, None, :]
                    - np.linspace(PREGRASP_BACK, 0.0, SWEPT_STEPS)[:, None, None] * approach
                    + hand_R[None]
                ).reshape(-1, 3)
                if swept[:, 2].min() < z_table + TABLE_CLEAR or touches(swept):
                    continue
                out.append(dict(tool=tool, R=R, width=width, p1=p1, p2=p2, depth=a))
                break
    log(f"[grasp_cache] {N_SURFACE} surface points, {len(pairs)} antipodal pairs, {len(out)} grasps before dedupe")
    kept, kept_tool, kept_R = [], np.zeros((0, 3)), np.zeros((0, 3, 3))
    cos_min = np.cos(np.radians(DEDUPE_DEG))
    for c in out:
        near = np.linalg.norm(kept_tool - c["tool"], axis=1) < DEDUPE_POS
        if near.any():
            # rotation angle between frames from the trace: cos = (tr(R1^T R2) - 1) / 2
            tr = np.einsum("nij,ij->n", kept_R[near], c["R"])
            if ((tr - 1) / 2 > cos_min).any():
                continue
        kept.append(c)
        kept_tool = np.vstack([kept_tool, c["tool"]])
        kept_R = np.concatenate([kept_R, c["R"][None]])
    log(f"[grasp_cache] {len(kept)} grasps after dedupe")
    return kept, dense


# ----------------------------------------------------------------------------------------------- validation


class HandModel:
    """The right Dex1 on a mocap carrier, the object and the table top, from the scene's own MJCF."""

    def __init__(self, scene_xml: str):
        src = mujoco.MjSpec.from_file(scene_xml)
        spec = mujoco.MjSpec()
        spec.option.timestep = DT
        spec.copy_during_attach = True
        w = spec.worldbody
        # The wrist is a free body welded to a mocap carrier, not a child of it: geoms of a mocap body teleport with
        # zero velocity and their friction never carries a load (the first build lifted nothing, 0/40, 2026-10-10).
        carrier = w.add_body(name="carrier", mocap=True)
        wrist = w.add_frame().attach_body(src.body("right_wrist_yaw_link"), "h_", "")
        wrist.pos, wrist.quat = (
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )  # its 0.046 m offset from the wrist pitch link
        wrist.add_freejoint()
        spec.add_equality(
            type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1="carrier",
            name2="h_right_wrist_yaw_link", solref=[WELD_SOLREF, 1.0], solimp=WELD_SOLIMP,
        )  # fmt: skip
        w.add_frame().attach_body(src.body(BODY), "o_", "")
        top = src.geom("dining_table_dining_table_top")
        tbody = src.body("dining_table_dining_table")
        w.add_geom(
            name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=top.size,
            pos=np.asarray(tbody.pos) + np.asarray(top.pos), rgba=[0.6, 0.5, 0.4, 1],
        )  # fmt: skip
        for a in list(spec.actuators):
            if "wrist_yaw" in a.name:
                spec.delete(a)
        for s in list(spec.sensors):
            if "wrist_yaw" in s.name:
                spec.delete(s)
        for j in list(spec.joints):
            if "wrist_yaw" in j.name:
                spec.delete(j)
        self.m = m = spec.compile()
        self.d = mujoco.MjData(m)
        self.obj = m.body("o_" + BODY).id
        self.wrist = m.body("h_right_wrist_yaw_link").id
        self.obj_adr = m.jnt_qposadr[m.body_jntadr[self.obj]]
        self.wrist_adr = m.jnt_qposadr[m.body_jntadr[self.wrist]]
        self.fingers = [m.joint(f"h_right_hand_gripper_{i}_joint").id for i in (1, 2)]
        self.f_qadr = [m.jnt_qposadr[j] for j in self.fingers]
        self.f_vadr = [m.jnt_dofadr[j] for j in self.fingers]
        self.acts = [m.actuator(f"h_right_hand_gripper_{i}").id for i in (1, 2)]
        self.finger_bodies = [m.body(f"h_right_dex1_finger_{i}_link").id for i in (1, 2)]
        hand_bodies = {self.wrist, *self.finger_bodies}
        self.hand_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in hand_bodies}
        self.obj_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.obj}
        self.table_geom = m.geom("table_top").id
        # the object settled on the table with the hand away (the scene's body pos sits ~5 mm above its rest)
        mujoco.mj_resetData(m, self.d)
        self.d.mocap_pos[0] = [5.0, 5.0, 5.0]
        self.d.qpos[self.wrist_adr : self.wrist_adr + 3] = [5.0, 5.0, 5.0]
        for _ in range(int(1.0 / DT)):
            mujoco.mj_step(m, self.d)
        self.rest_pos = self.d.xpos[self.obj].copy()
        self.rest_quat = self.d.xquat[self.obj].copy()
        self.rest_R = Rotation.from_quat(self.rest_quat[[1, 2, 3, 0]]).as_matrix()

    def _contacts(self):
        """(finger 1 touching the object, finger 2 touching, hand touching the table, deepest hand penetration)."""
        m, d = self.m, self.d
        f1 = f2 = table = False
        deepest = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            hand = g1 in self.hand_geoms or g2 in self.hand_geoms
            if not hand:
                continue
            other = g2 if g1 in self.hand_geoms else g1
            hg = g1 if g1 in self.hand_geoms else g2
            if other in self.obj_geoms:
                b = m.geom_bodyid[hg]
                f1 |= b == self.finger_bodies[0]
                f2 |= b == self.finger_bodies[1]
            if other == self.table_geom:
                table = True
            deepest = min(deepest, float(c.dist))
        return f1, f2, table, deepest

    def _object_in_hand(self):
        R = self.d.xmat[self.wrist].reshape(3, 3)
        return R.T @ (self.d.xpos[self.obj] - self.d.xpos[self.wrist])

    def run(self, tool, R, log=None) -> dict:
        """Close, lift, hold and shake at one grasp (body-frame tool pose); the validation record."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        tool_w = self.rest_pos + self.rest_R @ tool
        R_w = self.rest_R @ R
        wrist = tool_w - R_w @ OFFSET
        quat = Rotation.from_matrix(R_w).as_quat()[[3, 0, 1, 2]]
        d.mocap_pos[0], d.mocap_quat[0] = wrist, quat
        d.qpos[self.wrist_adr : self.wrist_adr + 3] = wrist
        d.qpos[self.wrist_adr + 3 : self.wrist_adr + 7] = quat
        d.qpos[self.obj_adr : self.obj_adr + 3] = self.rest_pos
        d.qpos[self.obj_adr + 3 : self.obj_adr + 7] = self.rest_quat
        for a in self.f_qadr:
            d.qpos[a] = JAW_OPEN
        mujoco.mj_forward(m, d)
        f1, f2, table, deepest = self._contacts()
        rec = dict(start_penetration_m=-deepest, start_touching=bool(f1 or f2 or table))
        if deepest < -0.003:
            rec.update(closed=False, held=False, why="hand placed into the object or table")
            return rec
        z0 = float(d.xpos[self.obj][2])
        wrist0 = wrist.copy()
        target = JAW_OPEN
        t = 0.0
        phase, began = "close", 0.0
        rel_lift_end = None
        contact_time = None
        forces = []
        max_steps = int((CLOSE_S + CONTACT_TIMEOUT_S + LIFT_S + HOLD_S + 2 * SHAKE_S + 1.0) / DT) + 10
        for _ in range(max_steps):
            el = t - began
            if phase == "close":
                target = JAW_OPEN + min(1.0, el / CLOSE_S) * (JAW_CLOSED - JAW_OPEN)
                f1, f2, _, _ = self._contacts()
                if f1 and f2 and contact_time is None:
                    contact_time = t
                if el >= CLOSE_S:
                    if f1 and f2:
                        rec["closed"], rec["contact_s"] = True, contact_time
                        rec["finger_q"] = [float(d.qpos[a]) for a in self.f_qadr]
                        rec["finger_force_n"] = [float(abs(d.qfrc_actuator[v])) for v in self.f_vadr]
                        rec["sag_m"] = float(d.mocap_pos[0][2] - d.xpos[self.wrist][2])
                        phase, began = "lift", t
                    elif el >= CLOSE_S + CONTACT_TIMEOUT_S:
                        rec.update(closed=False, held=False, why="no bilateral pad contact after the close")
                        rec["finger_q"] = [float(d.qpos[a]) for a in self.f_qadr]
                        return rec
            elif phase == "lift":
                d.mocap_pos[0] = wrist0 + [0.0, 0.0, LIFT_M * minjerk(el / LIFT_S)]
                if el >= LIFT_S:
                    phase, began = "hold", t
            elif phase == "hold":
                if el >= HOLD_S:
                    f1, f2, _, _ = self._contacts()
                    rise = float(d.xpos[self.obj][2]) - z0
                    rec.update(rise_m=rise, touching_after_hold=bool(f1 and f2))
                    if not (f1 and f2) or rise < RISE_MIN:
                        rec.update(held=False, why="lost during the lift" if not (f1 and f2) else "not lifted")
                        return rec
                    rel_lift_end = self._object_in_hand()
                    phase, began = "shake", t
            elif phase == "shake":
                axis = 0 if el < SHAKE_S else 1
                u = el if el < SHAKE_S else el - SHAKE_S
                off = np.zeros(3)
                off[axis] = SHAKE_AMP * np.sin(2 * np.pi * SHAKE_HZ * u) * min(1.0, u / 0.3)
                d.mocap_pos[0] = wrist0 + [0.0, 0.0, LIFT_M] + off
                if el >= 2 * SHAKE_S:
                    f1, f2, _, _ = self._contacts()
                    slip = float(np.linalg.norm(self._object_in_hand() - rel_lift_end))
                    held = bool(f1 and f2) and slip < SLIP_FULL
                    rec.update(slip_m=slip, held=held, touching_end=bool(f1 and f2))
                    rec["finger_force_end_n"] = [float(abs(d.qfrc_actuator[v])) for v in self.f_vadr]
                    rec["sag_end_m"] = float(d.mocap_pos[0][2] - d.xpos[self.wrist][2])
                    rec["score"] = (1.0 - slip / SLIP_FULL) if held else 0.0
                    rec["why"] = "" if held else ("lost in the shake" if not (f1 and f2) else "slipped in the shake")
                    return rec
            for k, (qa, va, act) in enumerate(zip(self.f_qadr, self.f_vadr, self.acts)):
                tau = KP * (target - d.qpos[qa]) + KD * (0.0 - d.qvel[va])
                d.ctrl[act] = float(np.clip(tau, -FORCE_MAX, FORCE_MAX))
            mujoco.mj_step(m, d)
            t += DT
            if len(forces) < 4:
                forces.append(float(d.actuator_force[self.acts[0]]))
        rec.update(held=False, why="validation did not finish")
        return rec


# ----------------------------------------------------------------------------------------------- driver


def build(oid: int, max_validate: int, seed: int = 0, log=print) -> dict:
    scene = ROOT / "scenes" / f"obj_{oid:03d}"
    xml = str(scene / "scene_locomani.xml")
    m = mujoco.MjModel.from_xml_path(xml)
    body = m.body(BODY).id
    rng = np.random.default_rng(seed + oid)
    t0 = time.time()
    cands, dense = sample(m, body, rng, log)
    t_sample = time.time() - t0
    rng.shuffle(cands)  # the validation budget spread over the surface, not the first sampled region
    cands = cands[:max_validate]
    hm = HandModel(xml)
    t1 = time.time()
    for i, c in enumerate(cands):
        c["validation"] = hm.run(c["tool"], c["R"])
        if (i + 1) % 50 == 0:
            log(
                f"[grasp_cache] validated {i + 1}/{len(cands)}, held {sum(x['validation'].get('held', False) for x in cands[: i + 1])}"
            )
    t_val = time.time() - t1
    held = [c for c in cands if c["validation"].get("held")]
    closed = [c for c in cands if c["validation"].get("closed")]
    whys = {}
    for c in cands:
        whys[c["validation"].get("why", "")] = whys.get(c["validation"].get("why", ""), 0) + 1
    n = len(cands)
    np.savez_compressed(
        scene / "grasp_cache.npz",
        tool=np.array([c["tool"] for c in cands]).reshape(n, 3),
        R=np.array([c["R"] for c in cands]).reshape(n, 3, 3),
        width=np.array([c["width"] for c in cands]),
        p1=np.array([c["p1"] for c in cands]).reshape(n, 3),
        p2=np.array([c["p2"] for c in cands]).reshape(n, 3),
        depth=np.array([c["depth"] for c in cands]),
        closed=np.array([bool(c["validation"].get("closed", False)) for c in cands]),
        held=np.array([bool(c["validation"].get("held", False)) for c in cands]),
        score=np.array([float(c["validation"].get("score", 0.0)) for c in cands]),
        slip=np.array([float(c["validation"].get("slip_m", np.nan)) for c in cands]),
        rise=np.array([float(c["validation"].get("rise_m", np.nan)) for c in cands]),
        force=np.array([c["validation"].get("finger_force_n", [np.nan, np.nan]) for c in cands]).reshape(n, 2),
        surface=dense.astype(np.float32),
        rest_pos=hm.rest_pos,
        rest_quat=hm.rest_quat,
    )
    summary = dict(
        object=oid,
        sampled=n,
        closed=len(closed),
        held=len(held),
        why=whys,
        sample_seconds=t_sample,
        validate_seconds=t_val,
        width_held=[float(c["width"]) for c in held][:50],
        force_held=[c["validation"]["finger_force_n"] for c in held][:50],
        slip_held=sorted(float(c["validation"]["slip_m"]) for c in held)[:50],
    )
    (scene / "grasp_cache_summary.json").write_text(json.dumps(summary, indent=1))
    log(
        f"[grasp_cache] obj {oid:03d}: {n} sampled, {len(closed)} closed, {len(held)} held; sample {t_sample:.0f} s, "
        f"validate {t_val:.0f} s; why {whys}"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("object", type=int)
    ap.add_argument("--max-validate", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.object, args.max_validate, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
