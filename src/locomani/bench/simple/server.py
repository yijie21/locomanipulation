"""The environment server of the SIMPLE bed (TODO 49.2; moved from escalation_20261116/official_server_esc.py). Runs under
SIMPLE's own venv, which has no `locomani`: this file imports nothing of the package except judge.py, which it loads BY PATH
(judge.py is numpy only, so the Watcher runs here, where the physics is). Its three arguments replace the old environment
switches: <socket> <released-episodes folder, never opened> <env names json: task -> SIMPLE's gym name>. A "reset" message
(a released episode) is refused: no released file is read by this bed. SIMPLE_TARGET_ASSET (draws only) is not carried over.

Copy of simple_eval_20261112/official_server.py for the escalation bed (2026-11-17), with three additions:
("reset_cfg", task, state_dict): reset to a layout the client supplies (SIMPLE's reset takes the whole environment
configuration, the judge is untouched); ("truth",): the poses of the scene's objects and whether the target touches a
hand, for the record only; ("timing",): seconds spent in the last steps. Everything else is as in the original.

SIMPLE's own environment behind a socket: MuJoCo physics, Isaac Sim rendering, the task's step limit and success check, all
as in third_party/SIMPLE/src/simple/cli/eval_decoupled_wbc.py. The client (official.py) sends joint targets and gains for the
43 motors each 20 ms step and gets back the joint readings, the base state (IMU) and the episode's flags; the head stereo pair
on request. Run in SIMPLE's venv from its root:

    cd third_party/SIMPLE && MUJOCO_GL=egl OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python <this file> <socket path> <episodes root> <env names json>
"""

import faulthandler
import glob
import importlib.util
import json
import os
import sys
import time
import types
from multiprocessing.connection import Listener

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [
    p for p in sys.path if os.path.abspath(p or ".") != HERE
]  # this folder's module names must not shadow SIMPLE's


def _judge():
    """judge.py by its file: this process has SIMPLE's venv and its `src` on the path, not this package."""
    spec = importlib.util.spec_from_file_location("bed_judge", os.path.join(HERE, "judge.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bed_judge"] = mod  # (dataclass-free, but the Watcher pickles by name)
    spec.loader.exec_module(mod)
    return mod


def _statecopy():
    """statecopy.py by its file, as judge.py: the capture / restore of live objects (snap / restore messages)."""
    spec = importlib.util.spec_from_file_location("bed_statecopy", os.path.join(HERE, "statecopy.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bed_statecopy"] = mod
    spec.loader.exec_module(mod)
    return mod


SC = _statecopy()
C = _judge()  # the Watcher runs here, where the physics is: the skill package cannot reach it

faulthandler.dump_traceback_later(900, repeat=True)
faulthandler.enable()  # a crash in the renderer leaves its Python stack in the log
sock, xdir, env_names = sys.argv[1], sys.argv[2], sys.argv[3]

import gymnasium as gym  # noqa: E402
import simple.envs  # noqa: E402, F401
from gymnasium.wrappers import TimeLimit  # noqa: E402
from simple.cli.eval_decoupled_wbc import _make_sonic_config  # noqa: E402
from simple.core.action import ActionCmd  # noqa: E402


class Episode:
    def __init__(self, task):
        self.task = task
        name = json.load(open(env_names))[
            task
        ]  # SIMPLE's environment name of the task from a file; xdir is not touched
        kw = {}
        raw = gym.make(f"simple/{name}", sim_mode="mujoco_isaac", render_hz=50, headless=True,
                       sonic_config=_make_sonic_config(), **kw)  # fmt: skip
        self.u = raw.unwrapped
        self.limit = self.u.task.metadata.get("max_episode_steps")
        self.env = TimeLimit(raw, max_episode_steps=self.limit)
        self.u.sync_isaac_every_substep = (
            False  # the renderer follows the physics once per 20 ms step, as replay_wbc.py does
        )
        self.rob = self.u.task.robot
        self.amo = not hasattr(
            self.rob, "num_body_dof"
        )  # the MP tasks' robot (AMO): legs by its own policy, not torques
        if self.amo:
            self.pend = None
            self.rob.apply_action = lambda a: None  # the command arrives in self.pend, the torques are ours:
            self.rob.step = self.amo_step  # robot.step is what the engine calls once per control step
            self.act = None
        else:
            n = self.rob.num_body_dof
            cmd = lambda k: types.SimpleNamespace(
                motor_cmd=[types.SimpleNamespace(q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0) for _ in range(k)]
            )  # noqa: E731
            self.low, self.lh, self.rh = cmd(n), cmd(7), cmd(7)
            self.act = ActionCmd(
                "wbc_torque", low_cmd=self.low, use_sensor=False, left_hand_cmd=self.lh, right_hand_cmd=self.rh
            )
        self.frames, self.steps, self.obs = [], 0, None
        self.art = (
            task in C.ART
        )  # the four articulated tasks: the ArtWatcher (joints, hand / body contacts with the moving parts)
        self.comply, self.comply_done = (C.ArtWatcher(task) if self.art else C.Watcher(0.02)), False
        self.art_move, self.art_cols = set(), []  # the asset's moving bodies (ids), its joints' qpos addresses
        self.art_hand, self.art_body = (
            False,
            set(),
        )  # what touched a moving body during this control step's physics steps
        self.prof, self.t_step = None, [0.0, 0]
        self.snaps: dict = {}  # TODO 52: id -> saved state (snap / restore)
        self.first_contacts: list = []
        self.hits: dict = {}  # body name -> first counted step it touched the target
        self.hand_log: dict = {}  # counted step -> bits of what touched the target (1 left arm/hand, 2 right, 4 table, 8 container)
        if not self.amo:
            self.u.substep_callback = self.on_sub

    def phys(self):
        """Contact settings and masses of the target, the table and one finger, as the environment has them (for the notes)."""
        mj = self.u.mujoco
        m = mj.mjModel
        out = {}
        for name in ("target", "table", "right_hand_index_1_link"):
            try:
                b = mj.mj_objects[name].id if name in mj.mj_objects else m.body(name).id
            except Exception:
                continue
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
            out[name] = dict(
                mass=float(m.body_mass[b]),
                ngeom=len(gs),
                friction=m.geom_friction[gs[:1]].round(3).tolist() if gs else None,
                condim=m.geom_condim[gs[:1]].tolist() if gs else None,
                solref=m.geom_solref[gs[:1]].round(4).tolist() if gs else None,
                solimp=m.geom_solimp[gs[:1]].round(4).tolist() if gs else None,
                margin=m.geom_margin[gs[:1]].tolist() if gs else None,
            )
        import mujoco

        jp = {}
        for n in (
            "right_hand_thumb_1_joint",
            "right_hand_index_0_joint",
            "right_hand_middle_1_joint",
            "right_elbow_joint",
            "right_shoulder_pitch_joint",
            "waist_pitch_joint",
            "waist_yaw_joint",
            "right_wrist_pitch_joint",
        ):
            j = m.joint(n)
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            dof = int(j.dofadr[0])
            jp[n] = dict(damping=float(m.dof_damping[dof]), armature=float(m.dof_armature[dof]),
                         frictionloss=float(m.dof_frictionloss[dof]), range=m.jnt_range[j.id].round(3).tolist(),
                         actuator=a, gear=m.actuator_gear[a][:1].tolist() if a >= 0 else None,
                         ctrlrange=m.actuator_ctrlrange[a].tolist() if a >= 0 else None,
                         forcerange=m.actuator_forcerange[a].tolist() if a >= 0 else None)  # fmt: skip
        out["joints"] = jp
        out["opt"] = dict(
            timestep=float(m.opt.timestep),
            cone=int(m.opt.cone),
            impratio=float(m.opt.impratio),
            noslip=int(m.opt.noslip_iterations),
            integrator=int(m.opt.integrator),
        )
        return out

    def campose(self):
        """Where each camera is: Isaac's own (the images come from it) and MuJoCo's model of it, world frame (position, viewing
        direction); the robot's pelvis for reference. Record only."""
        mj = self.u.mujoco
        m, d = mj.mjModel, mj.mjData
        out = dict(pelvis=np.asarray(d.qpos[:7]).tolist(), isaac={}, mujoco={})
        for k, cam in (getattr(self.u.isaac, "cameras", None) or {}).items():
            try:
                p, q = cam.get_world_pose()
                w, x, y, z = (float(v) for v in q)
                Rm = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                               [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                               [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip
                out["isaac"][k] = dict(
                    pos=np.asarray(p, float).tolist(), forward=Rm[:, 0].tolist(), up=Rm[:, 2].tolist()
                )
            except Exception as e:
                out["isaac"][k] = repr(e)
        for i in range(m.ncam):
            out["mujoco"][m.camera(i).name] = dict(pos=d.cam_xpos[i].tolist(), forward=(-d.cam_xmat[i].reshape(3, 3)[:, 2]).tolist(),
                                                   body=m.body(int(m.cam_bodyid[i])).name)  # fmt: skip
        return out

    def tp_setup(self, pos, target, focal=12.0):
        """Videos only: a camera of our own in the Isaac scene (640 x 360) at `pos` looking at `target` (world frame, z up); created once
        per server, moved on every call. -> the camera's pose as Isaac reports it."""
        from omni.isaac.sensor import Camera

        iso = self.u.isaac
        if getattr(self, "tp", None) is None:
            self.tp = Camera(prim_path=f"{iso.workspace_prim_path}/ThirdPerson", resolution=(640, 360))
            self.tp.initialize()
        pos, target = np.asarray(pos, float), np.asarray(target, float)
        f = target - pos
        f /= np.linalg.norm(f)
        left = np.cross([0.0, 0.0, 1.0], f)
        left /= np.linalg.norm(left)
        up = np.cross(f, left)
        Rm = np.stack([f, left, up], axis=1)  # camera axes "world": x forward, y left, z up
        w = np.sqrt(max(0.0, 1 + Rm[0, 0] + Rm[1, 1] + Rm[2, 2])) / 2
        q = np.array(
            [w, (Rm[2, 1] - Rm[1, 2]) / (4 * w), (Rm[0, 2] - Rm[2, 0]) / (4 * w), (Rm[1, 0] - Rm[0, 1]) / (4 * w)]
        )
        self.tp.set_world_pose(position=pos, orientation=q, camera_axes="world")
        self.tp.set_focal_length(float(focal))
        p_, q_ = self.tp.get_world_pose(camera_axes="world")
        return dict(pos=np.asarray(p_).tolist(), quat=np.asarray(q_).tolist())

    def tp_frame(self, want_head=True):
        """Videos only: the third-person camera's colour image (None before tp_setup) and the head camera's left image of the last step."""
        out = dict(tp=None, head=None)
        if getattr(self, "tp", None) is not None:
            out["tp"] = np.asarray(self.tp.get_rgba())[..., :3].astype(np.uint8)
        if want_head and self.obs is not None:
            out["head"] = np.asarray(self.obs["head_stereo_left"]).astype(np.uint8)
        return out

    def drives(self):
        """The model's springs and actuators per joint group: joint stiffness and spring reference, passive damping, the actuator's
        gain / bias (a torque motor has gain 1 and no bias; a position servo has bias -kp q), force range, and the control value now."""
        import mujoco

        mj = self.u.mujoco
        m, d = mj.mjModel, mj.mjData
        groups = dict(
            hand=("hand",), arm=("shoulder", "elbow", "wrist"), waist=("waist",), leg=("hip", "knee", "ankle")
        )
        out = {}
        for g, pats in groups.items():
            rows = []
            for i in range(1, 44):
                j = m.joint(i)
                if not any(p in j.name for p in pats):
                    continue
                a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j.name)
                rows.append(dict(joint=j.name, stiffness=float(m.jnt_stiffness[i]), springref=float(m.qpos_spring[int(m.jnt_qposadr[i])]),
                                 damping=float(m.dof_damping[int(m.jnt_dofadr[i])]), armature=float(m.dof_armature[int(m.jnt_dofadr[i])]),
                                 act=None if a < 0 else dict(gaintype=int(m.actuator_gaintype[a]), gain=float(m.actuator_gainprm[a][0]),
                                                             biastype=int(m.actuator_biastype[a]), bias=m.actuator_biasprm[a][:3].tolist(),
                                                             forcerange=m.actuator_forcerange[a].tolist(), forcelimited=int(m.actuator_forcelimited[a]),
                                                             ctrlrange=m.actuator_ctrlrange[a].tolist(), ctrl=float(d.ctrl[a]))))  # fmt: skip
            out[g] = rows
        return out

    def geoms(self, pats):
        """Every geom of the bodies whose name contains one of `pats` (plus the target's and the table's): type, size, bounding box
        half extents (geom_aabb), world position, body, contact settings, mass of the body; and the table's edge geometry."""
        mj = self.u.mujoco
        m, d = mj.mjModel, mj.mjData
        out = {}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            n = m.body(b).name
            if any(p in n for p in pats):
                out.setdefault(n, []).append(dict(id=g, type=int(m.geom_type[g]), size=m.geom_size[g].round(4).tolist(),
                                                  aabb=m.geom_aabb[g][3:].round(4).tolist(), pos=d.geom_xpos[g].round(4).tolist(),
                                                  contype=int(m.geom_contype[g]), conaff=int(m.geom_conaffinity[g]), condim=int(m.geom_condim[g]),
                                                  friction=m.geom_friction[g].round(4).tolist(), margin=float(m.geom_margin[g]),
                                                  solref=m.geom_solref[g].round(4).tolist(), solimp=m.geom_solimp[g].round(4).tolist(),
                                                  mass=float(m.body_mass[b])))  # fmt: skip
        return out

    def geom_points(self, g):
        """World vertices of one collision geom: a mesh's vertices, any other shape the corners of its local bounding box."""
        m, d = self.u.mujoco.mjModel, self.u.mujoco.mjData
        Rg = d.geom_xmat[g].reshape(3, 3)
        k = int(m.geom_dataid[g])
        if int(m.geom_type[g]) == 7 and k >= 0:  # mjGEOM_MESH
            v = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        else:
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            v = c + h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return (d.geom_xpos[g] + v @ Rg.T).astype(np.float64)

    def body_geoms(self, b, subtree=False):
        """The collision geoms (contype or conaffinity set) of body `b` (and its descendants), each as an (N, 3) array of world
        vertices. Privileged: target=truth only (messages "verts", "art_verts")."""
        m = self.u.mujoco.mjModel
        ids = {int(b)}
        if subtree:
            par = m.body_parentid
            ids = {x for x in range(m.nbody) if any(y == int(b) for y in self._chain(par, x))}
        return [self.geom_points(g) for g in range(m.ngeom)
                if int(m.geom_bodyid[g]) in ids and (int(m.geom_contype[g]) or int(m.geom_conaffinity[g]))]  # fmt: skip

    @staticmethod
    def _chain(par, x):
        while x > 0:
            yield int(x)
            x = int(par[x])
        yield 0

    def verts(self, names):
        """{name: [per-geom world vertices]} for scene objects (keys of mj_objects: "target", "container", ...) or bodies by name."""
        mj = self.u.mujoco
        out = {}
        for n in names:
            b = mj.mj_objects[n].id if n in mj.mj_objects else mj.mjModel.body(n).id
            out[n] = self.body_geoms(b)
        return out

    def art_verts(self):
        """The articulated asset: {"obj": per-geom world vertices of the whole asset (static frame and moving parts),
        "moving": {body name: per-geom vertices}} (the moving bodies are art_move's)."""
        m = self.u.mujoco.mjModel
        if not self.art_move:
            return dict(obj=[], moving={})
        first = sorted(self.art_move)[0]
        root = first
        for x in self._chain(m.body_parentid, first):  # the asset's root: the highest ancestor below the world
            if x > 0:
                root = x
        depth = {
            m.body(b).name: sum(int(x) in self.art_move for x in self._chain(m.body_parentid, b)) - 1
            for b in self.art_move
        }
        return dict(
            obj=self.body_geoms(root, subtree=True),
            moving={m.body(b).name: self.body_geoms(b) for b in sorted(self.art_move)},
            depth=depth,  # moving ancestors of each moving body: a hinged knob on a hinged leaf is 1
        )

    def feet(self):
        """Which ankle links touch what (contact counts), and the pelvis, for the walk trace."""
        mj = self.u.mujoco
        m, d = mj.mjModel, mj.mjData
        out = {}
        for i in range(d.ncon):
            a, b = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            for x, y in ((a, b), (b, a)):
                n = m.body(int(x)).name
                if "ankle_roll" in n:
                    out[n.split("_")[0]] = out.get(n.split("_")[0], 0) + 1
        return out

    def phys2(self):
        """Structure the mirror must match for the legs: the integrator and solver, the floor, the feet, the legs' joint
        damping/armature/friction loss, the robot's mass."""
        mj = self.u.mujoco
        m = mj.mjModel
        o = m.opt
        out = dict(opt=dict(timestep=o.timestep, integrator=int(o.integrator), cone=int(o.cone), impratio=o.impratio, iterations=int(o.iterations),
                            ls_iterations=int(o.ls_iterations), noslip=int(o.noslip_iterations), tolerance=o.tolerance, gravity=o.gravity.tolist(),
                            solver=int(o.solver), jacobian=int(o.jacobian)))  # fmt: skip
        names = [m.joint(i).name for i in range(1, 44)]
        out["joints"] = {n: [float(m.dof_damping[int(m.joint(n).dofadr[0])]), float(m.dof_armature[int(m.joint(n).dofadr[0])]),
                             float(m.dof_frictionloss[int(m.joint(n).dofadr[0])])] for n in names}  # fmt: skip
        pel = m.body("pelvis").id
        sub = [b for b in range(m.nbody) if int(m.body_rootid[b]) == int(m.body_rootid[pel])]
        out["robot_mass"] = float(sum(m.body_mass[b] for b in sub))
        feet = {}
        for g in range(m.ngeom):
            n = m.body(int(m.geom_bodyid[g])).name
            if "ankle_roll" in n or m.geom_bodyid[g] == 0:
                feet.setdefault(n, []).append([int(m.geom_type[g]), int(m.geom_contype[g]), int(m.geom_conaffinity[g]), int(m.geom_condim[g]),
                                               m.geom_friction[g].round(4).tolist(), m.geom_solref[g].round(4).tolist(), m.geom_solimp[g].round(4).tolist(),
                                               m.geom_margin[g].round(4).tolist(), m.geom_size[g].round(4).tolist()])  # fmt: skip
        out["feet_floor"] = feet
        return out

    def on_sub(self):
        """Every physics step: the first bodies other than the table that touch the target (for the record's loss analysis)."""
        mj = self.u.mujoco
        m, d = mj.mjModel, mj.mjData
        if self.art:
            self.art_sub(m, d)
        tb = mj.mj_objects["target"].id
        cid = mj.mj_objects["container"].id if "container" in mj.mj_objects else -1
        for i in range(d.ncon):
            a, b = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            o = b if a == tb else a if b == tb else None
            if o is not None:
                n = m.body(int(o)).name
                bit = 1 if n.startswith("left_") else 2 if n.startswith("right_") else 4 if n.startswith("table") else 8 if int(o) == cid else 0  # fmt: skip
                if bit:
                    self.hand_log[self.steps] = self.hand_log.get(self.steps, 0) | bit
                if n not in ("table", "table2", "world") and n not in self.hits and len(self.hits) < 30:
                    self.hits[n] = self.steps
                    g = d.contact[i]
                    self.first_contacts.append(dict(step=self.steps, body=n, geoms=[int(g.geom1), int(g.geom2)], pos=g.pos.tolist(),
                                                    normal=g.frame[:3].tolist(), dist=float(g.dist)))  # fmt: skip

    def art_setup(self):
        """After a reset: the asset's joints (name -> qpos address) and its moving bodies = the subtrees below the bodies its hinges
        move (oven: both doors; door: the leaf; faucet: the handle; trash can: the lid). The static frame is not in it."""
        m = self.u.mujoco.mjModel
        self.art_cols = [
            (m.joint(i).name, int(m.jnt_qposadr[i]))
            for i in range(m.njnt)
            if m.joint(i).name.startswith("articulate_joint")
        ]
        roots = {
            int(m.jnt_bodyid[i])
            for i in range(m.njnt)
            if m.joint(i).name.startswith("articulate_joint") and int(m.jnt_type[i]) == 3
        }  # hinges only (3 = mjJNT_HINGE; the oven's racks slide)
        par = m.body_parentid
        mov = set()
        for b in range(1, m.nbody):
            x = b
            while x > 0:
                if x in roots:
                    mov.add(b)
                    break
                x = int(par[x])
        self.art_move = mov
        self.art_hand, self.art_body = False, set()

    def art_box(self, b):
        """World bounding box [lo, hi] of the collision geoms of body `b` (their local boxes rotated into the world); record / tests only."""
        m, d = self.u.mujoco.mjModel, self.u.mujoco.mjData
        lo, hi = np.full(3, 1e9), np.full(3, -1e9)
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != b or int(m.geom_conaffinity[g]) == 0 and int(m.geom_contype[g]) == 0:
                continue
            c, h, Rg = m.geom_aabb[g][:3], m.geom_aabb[g][3:], d.geom_xmat[g].reshape(3, 3)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        w = d.geom_xpos[g] + Rg @ (c + h * np.array([sx, sy, sz]))
                        lo, hi = np.minimum(lo, w), np.maximum(hi, w)
        return [lo.tolist(), hi.tolist()]

    def art_q(self):
        d = self.u.mujoco.mjData
        return {n: float(d.qpos[a]) for n, a in self.art_cols}

    def art_sub(self, m, d):
        """Every physics step: a robot link (hand or other) in contact with a moving body of the articulated asset."""
        cls = C._classes(m)
        for i in range(d.ncon):
            g = d.contact[i]
            a, b = int(m.geom_bodyid[g.geom1]), int(m.geom_bodyid[g.geom2])
            for x, y in ((a, b), (b, a)):
                if y in self.art_move and x not in self.art_move and cls[x]:
                    if cls[x] in (C.L, C.R):
                        self.art_hand = True
                    else:
                        self.art_body.add(m.body(x).name)

    def amo_step(self, command=None, replay=False, eval=False):
        """The AMO robot's control step with torques from the client's targets: 10 physics steps (20 ms), torque = kp (q* - q) - kd dq
        recomputed each physics step and clipped at the robot's limits, as g1_wholebody.step does with its own policy's targets."""
        import mujoco

        m, d = self.u.mujoco.mjModel, self.u.mujoco.mjData
        q, kp, kd = self.pend
        lim = self.lim
        for _ in range(int(round(0.02 / m.opt.timestep))):
            if getattr(
                self, "hand_gain", False
            ):  # SIMPLE's own PD of the AMO hand joints (g1_wholebody.stiffness / damping)
                kp = np.where(self.hand_mask, self.hand_kp, kp)
                kd = np.where(self.hand_mask, self.hand_kd, kd)
            tau = kp * (q - d.qpos[7 : 7 + 43]) - kd * d.qvel[6 : 6 + 43]
            u = np.clip(tau, -lim, lim)
            if getattr(
                self, "hand_limit", False
            ):  # the teleop Dex3 effort limits (thumb_0 2.45, the others 0.7 N m), what the mirror applies
                u = np.where(self.hand_mask, np.clip(tau, -self.hand_clip, self.hand_clip), u)
            if getattr(
                self, "arm_pos", False
            ):  # the arms' actuators are position servos (gain 1000, bias -1000 q - 10 qdot): their control is the target angle
                pos = self.servo
                u = np.where(pos, np.clip(q, self.ctrl_lo, self.ctrl_hi), u)
            d.ctrl[self.cidx] = self.last_tau = u
            mujoco.mj_step(m, d, nstep=1)
            self.on_sub()  # first contacts with the target, as the sonic robot's substep callback does
        self.rob.count += 1

    def state(self):
        m, d = self.u.mujoco.mjModel, self.u.mujoco.mjData
        return dict(qpos=d.qpos[: 7 + 43].copy(), qvel=d.qvel[: 6 + 43].copy())

    def reset(self, level, idx):
        (f,) = glob.glob(f"{xdir}/{self.task}/*/dr-level-{level}/meta/episodes.jsonl")
        ep = [json.loads(line) for line in open(f)][idx]
        conf = ep["environment_config"]
        return self.reset_with(json.loads(conf) if isinstance(conf, str) else conf)

    def art_joints(self):
        """The articulated asset's joint values (articulate_joint_*), only the four articulated tasks have them; {} otherwise."""
        m, d = self.u.mujoco.mjModel, self.u.mujoco.mjData
        out = {}
        for i in range(m.njnt):
            n = m.joint(i).name
            if n.startswith("articulate_joint"):
                out[n] = float(d.qpos[int(m.jnt_qposadr[i])])
        return out

    def truth(self):
        mj = self.u.mujoco
        m, d = mj.mjModel, mj.mjData
        out = {k: np.concatenate([np.asarray(v.xpos), np.asarray(v.xquat)]) for k, v in mj.mj_objects.items()}
        tb = mj.mj_objects["target"].id if hasattr(mj.mj_objects["target"], "id") else None
        held, touch = False, set()
        if tb is not None:
            for i in range(d.ncon):
                a, b = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
                o = b if a == tb else a if b == tb else None
                if o is not None:
                    touch.add(m.body(int(o)).name)
                    held = held or "hand" in m.body(int(o)).name
        return dict(
            art=self.art_joints(),
            objects=out,
            held=held,
            touch=sorted(touch),
            hits=dict(self.hits),
            first_contacts=list(self.first_contacts),
            hand_log=dict(self.hand_log),
        )

    def fresh(self, level, seed, target=None):
        """Additive: a layout drawn by SIMPLE's own randomisation (reset with state_dict None at dr_level `level`, seeded);
        -> its environment configuration (task.state_dict(), what control.py stores), the instruction, and the robot's and the
        objects' poses as drawn, so that reset_cfg of the configuration can be compared with them."""
        import copy
        import random

        if (
            target is not None
        ):  # additive, draws only: this draw's target asset "<res>:<id>" (and no duplicate among the distractors)
            dr = self.u.task.dr
            rnd = dr.get_randomizer("target")
            rnd.cfg.asset_id = target
            rnd.res_id, rnd.obj_id = target.split(":")
            dis = dr.get_randomizer("distractors")
            if dis is not None:
                dis.cfg.exclude = [rnd.obj_id]
        np.random.seed(seed % 2**32)
        random.seed(seed)
        self.obs, _ = self.env.reset(seed=seed, options={"state_dict": None, "dr_level": level})
        st = self.state()
        objs = {
            k: np.concatenate([np.asarray(v.xpos), np.asarray(v.xquat)]) for k, v in self.u.mujoco.mj_objects.items()
        }
        return dict(
            cfg=copy.deepcopy(self.u.task.state_dict()),
            instruction=self.u.task.instruction,
            qpos=st["qpos"],
            objects=objs,
            art=self.art_joints(),
        )

    def reset_with(self, conf):
        t0 = time.time()
        self.obs, _ = self.env.reset(options={"state_dict": conf})
        self.reset_s = time.time() - t0
        self.hits = {}
        self.hand_log = {}
        self.first_contacts = []
        self.comply, self.comply_done = (
            (C.ArtWatcher(self.task) if self.art else C.Watcher(0.02)),
            False,
        )  # a new episode's watcher
        if self.art:
            self.art_setup()
        m = self.u.mujoco.mjModel
        if self.amo:
            self.rob = self.u.task.robot  # in case a reset built a new robot object: the same two patches
            self.rob.apply_action, self.rob.step = (lambda a: None), self.amo_step
            import mujoco

            jn = [m.joint(i).name for i in range(1, 44)]
            an = [m.actuator(i).name for i in range(m.nu)]
            ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in jn]
            assert min(ids) >= 0, f"actuators {m.nu}: {an[:4]} ... joints {jn[:4]}"
            self.cidx = np.array(ids)  # the actuator of each joint, by name
            by = dict(zip(self.rob.joints_names, np.asarray(self.rob.torque_limits, float)))
            self.servo = np.array(
                [int(m.actuator_biastype[i]) == 1 for i in ids]
            )  # joints whose actuator is a position servo
            self.ctrl_lo = np.array([m.actuator_ctrlrange[i][0] for i in ids])
            self.ctrl_hi = np.array([m.actuator_ctrlrange[i][1] for i in ids])
            sk = dict(zip(self.rob.joints_names, np.asarray(self.rob.stiffness, float)))
            dk = dict(zip(self.rob.joints_names, np.asarray(self.rob.damping, float)))
            self.hand_mask = np.array(["hand" in n for n in jn])
            self.hand_clip = np.array([2.45 if "thumb_0" in n else 0.7 for n in jn])
            self.hand_kp = np.array([sk[n] for n in jn])
            self.hand_kd = np.array([dk[n] for n in jn])
            self.lim = np.array([by[n] for n in jn])  # the robot's list is in its own joint order, not the model's
            self.rob.command, self.rob.is_replay, self.rob.is_eval = (
                [0.0] * 8,
                False,
                False,
            )  # the engine steps by robot.step
            self.frames, self.steps = [], 0
            return dict(self.state(), names=[m.joint(i).name for i in range(1, 44)], limit=self.limit,
                        torque_limit=self.lim, instruction=self.u.task.instruction)  # fmt: skip
        self.frames, self.steps = [], 0
        idx = lambda a: (np.asarray(a) - 1).tolist()  # noqa: E731
        return dict(self.state(), names=[m.joint(i).name for i in range(1, 44)], body=idx(self.rob.body_joint_index),
                    left=idx(self.rob.left_hand_index), right=idx(self.rob.right_hand_index), limit=self.limit,
                    torque_limit=np.asarray(self.rob.torque_limit), instruction=self.u.task.instruction)  # fmt: skip

    # ---- TODO 52: saved states (snap / restore). Everything the stepping changes, on this side of the socket.

    def state_objects(self) -> dict:
        u = self.u
        out = dict(ep=self, u=u, mj=u.mujoco, tl=self.env, robot=u.task.robot, task=u.task)
        if getattr(u, "isaac", None) is not None:
            out["isaac"] = u.isaac
        return out

    SKIP = dict(
        ep=(
            "frames",
            "obs",
            "prof",
            "snaps",
            "n_snap",
            "_last_frame",
        ),  # (_last_frame: the real frame the lazy observation shows, set at setup)
        u=("viewer",),
        mj=("renderers", "viewer"),
        isaac=(),
        tl=(),
        robot=(),
        task=(),
    )

    def snap(self) -> int:
        owned = (C.Watcher, C.ArtWatcher)
        models = (self.u.mujoco.mjModel,)
        snap = {}
        for n, o in self.state_objects().items():
            snap[n] = SC.capture(o, skip=self.SKIP.get(n, ()), owned=owned, models=models)
        self.n_snap = getattr(self, "n_snap", 0) + 1
        self.snaps[self.n_snap] = snap
        return self.n_snap

    # callables the saved states hold that are set once while the episode is set up and never change: the live ones stay
    STATIC_FN = ("ep._render0", "ep._orch0", "u._render_frame", "u.substep_callback")

    def _keep_fn(self, f, path):
        if path in self.STATIC_FN:
            return SC.KEEP
        raise SC.Unpersistable(f"callable at {path}")

    def dump(self, sid: int):
        """TODO 53: the saved state `sid` as bytes (a file of the memo's disk store), or the reason it cannot be."""
        if sid not in self.snaps:
            return f"no state {sid}"
        try:
            return SC.dumps(self.snaps[sid], fn=self._keep_fn)
        except Exception as e:  # (never the server's end: the caller keeps the state in memory)
            return f"unpersistable: {e!r}"

    def unpersistable(self, sid: int) -> list:
        """TODO 53 (a diagnostic): what in the saved state `sid` the disk format would refuse."""
        return SC.unpersistable(self.snaps[sid])

    def load(self, blob: bytes):
        """TODO 53: a state written by `dump` (this process or another) as a new saved state: its id, or the reason it fails."""
        try:
            snap = SC.loads(blob, owned=(C.Watcher, C.ArtWatcher))
        except Exception as e:
            return f"unreadable: {e!r}"
        self.n_snap = getattr(self, "n_snap", 0) + 1
        self.snaps[self.n_snap] = snap
        return self.n_snap

    def snap_diff(self, sid: int) -> list:
        """The attributes whose current value differs from the saved state `sid` (a check, TODO 52): ["object.attr", ...]."""
        owned = (C.Watcher, C.ArtWatcher)
        models = (self.u.mujoco.mjModel,)
        out = []
        for n, o in self.state_objects().items():
            now = SC.capture(o, skip=self.SKIP.get(n, ()), owned=owned, models=models)
            for k, p in self.snaps[sid][n].items():
                if p.kind != "ref" and k in now and SC.fingerprint({k: p}) != SC.fingerprint({k: now[k]}):
                    out.append(f"{n}.{k}")
        return out

    def restore(self, sid: int):
        if sid not in self.snaps:
            return f"no state {sid}"
        models = (self.u.mujoco.mjModel,)
        try:
            for n, o in self.state_objects().items():
                SC.restore(o, self.snaps[sid][n], models=models)
        except (
            Exception
        ) as e:  # (a state read from disk that does not fit this episode: the caller must not go on from here)
            return f"restore failed: {e!r}"
        mj, u = self.u.mujoco, self.u
        # (no mj_forward here: sensors and body poses are one substep behind the state after a step, and they were restored as such)
        if mj.render_step > 5:  # the update MujocoEngine.step makes of the layout's poses after each step
            for objtype, mj_obj in mj.mj_objects.items():
                u.task.layout.actors[objtype].pose.position = list(mj_obj.xpos)
                u.task.layout.actors[objtype].pose.quaternion = list(mj_obj.xquat)
            u.task.layout.actors["robot"].pose.position = list(np.round(mj.mjData.qpos[:3], 3))
            u.task.layout.actors["robot"].pose.quaternion = list(np.round(mj.mjData.qpos[3:7], 3))
        if (
            getattr(u, "isaac", None) is not None
        ):  # the renderer's scene follows the restored state before any image is read
            step_id = u.isaac.step_id
            u.isaac.step(mj)
            u.isaac.step_id = step_id
            if not getattr(self, "lazy", False):
                self.obs = u._get_obs()  # every-step rendering: `eyes` reads the last step's frame
        return True

    def step(self, q, kp, kd, counted, film):
        if self.amo:
            self.pend = (np.asarray(q, float), np.asarray(kp, float), np.asarray(kd, float))
            self.act = self.act or ActionCmd("amo_torque")
        else:
            b, lft, rgt = (
                np.asarray(a) - 1
                for a in (self.rob.body_joint_index, self.rob.left_hand_index, self.rob.right_hand_index)
            )
            for cmd, ids in ((self.low, b), (self.lh, lft), (self.rh, rgt)):
                for c, i in zip(cmd.motor_cmd, ids):
                    c.q, c.kp, c.kd = float(q[i]), float(kp[i]), float(kd[i])
        t_in = time.perf_counter()
        if self.prof is not None:
            self.prof.enable()
        if self.art:
            self.art_hand, self.art_body = False, set()  # filled by on_sub during the step below
            if counted and not self.comply.q0:
                self.comply.start(self.art_q())  # the joints when the counted time begins
        if counted:
            self.obs, rew, term, trunc, _ = self.env.step(self.act)
            self.steps += 1
            if (
                self.art and not self.comply_done
            ):  # the articulated tasks: joints and contacts of this step, up to the check
                self.comply.feed(self.steps * 0.02, self.art_q(), self.art_hand, self.art_body, bool(self.u._success))
                self.comply_done = bool(self.u._success)
            elif (
                not self.comply_done
            ):  # comply_simple: one snapshot per counted step, up to the step the check first holds
                mj = self.u.mujoco
                cont = mj.mj_objects["container"].id if "container" in mj.mj_objects else None
                pick = self.task in C.PICK  # a pick is judged per step (held run of 0.2 s), others at the check
                snap = C.snapshot(mj.mjModel, mj.mjData, mj.mj_objects["target"].id, cont)
                self.comply.feed(self.steps * 0.02, snap, bool(self.u._success) if pick else None)
                self.comply_done = (self.comply.t_held is not None) if pick else bool(self.u._success)
        else:  # stabilising before the episode: the raw environment, as eval_decoupled_wbc.py does
            self.obs, rew, term, trunc, _ = self.u.step(self.act)
            term = False
        if self.prof is not None:
            self.prof.disable()
        self.t_step[0] += (
            time.perf_counter() - t_in
        )  # additive timing (message `timing`): seconds inside the environment's step
        self.t_step[1] += 1
        if film and (counted and self.steps % 2 == 0 or not counted and len(self.frames) < 2):
            self.frames.append(np.asarray(self.obs["head_stereo_left"]).copy())
        return dict(
            self.state(),
            terminated=bool(term),
            truncated=bool(trunc),
            success=bool(self.u._success),
            steps=self.steps,
            held_t=self.comply.t_held,  # the pick rule's held run reached 0.2 s (first step of it), else None
            reward=float(rew),
        )

    def video(self, path):
        import cv2

        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
        for f in self.frames:
            w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        w.release()
        return len(self.frames)


ep = None
with Listener(sock, family="AF_UNIX") as lis:
    print("listening", sock, flush=True)
    while True:
        with lis.accept() as c:
            try:
                while True:
                    msg = c.recv()
                    k = msg[0]
                    if k == "reset":
                        raise RuntimeError("no released episode: the bed opens practice layouts only (reset_cfg)")
                        if ep is None or ep.task != msg[1]:
                            if ep is not None:
                                ep.env.close()
                            ep = Episode(msg[1])
                        c.send(ep.reset(msg[2], msg[3]))
                    elif k == "reset_cfg":
                        if ep is None or ep.task != msg[1]:
                            if ep is not None:
                                ep.env.close()
                            ep = Episode(msg[1])
                        c.send(dict(ep.reset_with(msg[2]), reset_s=ep.reset_s))
                    elif k == "fresh":  # additive: ("fresh", task, dr_level, seed)
                        if ep is None or ep.task != msg[1]:
                            if ep is not None:
                                ep.env.close()
                            ep = Episode(msg[1])
                        try:
                            c.send(ep.fresh(msg[2], msg[3], *(msg[4:5])))
                        except Exception as e:  # SIMPLE's own sampler can draw a layout its model builder refuses
                            c.send(dict(error=f"{type(e).__name__}: {e}"[:300]))
                    elif (
                        k == "lazy_render"
                    ):  # additive, default off: no camera rendering in the steps; `eyes` renders on request
                        if not hasattr(ep, "_render0"):
                            ep._render0 = ep.u._render_frame
                        if not hasattr(ep, "_orch0"):
                            import omni.replicator.core as rep

                            ep._rep, ep._orch0 = rep, rep.orchestrator.step
                            ep.lazy, ep.force = False, False
                            ep._rep.orchestrator.step = lambda *a, **k: (
                                ep._orch0(*a, **k) if (ep.force or not ep.lazy) else None
                            )
                        ep.lazy = bool(msg[1])
                        ep.warm = (
                            int(msg[2]) if len(msg) > 2 else 0
                        )  # extra renders of the same state before `eyes` reads (TODO 52)
                        ep.sub = int(msg[3]) if len(msg) > 3 else 1  # rt_subframes of every `eyes` render
                        if (
                            ep.lazy
                        ):  # the observation keeps its image keys (gymnasium's checker) with the last real frame
                            ep._last_frame = ep._render0()
                        ep.u._render_frame = (lambda: ep._last_frame) if ep.lazy else ep._render0
                        c.send(ep.lazy)
                    elif k == "snap":  # TODO 52: save the whole state of the episode; -> an id
                        c.send(ep.snap())
                    elif k == "snap_dump":  # TODO 53: ("snap_dump", id) -> bytes
                        c.send(ep.dump(msg[1]))
                    elif k == "snap_unpersistable":  # TODO 53 diagnostic: ("snap_unpersistable", id) -> [(path, why)]
                        c.send(ep.unpersistable(msg[1]))
                    elif k == "snap_load":  # TODO 53: ("snap_load", bytes) -> a new id
                        c.send(ep.load(msg[1]))
                    elif k == "restore":  # TODO 52: ("restore", id): put it back, the renderer's scene synced
                        c.send(ep.restore(msg[1]))
                    elif (
                        k == "snap_diff"
                    ):  # TODO 52: ("snap_diff", id) -> the attributes that differ from the saved state
                        c.send(ep.snap_diff(msg[1]))
                    elif k == "snap_free":  # TODO 52: ("snap_free", [ids])
                        for i in msg[1]:
                            ep.snaps.pop(i, None)
                        c.send(True)
                    elif k == "render":  # record only: stop (False) or resume (True) the cameras' rendering in the step
                        if not hasattr(ep, "_render0"):
                            ep._render0 = ep.u._render_frame
                        ep.u._render_frame = ep._render0 if msg[1] else (lambda: {})
                        c.send(bool(msg[1]))
                    elif (
                        k == "amo_arm_mode"
                    ):  # default off: the AMO robot's arm joints get their target angle, not a torque
                        ep.arm_pos = bool(msg[1])
                        c.send(ep.arm_pos)
                    elif (
                        k == "amo_hand_gain"
                    ):  # default off: the AMO hand joints use SIMPLE's own PD gains (the clip is its own already)
                        ep.hand_gain = bool(msg[1])
                        c.send(ep.hand_gain)
                    elif k == "amo_hand_limit":  # default off: the AMO hand joints clipped at the teleop Dex3 limits
                        ep.hand_limit = bool(msg[1])
                        c.send(ep.hand_limit)
                    elif (
                        k == "timing"
                    ):  # additive: seconds and count inside env.step so far; ("timing", "prof") starts a cProfile of the steps, ("timing", "stop") returns its top
                        if len(msg) > 1 and msg[1] == "prof":
                            import cProfile

                            ep.prof = cProfile.Profile()
                            c.send("profiling")
                        elif len(msg) > 1 and msg[1] == "stop":
                            import io
                            import pstats

                            ep.prof.disable()
                            out = io.StringIO()
                            pstats.Stats(ep.prof, stream=out).sort_stats("cumulative").print_stats(28)
                            ep.prof = None
                            c.send(out.getvalue())
                        else:
                            c.send(dict(step_s=ep.t_step[0], steps=ep.t_step[1]))
                    elif (
                        k == "art_info"
                    ):  # additive: the articulated asset's joints (name, qpos address, range) and moving bodies (names)
                        m = ep.u.mujoco.mjModel
                        d = ep.u.mujoco.mjData
                        c.send(
                            dict(
                                cols=ep.art_cols,
                                moving=sorted(m.body(b).name for b in ep.art_move),
                                pos={m.body(b).name: d.xpos[b].tolist() for b in ep.art_move},
                                boxes={m.body(b).name: ep.art_box(b) for b in ep.art_move},
                                types={n: int(m.jnt_type[m.joint(n).id]) for n, _ in ep.art_cols},
                                geoms={
                                    m.body(b).name: [
                                        [int(m.geom_contype[g]), int(m.geom_conaffinity[g])]
                                        for g in range(m.ngeom)
                                        if m.geom_bodyid[g] == b
                                    ][:3]
                                    for b in ep.art_move
                                },
                                ranges={n: m.jnt_range[m.joint(n).id].tolist() for n, _ in ep.art_cols},
                            )
                        )
                    elif k == "comply":  # additive: the watcher's state (comply_simple.Watcher.state()) of this episode
                        c.send(ep.comply.state())
                    elif k == "drives":
                        c.send(ep.drives())
                    elif k == "geoms":
                        c.send(ep.geoms(msg[1]))
                    elif k == "campose":
                        c.send(ep.campose())
                    elif k == "feet":
                        c.send(ep.feet())
                    elif k == "phys2":
                        c.send(ep.phys2())
                    elif k == "phys":
                        c.send(ep.phys())
                    elif k == "tp_setup":  # additive, only for the videos: a third-person camera in the Isaac scene
                        c.send(ep.tp_setup(*msg[1:]))
                    elif k == "tp_frame":
                        c.send(ep.tp_frame(*msg[1:]))
                    elif k == "truth":
                        c.send(ep.truth())
                    elif k == "step":
                        c.send(ep.step(*msg[1:]))
                    elif (
                        k == "rgbd_setup"
                    ):  # sensor=rgbd only: Isaac's depth annotator on the two head cameras (cli/render.py's call)
                        if not getattr(
                            ep, "rgbd", False
                        ):  # once per Episode object: the annotator stays attached over resets
                            for n in (
                                "head_stereo_left",
                                "head_stereo_right",
                            ):  # SIMPLE's engines/isaacsim.py keeps its cameras in u.isaac.cameras, by these names
                                ep.u.isaac.cameras[n].add_distance_to_image_plane_to_frame()
                            ep.rgbd = True
                        c.send(True)
                    elif (
                        k == "eyes_d"
                    ):  # sensor=rgbd: both head images and both rendered depths from the SAME render, lazy render as "eyes"

                        def _depths():
                            """The annotator's own data, not Camera.get_depth(): that reads the camera's frame dict, which the
                            camera updates only when its sim-time throttle passes, and after a reset (sim time restarts) it stays
                            on the previous episode's last frame (found on the 2nd episode of a server)."""
                            out = []
                            for n in ("head_stereo_left", "head_stereo_right"):
                                ann = ep.u.isaac.cameras[n]._custom_annotators["distance_to_image_plane"]
                                d = ann.get_data()
                                out.append(
                                    None if d is None else np.asarray(d, np.float32).reshape(d.shape[0], d.shape[1])
                                )
                            return out

                        import omni.replicator.core as rep

                        step = getattr(ep, "_orch0", None) or rep.orchestrator.step
                        if getattr(ep, "lazy", False):
                            ep.force = True
                            try:
                                for _ in range(1 + getattr(ep, "warm", 0)):
                                    step(rt_subframes=getattr(ep, "sub", 1), pause_timeline=False)
                                fr = ep._render0()
                            finally:
                                ep.force = False
                        else:
                            fr = dict(
                                head_stereo_left=ep.obs["head_stereo_left"],
                                head_stereo_right=ep.obs["head_stereo_right"],
                            )
                        dl, dr = _depths()
                        for _ in range(3):
                            if dl is not None and dr is not None:
                                break
                            step(
                                rt_subframes=1, pause_timeline=False
                            )  # the annotator holds nothing until a render has run
                            dl, dr = _depths()
                        if dl is None or dr is None:
                            raise RuntimeError("eyes_d: the head cameras' depth annotator returned nothing")
                        c.send(
                            dict(
                                left=np.asarray(fr["head_stereo_left"]),
                                right=np.asarray(fr["head_stereo_right"]),
                                depth_left=np.asarray(dl, np.float32).copy(),
                                depth_right=np.asarray(dr, np.float32).copy(),
                            )
                        )
                    elif k == "verts":  # target=truth only: world vertices of scene bodies, per collision geom
                        c.send(ep.verts(msg[1]))
                    elif (
                        k == "art_verts"
                    ):  # target=truth only: the articulated asset's vertices (whole, and per moving body)
                        c.send(ep.art_verts())
                    elif k == "eyes" and getattr(ep, "lazy", False):
                        ep.force = True  # one render of the state now (the steps in between rendered nothing)
                        try:
                            for _ in range(
                                1 + getattr(ep, "warm", 0)
                            ):  # `warm` extra renders of the same state: the renderer's
                                ep._orch0(
                                    rt_subframes=getattr(ep, "sub", 1), pause_timeline=False
                                )  # temporal state settles
                            fr = ep._render0()
                        finally:
                            ep.force = False
                        c.send(dict(left=np.asarray(fr["head_stereo_left"]), right=np.asarray(fr["head_stereo_right"])))
                    elif (
                        k == "eyes_seq"
                    ):  # diagnostic (TODO 52): n renders of the SAME state, both head images after each
                        out = []
                        if not hasattr(ep, "_orch0"):
                            raise RuntimeError(
                                "eyes_seq needs a lazy_render message first (it installs the render hook)"
                            )
                        ep.force = True
                        try:
                            for _ in range(int(msg[1])):
                                ep._orch0(rt_subframes=int(msg[2]) if len(msg) > 2 else 1, pause_timeline=False)
                                fr = ep._render0()
                                out.append(
                                    (
                                        np.asarray(fr["head_stereo_left"]).copy(),
                                        np.asarray(fr["head_stereo_right"]).copy(),
                                    )
                                )
                        finally:
                            ep.force = False
                        c.send(out)
                    elif k == "eyes":
                        c.send(
                            dict(
                                left=np.asarray(ep.obs["head_stereo_left"]),
                                right=np.asarray(ep.obs["head_stereo_right"]),
                            )
                        )
                    elif k == "video":
                        c.send(ep.video(msg[1]))
                    elif k == "quit":
                        raise SystemExit
            except EOFError:
                pass
