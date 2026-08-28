"""Privileged simulator state on ZMQ :5581 — the agent's substitute for eyes.

LEGS-style scripted collection needs ground-truth object poses. The MuJoCo sim
runs in its own process (run_sim_loop.py), so the state has to cross a process
boundary; the camera already crosses on :5555 and the control stream on :5556,
so this is one more PUB in the same style.

    sim process   --(:5581 privileged state, 50 Hz)-->   agent process

JSON rather than the packed binary format the control topics use: the payload
is a few hundred bytes at 50 Hz, and being able to `nc` the port and read it is
worth more than the bytes.

Convention throughout: positions are world-frame metres, quaternions are
scalar-first (w, x, y, z) to match MuJoCo.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import zmq

GHOST_BODY = "ghost_hand"

# Arm-pin state shared between the episode-control server (which sets it over
# :5582) and the wrapped bridge GetAction in run_sim_locomani (which applies
# it to the arm motors' position targets every physics step). 7 joint values
# per side (shoulder p/r/y, elbow, wrist r/p/y) or None = released.
ARM_PIN: dict = {"left": None, "right": None, "waist": None}
# Sim-side OBJECT pin: {name: [x, y, z, qw, qx, qy, qz]} or empty. The episode
# server rewrites those free joints every poll, which holds an object exactly
# where a diagnostic put it — gravity, contact and the robot all stop moving
# it. Used by diag_reach_error.py to measure delivery error against a target
# that cannot be knocked away.
OBJ_PIN: dict = {}
GHOST_PARK = (0.0, 0.0, -5.0)

# Per-instance overrides so several stacks can share the host (parallel
# data collection); instance i uses base + 10*i via start_stack.sh.
PRIVILEGED_PORT = int(os.environ.get("TF_PRIVILEGED_PORT", "5581"))
EPISODE_PORT = int(os.environ.get("TF_EPISODE_PORT", "5582"))
# SCENE_RANDOM=0 pins the appearance, which is what a controlled comparison
# wants (two runs differing only in the thing under test) and what a replay of
# an old capture needs. On by default: a dataset where every frame shares one
# lighting rig and one table colour teaches the model to rely on both.
SCENE_RANDOM = os.environ.get("SCENE_RANDOM", "1") != "0"

# Palm frame. The G1's three-finger hand hangs off the wrist yaw link, so this
# is the frame a pre-grasp offset is naturally expressed in.
PALM_BODY = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}

# Everything distal of the wrist — used for the "is it actually grasped"
# contact test.
HAND_LINK_PREFIX = ("left_hand_", "right_hand_",
                    "left_dex1_", "right_dex1_")

# Order matters: it is the order the values in a `planner` message's
# left/right_hand_joints field are interpreted in. Per hand type (HAND env):
# dex3 = 7 finger joints; dex1 = the two prismatic gripper fingers.
if os.environ.get("HAND", "dex1") == "dex1":
    HAND_JOINTS = ["gripper_1", "gripper_2"]
else:
    HAND_JOINTS = ["thumb_0", "thumb_1", "thumb_2",
                   "middle_0", "middle_1", "index_0", "index_1"]


def _body_pose(model, data, bid: int):
    return (data.xpos[bid].tolist(), data.xquat[bid].tolist())


class PrivilegedPublisher:
    """Binds :5581 and publishes ground truth at whatever rate it is stepped."""

    def __init__(self, model, data, object_bodies: dict[str, str],
                 port: int = PRIVILEGED_PORT, surfaces: list[str] | None = None):
        self._seq = 0
        import mujoco

        self._mj = mujoco
        self.model = model
        self.data = data

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        # CONFLATE on the subscriber, not here: the agent wants the newest
        # state, never a queue of stale ones.
        self.sock.bind(f"tcp://*:{port}")

        def bid(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

        self.objects = {k: bid(v) for k, v in object_bodies.items()}
        self.missing = [k for k, v in self.objects.items() if v < 0]
        self.objects = {k: v for k, v in self.objects.items() if v >= 0}
        self.palms = {s: bid(n) for s, n in PALM_BODY.items()}
        self.pelvis = bid("pelvis")

        # Ego camera, for the VLA data-quality predicate: an episode whose RGB
        # never shows the object teaches a policy to act blind. The exporter
        # renders "head_camera" at 640x480.
        self.cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        self.cam_aspect = 640.0 / 480.0

        # The reach-zone carpet (debug visual injected by run_sim_locomani)
        # follows the pelvis: xy + yaw only, like the map's pelvis frame.
        zb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "reach_zone")
        self.zone_mocap = int(model.body_mocapid[zb]) if zb >= 0 else -1

        surface_bodies = {bid(n) for n in surfaces} if surfaces else set()

        # geom -> True for every geom on a surface (table/counter) body, so
        # the state can report robot-vs-furniture contacts — the arm swinging
        # into the table edge is otherwise invisible in the logs.
        self.surf_geoms = {g for g in range(model.ngeom)
                           if model.geom_bodyid[g] in surface_bodies}

        # geom -> "left" / "right" for every geom on a hand link.
        self.hand_geom_side = {}
        # geom -> short link name, resolved once. Looking these up per contact
        # per step would put a name lookup in the 50 Hz publish loop.
        self.hand_geom_link = {}
        for g in range(model.ngeom):
            name = model.body(model.geom_bodyid[g]).name
            for prefix in HAND_LINK_PREFIX:
                if name.startswith(prefix):
                    self.hand_geom_side[g] = prefix.split("_")[0]
                    self.hand_geom_link[g] = (name.replace("_link", "")
                                              .replace("left_hand_", "L:")
                                              .replace("right_hand_", "R:"))
        # geom -> short body name for EVERY robot geom, not just the hand's.
        # `touching` stays hand-only, because that is what confirms a grasp,
        # but the *diagnostic* must not be: a can that topples with "nothing"
        # touching it was hit by the forearm or the wrist, and restricting the
        # report to the hand hides exactly the contact worth knowing about.
        obj_bodies = set(self.objects.values())
        self.robot_geom_link = {}
        for g in range(model.ngeom):
            b = model.geom_bodyid[g]
            # Surfaces are excluded along with the objects and the world: a can
            # standing on a table touches the table always, and reporting that
            # buries the one contact worth seeing.
            if b in obj_bodies or b == 0 or b in surface_bodies:
                continue
            nm = model.body(b).name
            self.robot_geom_link[g] = (nm.replace("_link", "")
                                       .replace("left_hand_", "L:")
                                       .replace("right_hand_", "R:")
                                       .replace("left_", "L.")
                                       .replace("right_", "R."))

        # geom -> object key, for the contact test on the other side.
        self.obj_geom_key = {}
        for key, b in self.objects.items():
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == b:
                    self.obj_geom_key[g] = key

        # Hand joint qpos, so the agent can confirm the fingers actually moved
        # rather than trusting that a command was applied.
        self.hand_qadr = {}
        for side in ("left", "right"):
            adrs = []
            for j in HAND_JOINTS:
                jid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_{j}_joint")
                adrs.append(int(model.jnt_qposadr[jid]) if jid >= 0 else -1)
            self.hand_qadr[side] = adrs

        # Robot joint qpos addresses, MuJoCo order — the streamed-motion path
        # needs the measured 29-vector to seed IK and to hold the un-commanded
        # joints at their current angles.
        try:
            from g1_joints import MUJOCO_JOINT_NAMES
            self.joint_qadr = []
            for n in MUJOCO_JOINT_NAMES:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                self.joint_qadr.append(int(model.jnt_qposadr[jid]) if jid >= 0
                                       else -1)
        except ImportError:
            self.joint_qadr = []

        print(f"[privileged] :{port} tracking {list(self.objects)}"
              + (f" (MISSING {self.missing})" if self.missing else ""))

    def _contacts(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Which hand, and which of its links, is touching each object.

        The link names are what tell a failed grasp apart from a swipe. "The
        right hand touched it" is true both when the object is seated in the
        jaw and when a fingertip has just knocked it across the table; the link
        that made contact first says which.
        """
        touching: dict[str, set] = {k: set() for k in self.objects}
        links: dict[str, set] = {k: set() for k in self.objects}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                key = self.obj_geom_key.get(a)
                if key is None:
                    continue
                side = self.hand_geom_side.get(b)
                if side is not None:
                    touching[key].add(side)
                link = self.robot_geom_link.get(b)
                if link is not None:
                    links[key].add(link)

        return ({k: sorted(v) for k, v in touching.items()},
                {k: sorted(v) for k, v in links.items()})

    def state(self) -> dict:
        d = self.data
        pelvis_pos, pelvis_quat = _body_pose(self.model, d, self.pelvis)
        # Yaw from the pelvis quaternion (w, x, y, z).
        w, x, y, z = pelvis_quat
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        if self.zone_mocap >= 0:
            # Full 3D pose incl. HEIGHT: the volume rides down with a squat.
            d.mocap_pos[self.zone_mocap] = [pelvis_pos[0], pelvis_pos[1],
                                            pelvis_pos[2]]
            d.mocap_quat[self.zone_mocap] = [np.cos(yaw / 2), 0.0, 0.0,
                                             np.sin(yaw / 2)]

        contacts, contact_links = self._contacts()
        objects = {}
        for key, b in self.objects.items():
            pos, quat = _body_pose(self.model, d, b)
            objects[key] = {
                "pos": pos, "quat": quat, "touching": contacts.get(key, []),
                "touch_links": contact_links.get(key, []),
            }
            if self.cam_id >= 0:
                # Normalized image coords in [-1, 1]: the MuJoCo camera looks
                # along its frame's -z, +x right, +y up.
                cp = d.cam_xpos[self.cam_id]
                cm = d.cam_xmat[self.cam_id].reshape(3, 3)
                v = cm.T @ (np.asarray(pos) - cp)
                if v[2] < -1e-6:
                    ty = float(np.tan(0.5 * np.deg2rad(
                        self.model.cam_fovy[self.cam_id])))
                    objects[key]["cam"] = [
                        float(v[0] / -v[2] / (ty * self.cam_aspect)),
                        float(v[1] / -v[2] / ty)]
                else:
                    objects[key]["cam"] = None    # behind the camera

        palms = {}
        for side, b in self.palms.items():
            pos, quat = _body_pose(self.model, d, b)
            palms[side] = {
                "pos": pos, "quat": quat,
                "hand_q": [float(d.qpos[a]) if a >= 0 else 0.0
                           for a in self.hand_qadr[side]],
            }

        return {
            "t": float(d.time),
            "wall": time.time(),
            "pelvis": {"pos": pelvis_pos, "quat": pelvis_quat, "yaw": yaw,
                       "vel": d.qvel[:3].tolist()},
            "palms": palms,
            "objects": objects,
            "furniture_touch": sorted({
                self.robot_geom_link[b]
                for i in range(d.ncon)
                for a, b in ((d.contact[i].geom1, d.contact[i].geom2),
                             (d.contact[i].geom2, d.contact[i].geom1))
                if a in self.surf_geoms and b in self.robot_geom_link}),
            "joints": [float(d.qpos[a]) if a >= 0 else 0.0
                       for a in self.joint_qadr],
            "fallen": bool(pelvis_pos[2] < 0.4),
        }

    def publish(self) -> None:
        # Monotonic per-process sequence number: the ONE shared key between
        # this stream and everything that samples it. The exporter CONFLATEs
        # this stream and records nothing about which message it used, which
        # is why pose<->parquet alignment was structurally impossible
        # (docs/pose_parquet_alignment.md: seven offline methods, no
        # diagonal). Restarts restart it at 0 -- consumers must treat a
        # backwards jump as a new lineage, not wraparound.
        d = self.state()
        d["seq"] = self._seq
        self._seq += 1
        self.sock.send_string(json.dumps(d))

    def close(self) -> None:
        self.sock.close()
        self.ctx.term()


class EpisodeControlServer:
    """Sim-side reset channel on :5582.

    Two hundred episodes need two hundred resets, and each one has to place the
    objects and the robot at freshly sampled poses — so reset and perturbation
    are the same operation, not two. Doing it in the sim process is what makes
    it atomic with respect to physics: the agent cannot observe a half-reset
    world, because `mj_resetData` and the pose writes happen between two steps.

    PULL rather than REP: the agent does not need a reply it would only ignore,
    and REP would deadlock the sim loop if the agent died mid-request.
    """

    def __init__(self, model, data, meta: dict, default_dof_angles=None,
                 port: int = EPISODE_PORT, recorder=None):
        import mujoco

        # The ghost hand is a mocap body: no joints, no collision, teleported
        # by writing mocap_pos. -1 when the scene predates it.
        gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GHOST_BODY)
        self.ghost_mocap = int(model.body_mocapid[gb]) if gb >= 0 else -1
        # Built on first use: snapshotting the pristine appearance costs a
        # copy of the texture block, and a run with SCENE_RANDOM=0 should not
        # pay it.
        self._rand = None
        self._rand_mod = None

        self._mj = mujoco
        self.model = model
        self.data = data
        self.meta = meta
        # Resetting to qpos0 does not work: qpos0 is every joint at zero, i.e.
        # straight-legged, and the controller cannot hold that — the robot
        # pitches forward and is flat on the floor inside one second. The
        # controller's own nominal stance (hips -0.1, knees +0.3, ankles -0.2)
        # is the pose it is built to stabilise. The mapping to qpos is
        # positional, the same convention the sim's PD loop uses
        # (base_sim.py indexes qpos[qpos_offset : qpos_offset + num_body_dof]).
        self.default_dof_angles = (
            np.asarray(default_dof_angles, dtype=float)
            if default_dof_angles is not None else None)

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(f"tcp://*:{port}")

        # qpos address of each free object, and of the robot's floating base.
        self.obj_qadr = {}
        for name, info in meta["objects"].items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, info["body"])
            if bid < 0:
                continue
            jadr = model.body_jntadr[bid]
            if jadr >= 0 and model.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE:
                self.obj_qadr[name] = int(model.jnt_qposadr[jadr])

        # A reset is also the natural episode boundary for video: one file per
        # episode, started by whoever asked for the reset, named by its label.
        self.recorder = recorder
        self.qpos0 = model.qpos0.copy()
        self.root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        print(f"[episode] :{port} can reset {sorted(self.obj_qadr)}")

    @staticmethod
    def _yaw_quat(yaw: float):
        return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

    def _seat_on_floor(self) -> None:
        """Drop the robot so its lowest foot geom rests on z = 0.

        Bending the knees raises the feet relative to the pelvis, so keeping the
        pelvis at its nominal 0.793 would leave the robot hanging a few
        centimetres in the air and drop it the moment physics resumes.
        """
        mujoco = self._mj
        mujoco.mj_forward(self.model, self.data)
        lowest = min(
            (float(self.data.geom_xpos[g][2] - self.model.geom_rbound[g])
             for g in range(self.model.ngeom)
             if self.model.body_rootid[self.model.geom_bodyid[g]] == self.root_id),
            default=0.0,
        )
        self.data.qpos[2] -= lowest

    def _appearance(self, seed) -> None:
        """Re-roll what the ego camera sees, from one integer.

        Lives here, not in the agent, because the model is here: it is a
        mutation of mjModel fields between two physics steps, exactly like the
        pose writes above, and just as atomic. Nothing about it touches the
        control path -- geometry, masses and contacts are untouched -- so an
        episode looks different and behaves identically.
        """
        if seed is None or not SCENE_RANDOM:
            return
        if self._rand is None:
            import scene_random
            self._rand = scene_random.SceneRandomizer(self.model, self._mj)
            self._rand_mod = scene_random
        print(f"[episode] appearance: "
              f"{self._rand.apply(self._rand_mod.sample(int(seed)))}")

    def _apply(self, req: dict) -> None:
        mujoco = self._mj
        self._appearance(req.get("appearance"))
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.qpos0

        if self.default_dof_angles is not None:
            n = len(self.default_dof_angles)
            self.data.qpos[7: 7 + n] = self.default_dof_angles

        for name, delta in (req.get("objects") or {}).items():
            adr = self.obj_qadr.get(name)
            if adr is None:
                continue
            # dxy is a nudge from the nominal pose; dyaw replaces the heading,
            # so a perturbation set never compounds across episodes.
            dx, dy = delta.get("dxy", (0.0, 0.0))
            self.data.qpos[adr + 0] += dx
            self.data.qpos[adr + 1] += dy
            if "yaw" in delta:
                self.data.qpos[adr + 3: adr + 7] = self._yaw_quat(float(delta["yaw"]))

        robot = req.get("robot") or {}
        if robot:
            dx, dy = robot.get("dxy", (0.0, 0.0))
            self.data.qpos[0] += dx
            self.data.qpos[1] += dy
            if "yaw" in robot:
                self.data.qpos[3:7] = self._yaw_quat(float(robot["yaw"]))

        self._seat_on_floor()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def poll(self) -> None:
        """Non-blocking; call once per sim step."""
        if OBJ_PIN:
            self._apply_obj_pin()
        while self.sock.poll(0):
            try:
                req = json.loads(self.sock.recv_string())
            except Exception as e:  # a malformed request must not kill the sim
                print(f"[episode] bad request: {e}")
                continue
            if req.get("cmd") == "reset":
                self._apply(req)
                self._place_ghost(None, None)      # a new episode has no plan yet
                print(f"[episode] reset applied: {req.get('label', '')}")
            elif req.get("cmd") == "startvideo":
                # Deliberately *not* on reset. The agent stands still for
                # STAND_SETTLE seconds after every reset while the controller
                # settles the stance, and recording that put four seconds of a
                # motionless robot at the head of every clip — and of every
                # trajectory, had the exporter been wired the same way. The
                # caller starts the video once the robot is standing, so frame
                # one is already the task.
                if self.recorder is not None:
                    self.recorder.start(req.get("video") or "episode")
            elif req.get("cmd") == "ghost":
                # Park the ghost when no pose is given, so a stale plan from
                # the last episode is never left hanging in the shot.
                self._place_ghost(req.get("pos"), req.get("quat"))
            elif req.get("cmd") == "pin_arms":
                # Sim-side arm pin: the walking policy refuses to hold the
                # elbow past ~1.5 rad (its gait prior wins over any joint
                # reference, commanded up to the 2.09 joint limit), so the
                # level-forearm walking pose CANNOT be produced through the
                # policy. Instead the bridge's GetAction is wrapped
                # (run_sim_locomani) to overwrite the arm motors' position
                # targets with these values — the arm motors run their own
                # stiff position loop, exactly as real hardware allows,
                # while the policy keeps the legs and waist.
                ARM_PIN["left"] = req.get("left")
                ARM_PIN["right"] = req.get("right")
                ARM_PIN["waist"] = req.get("waist")
                print(f"[episode] arm pin: "
                      f"L={'on' if ARM_PIN['left'] else 'off'} "
                      f"R={'on' if ARM_PIN['right'] else 'off'}")
            elif req.get("cmd") == "pin_object":
                name, pose = req.get("name"), req.get("pose")
                if pose is None:
                    OBJ_PIN.pop(name, None)
                else:
                    OBJ_PIN[name] = [float(v) for v in pose]
                    self._apply_obj_pin()
            elif req.get("cmd") == "endvideo":
                # Nothing else closes the writer. A recording is only ever
                # stopped by the *next* reset, so the last episode of every
                # batch used to be left with no moov atom — an unplayable file,
                # and precisely the episode most worth watching.
                if self.recorder is not None:
                    self.recorder.stop()

    def _apply_obj_pin(self) -> None:
        """Hold every pinned object at its commanded pose, velocity zero."""
        for name, pose in OBJ_PIN.items():
            adr = self.obj_qadr.get(name)
            if adr is None:
                continue
            self.data.qpos[adr:adr + 7] = pose
            dadr = adr - 7 if adr >= 7 else adr      # free joint: 6 dof
            bid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY,
                                      self.meta["objects"][name]["body"])
            jadr = self.model.body_jntadr[bid]
            if jadr >= 0:
                v0 = int(self.model.jnt_dofadr[jadr])
                self.data.qvel[v0:v0 + 6] = 0.0

    def _place_ghost(self, pos, quat) -> None:
        if self.ghost_mocap < 0:
            return
        self.data.mocap_pos[self.ghost_mocap] = (
            pos if pos is not None else GHOST_PARK)
        self.data.mocap_quat[self.ghost_mocap] = (
            quat if quat is not None else [1.0, 0.0, 0.0, 0.0])

    def close(self) -> None:
        self.sock.close()
        self.ctx.term()


class EpisodeControlClient:
    """Agent-side reset sender."""

    def __init__(self, host: str = "localhost", port: int = EPISODE_PORT):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.LINGER, 1000)
        self.sock.connect(f"tcp://{host}:{port}")
        time.sleep(0.3)          # PUSH drops until the connection is up

    def reset(self, objects: dict | None = None, robot: dict | None = None,
              label: str = "", appearance: int | None = None) -> None:
        """`appearance` re-rolls materials, lighting and the sky for this
        episode. One integer, so the manifest can record it and a replay can
        reproduce the exact frame."""
        self.sock.send_string(json.dumps(
            {"cmd": "reset", "objects": objects or {}, "robot": robot or {},
             "label": label, "appearance": appearance}))

    def pin_arms(self, left=None, right=None, waist=None) -> None:
        """Pin (or with None release) the sim-side arm position loops."""
        self.sock.send_string(json.dumps({
            "cmd": "pin_arms",
            "waist": list(map(float, waist)) if waist is not None else None,
            "left": list(map(float, left)) if left is not None else None,
            "right": list(map(float, right)) if right is not None else None,
        }))

    def pin_object(self, name: str, pos=None, quat=None) -> None:
        """Hold `name` at (pos, quat) every sim step; no args releases it."""
        pose = None
        if pos is not None:
            q = [1.0, 0.0, 0.0, 0.0] if quat is None else list(map(float, quat))
            pose = list(map(float, pos)) + q
        self.sock.send_string(json.dumps({
            "cmd": "pin_object", "name": name, "pose": pose}))

    def set_ghost(self, pos=None, quat=None) -> None:
        """Show the planned grasp pose in the render; no args parks it."""
        self.sock.send_string(json.dumps({
            "cmd": "ghost",
            "pos": list(pos) if pos is not None else None,
            "quat": list(quat) if quat is not None else None}))

    def start_video(self, name: str) -> None:
        """Begin a recording. Call once the robot is standing, not at reset."""
        self.sock.send_string(json.dumps({"cmd": "startvideo", "video": name}))

    def end_video(self) -> None:
        """Finalise the recording in progress, so the file is playable."""
        self.sock.send_string(json.dumps({"cmd": "endvideo"}))
        time.sleep(0.5)          # let the sim close the writer before we exit

    def close(self) -> None:
        self.sock.close()
        self.ctx.term()


class PrivilegedSubscriber:
    """Agent-side reader. CONFLATE, so `latest()` never returns stale state."""

    def __init__(self, host: str = "localhost", port: int = PRIVILEGED_PORT):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.connect(f"tcp://{host}:{port}")
        self._last = None

    def latest(self, timeout_ms: int = 200) -> dict | None:
        if self.sock.poll(timeout_ms):
            self._last = json.loads(self.sock.recv_string())

        return self._last

    def wait(self, timeout_s: float = 10.0) -> dict:
        """Block until the sim is actually publishing, then return one state."""
        end = time.time() + timeout_s
        while time.time() < end:
            s = self.latest(200)
            if s is not None:
                return s
        raise TimeoutError(
            f"no privileged state on :{PRIVILEGED_PORT} after {timeout_s:.0f}s "
            "— is run_sim_locomani.py running?"
        )

    def close(self) -> None:
        self.sock.close()
        self.ctx.term()


class PoseSidecar:
    """Per-round world-pose recorder, for offline re-rendering.

    The raw exporter session records NO world-frame robot position (32 columns,
    checked 2026-08-26: root_orientation is orientation only, object poses are
    robot-frame) -- so nothing collected before this class existed can be
    re-rendered by a mirror renderer (UE/3DGS): placing the scene needs the
    robot's pose relative to the furniture. The data was always computed --
    this same module publishes pelvis pos/quat every frame on :5581 -- it was
    just never persisted. This subscribes on its own socket (PUB/SUB, no
    interference with the gym's poll) and appends one jsonl row per message:

        {"t": <sim s>, "wall": <unix s>, "pelvis": [x,y,z,qw,qx,qy,qz],
         "objects": {key: [x,y,z,qw,qx,qy,qz]}}

    Both clocks on purpose: the exporter's parquet timestamps live on a
    different clock, and alignment is an offline problem (docs/ROADMAP.md).
    Fail-safe by design -- every entry point swallows its own errors, because
    a broken sidecar must cost a warning, never an episode.
    """

    def __init__(self, out_dir, host: str = "localhost", port: int | None = None):
        import os as _os, threading as _th
        self.dir = out_dir
        self._lock = _th.Lock()
        self._fh = None
        self._alive = True
        try:
            _os.makedirs(out_dir, exist_ok=True)
            ctx = zmq.Context.instance()
            self.sock = ctx.socket(zmq.SUB)
            self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
            self.sock.RCVTIMEO = 500
            self.sock.connect(f"tcp://{host}:{port or PRIVILEGED_PORT}")
            t = _th.Thread(target=self._loop, daemon=True)
            t.start()
        except Exception as e:                                # noqa: BLE001
            print(f"[sidecar] disabled: {e}")
            self._alive = False

    def _loop(self):
        while self._alive:
            try:
                msg = json.loads(self.sock.recv_string())
            except Exception:
                continue
            with self._lock:
                if self._fh is None:
                    continue
                try:
                    pel = msg["pelvis"]
                    if msg.get("seq") is None and not getattr(
                            self, "_warned_no_seq", False):
                        self._warned_no_seq = True
                        print("[sidecar] privileged messages carry no seq "
                              "(old publisher?) -- pose<->parquet alignment "
                              "will not be possible for this session")
                    row = {"seq": msg.get("seq"),
                           "t": msg.get("t"), "wall": msg.get("wall"),
                           "pelvis": list(pel["pos"]) + list(pel["quat"]),
                           "objects": {k: list(o["pos"]) + list(o["quat"])
                                       for k, o in msg.get("objects", {}).items()}}
                    self._fh.write(json.dumps(row) + "\n")
                except Exception:
                    pass

    def start(self, round_i: int, attempt: int):
        import os as _os
        with self._lock:
            try:
                if self._fh:
                    self._fh.close()
                # PID in the name: chunked collection restarts the gym per
                # chunk and each restart numbers rounds from 0 again, so a
                # shared directory overwrote earlier chunks' files (round 4's
                # DAgger kept 22 of 53). The pid disambiguates the chunk.
                self._fh = open(_os.path.join(
                    self.dir,
                    f"p{_os.getpid()}_round_{round_i:03d}_attempt_{attempt}.jsonl"),
                    "w")
            except Exception as e:                            # noqa: BLE001
                print(f"[sidecar] start failed: {e}")
                self._fh = None

    def stop(self):
        with self._lock:
            try:
                if self._fh:
                    self._fh.close()
            except Exception:
                pass
            self._fh = None
