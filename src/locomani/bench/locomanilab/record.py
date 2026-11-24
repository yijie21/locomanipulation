"""The recorder of the LocoManiLab camera track: every reading of the simulator's truth that goes into an episode's
record but into no decision (TODO 49.4; moved from record_lml.py). The skill package holds none of these reads; it leaves its own
estimates in `env.row` and the bed calls the recorder at the moments the old skill files made the comparisons:

    rec = Recorder(env)
    rec.on_scene()                           from the bed's Belief.scene (the camera error of every scene call)
    rec.on_mark(name)                        from the bed's Env.mark
    rec.after_approach() / after_pick()      the package's flows call `env.flow_done("approach" | "pick")` at their ends
                                             (also on an exception); the facade forwards it here

The fields it writes are the ones of the hand-tuned track's rows: cam_err, nav.{cam_errs, end_dist_cm, true_centre,
centre_err_cm}, foot_err_cm, dest.{true_centre, true_rim, err_cm, rim_err_cm}, release_cm, open_vec_cm, held_true.
A comparison that fails is written as `record_error` (counted, not hidden); the episode goes on.
The camera's true pose of the last image is `sim.last_pose` (LmlSim.camera_image keeps it for this and for nothing else).
"""

import numpy as np

from locomani.bench.locomanilab import truth as SL

SIDE = "right"


class Recorder:
    """One episode's recorder."""

    def __init__(self, env):
        self.env, self.errs, self.dest_truth = env, [], None

    # ---- the camera: own estimate of the camera's pose against the true one
    def on_scene(self) -> None:
        try:
            self._cam_error()
        except Exception as e:  # noqa: BLE001
            self.env.row.setdefault("record_error", {})["cam"] = str(e)[:80]

    def _cam_error(self) -> None:
        sim, bel, row = self.env.sim, self.env.bel, self.env.row
        rel_p, rel_R = bel.kin(sim.d.qpos[sim.qadr])
        ps, Rs = bel._pelvis_sim()
        cam_p, cam_R = ps + Rs @ rel_p, Rs @ rel_R
        true_p, true_R = sim.last_pose
        dR = cam_R.T @ true_R
        last = dict(pos_cm=(100 * (cam_p - true_p)).round(2).tolist(),
                    err_cm=round(float(100 * np.linalg.norm(cam_p - true_p)), 2),
                    angle_deg=round(float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))), 2))  # fmt: skip
        self.errs.append((last["err_cm"], last["angle_deg"]))
        row.setdefault("cam_err", last)

    # ---- the bed's marks
    def on_mark(self, name: str) -> None:
        try:
            if name == "before_carry":
                self._foot()
                self._dest_truth()
            elif name == "before_open":
                self._release()
                self._dest_compare()
        except Exception as e:  # noqa: BLE001
            self.env.row.setdefault("record_error", {})[name] = f"{type(e).__name__}: {e}"[:80]

    def _foot(self) -> None:
        """The object's foot in the palm's frame as it is now, against the one the skill worked out from its seen cloud."""
        env = self.env
        O = SL.body_points(env.sim, env.bodies)
        pj, Rj = env.mo.jaw_centre(SIDE)
        true_foot = Rj.T @ (np.r_[0.5 * (O.min(0) + O.max(0))[:2], O[:, 2].min()] - pj)
        env.row["foot_err_cm"] = (100 * (np.asarray(env.row["foot"]) - true_foot)).round(1).tolist()

    def _dest_truth(self) -> None:
        """The task's receptacle in the robot's frame (the object is in the hand, as when the skill places)."""
        env = self.env
        rc = SL.receptacle(env)
        self.dest_truth = None if rc is None else (env.bel.from_sim(np.r_[rc[0], 0.0])[:2], float(rc[1]))

    def _dest_compare(self) -> None:
        env = self.env
        d = env.row.get("dest")
        if d is None or self.dest_truth is None:
            return
        tc, tr = self.dest_truth
        d.update(true_centre=np.round(tc, 3).tolist(), true_rim=round(tr, 3),
                 err_cm=round(float(100 * np.linalg.norm(np.asarray(d["centre"]) - tc)), 1),
                 rim_err_cm=round(float(100 * (d["rim"] - tr)), 1))  # fmt: skip

    def _release(self) -> None:
        """The hand against the release pose of the skill's plan, in the simulator's frame."""
        env = self.env
        plan = env.row.get("place")
        if not plan or "release" not in plan:
            return
        hand = env.mo.jaw_centre(SIDE)[0]
        goal = env.bel.to_sim(np.asarray(plan["release"]))
        env.row["release_cm"] = float(100 * np.linalg.norm(hand - goal))
        env.row["open_vec_cm"] = (100 * (hand - goal)).round(1).tolist()

    # ---- after the skill's flows
    def after_approach(self) -> None:
        """The robot's end place against the fixture's points, and the seen centre against the true one."""
        env, row = self.env, self.env.row
        nav = row.get("nav")
        if nav is None:
            return
        nav["cam_errs"] = list(self.errs)
        try:
            sim, bel = env.sim, env.bel
            Vt = SL.body_points(sim, env.bodies)
            nav["end_dist_cm"] = round(float(100 * np.linalg.norm(Vt[:, :2] - sim.d.qpos[:2], axis=1).min()), 1)
            mid = 0.5 * (Vt.min(0) + Vt.max(0))
            nav["true_centre"] = bel.from_sim(np.r_[mid[:2], 0.0])[:2].round(2).tolist()
            c = nav.get("centre")
            nav["centre_err_cm"] = None if c is None else round(float(100 * np.linalg.norm(bel.to_sim(np.r_[c, 0.0])[:2] - mid[:2])), 1)  # fmt: skip
        except Exception as e:  # noqa: BLE001
            nav["record_error"] = str(e)[:80]

    def after_pick(self) -> None:
        """Whether the hand's contacts say the object is held after the pick."""
        self.env.row["held_true"] = bool(self.env.snapshot()["held"])


# ---- articulated tasks (fork: bed2 doors, bed3 slide joints): the per-step record of a door or drawer. Record only, never
# visible to the package. Only relative quantities (fractions, seconds, palm-to-handle offsets, pelvis drop): no room frame.
ART_KEYS = (
    "action",
    "goal",
    "frac_start",
    "frac_max",
    "frac_min",
    "frac_end",
    "t_first_move",
    "t_goal",
    "hold_best_s",
    "touch",
    "t_touch",
    "hand_min_cm",
    "hand_min_side",
    "hand_min_t",
    "palm_dz_cm",
    "palm_dxy_cm",
    "palm_best_dz_cm",
    "grip_on_handle",
    "pelvis_drop_cm",
    "moved_by",
)  # fmt: skip  (bed2's eighteen, then the two of bed3)
HINGE_RAD = 1.5708  # a door's range when the spec names none: fraction = clip(q / 1.5708, 0, 1)
# the body parts a contact is named by; HAND, FOREARM, ... are the entries of this tuple
GROUPS = ("hand", "forearm", "upper_arm", "torso", "pelvis", "thigh", "knee", "shin", "foot")
LEG_PARTS = ("thigh", "knee", "shin", "foot")  # a drawer moved by one of these is not worked by a hand
KNEE_CM = 10.0  # a contact on the knee link this near the knee joint is `knee`, farther down `shin`
GRIP_NEAR_M = 0.04  # a hand closed with its palm this near the handle holds it
CLOSED_RAD = 1.0  # the finger joints' summed distance from the open reading above which a hand counts as closed
MOVE_FRAC = 0.01
MOVING_M = 2e-5  # the joint moved this much (m or rad) in a step: the drawer / door was moving


def _robot_group(name: str) -> int | None:
    """Index into GROUPS for a body of the robot's own model (the knee link is `knee` here, split by contact point later);
    None for any other name."""
    if "_hand_" in name or name.endswith("wrist_yaw_link"):
        return 0
    if "elbow" in name or "wrist_roll" in name or "wrist_pitch" in name:
        return 1
    if "shoulder" in name:
        return 2
    if name == "torso_link" or name.startswith("waist"):
        return 3
    if name == "pelvis":
        return 4
    if "hip" in name:
        return 5
    if "knee" in name:
        return 6
    if "ankle" in name:
        return 8
    return None


class ArtRecorder:
    """One articulated episode's recorder: the joint's fraction, the handle, both palms, the contacts with the moving part
    (which body parts touch it, and which touched it while it moved) and how far the pelvis went down."""

    def __init__(self, env):
        self.env, spec = env, env.spec
        sim, m = env.sim, env.sim.m
        import mujoco

        self.mj = mujoco
        goal = spec["goal"]
        self.action = (
            "open_put_close" if spec["family"] == "articulation_pnp" else "close" if goal["op"] == "<=" else "open"
        )
        self.close, self.thr, self.hold_n = goal["op"] == "<=", float(goal["threshold"]), int(goal["hold_steps"])
        self.pnp = spec["family"] == "articulation_pnp"
        self.stages = spec.get("stages") or []
        self.open_thr = float(self.stages[0]["threshold"]) if self.pnp else None
        self.open_n = int(self.stages[0]["hold_steps"]) if self.pnp else 0
        self.nav = spec["family"] == "navigation_articulation"
        self.adr = int(m.jnt_qposadr[m.joint(spec["joint"]).id])
        self.q_closed = float(spec.get("closed_qpos", 0.0))
        self.q_open = float(spec.get("open_qpos", HINGE_RAD))
        door = m.body(spec["door_body"]).id
        tree = SL.subtree(m, door)
        hb = m.body(spec["handle_body"]).id
        self.hgeoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == hb]
        # a long thin handle (a bar: capsule or cylinder) is a segment; the palm's distance is taken to the segment
        self.bar = next((g for g in self.hgeoms if m.geom_type[g] in (3, 5) and m.geom_size[g][1] > 0.08), None)
        self.apple = None if not self.pnp else m.body(spec["object_body"]).id
        root = m.body("pelvis").id
        grp = np.full(m.nbody, -2, dtype=int)  # -1 the moving part's subtree, 0.. GROUPS' index, -2 anything else
        for b in SL.subtree(m, root):
            g = _robot_group(m.body(b).name)
            grp[b] = -2 if g is None else g
        for b in tree:
            grp[b] = -1
        self.grp = grp
        self.knee_bodies = {b for b in range(m.nbody) if grp[b] == 6}
        self.apple_bodies = SL.subtree(m, self.apple) if self.pnp else set()
        self.site = None
        self.hopen = np.asarray(sim.hand_open, float)
        self.dt = 0.02
        f0 = self.frac()
        self.f0, self.fmax, self.fmin, self.fend = f0, f0, f0, f0
        self.q_prev = float(sim.d.qpos[self.adr])
        self.t_move = self.t_goal = self.t_touch = None
        self.run_n = self.best_n = 0
        self.opened_n = 0  # pnp: the run of steps with the drawer open past stage 1's threshold
        self.stage2 = False  # pnp: stage 1 completed
        self.touch: set = set()
        self.moved_by: set = set()
        self.prev_got: set = set()
        self.hand_min = None  # (cm, side, t, dz_cm, dxy_cm)
        self.best_dz = None
        self.grip = False
        self.z0 = None
        self.drop = 0.0

    def frac(self) -> float:
        q = float(self.env.sim.d.qpos[self.adr])
        return float(np.clip((q - self.q_closed) / (self.q_open - self.q_closed), 0.0, 1.0))

    def handle(self) -> np.ndarray:
        return self.env.sim.d.geom_xpos[self.hgeoms].mean(axis=0)

    def nearest(self, p: np.ndarray) -> np.ndarray:
        """The handle's nearest point to `p`: on the bar's axis when the handle is a bar, else its middle."""
        if self.bar is None:
            return self.handle()
        d = self.env.sim.d
        c, ax, half = d.geom_xpos[self.bar], d.geom_xmat[self.bar].reshape(3, 3)[:, 2], float(self.env.sim.m.geom_size[self.bar][1])  # fmt: skip
        return c + float(np.clip((p - c) @ ax, -half, half)) * ax

    def _group(self, m, d, c, body: int) -> int:
        g = int(self.grp[body])
        if g == 6 and body in self.knee_bodies:  # the knee link: the knee near its joint, the shin below
            g = 6 if float(np.linalg.norm(c.pos - d.xpos[body])) * 100 <= KNEE_CM else 7
        return g

    def step(self, t: float) -> None:
        env, sim = self.env, self.env.sim
        d = sim.d
        f = self.frac()
        q = float(d.qpos[self.adr])
        moving = abs(q - self.q_prev) > MOVING_M
        self.q_prev = q
        self.fend, self.fmax, self.fmin = f, max(self.fmax, f), min(self.fmin, f)
        if self.t_move is None and abs(f - self.f0) > MOVE_FRAC:
            self.t_move = t
        z = float(d.qpos[2])  # the pelvis (the free base's z)
        if self.z0 is None:
            self.z0 = z
        self.drop = max(self.drop, self.z0 - z)
        if (
            self.pnp
        ):  # stage 1: open past the threshold for open_n steps; then stage 2: the apple inside and the drawer closed
            self.opened_n = self.opened_n + 1 if f >= self.open_thr else 0
            self.stage2 = self.stage2 or self.opened_n >= self.open_n
            inside = self.stage2 and f <= self.thr and self._apple_inside()  # (cheap tests first)
        else:
            inside = (f <= self.thr if self.close else f >= self.thr) and (bool(sim.success) if self.nav else True)
        if inside:
            if self.t_goal is None:
                self.t_goal = t
            self.run_n += 1
            self.best_n = max(self.best_n, self.run_n)
        else:
            self.run_n = 0
        h = self.handle()
        for i, side in enumerate(("left", "right")):
            p = env.mo.jaw_centre(side)[0]
            near = self.nearest(p)
            dist = float(np.linalg.norm(p - near))
            dz = float(p[2] - near[2])
            if self.hand_min is None or dist < self.hand_min[0]:
                self.hand_min = (dist, side, t, dz, float(np.linalg.norm((p - near)[:2])))
            self.best_dz = dz if self.best_dz is None else max(self.best_dz, dz)
            qf = d.qpos[sim.gq[7 * i : 7 * i + 7]]
            if dist < GRIP_NEAR_M and float(np.abs(qf - self.hopen[7 * i : 7 * i + 7]).sum()) > CLOSED_RAD:
                self.grip = True
        m, got = sim.m, set()
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            a, b = int(self.grp[b1]), int(self.grp[b2])
            if (a == -1) != (b == -1):
                o, ob = (b, b2) if a == -1 else (a, b1)
                if o >= 0:
                    got.add(self._group(m, d, c, ob))
        if got:
            if self.t_touch is None:
                self.t_touch = t
            self.touch |= got
        if moving:  # the parts in contact at this step or the one before it (an impulse can end within a step)
            self.moved_by |= got | self.prev_got
        self.prev_got = got

    def _apple_inside(self) -> bool:
        """pnp stage 2's own test, on the local model: the apple's bounding-box centre in the drawer's receptacle site box
        (as the benchmark's `_apple_inside_drawer`). Record only."""
        m, d = self.env.sim.m, self.env.sim.d
        if self.site is None:
            door = m.body(self.env.spec["door_body"]).id
            ids = [i for i in range(m.nsite) if int(m.site_bodyid[i]) == door and "eceptacle" in (self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_SITE, i) or "")]  # fmt: skip
            self.site = ids[0] if ids else -1
        if self.site < 0:
            return False
        O = SL.body_points(self.env.sim, self.apple_bodies)
        local = (0.5 * (O.min(0) + O.max(0)) - d.site_xpos[self.site]) @ d.site_xmat[self.site].reshape(3, 3)
        return bool(np.all(np.abs(local) <= m.site_size[self.site]))

    def result(self) -> dict:
        r2 = lambda v: None if v is None else round(float(v), 2)  # noqa: E731
        hm = self.hand_min
        thr = "%.2f" % self.thr
        if self.pnp:
            goal = "fraction >= %.2f held %.1f s, then apple inside and fraction <= %s held %.1f s" % (
                self.open_thr, self.open_n * self.dt, thr, self.hold_n * self.dt)  # fmt: skip
        else:
            goal = "fraction %s %s" % ("<=" if self.close else ">=", thr)
            goal += " and robot within 0.75 m" if self.nav else " held %.1f s" % (self.hold_n * self.dt)
        out = dict(
            action=self.action, goal=goal, frac_start=round(self.f0, 3), frac_max=round(self.fmax, 3),
            frac_min=round(self.fmin, 3), frac_end=round(self.fend, 3), t_first_move=r2(self.t_move),
            t_goal=r2(self.t_goal), hold_best_s=round(self.best_n * self.dt, 2),
            touch=[GROUPS[g] for g in sorted(self.touch)], t_touch=r2(self.t_touch),
            hand_min_cm=None if hm is None else round(100 * hm[0], 1), hand_min_side=None if hm is None else hm[1],
            hand_min_t=None if hm is None else r2(hm[2]), palm_dz_cm=None if hm is None else round(100 * hm[3], 1),
            palm_dxy_cm=None if hm is None else round(100 * hm[4], 1),
            palm_best_dz_cm=None if self.best_dz is None else round(100 * self.best_dz, 1), grip_on_handle=bool(self.grip),
            pelvis_drop_cm=round(100 * max(0.0, self.drop), 1), moved_by=[GROUPS[g] for g in sorted(self.moved_by)],
        )  # fmt: skip
        assert tuple(out) == ART_KEYS
        return out
