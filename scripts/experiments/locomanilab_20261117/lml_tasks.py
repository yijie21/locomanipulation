"""Step 3 (2026-11-18): four ways of placing the hand on LocoManiLab's standing close tasks, ScaleBFM in the benchmark
host, privileged targets from the task's own adapter (humanoid_benchmark rl/kimodo/heft/adapters.goals_for).

Tasks: the close tasks whose adapter waypoints all lie inside the self model's command distribution with the robot
standing where the scene parks it (scan 2026-11-18): IThorKitchenCloseDrawer_scene13, IThorKitchenCloseCabinet_scene13,
IThorBedroomCloseTopDrawer_scene329, IThorKitchenCloseOven_scene13 (wrist pitch -6.6 deg asked, clamped to 0). Episode i
uses the benchmark's reset jitter (robot 0.05 m, `apply_initial_pose_jitter`, rng seed i) and the task's own success
predicate (`_check_success`: the joint under the close fraction, held HOLD_STEPS).

An episode is the adapter's waypoints (touch, then press(es)) done as consecutive reaches exactly as in the self model's
data (lml_selfmodel protocol 2 / lml_host): the stance held START_HOLD_S, then per waypoint PRE_S, the reach planned from
the measured state at its command row, POST_S. The waypoint becomes a command in the data's parametrization: the hand
point (13 cm ahead of wrist_yaw_link) at the waypoint's grasp point, the wrist x axis along its approach direction
(dyaw clamped to +-60 deg, pitch to 0..90 deg), standing (no pelvis height asked). Methods differ only in the command:

  alone       the waypoint itself
  lookup      minus the median landing offset of training reaches with a similar command (same hand, standing, target,
              wrist turn and pitch nearby; the box doubles until 20 match) -- a table, no state
  feedback    the waypoint, plus an integral servo during the reach and its hold (Motor's: ki 2/s, cap 8 cm per axis,
              every 0.1 s) that moves the reference hand target by the measured miss; the hand is measured as a real
              robot would: joint encoders + the IMU's orientation, the pelvis located by leg odometry from the feet
              planted at the command (no simulator positions)
  selfmodel   the self model's position fixed point (wrist_invert.fixed_point: t <- t + goal - predicted, 6 steps, the
              candidate with the least predicted miss + spread); every candidate is planned from the measured state and
              its features are built by the same code that records the data (lml_selfmodel.segments2 + data.derive)

    python lml_tasks.py table --out DIR                   # lookup table + pickled self models from the host data
    python lml_tasks.py run --out DIR --task T --method M --episodes 0:50 --worker K
    python lml_tasks.py summary --out DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lml_host as H  # noqa: E402
import lml_selfmodel as L  # noqa: E402

REACHWM = HERE.parent / "reachwm_20261030"
SELFWM = HERE.parent / "selfwm_20261029"
HOSTDATA = Path(os.environ.get("LML_HOSTDATA", "/wuji-vefps-D/wuji-il/yijie/loco_outputs/lml_hostdata_20261118"))
TASKS = {  # benchmark/tasks.yaml: scene, step budget, genuine_min_steps
    "IThorKitchenCloseDrawer": dict(scene=13, max_steps=389, genuine=50),
    "IThorKitchenCloseCabinet": dict(scene=13, max_steps=300, genuine=38),
    "IThorBedroomCloseTopDrawer": dict(scene=329, max_steps=318, genuine=44),
    "IThorKitchenCloseOven": dict(scene=13, max_steps=643, genuine=62),
}
METHODS = ("alone", "lookup", "feedback", "selfmodel", "selfmodel_s1")
ITERS = 6  # pose_invert.ITERS_POSE
LAMBDA = 1.0  # invert_eval.LAMBDA: predicted spread's weight
SERVO_KI, SERVO_CAP, SERVO_EVERY = 2.0, 0.08, 5  # scalebfm MotorConfig servo_ki / servo_cap; every 5 ticks = 0.1 s
JITTER = 0.05  # tasks.yaml reset_jitter


def reachwm():
    """The self model's data / model code (REACHWM_* flags as the models were trained with)."""
    for k, v in (("REACHWM_GOAL_REF", "1"), ("REACHWM_APPROACH", "1"), ("REACHWM_QFIX", "1")):
        os.environ.setdefault(k, v)
    sys.path[:0] = [str(REACHWM), str(SELFWM)]
    import data as DT
    import models as M

    return DT, M


# -- the lookup table and the pickled models -------------------------------------------------------------------------
def table(out: Path) -> None:
    import torch

    DT, M = reachwm()
    import train as TR

    D = DT.load(str(HOSTDATA / "data_scalebfm" / "all.npz"))
    tr = DT.splits(D)["train"]
    tr = tr[~D["fell"][tr].astype(bool) & (D["pelvis_z"][tr] < 0)]  # standing reaches, as the task commands
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "lookup.npz", side=np.sign(D["side"][tr]).astype(np.int8), pitch=D["pitch"][tr].astype(np.float32),
                        dyaw=D["dyaw"][tr].astype(np.float32), target=D["target"][tr].astype(np.float32),
                        off=(D["pos_f"][tr] - D["target"][tr]).astype(np.float32))  # fmt: skip
    for s in (0, 1):
        m = TR.load_model("E3_ensemble", 1.0, D, "cpu", HOSTDATA / f"models_scalebfm_s{s}")
        m.qfix = DT.QFIX
        torch.save(m, out / f"wm_s{s}.pt")
    print(f"lookup.npz: {len(tr)} standing training reaches; wm_s0.pt, wm_s1.pt from {HOSTDATA}", flush=True)


def lookup(T: dict, side: int, goal: np.ndarray, dyaw: float, pitch: float) -> tuple[np.ndarray, int]:
    base = T["side"] == side
    t = T["target"]
    for k in (1.0, 2.0, 4.0, 8.0):
        m = base & (np.abs(t[:, 2] - goal[2]) < 0.05 * k) & (np.abs(t[:, 0] - goal[0]) < 0.10 * k)
        m &= (np.abs(t[:, 1] - goal[1]) < 0.10 * k) & (np.abs(T["dyaw"] - dyaw) < np.radians(15) * k)
        m &= np.abs(T["pitch"] - pitch) < np.radians(15) * k
        if m.sum() >= 20:
            break
    return np.median(T["off"][m], axis=0).astype(np.float64), int(m.sum())


# -- the host with a task scene --------------------------------------------------------------------------------------
def _private_scene_cache() -> None:
    """robocasa rewrites <scene>.robocasa.xml on every build; parallel workers on one scene then read half-written files.
    Each process writes its own copy next to the original (relative mesh paths unchanged), removed once loaded."""
    sys.path[:0] = [str(H.HB), str(L.G1B)]
    import robocasa.models.arenas.molmospaces_arena as A

    orig = A.preprocess_scene_xml

    def private(src_xml, dst_xml=None):
        if dst_xml is None:
            base = os.path.splitext(os.path.basename(src_xml))[0]
            dst_xml = os.path.join(os.path.dirname(src_xml), f"{base}.robocasa.pid{os.getpid()}.xml")
            _private_scene_cache.written.append(dst_xml)
        return orig(src_xml, dst_xml)

    A.preprocess_scene_xml = private


_private_scene_cache.written = []


class TaskDriver(H.HostDriver):
    def __init__(self, task: str, seed: int = 0):
        _private_scene_cache()
        self.task = task
        super().__init__(env_name=task, seed=seed, scene_ids=TASKS[task]["scene"], fixed_scene_seed=0)
        for f in _private_scene_cache.written:
            if os.path.exists(f):
                os.remove(f)
        import mujoco

        m = self.m
        self.feet = [mujoco.mj_name2id(self.pl.ik.model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")]
        self.wrist_fk = {s: mujoco.mj_name2id(self.pl.ik.model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_link") for s in ("left", "right")}
        assert m is not None

    def reset_episode(self, ep: int) -> None:
        from robocasa.utils.scene_initial_jitter import apply_initial_pose_jitter

        self.env.reset()
        ok = apply_initial_pose_jitter(self.env, robot_radius=JITTER, object_radius=0.0, rng=np.random.default_rng(ep),
                                       include_robot=True, include_objects=False)  # fmt: skip
        self.mj.mj_forward(self.m, self.d)
        self.comp = self.env.robots[0].composite_controller
        self.buf = None
        self.jitter_ok = bool(ok)

    def waypoints(self) -> list[dict]:
        from rl.kimodo.heft import adapters as AD

        g = AD.goals_for(self.env)
        out = []
        for w in g.waypoints:
            gp = np.asarray(w.grasp_point, float)
            ad = np.asarray(w.approach_dir if w.approach_dir is not None else AD.approach_from_root(gp, self.d.xpos[self.pelvis]), float)
            out.append(dict(goal=gp, ad=ad / np.linalg.norm(ad), side="left" if w.eef == "LeftHand" else "right", label=w.label,
                            needs_root=bool(w.needs_root)))  # fmt: skip
        self.notes = g.notes
        return out

    def joint_fraction(self) -> float:
        return float(self.env.target_semantic_joint.fraction(self.env.sim))

    # -- leg odometry (the feedback method's sensor) -------------------------------------------------------------------
    def body_rel(self, q_mj: np.ndarray) -> dict:
        """Feet (mean of the two ankle_roll_links) and both hand points relative to the pelvis, pelvis frame, by FK of
        joint angles on robot_fk.xml -- what encoders give."""
        mj, m, d = self.pl.mj, self.pl.ik.model, self.pl.d
        d.qpos[:] = 0
        d.qpos[self.pl.ik.base_qadr + 3] = 1.0
        d.qpos[self.pl.ik.qadr] = q_mj
        mj.mj_kinematics(m, d)
        out = dict(feet=np.mean([d.xpos[b] for b in self.feet], axis=0).copy())
        for s, b in self.wrist_fk.items():
            out[s] = d.xpos[b] + d.xmat[b].reshape(3, 3) @ L.HAND_OFF
        return out


# -- the command -----------------------------------------------------------------------------------------------------
def command(x: dict, wp: dict) -> tuple[dict, np.ndarray]:
    """The waypoint as a data-distribution command at the measured state x; and the goal in the command's frame (heading
    frame of the measured pelvis, xy relative, z world -- the frame of the data's `target` and `pos_f`)."""
    p, R = x["root"][:3].astype(float), L._qmat(x["root"][3:])
    yaw = L.yaw_of(R)
    Rh = L.rot_z(yaw)
    gh = np.r_[(Rh.T @ np.r_[wp["goal"][:2] - p[:2], 0.0])[:2], wp["goal"][2]]
    ad = wp["ad"]
    dyaw = (math.atan2(ad[1], ad[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
    pitch = math.asin(float(np.clip(-ad[2], -1.0, 1.0)))
    c = dict(side=wp["side"], target=gh.copy(), dyaw=float(np.clip(dyaw, -math.radians(60), math.radians(60))),
             pitch=float(np.clip(pitch, 0.0, math.pi / 2)), pelvis_z=None)  # fmt: skip
    return c, gh


def clip_t(t: np.ndarray, side: str) -> np.ndarray:
    """wrist_invert.clip_t: the target kept inside the data's WIDE box (mirrored for the left hand)."""
    lo, hi = L.WIDE
    sd = 1.0 if side == "right" else -1.0
    y = np.clip(t[1] * sd, lo[1], hi[1]) * sd
    return np.r_[np.clip(t[0], lo[0], hi[0]), y, np.clip(t[2], lo[2], hi[2])]


class Predictor:
    """The self model on one candidate command: plan it from the measured state (as the drive will), build its features
    with the recording code, predict the landing."""

    def __init__(self, path: Path):
        import torch

        self.DT, self.M = reachwm()
        torch.set_num_threads(1)
        self.model = torch.load(path, map_location="cpu", weights_only=False).eval()
        self.DT.check_qfix(self.model)

    def features(self, dr: TaskDriver, rows: dict, f: int, frames: np.ndarray, c: dict, k: int, reset: int):
        """(plan, D) for command c at row f: the reach record and a one-segment dataset whose x0 / u are exactly what
        recording this reach would give (future rows are placeholders: they only feed the labels)."""
        blend, last, rec = dr.plan(rows[f], frames[f], c)
        rec.update(chain=min(k, 1), f_cmd=f, f_end=f + len(blend))
        cand = np.concatenate([frames[: f + 1], blend, L.hold(last, L.POST_S + 0.2)])
        f_last = rec["f_end"] + int(round(L.POST_S * L.FPS))
        fake = {i: rows[i] for i in range(f - int(round(L.PRE_S * L.FPS)), f + 1)}
        fake.update({i: rows[f] for i in range(f + 1, f_last + 1)})
        segs = L.segments2(fake, None, None, cand, [rec], reset, dr.pl)
        assert len(segs) == 1, "features: the placeholder segment was not built"
        D = L.pack(segs)
        D["walk"] = np.zeros((1, 3), np.float32)
        D["target_b"] = D["target"]
        self.DT.derive(D)
        return (blend, last, rec), D

    def predict(self, D: dict) -> tuple[np.ndarray, np.ndarray]:
        import torch

        GD = self.M.tensors(D, "cpu", step=False)
        with torch.no_grad():
            o = self.model.predict(self.M.sub(GD, torch.tensor([0]), False))
        return o["pos"][0].numpy().astype(np.float64), o["pos_std"][0].numpy().astype(np.float64)


# -- one episode ----------------------------------------------------------------------------------------------------
def episode(dr: TaskDriver, ep: int, method: str, T: dict | None, P: Predictor | None) -> dict:
    task = TASKS[dr.task]
    dr.reset_episode(ep)
    wps = dr.waypoints()
    frames = dr.start_frames()
    rows, wrec = {}, []
    f, k, next_cmd, end_frame = 0, 0, int(round(L.START_HOLD_S * L.FPS)), None
    fell, lost, why, success_at, fracs = None, None, "", None, []
    anchor, servo = None, None
    reset_id = 7000000 + ep
    while True:
        dr.push(frames[min(f, len(frames) - 1)])
        if dr.buf is None:
            dr.buf = dr._find_buffer()
        dr.step()
        x = rows[f] = dr.sample()
        fracs.append(dr.joint_fraction())
        if success_at is None and bool(dr.env._check_success()):
            success_at = f
        if float(x["root"][2]) < L.FALL_Z:
            fell, why = f, "height"
            break
        if anchor is not None:
            dev = (x["root"][:2] - anchor[0]) - (frames[min(f, len(frames) - 1), :2] - anchor[1])
            if float(np.linalg.norm(dev)) > L.LOST_M:
                lost, why = f, "lost"
                break
        if servo is not None and f <= servo["f_stop"] and (f - servo["f_cmd"]) % SERVO_EVERY == 0 and f > servo["f_cmd"]:
            frames = servo_step(dr, servo, x, frames, f)
        if k < len(wps) and f == next_cmd:
            wp = wps[k]
            c, gh = command(x, wp)
            tic = time.time()
            info = dict(k=k, label=wp["label"], side=wp["side"], goal_w=wp["goal"].tolist(), ad=wp["ad"].tolist(), goal_h=gh.tolist(),
                        dyaw_deg=math.degrees(c["dyaw"]), pitch_deg=math.degrees(c["pitch"]))  # fmt: skip
            if method == "lookup":
                off, n_nb = lookup(T, 1 if wp["side"] == "right" else -1, gh, c["dyaw"], c["pitch"])
                c["target"] = clip_t(gh - off, wp["side"])
                info.update(lookup_off_cm=(100 * off).tolist(), lookup_n=n_nb)
            if method.startswith("selfmodel"):
                t, evs = gh.copy(), []
                for it in range(ITERS + 1):
                    cc = dict(c, target=t.copy())
                    try:
                        plan, D = P.features(dr, rows, f, frames, cc, k, reset_id)
                    except (np.linalg.LinAlgError, ValueError):
                        evs.append(dict(t=t.copy(), l=np.inf))
                        break
                    pos, std = P.predict(D)
                    evs.append(dict(t=t.copy(), pos=pos, std=std, l=float(((pos - gh) ** 2).sum() + LAMBDA * (std**2).sum()),
                                    plan=plan, x0=D["x0"][0].copy()))  # fmt: skip
                    t = clip_t(t + (gh - pos), wp["side"])
                best = min(evs, key=lambda e: e["l"])
                if "plan" in best:
                    c["target"] = best["t"]
                    info.update(pred0_h=evs[0]["pos"].tolist(), pred_h=best["pos"].tolist(), pred_std_cm=(100 * best["std"]).tolist(),
                                pred_miss0_cm=100 * float(np.linalg.norm(evs[0]["pos"] - gh)),
                                pred_miss_cm=100 * float(np.linalg.norm(best["pos"] - gh)), iters=len(evs))  # fmt: skip
                else:  # no candidate could be planned: the waypoint as asked (counted)
                    info.update(model_fallback=True)
            if method.startswith("selfmodel") and "plan" in best:
                blend, last, rec = best["plan"]
                x0_live = best["x0"]
            else:
                blend, last, rec = dr.plan(x, frames[f], c)
                rec.update(chain=min(k, 1))
                x0_live = None
            info["think_s"] = time.time() - tic
            rec.update(chain=min(k, 1), f_cmd=f, f_end=f + len(blend))
            tail = L.POST_S + (L.PRE_S if k < len(wps) - 1 else 0.2) + L.MARGIN_S
            frames = np.concatenate([frames[: f + 1], blend, L.hold(last, tail)])
            anchor = (x["root"][:2].copy(), frames[f, :2].copy())
            post = int(round(L.POST_S * L.FPS))
            info.update(cmd_h=np.asarray(c["target"]).tolist(), f_cmd=f, f_end=rec["f_end"], f_win=rec["f_end"] + post,
                        x0_live=None if x0_live is None else x0_live.tolist(), _rec=rec)  # fmt: skip
            wrec.append(info)
            if method == "feedback":
                servo = servo_start(dr, x, wp, c, rec, last, f, rec["f_end"] + post)
                info["servo"] = servo["leads"]  # filled as the servo runs
            next_cmd = rec["f_end"] + post + int(round(L.PRE_S * L.FPS))
            end_frame = rec["f_end"] + post
            k += 1
        if k == len(wps) and end_frame is not None and f >= end_frame:
            break
        if f >= len(frames) - 1:
            why = "motion ended"
            break
        f += 1
    # the features the self model was given must be the ones recording this reach gives (selfmodel methods)
    if P is not None:
        for w in wrec:
            if w.get("x0_live") is None:
                continue
            segs = L.segments2(rows, None, None, frames, [w["_rec"]], reset_id, dr.pl)
            if segs:
                D = L.pack(segs)
                D["walk"] = np.zeros((1, 3), np.float32)
                D["target_b"] = D["target"]
                P.DT.derive(D)
                w["x0_check"] = float(np.abs(D["x0"][0] - np.asarray(w["x0_live"])).max())
    # per waypoint: where the hand point was at the end of its window
    for w in wrec:
        fw = min(w["f_win"], max(rows))
        pw, Rw = rows[fw]["wr"][w["side"]]
        hand = pw + Rw @ L.HAND_OFF
        e = hand - np.asarray(w["goal_w"])
        ad = np.asarray(w["ad"])
        along = float(e @ ad)
        w.update(hand_w=hand.tolist(), err_cm=100 * float(np.linalg.norm(e)), along_cm=100 * along,
                 perp_cm=100 * float(np.linalg.norm(e - along * ad)), complete=fw == w["f_win"])  # fmt: skip
        w.pop("_rec", None)
        w.pop("x0_live", None)
    steps = len(rows)
    return dict(task=dr.task, method=method, ep=ep, jitter_ok=dr.jitter_ok, notes=dr.notes, n_wp=len(wps), done_wp=len(wrec),
                success=success_at is not None, success_step=success_at,
                success_in_budget=success_at is not None and success_at + 1 <= task["max_steps"],
                genuine=success_at is not None and success_at + 1 >= task["genuine"],
                frac_final=fracs[-1], frac_min=min(fracs), frac_start=fracs[0], fell=fell, lost=lost, why=why, steps=steps,
                waypoints=wrec, _rows=rows, _frames=frames)  # fmt: skip


def servo_start(dr: TaskDriver, x: dict, wp: dict, c: dict, rec: dict, last, f: int, f_stop: int) -> dict:
    """The command frame: world axes, origin at the pelvis at the command. The goal is known there (the sighting); the
    feet are anchored there from encoders + IMU at the command."""
    R = L._qmat(x["root"][3:])
    rel = dr.body_rel(x["q"][dr.to_mj])
    # the plan lives in reference coordinates (lml_selfmodel.OnlineDriver._plan shifts it by `delta` onto the
    # reference root); the IK re-solve must be asked in those
    return dict(f_cmd=f, f_stop=f_stop, f_end=rec["f_end"], side=wp["side"], goal_c=wp["goal"] - x["root"][:3],
                feet_c=R @ rel["feet"], lead=np.zeros(3), goal_ref=wp["goal"] + np.asarray(rec["delta"]),
                rot_w=np.asarray(rec["rot_w"]), last=last, n=0, leads=[])  # fmt: skip


def servo_step(dr: TaskDriver, sv: dict, x: dict, frames: np.ndarray, f: int) -> np.ndarray:
    """One servo update: hand measured by leg odometry (feet planted since the command), the integral moves the hand
    target, the IK re-solves the final pose from the planned one, and the rest of the reference is shifted toward it."""
    R = L._qmat(x["root"][3:])  # IMU orientation
    rel = dr.body_rel(x["q"][dr.to_mj])
    pel_c = sv["feet_c"] - R @ rel["feet"]
    hand_c = pel_c + R @ rel[sv["side"]]
    e = sv["goal_c"] - hand_c
    sv["lead"] = np.clip(sv["lead"] + SERVO_KI * SERVO_EVERY * L.DT * e, -SERVO_CAP, SERVO_CAP)
    sv["n"] += 1
    sv["leads"].append((f, (100 * e).round(2).tolist(), (100 * sv["lead"]).round(2).tolist()))
    try:
        poses, _, _, _ = dr.pl.reach(sv["last"], sv["side"], sv["goal_ref"] + sv["lead"], sv["rot_w"], None)
    except (np.linalg.LinAlgError, ValueError):
        return frames
    new = poses[-1]
    old = sv["last"]
    dp, dq = new[0] - old[0], new[2] - old[2]
    dquat = (new[1] if float(np.dot(new[1], old[1])) >= 0 else -new[1]) - old[1]
    span = max(sv["f_end"] - f, 10)
    out = frames.copy()
    for j in range(f + 1, len(out)):
        w = L.minjerk(min(1.0, (j - f) / span))
        out[j, :3] += w * dp
        q = out[j, 3:7] + w * dquat
        out[j, 3:7] = q / np.linalg.norm(q)
        out[j, 7:] += w * dq
    sv["last"] = (old[0] + dp, new[1], old[2] + dq)
    return out


# -- runner ---------------------------------------------------------------------------------------------------------
def run(out: Path, task: str, method: str, eps: list[int], worker: int) -> None:
    od = out / "episodes" / task / method
    od.mkdir(parents=True, exist_ok=True)
    todo = [e for e in eps if not (od / f"ep{e:03d}.json").exists()]
    print(f"tasks {task} {method} worker {worker}: {len(eps) - len(todo)}/{len(eps)} episodes done", flush=True)
    if not todo:
        return
    T = dict(np.load(out / "lookup.npz")) if method == "lookup" else None
    P = Predictor(out / ("wm_s1.pt" if method == "selfmodel_s1" else "wm_s0.pt")) if method.startswith("selfmodel") else None
    dr = TaskDriver(task, seed=worker)
    for e in todo:
        t0 = time.time()
        r = episode(dr, e, method, T, P)
        r.pop("_rows"), r.pop("_frames")
        r["wall_s"] = round(time.time() - t0, 1)
        tmp = od / f".ep{e:03d}.json"
        tmp.write_text(json.dumps(r, default=_json))
        os.replace(tmp, od / f"ep{e:03d}.json")
        w0 = r["waypoints"][0] if r["waypoints"] else {}
        print(f"tasks {task} {method} ep {e}: success {r['success']} (step {r['success_step']}), frac {r['frac_start']:.2f}->{r['frac_final']:.2f}, "
              f"touch err {w0.get('err_cm', float('nan')):.1f} cm (perp {w0.get('perp_cm', float('nan')):.1f}), {r['why'] or 'ok'}, {r['wall_s']} s", flush=True)  # fmt: skip


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def summary(out: Path) -> None:
    rs = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(out / "episodes" / "*" / "*" / "ep*.json")))]
    print(f"{len(rs)} episodes")
    for task in TASKS:
        for method in METHODS:
            g = [r for r in rs if r["task"] == task and r["method"] == method]
            if not g:
                continue
            touch = [r["waypoints"][0] for r in g if r["waypoints"]]
            print(f"{task:28s} {method:12s} n {len(g):3d} | success {sum(r['success'] for r in g):3d} "
                  f"in budget {sum(r['success_in_budget'] for r in g):3d} | final frac median {np.median([r['frac_final'] for r in g]):.2f} "
                  f"| touch err median {np.median([w['err_cm'] for w in touch]):.1f} perp {np.median([w['perp_cm'] for w in touch]):.1f} "
                  f"along {np.median([w['along_cm'] for w in touch]):+.1f} cm | fell {sum(r['fell'] is not None for r in g)} lost {sum(r['lost'] is not None for r in g)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["table", "run", "summary"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", default="IThorKitchenCloseDrawer")
    ap.add_argument("--method", default="alone", choices=METHODS)
    ap.add_argument("--episodes", default="0:50")
    ap.add_argument("--worker", type=int, default=0)
    a = ap.parse_args()
    if a.phase == "table":
        table(a.out)
    elif a.phase == "run":
        lo, hi = (int(v) for v in a.episodes.split(":"))
        run(a.out, a.task, a.method, list(range(lo, hi)), a.worker)
    else:
        summary(a.out)
