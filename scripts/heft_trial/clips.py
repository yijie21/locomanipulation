"""TODO 51 stage A: the same Motor reference clips under ScaleBFM and under HEFT, on SIMPLE's own physics (the bed's environment
server, nothing rendered), no task. Run with the sim venv: PYTHONPATH=src <sim_python> scripts/heft_trial/clips.py --out DIR.

One Isaac server (one slot) serves both trackers; every clip starts from a fresh environment reset, the Motor's start hold, then the clip.
Metrics come from the environment's own state (the local model's qpos is the server's reading)."""

import argparse
import dataclasses
import json
import os
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

import mujoco
import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.scalebfm.heft import HeftPolicy
from locomani.backends.scalebfm.motor import SIDES, MotorConfig
from locomani.bench.simple import body as BODY
from locomani.bench.simple import client as C
from locomani.bench.simple import episodes as EP
from locomani.bench.simple import run as RUN
from locomani.bench.simple.config import SimpleConfig
from locomani.bench.simple.session import Session, take_slot

SEED = Path(RUN.__file__).parent / "seed"
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def rot_z(a):
    return R.rot_z(a)


class Rec:
    """Per-tick record wrapped round Motor.tick: pelvis pose, palm centres, feet, from the environment's state."""

    def __init__(self, sim, mo, film=False):
        self.sim, self.mo, self.rows, self.frames = sim, mo, [], []
        if film:  # the local model's own render (robot only: the scene's objects stay at their start in it), every 2nd tick = 25 fps
            sim.renderer = mujoco.Renderer(sim.m, height=360, width=640)
        self.film = film
        self.raw = mo.tick
        mo.tick = self.tick

    def tick(self):
        ok = self.raw()
        d, m = self.sim.d, self.sim.m
        q = d.qpos[3:7].copy()
        Rm = R.mat(q)
        self.rows.append(
            dict(
                t=self.mo.t,
                p=d.qpos[:3].copy(),
                q=q,
                R=Rm,
                jaw={s: tuple(np.asarray(x).copy() for x in self.mo.jaw_centre(s)) for s in SIDES},
                foot=[d.xpos[m.body(n).id].copy() for n in FEET],
            )  # fmt: skip
        )
        if self.film and len(self.rows) % 2 == 0:
            self.frames.append(self.sim.frame().copy())
        return ok


def tilt_deg(Rm):
    pitch = np.degrees(np.arcsin(np.clip(-Rm[2, 0], -1, 1)))
    roll = np.degrees(np.arctan2(Rm[2, 1], Rm[2, 2]))
    return roll, pitch


def sway(rows):
    """std of pelvis xy (cm) and roll / pitch (deg) over `rows`, and the drift of the pelvis over them."""
    if len(rows) < 3:
        return {}
    p = np.array([r["p"] for r in rows])
    rp = np.array([tilt_deg(r["R"]) for r in rows])
    return dict(
        xy_std_cm=round(float(100 * np.sqrt(p[:, 0].var() + p[:, 1].var())), 3),
        roll_std_deg=round(float(rp[:, 0].std()), 3),
        pitch_std_deg=round(float(rp[:, 1].std()), 3),
        drift_cm=round(float(100 * np.linalg.norm(p[-1, :2] - p[0, :2])), 2),
        z_range_cm=round(float(100 * (p[:, 2].max() - p[:, 2].min())), 2),
    )


def slip(rows, floor=0.015):
    """Foot slip (cm): summed horizontal travel of a foot between two ticks both spent within `floor` of its lowest height."""
    foot = np.array([r["foot"] for r in rows])  # T, 2, 3
    if len(foot) < 2:
        return 0.0
    low = foot[:, :, 2].min(axis=0)
    down = foot[:, :, 2] - low < floor
    both = down[1:] & down[:-1]
    step = np.linalg.norm(np.diff(foot[:, :, :2], axis=0), axis=2)
    return round(float(100 * (step * both).sum()), 1)


def write_video(frames, path):
    """25 fps h264 through ffmpeg, small: the bytes are checked by the caller (cap 500 000)."""
    import subprocess

    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25", "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "34", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]  # fmt: skip
    subprocess.run(cmd, input=np.stack(frames).tobytes(), check=True)


class Bed:
    def __init__(self, cfg, slot, link):
        self.cfg, self.slot, self.link = cfg, slot, link
        self.pkg = RUN.load_package(SEED)
        self.K = RUN.resolved(self.pkg.skill, "OpenOvenTeleop", {"feed_torque": True})
        self.pols = {}

    def policy(self, name):
        if name not in self.pols:
            R.OFF[:] = BODY.PALM
            self.pols[name] = HeftPolicy() if name == "heft" else R.Policy("cuda", ScaleBfmBackendConfig())
        return self.pols[name]

    def episode(self, tracker, episode_id):
        """A fresh reset of the episode with the tracker's policy and its own gains; -> (sim, mo)."""
        ep = EP.load(episode_id, self.cfg)
        info = EP.build(ep, Path(self.cfg.run_dir) / "scene_a", self.cfg.simple_root, self.cfg.visual_dir)
        pol = self.policy(tracker)
        with EP.no_model_cache():
            sim = C.OffCfgSim(pol.meta, ScaleBfmBackendConfig(scene_dir=info["dir"], object_name=""), self.link, None)
            sim.begin_cfg("OpenOvenTeleop", C.state_cfg(ep["cfg"]))
            self.link.send(("lazy_render", True))
            self.link.recv()
            mo = C.BedMotor(sim, pol, dataclasses.replace(MotorConfig(), **{k: self.K[k] for k in self.pkg.skill.MOTOR_KEYS}),
                            model_xml=info["dir"] + "/scene_locomani.xml")  # fmt: skip
        mo.reset(info["start"]["xy"], info["start"]["yaw"])
        EP.pull_objects(sim.m, sim.d, info)
        sim.pull()
        mo._stand()
        mo.stand_pose = mo.ref(0.0)
        mo.hold(self.K["start_hold"])
        return sim, mo, info


def to_world(p0, yaw, rel):
    return p0 + rot_z(yaw) @ np.asarray(rel, float)


def clip(bed, tracker, name, episode_id, steps, film=None):
    t0 = time.time()
    sim, mo, info = bed.episode(tracker, episode_id)
    rec = Rec(sim, mo, film=film is not None)
    p0, q0 = mo.pelvis()
    yaw0 = R.yaw_of(q0)
    out = dict(
        tracker=tracker, clip=name, episode=episode_id, start=dict(xy=np.round(p0[:2], 3).tolist(), yaw=round(yaw0, 3))
    )
    try:
        res = steps(mo, rec, p0, yaw0, out)
    except Exception as e:  # a fall in the middle of a primitive ends the clip, it is the measurement
        res = dict(error=repr(e))
    out.update(res if isinstance(res, dict) else {})
    out["fell"] = mo.fell_at is not None or sim.d.qpos[2] < 0.4
    out["fell_at"] = None if mo.fell_at is None else round(mo.fell_at, 2)
    out["seconds"] = round(mo.t, 2)
    out["wall_s"] = round(time.time() - t0, 1)
    out["foot_slip_cm"] = slip(rec.rows)
    out["_rows"] = rec.rows
    if film is not None and rec.frames:
        write_video(rec.frames, film)
        out["video"] = str(film)
    return out


# ---- the clips ------------------------------------------------------------------------------------------------------
def c_stand(mo, rec, p0, yaw0, out):
    n0 = len(rec.rows)
    mo.hold(10.0)
    return dict(sway=sway(rec.rows[n0:]))


def c_walk(mo, rec, p0, yaw0, out):
    """1 m along the travel line and 0.3 m to its side, away from the scene (the oven stands 1.1 m ahead of the start): a goal
    (-1.0, 0.3) of the start heading, so the walk turns toward it first, as a skill's walk does."""
    goal = to_world(p0, yaw0, (-1.0, 0.3, 0))[:2]
    o = mo.walk_to(goal, None, face="travel")
    mo.hold(1.0)
    p, q = mo.pelvis()
    travel = float(np.arctan2(goal[1] - p0[1], goal[0] - p0[0]))
    return dict(
        walk=dict(ok=o.ok, why=o.why, goal_dist_cm=round(float(100 * np.linalg.norm(goal - p0[:2])), 1)),
        end_err_cm=round(float(100 * np.linalg.norm(p[:2] - goal)), 1),
        end_yaw_err_deg=round(float(np.degrees(R.wrap(R.yaw_of(q) - travel))), 1),
        travelled_cm=round(float(100 * np.linalg.norm(p[:2] - p0[:2])), 1),
    )


def c_turn(mo, rec, p0, yaw0, out):
    o = mo.turn(yaw0 + np.pi / 2)
    mo.hold(1.0)
    p, q = mo.pelvis()
    return dict(turn=dict(ok=o.ok, why=o.why), yaw_err_deg=round(float(np.degrees(R.wrap(R.yaw_of(q) - yaw0 - np.pi / 2))), 2),
                end_drift_cm=round(float(100 * np.linalg.norm(p[:2] - p0[:2])), 1))  # fmt: skip


REACHES = {
    "chest": ("right", (0.42, -0.22, 0.27), None),
    "table": ("right", (0.45, -0.20, -0.06), None),
    "low": ("right", (0.32, -0.22, -0.46), 0.56),
    "edge": ("right", (0.62, -0.20, 0.12), None),
    "both": ("both", ((0.42, 0.20, 0.27), (0.42, -0.20, 0.27)), None),
}


def c_reach(which):
    side, rel, z = REACHES[which]

    def run(mo, rec, p0, yaw0, out):
        pz = p0[2]
        n0 = len(rec.rows)
        if side == "both":
            tgt = [to_world(p0, yaw0, r) + [0, 0, 0] for r in rel]
            o = mo.reach("both", tgt, None, servo=False)
        else:
            tgt = to_world(p0, yaw0, rel)
            o = mo.reach(side, tgt, None, servo=False, pelvis_z=z)
        n1 = len(rec.rows)
        mo.hold(1.5)
        r = rec.rows[-1]
        errs = {}
        sides = SIDES if side == "both" else (side,)
        for i, s in enumerate(sides):
            t = np.asarray(tgt[i] if side == "both" else tgt, float)
            pj = r["jaw"][s][0]
            world = pj - t
            # pelvis frame: the palm in the robot's own pelvis frame against the target in the REFERENCE pelvis's frame, so a
            # drift of the whole body does not count as an arm error
            rp = mo.ref(mo.t)
            Rr = R.mat(rp.quat)
            pel = r["R"].T @ (pj - r["p"]) - Rr.T @ (t - rp.pos)
            errs[s] = dict(world_cm=round(float(100 * np.linalg.norm(world)), 2), pelvis_cm=round(float(100 * np.linalg.norm(pel)), 2),
                           target=np.round(t, 3).tolist())  # fmt: skip
        return dict(reach=dict(ok=o.ok, why=o.why), palm=errs, sway_hold=sway(rec.rows[n1:]), sway_move=sway(rec.rows[n0:n1]),
                    pelvis_z=round(float(r["p"][2]), 3))  # fmt: skip

    return run


CLIPS = {"stand": c_stand, "walk": c_walk, "turn": c_turn, **{f"reach_{k}": c_reach(k) for k in REACHES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episode", default="OpenOvenTeleop/Q/0")
    ap.add_argument("--clips", default=",".join(CLIPS))
    ap.add_argument("--trackers", default="scalebfm,heft")
    ap.add_argument("--slots", default="3")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--video", default="", help="clips to film (local model render), e.g. walk,reach_both")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dataclasses.replace(
        SimpleConfig(), run_dir=str(out / "run"), slots=tuple(int(s) for s in a.slots.split(",")), lazy_render=True
    )
    ids = out / "ids.txt"
    ids.write_text(a.episode + "\n")
    ses = Session(cfg, "OpenOvenTeleop", ids, out / "rows.jsonl", stereo=False, look=False)
    ses.slot, lock = take_slot(cfg, 600)
    ses._locks.append(lock)
    print("slot", ses.slot, flush=True)
    results = []
    try:
        if not ses.start_server():
            print("server did not start", file=sys.stderr)
            return 2
        link = Client(str(ses.sock("isaac")), family="AF_UNIX")
        bed = Bed(cfg, ses.slot, link)
        for rep in range(a.repeat):
            for cl in a.clips.split(","):
                for tr in a.trackers.split(","):
                    film = out / f"{cl}_{tr}.mp4" if cl in a.video.split(",") else None
                    r = clip(bed, tr, cl, a.episode, CLIPS[cl], film)
                    rows = r.pop("_rows")
                    np.savez_compressed(out / f"trace_{tr}_{cl}_{rep}.npz", t=[x["t"] for x in rows], p=[x["p"] for x in rows],
                                        q=[x["q"] for x in rows], foot=[x["foot"] for x in rows],
                                        jawl=[x["jaw"]["left"][0] for x in rows], jawr=[x["jaw"]["right"][0] for x in rows])  # fmt: skip
                    r["rep"] = rep
                    results.append(r)
                    print(json.dumps(r, default=float)[:600], flush=True)
                    with open(out / "clips.jsonl", "a") as f:
                        f.write(json.dumps(r, default=float) + "\n")
    finally:
        ses.stop_all()
        for f in ses._locks:
            f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
