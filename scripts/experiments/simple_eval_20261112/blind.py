"""SIMPLE's six tasks with no simulator state in the robot's hands (2026-11-12).

What the robot has: the task's instruction, the head camera's colour and depth images, its joint readings, the IMU
(tilt and turn rate) and its own model. From these:

  the target       SAM 3 on the colour image with the object the instruction names; an instruction that names none
                   ("grasp the object on the tabletop") takes the largest thing standing on the table
  its grasp        GraspGen-X (Dex3) on the target's points, in the robot's frame at the start (grasps_graspgen.py)
  the table        the level of the points around the target's foot, and where that level begins ahead of the robot
  the basket       SAM 3 with BOWL (how an upstream VLM would name what the container looks like); for the task whose
                   second table is out of view the robot turns to look for it: behind first, then to either side
  pelvis and palm  leg odometry and joint readings (place.Belief); ScaleBFM is told the pelvis from odometry too
  finger contact   a finger's joints stop following the closing command (pick.close_by_feel)

The simulator's state is read in three places, none of which the robot acts on: placing the robot at the episode's
start, the tie between the simulator's frame and the robot's (place.Belief: the Motor takes goals in the simulator's
frame, so a goal is handed over as its offset from the pelvis), and the success check and the record (fields under
`row`, after the comment "record").

    look   BFM_ROOT=true MUJOCO_GL=egl PYTHONPATH=src python blind.py look <task>    -> <out>/cloud_blind/<name>.npz
    grasps (GraspGen-X venv) python grasps_graspgen.py <out>/cloud_blind --assets-dir ...
    run    BFM_ROOT=true MUJOCO_GL=egl PYTHONPATH=src python blind.py run <task> --out rows.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import episodes as EP  # noqa: E402
import pick as P  # noqa: E402
import place as PL  # noqa: E402
import see as SEE  # noqa: E402
from official import EpisodeOver, OffSim  # noqa: E402, I001

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.backends.scalebfm.motor import Motor, MotorConfig  # noqa: E402

OFFICIAL: dict = {}  # --official: SIMPLE's own environment is the physics and the eyes (official.py)

SIDE = PL.SIDE
CLOUD = "cloud_blind"
BOWL = ("gray bowl", "basket", "bowl", "container")
PLACE = ("LocomotionPickBetweenTablesTeleop", "HandoverTeleop")
SEARCH = (np.pi, 0.5 * np.pi, -0.5 * np.pi)  # rad from the start heading: where the robot turns to look for the basket
OWN = 0.2  # m ahead of the pelvis: nearer points are the robot's own body
HELD = 0.6  # m ahead of the pelvis: with an object carried in front, nearer points are the hand and the object


def named(instruction: str):
    """The object an instruction names, or None ("the object")."""
    s = instruction.lower().strip()
    for pat in (r"pick up the ([a-z ]+?)(?: from|,|\.|$)", r"hand over (?:the )?([a-z ]+?) from"):
        m = re.search(pat, s)
        if m:
            return m.group(1).strip()
    return None


def target_mask(instruction: str, rgb, pts, ok, ahead, after: float):
    name = named(instruction)
    if name:
        m, score, _ = SEE.find(rgb, [name])
        return m, score
    return SEE.largest_thing(pts, ok, ahead, after), 1.0


def basket_from(seen: np.ndarray, ahead):
    """The basket's centre (xy), rim height and near side (along `ahead`) from its seen points."""
    lo, hi = np.percentile(seen, [1, 99], axis=0)
    return 0.5 * (lo + hi)[:2], float(hi[2]), float(np.percentile(seen[:, :2] @ ahead, 1))


def setup(pol, ep):
    info = EP.build(ep)
    cfg = ScaleBfmBackendConfig(scene_dir=info["dir"], object_name="")
    if not OFFICIAL:
        sim = R.Sim(pol.meta, backend=cfg)
    else:
        sim = OffSim(pol.meta, cfg, OFFICIAL["link"], OFFICIAL["depth"], film=bool(OFFICIAL["film"]))
        st = sim.begin(OFFICIAL["task"], ep["level"], ep["index"])
        print("official start", st["qpos"][:3].round(3), "layout start", np.round(info["start"]["xy"], 3), "limit", st["limit"],
              "torque limits differ by", float(np.abs(sim.limits["theirs"] - sim.limits["ours"]).max()), flush=True)  # fmt: skip
    mo = Motor(sim, pol, MotorConfig(), model_xml=info["dir"] + "/scene_locomani.xml")
    bel = PL.Belief(sim, mo, True)
    mo.reset(info["start"]["xy"], info["start"]["yaw"])
    EP.pull_objects(sim.m, sim.d, info)
    if OFFICIAL:  # the robot stands where SIMPLE put it; the hold is SIMPLE's stabilising phase, outside the step limit
        sim.pull()
        mo._stand()
        mo.stand_pose = mo.ref(0.0)
    mo.hold(1.0)
    if OFFICIAL:
        sim.counted = True
    bel.start()
    return info, sim, mo, bel


def first_look(ep, bel):
    """The start's look: the target's points and the table under it, all in the robot's frame."""
    rgb, pts, ok = bel.scene()
    pb, yaw = bel.pelvis()
    ahead = np.array([np.cos(yaw), np.sin(yaw)])
    mask, score = target_mask(ep["instruction"], rgb, pts, ok, ahead, pb[:2] @ ahead + OWN)
    out = dict(rgb=rgb, pts=pts, ok=ok, pb=pb, yaw=yaw, ahead=ahead, mask=mask, score=score)
    if mask is None or (mask & ok).sum() < 100:
        return dict(out, mask=None)
    keep = mask & ok
    eyes = getattr(bel.sim, "last_eyes", None)
    if eyes is not None:  # stereo depth: off the mask's rim, where two surfaces mix
        from scipy.ndimage import binary_erosion

        keep = binary_erosion(mask, iterations=1) & ok
    cloud = pts[keep]
    cloud = cloud[np.linalg.norm(cloud - np.median(cloud, axis=0), axis=1) < 0.3]
    return dict(out, cloud=cloud, top=SEE.support_of(pts, ok, mask))


def look(a, pol) -> None:
    (EP.OUT / CLOUD).mkdir(parents=True, exist_ok=True)
    eps = EP.episodes(a.task)
    for ep in [eps[i] for i in a.only] if a.only else eps[: a.n]:
        info, sim, mo, bel = setup(pol, ep)
        name = f"{a.task}_L{ep['level']}_e{ep['index']}"
        s = first_look(ep, bel)
        b = sim.m.body(info["target"]).id
        ctrue, dtrue, true = SEE.head_points(sim, b)  # record: how much of the target the mask covers
        if hasattr(sim, "last_eyes") and s["mask"] is not None:  # record: the stereo depth against the model's
            ps_, Rs_ = bel._pelvis_sim()
            if __import__("os").environ.get("LOOK_DUMP"):
                np.savez_compressed(__import__("os").environ["LOOK_DUMP"], dtrue=dtrue, true=true, mask=s["mask"], pts=s["pts"], ok=s["ok"],
                                    cam_p=sim.d.cam_xpos[sim.m.camera(EP.HEAD_CAM).id], cam_R=sim.d.cam_xmat[sim.m.camera(EP.HEAD_CAM).id].reshape(3, 3),
                                    fovy=sim.m.cam_fovy[sim.m.camera(EP.HEAD_CAM).id], **sim.last_eyes)  # fmt: skip
            rp_, rR_ = bel.kin(sim.d.qpos[sim.qadr])
            cid_ = sim.m.camera(EP.HEAD_CAM).id
            dR_ = (Rs_ @ rR_).T @ sim.d.cam_xmat[cid_].reshape(3, 3)
            print("  camera: belief minus model cm", (100 * (ps_ + Rs_ @ rp_ - sim.d.cam_xpos[cid_])).round(2), "angle deg",
                  round(float(np.degrees(np.arccos(np.clip((np.trace(dR_) - 1) / 2, -1, 1)))), 2),
                  "pelvis belief", ps_.round(3), "true", sim.d.qpos[:3].round(3),
                  "tilt deg belief", round(float(np.degrees(np.arccos(Rs_[2, 2]))), 2), "true", round(float(np.degrees(np.arccos(R.mat(sim.d.qpos[3:7])[2, 2]))), 2), flush=True)  # fmt: skip
            both, e = s["mask"] & true, sim.last_eyes
            sg_ok = both & (e["sg"] > 0)
            wtrue = bel.from_sim(
                sim.d.cam_xpos[sim.m.camera(EP.HEAD_CAM).id]
                + ctrue @ sim.d.cam_xmat[sim.m.camera(EP.HEAD_CAM).id].reshape(3, 3).T
            )
            print("  depth cm: fused", round(100 * float(np.median((e["depth"] - dtrue)[both])), 2), "matched",
                  round(100 * float(np.median((e["sg"] - dtrue)[sg_ok])), 2), "filled", round(100 * float(np.median((e["lb"] - dtrue)[both])), 2),
                  "| true cloud lo", np.percentile(wtrue, 1, axis=0).round(3), "hi", np.percentile(wtrue, 99, axis=0).round(3),
                  "| seen lo", np.percentile(s["cloud"], 1, axis=0).round(3), "hi", np.percentile(s["cloud"], 99, axis=0).round(3), flush=True)  # fmt: skip
        if s["mask"] is None:
            print(name, "no target seen", flush=True)
            continue
        iou = float((s["mask"] & true).sum() / max(1, (s["mask"] | true).sum()))
        np.savez_compressed(EP.OUT / CLOUD / f"{name}.npz", pts=s["cloud"], p_o=np.median(s["cloud"], axis=0),
                            R_o=np.eye(3), rgb=s["rgb"], mask=s["mask"], iou=iou, score=s["score"], true_mask=true)  # fmt: skip
        print(name, "px", len(s["cloud"]), "score", round(s["score"], 2), "iou", round(iou, 3), flush=True)


def run(a, pol) -> None:
    rows, cur = [], {}
    eps = EP.episodes(a.task)

    def one(ep):
        info, sim, mo, bel = setup(pol, ep)
        cur.update(sim=sim, mo=mo)
        name = f"{a.task}_L{ep['level']}_e{ep['index']}"
        row = dict(level=ep["level"], index=ep["index"])
        b = sim.m.body(info["target"]).id  # record
        z0 = float(sim.d.xpos[b][2])  # record
        q0 = sim.d.xquat[b].copy()  # record

        def fail(why):
            rows.append(dict(row, success=False, why=why, fell=mo.fell_at, t=float(mo.t)))
            print(rows[-1], flush=True)

        if "container" in info["objects"]:  # record: what first touches the basket, and when
            bk_, tick_, hit = sim.m.body(info["objects"]["container"]["body"]).id, mo.tick, []

            def tick():
                out = tick_()
                for i in range(sim.d.ncon if not hit else 0):
                    g1, g2 = sim.m.geom_bodyid[sim.d.contact[i].geom1], sim.m.geom_bodyid[sim.d.contact[i].geom2]
                    other = g2 if g1 == bk_ else g1 if g2 == bk_ else None
                    if other is not None and sim.m.body(other).name not in ("table", "table2", "world"):
                        hit.append((round(float(mo.t), 2), sim.m.body(other).name))
                        row["basket_hit"] = hit[0]
                        break
                return out

            mo.tick = tick
        s = first_look(ep, bel)
        if s["mask"] is None or not (EP.OUT / CLOUD / f"grasp_{name}.npz").exists():
            fail("no target seen")
            return
        pb, yaw, ahead, pts, ok = s["pb"], s["yaw"], s["ahead"], s["pts"], s["ok"]
        left = np.array([-ahead[1], ahead[0]])
        T_wo = np.eye(4)
        T_wo[:3, 3] = np.load(EP.OUT / CLOUD / f"{name}.npz")["p_o"]  # the frame the grasps were worked out in
        placing = a.task in PLACE
        found = P.graspgen_grasp(name, T_wo, yaw, cloud=CLOUD,
                                 **(dict(clear=PL.HELD_ABOVE, roll_range=PL.FROM_SIDE) if placing else {}))  # fmt: skip
        if found is None:
            fail("no grasp")
            return
        T_og, T_op, f_close, f_pre, row["score"] = found
        Tg, Tp = T_wo @ T_og, T_wo @ T_op
        stance = Tg[:2, 3] - PL.rot_z(yaw)[:2, :2] @ [a.stand, a.side_off]
        edge = SEE.near_edge(pts, ok, s["top"], ahead, stance @ left, pb[:2] @ ahead + OWN)
        if edge is not None:
            edge -= float(__import__("os").environ.get("EDGE_BIAS", 0))
        row["edge"] = edge

        def keep_off(xy, limit):  # no nearer to the table than the gap
            over = xy @ ahead - (limit - a.table_gap)
            return xy - max(0.0, over) * ahead

        if edge is not None:
            stance = keep_off(stance, edge)
        row["walk_cm"], _ = bel.walk(stance, yaw, face="fixed")
        to_sim = lambda M: np.block([[bel.to_sim_R(M[:3, :3]), bel.to_sim(M[:3, 3])[:, None]], [np.eye(4)[3:]]])  # noqa: E731
        feel = [] if a.close == "feel" else None
        row.update(
            P.pick(
                mo, sim, SIDE, to_sim(Tg), to_sim(Tp), f_close, f_pre, feel=feel, body=None if a.close == "feel" else b
            )
        )
        row["tilt_deg"] = float(
            np.degrees(np.arccos(np.clip(sim.d.xmat[b].reshape(3, 3)[2, 2] / max(1e-9, abs(R.mat(q0)[2, 2])), -1, 1)))
        )  # record
        row["stopped"] = sorted({f for _, f, r in (feel or []) if r is not None and r < 0.3})
        row["rise_cm"] = float(100 * (sim.d.xpos[b][2] - z0))  # record
        if not placing:
            row.update(success=bool(EP.lifted(a.task, z0, float(sim.d.xpos[b][2]))), touch=P.touching(sim, b),
                       fell=mo.fell_at, t=float(mo.t))  # fmt: skip
            rows.append(row)
            print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            return

        # the object's foot in the palm's frame at the grasp, taken as fixed to the palm from then on
        lo, hi = np.percentile(s["cloud"], [1, 99], axis=0)  # the head sees the front and the top: their span
        foot = Tg[:3, :3].T @ (np.r_[0.5 * (lo + hi)[:2], s["top"]] - Tg[:3, 3])  # it stands on the table
        pb, yb = bel.pelvis()
        palm, Rm = bel.palm(SIDE)
        seen, turn_dump = None, None
        if a.task == "HandoverTeleop":  # the basket stands on the same table: it was in the start's look
            yaw2 = yaw
            m, row["basket_score"], _ = SEE.find(s["rgb"], BOWL)
            if m is not None and (m & ok).sum() >= 200:
                seen = pts[m & ok]
        else:  # the other table is out of view: carry the object and turn to look for the basket
            bel.reach(SIDE, np.r_[pb[:2] + PL.rot_z(yb)[:2, :2] @ PL.CARRY, palm[2]], Rm, speed=0.15)
            for k, turn in enumerate(SEARCH):
                yaw2 = yaw + turn
                bel.walk(bel.pelvis()[0][:2], yaw2, face="fixed")
                rgb2, pts2, ok2 = bel.scene()
                turn_dump = rgb2
                m, row["basket_score"], _ = SEE.find(rgb2, BOWL)
                if m is not None and (m & ok2).sum() >= 200:
                    seen, pts, ok, row["turns"] = pts2[m & ok2], pts2, ok2, k + 1
                    break
        if a.dump and turn_dump is not None:
            from PIL import Image

            Image.fromarray(turn_dump).save(f"{a.dump}/{name}_far.png")
        if seen is None:
            fail("no basket seen")
            return
        turn = yaw2 - yaw
        ahead2 = np.array([np.cos(yaw2), np.sin(yaw2)])
        centre, rim, near = basket_from(seen, ahead2)
        bk = sim.m.body(info["objects"]["container"]["body"]).id  # record
        row["far_cm"] = float(100 * np.linalg.norm(centre - bel.from_sim(sim.d.xpos[bk])[:2]))  # record
        Rr = PL.rot_z(turn) @ Tg[:3, :3]  # the same hand attitude seen from the robot
        over = np.r_[centre, rim + PL.OVER_RIM + 0.06] - Rr @ foot  # the palm with the object upright over the basket
        release = np.r_[centre, rim - PL.SEAT] - Rr @ foot  # and with its foot at the rim
        stance2 = release[:2] - PL.rot_z(yaw2)[:2, :2] @ [a.stand, PL.PLACE_SIDE]
        if turn:
            pb, _ = bel.pelvis()
            top2 = float(np.percentile(seen[:, 2], 2))
            edge2 = SEE.near_edge(
                pts, ok, top2, ahead2, stance2 @ np.array([-ahead2[1], ahead2[0]]), pb[:2] @ ahead2 + HELD
            )
        else:
            edge2 = edge
        limit = min(near, edge2) if edge2 is not None else near
        gap0 = stance2 @ ahead2
        stance2 = stance2 - max(0.0, gap0 - (limit - a.table_gap)) * ahead2
        if not turn:
            pb, yb = bel.pelvis()
            bel.reach(SIDE, np.r_[pb[:2] + PL.rot_z(yb)[:2, :2] @ PL.CARRY, max(palm[2], over[2])], Rm, speed=0.15)
        row["dbg"] = dict(
            stance=stance.round(3).tolist(),
            stance2=stance2.round(3).tolist(),
            edge=edge,
            edge2=edge2,
            near=near,
            centre=centre.round(3).tolist(),
            rim=rim,
            release=release.round(3).tolist(),
            over=over.round(3).tolist(),
            foot=foot.round(3).tolist(),
            true_c=bel.from_sim(sim.d.xpos[bk]).round(3).tolist(),
        )  # record
        short = PL.LOOK_FROM if turn else 0.0  # after a long walk: stop short, look again, walk the rest
        row["walk2_cm"], row["walk2_deg"] = bel.walk(stance2 - short * ahead2, yaw2, face="travel" if turn else "fixed",
                                                     speed=a.speed if turn else None)  # fmt: skip
        row["carried"] = bool(P.touching(sim, b))  # record
        row["at_near"] = dict(
            goal=(stance2 - short * ahead2).round(3).tolist(),
            yaw2=round(float(yaw2), 3),
            belief=[np.round(v, 3).tolist() for v in bel.pelvis()],
            true=bel.pt.round(3).tolist() + [round(float(bel.yt), 3)],
            tilt=round(float(np.degrees(np.arccos(R.mat(sim.d.qpos[3:7])[2, 2]))), 1),
        )  # record
        pb, yb = bel.pelvis()  # the object held aside, a look at the basket from near
        bel.reach(
            SIDE, np.r_[pb[:2] + PL.rot_z(yb)[:2, :2] @ PL.ASIDE, max(bel.palm(SIDE)[0][2], over[2])], Rr, speed=0.2
        )
        rgb3, pts3, ok3 = bel.scene()
        m, _, _ = SEE.find(rgb3, BOWL)
        row["basket_px"] = int((m & ok3).sum()) if m is not None else 0
        if a.dump:
            from PIL import Image

            im = rgb3.copy()
            if m is not None:
                im[m] = (0.5 * im[m] + [0, 127, 0]).astype(np.uint8)
            Image.fromarray(im).save(f"{a.dump}/{name}_near.png")
        if row["basket_px"] >= 200:
            c3, rim3, _ = basket_from(pts3[m & ok3], ahead2)
            shift = np.r_[c3 - centre, rim3 - rim]
            row["look_cm"] = [float(v) for v in 100 * shift]
            over, release, stance2, centre, rim = over + shift, release + shift, stance2 + shift[:2], c3, rim3
        if short:
            row["walk3_cm"], _ = bel.walk(stance2, yaw2, face="fixed")
        bel.reach(SIDE, over, Rr, speed=0.15)
        bel.reach(SIDE, release, Rr, speed=0.05)
        row["over_cm"] = [float(v) for v in 100 * (sim.d.xpos[b][:2] - sim.d.xpos[bk][:2])]  # record
        for u in np.linspace(0, 1, 15):
            sim.fingers_cmd[SIDE] = f_close + u * (PL.WIDE - f_close)
            mo.hold(0.02)
        mo.hold(1.0)  # the finger joints follow their targets with a time constant of 0.4 s
        p, Rm = bel.palm(SIDE)
        bel.reach(SIDE, p - 0.04 * Rm[:, 1], Rm, speed=0.05)  # off the object against the palm's normal,
        p, Rm = bel.palm(SIDE)
        bel.reach(SIDE, p - 0.08 * Rm[:, 1] + [0, 0, 0.08], Rm, speed=0.1)  # then away and up, clear of the rim
        run_, best, basket = 0, 0, sim.m.body(bk).name  # record: SIMPLE's reward, the object on the basket in a row
        row["t_done"] = None
        for _ in range(75):
            mo.hold(0.02)
            on = PL.contact(sim, b, lambda n: n == basket)
            free = not PL.contact(sim, b, lambda n: "hand" in n)
            run_ = run_ + 1 if on and (free or not PL.HANDS_OFF[a.task]) else 0
            best = max(best, run_)
            if run_ == PL.HELD[a.task] and row["t_done"] is None:
                row["t_done"] = float(mo.t)
        row.update(success=bool(best >= PL.HELD[a.task]), fell=mo.fell_at, t=float(mo.t),
                   basket_cm=float(100 * np.linalg.norm(sim.d.xpos[bk][:2] - info["objects"]["container"]["pos"][:2])))  # fmt: skip
        rows.append(row)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    for ep in [eps[i] for i in a.only] if a.only else eps[: a.n]:
        cur.clear()
        try:
            one(ep)
            if OFFICIAL:  # the script is through: SIMPLE's check keeps running until its verdict or the step limit
                for _ in range(100):
                    cur["mo"].hold(0.02)
        except EpisodeOver:
            pass
        if OFFICIAL:
            sim, key = cur["sim"], dict(level=ep["level"], index=ep["index"])
            if not rows or any(rows[-1].get(k) != v for k, v in key.items()):
                rows.append(dict(key, why="episode over mid-way"))
            over = sim.over or dict(success=False, steps=sim.steps, truncated=False)
            rows[-1].update(
                success=bool(over["success"]), steps=over["steps"], truncated=over["truncated"], official=True
            )
            if OFFICIAL["film"]:
                sim.video(f"{OFFICIAL['film']}/{a.task}_L{ep['level']}_e{ep['index']}.mp4")
            print(
                "official",
                {k: rows[-1].get(k) for k in ("level", "index", "success", "steps", "truncated", "why")},
                flush=True,
            )
    print("success", sum(r["success"] for r in rows), "of", len(rows))
    if a.out:
        json.dump(rows, open(a.out, "w"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("look", "run"))
    ap.add_argument("task")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--stand", type=float, default=0.27)
    ap.add_argument("--side-off", type=float, default=P.SIDE_OFF)
    ap.add_argument("--table-gap", type=float, default=0.2)
    ap.add_argument("--speed", type=float, default=0.5, help="m/s, the long walk between the tables")
    ap.add_argument(
        "--close", default="feel", choices=("feel", "contact"), help="contact: the simulator's contacts, a diagnostic"
    )
    ap.add_argument("--dump", help="a folder for the near look's image")
    ap.add_argument("--only", type=int, nargs="*", help="positions in the task's list of episodes")
    ap.add_argument("--out")
    ap.add_argument(
        "--official", help="socket of official_server.py: SIMPLE's environment is the physics and the camera"
    )
    ap.add_argument("--depth", help="socket of depth_server.py")
    ap.add_argument("--film", help="a folder for the head camera's video of each episode (Isaac Sim)")
    a = ap.parse_args()
    if a.official:
        from multiprocessing.connection import Client

        global CLOUD
        CLOUD = "cloud_isaac"
        OFFICIAL.update(
            link=Client(a.official, family="AF_UNIX"), depth=Client(a.depth, family="AF_UNIX"), film=a.film, task=a.task
        )
    R.OFF[:] = P.PALM
    pol = R.Policy("cuda", ScaleBfmBackendConfig())
    (look if a.stage == "look" else run)(a, pol)


if __name__ == "__main__":
    main()
