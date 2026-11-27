"""selfmodel_faildiag_20261126 question 3: HEFT's in-place reaches in LML with a per-step log (measurement only).

  d3   data3's own in-place host (EmptyEnv) and resets, rerun in order from the start of a shard (the host is
       stateful, so a reset is reproduced only after its shard's earlier ones): reset_plan / run_reset / segments of
       data3's frozen copy (--code), host seed SEED + shard, ids shard, shard + 30, ...; each reset's segments are
       compared array for array with data3's chunk (w<shard>_<chunk>.npz) -> reproduced or not, in the record.
  ha   one HumanoidArena reach command (selfmodel_hacollect's HEFT video episodes: side, target in the heading frame,
       dyaw / pitch / roll, pelvis height, duration) given to LML's in-place host from a fresh stand: after data3's
       settle rule, the command placed by the measured pelvis (as HA does), planned by plan_reach_b (path_reach, the
       planner HA's collection uses), the speed set so that the reference lasts HA's reach_s; then 9 s of hold.

Per step (after the step): pelvis pose (h.pelvis_pose(): the host pelvis body + R c) and root velocity, both wrists, both ankle_roll_link positions, each foot's
normal force on the world, the 29 joints (SCALEBFM order), the frame pushed (our frame; its 29 joints in MuJoCo order) and the buffer's transform (yaw offset, source
and destination xy), optionally the full qpos (videos).

    cd $G; PYTHONPATH=$G:$SRC python fd_heft.py d3 --code <d3 code dir> --shard 0 --n 60 --out <dir>
    cd $G; PYTHONPATH=$G:$SRC python fd_heft.py ha --code <d3 code dir> --ha-npz <npz> --chains 0,2 --out <dir>
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


def load_d3(code: Path):
    sys.path[:0] = [str(code / "selfmodel_data3_20261125"), str(code / "locomanilab_20261117"), str(code / "hardgrasp_20261125")]
    import d3_collect as D3

    return D3


class HLog:
    def __init__(self, h, qpos: bool):
        mj, m = h.mj, h.m
        self.h, self.mj, self.m, self.qpos = h, mj, m, qpos
        bn = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
        self.robot = np.array([n.startswith("robot0_") for n in bn])
        self.foot = [next(i for i, n in enumerate(bn) if n.endswith(f"{s}_ankle_roll_link")) for s in ("left", "right")]
        ft = np.full(m.nbody, -1)
        for k, b in enumerate(self.foot):
            for i in range(m.nbody):
                j = i
                while j > 0:
                    if j == b:
                        ft[i] = k
                        break
                    j = m.body_parentid[j]
        self.ft = ft
        self.f6 = np.zeros(6)
        self.rows, self.qp, self.fw = [], [], None

    def clear(self):
        self.rows, self.qp, self.fw = [], [], None

    def push(self, fw):
        self.fw = np.asarray(fw, np.float64).copy()

    def record(self):
        h, mj, m, d = self.h, self.mj, self.m, self.h.d
        ff = np.zeros(2)
        nc = int(d.ncon)
        if nc:
            b1 = m.geom_bodyid[np.asarray(d.contact.geom1[:nc])]
            b2 = m.geom_bodyid[np.asarray(d.contact.geom2[:nc])]
            for i in range(nc):
                a, b = int(b1[i]), int(b2[i])
                ka, kb = self.ft[a], self.ft[b]
                if (ka >= 0) == (kb >= 0):
                    continue
                k, o = (ka, b) if ka >= 0 else (kb, a)
                if self.robot[o]:
                    continue
                mj.mj_contactForce(m, d, i, self.f6)
                ff[k] += abs(float(self.f6[0]))
        try:
            yo, src, dst = h.transform()
            tr = np.r_[float(yo), np.asarray(src, float)[:2], np.asarray(dst, float)[:2]]
        except Exception:  # noqa: BLE001 -- the buffer is found on the first step
            tr = np.full(5, np.nan)
        pp, _ = h.pelvis_pose()  # the pelvis in robot_fk's convention (the host body + R c), not the host body's origin
        row = np.r_[pp, d.xquat[h.pelvis], d.qvel[h.rv : h.rv + 6], d.xpos[h.wrist["left"]], d.xpos[h.wrist["right"]],
                    d.xpos[self.foot[0]], d.xpos[self.foot[1]], ff, d.qpos[h.qadr], self.fw if self.fw is not None else np.full(36, np.nan), tr,
                    d.xquat[h.wrist["left"]], d.xquat[h.wrist["right"]]]  # fmt: skip
        self.rows.append(row.astype(np.float64))
        if self.qpos:
            self.qp.append(d.qpos.astype(np.float32).copy())

    @staticmethod
    def cols(L) -> list[str]:
        c = [f"pel_{a}" for a in "xyz"] + [f"pel_q{a}" for a in "wxyz"] + [f"pel_v{i}" for i in range(6)]
        c += [f"wl_{a}" for a in "xyz"] + [f"wr_{a}" for a in "xyz"] + [f"fl_{a}" for a in "xyz"] + [f"fr_{a}" for a in "xyz"]
        c += ["force_l", "force_r"] + [f"q_{n}" for n in L.SCALEBFM_ORDER] + [f"ref_{i}" for i in range(7)]
        c += [f"refq{i}" for i in range(29)] + ["tr_yaw", "tr_src_x", "tr_src_y", "tr_dst_x", "tr_dst_y"]
        c += [f"wl_q{a}" for a in "wxyz"] + [f"wr_q{a}" for a in "wxyz"]
        return c


def attach(h, log: HLog) -> None:
    push0, step0 = h.push, h.step

    def push(fw):
        log.push(fw)
        push0(fw)

    def step():
        step0()
        log.record()

    h.push, h.step = push, step


def _j(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


def save(out: Path, name: str, log: HLog, L, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    arr = dict(rows=np.array(log.rows), cols=np.array(HLog.cols(L)), meta=np.array(json.dumps(meta, default=_j)))
    if log.qpos:
        arr["qpos"] = np.array(log.qp)
    tmp = out / f".{name}.tmp.npz"
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, out / f"{name}.npz")


def compare(D3, segs: list[dict], orig: Path, reset: int) -> dict:
    """data3's chunk vs ours for one reset: every array of the main file, per segment rows and per reference rows; the
    one-hand segments against data/<chunk>, the two-hand ones against data_both/<chunk> (data3's split)."""
    if not segs:
        return dict(n=0)
    out = dict(n=len(segs), same=True, diff=[])
    for both, f in ((False, orig), (True, orig.parent.parent / "data_both" / orig.name)):
        part = [s for s in segs if (s["hands"] == 2) == both]
        if not part:
            continue
        z = np.load(f)
        mine, _ = D3.pack(part)
        idx = np.flatnonzero(z["reset"] == reset)
        if len(idx) != len(part):
            out.update(same=False, why=f"segment count ({'both' if both else 'one'} hand: {len(part)} vs {len(idx)})")
            continue
        rs = np.r_[0, np.cumsum(z["n_steps"])]
        rr = np.r_[0, np.cumsum(z["n_ref"])] if "n_ref" in z.files else None
        for k in mine:
            if k not in z.files:
                continue
            a = mine[k]
            if k in D3.ROW_KEYS:
                b = np.concatenate([z[k][rs[i] : rs[i + 1]] for i in idx])
            elif k in D3.REF_KEYS and rr is not None:
                b = np.concatenate([z[k][rr[i] : rr[i + 1]] for i in idx])
            else:
                b = z[k][idx]
            if a.shape != b.shape or not np.array_equal(a, b):
                out["same"] = False
                out["diff"].append(("both:" if both else "") + k)
    out["diff"] = out["diff"][:12]
    return out


def run_d3(a) -> None:
    D3 = load_d3(Path(a.code))
    L = D3.L
    table = json.loads(Path(D3.TABLE).read_text())
    h = D3.make_host("heft", D3.SEED + a.shard)
    rig = D3.Rig(h)
    log = HLog(h, a.qpos)
    attach(h, log)
    ids = list(range(a.shard, D3.RESETS, a.shards))
    out = Path(a.out)
    js = [int(v) for v in a.js.split(",")] if a.js else list(range(a.n))
    retime = {}  # revision 3: {"chain": c, "dur": T} for the reset being run (None: data3's own timing)
    plan0 = D3.plan_reach
    calls = [0]

    def plan_retimed(h_, x, ref_frame, cmd):
        blend, last, rec = plan0(h_, x, ref_frame, cmd)
        k = calls[0]
        calls[0] += 1
        if retime.get("chain") != k or not retime.get("dur"):
            return blend, last, rec
        RT, RP, RQ, delta = np.asarray(rec["RT"]), np.asarray(rec["RP"]), np.asarray(rec["RQ"]), np.asarray(rec["delta"])
        shifted = [(ref_frame[:3].astype(float), ref_frame[3:7].astype(float), ref_frame[7:].astype(float))]
        shifted += [(RP[i][:3] + delta, RP[i][3:7], RQ[i]) for i in range(1, len(RP))]
        durs = np.diff(RT) * float(retime["dur"]) / float(RT[-1])
        fast = L.blend_frames(shifted, list(durs))
        # the same waypoints in T s, then the last one held to the original length: the reach's window (f_end, the
        # settle wait, the label and the next command) is the original's, only the motion is faster
        pad = max(0, len(blend) - len(fast))
        out_b = np.concatenate([fast, np.repeat(fast[-1:], pad, 0)]) if pad else fast
        rec = dict(rec, motion_s=float(retime["dur"]), reach_s_orig=float(RT[-1]), motion_f=len(fast), pad_f=pad, RT=np.r_[0.0, np.cumsum(durs)])
        return out_b, shifted[-1], rec

    D3.plan_reach = plan_retimed
    variants = [None] + [float(v) for v in a.durs.split(",")] if (a.durs and a.target_j < 0) else [None]
    for j in js:
        vs = variants
        if a.target_j >= 0:  # revision 3: the shard's resets in order up to the target (the same start state), then the target retimed
            vs = [None] if j != a.target_j else ([float(a.durs)] if a.durs and a.durs != "orig" else [None])
        for T in vs:
            retime.clear()
            if a.chain >= 0 and T is not None:
                retime.update(chain=a.chain, dur=T)
            calls[0] = 0
            i = ids[j]
            reset = D3.ID0 + i
            log.clear()
            t0 = time.time()
            plan = D3.reset_plan(np.random.default_rng(D3.SEED + i), table, True)
            rows, info, frames, reaches = D3.run_reset(h, rig, plan)
            segs = D3.segments(rows, info, frames, reaches, reset, h.pl, rig)
            cmp = compare(D3, segs, Path(a.orig) / f"w{a.shard:03d}_{j // D3.CHUNK:03d}.npz", reset) if T is None else dict(n=len(segs), same=None, why="retimed")
            rch = [dict(f_cmd=r["f_cmd"], f_end=r["f_end"], f_label=r.get("f_label"), mode=r["mode"], pelvis_z=r["pelvis_z"], reach_s=r["reach_s"],
                        reach_s_orig=r.get("reach_s_orig"), motion_s=r.get("motion_s"), motion_f=r.get("motion_f"), pad_f=r.get("pad_f"), hands={s: dict(cmd_target=v["cmd_target"], target_w=v.get("target_w"), ik_err=v.get("ik_err"), dyaw=v["dyaw"],
                        pitch=v["pitch"], roll=v.get("roll"), core=v.get("core")) for s, v in r["hands"].items()}, RT=r.get("RT"), RP=r.get("RP"), RQ=r.get("RQ")) for r in reaches]  # fmt: skip
            meta = dict(mode="d3", c=h.c, shard=a.shard, j=j, reset=reset, retime=dict(retime) or None, walked=bool(info["walked"]), fell=info["fell"], lost=info["lost"],
                        why=info["why"], steps=len(log.rows), delay_f=info.get("delay_f"), reaches=rch, compare=cmp, wall_s=time.time() - t0)  # fmt: skip
            tag = "" if T is None else f"_c{a.chain}_T{T:.1f}"
            if a.target_j >= 0 and j == a.target_j and T is None:
                tag = f"_c{a.chain}_Torig"
            if a.target_j < 0 or j == a.target_j:
                save(out, f"d3_w{a.shard:03d}_r{reset}{tag}", log, L, meta)
            print(f"fd_heft d3 shard {a.shard} reset {reset}{tag}: walked {info['walked']} lost {info['lost']} reaches {len(reaches)} "
                  f"segments {len(segs)} reproduced {cmp.get('same')} {cmp.get('diff', cmp.get('why', ''))} {time.time() - t0:.1f} s", flush=True)  # fmt: skip

WAIST_PITCH_MJ = 14  # waist_pitch_joint in the frame's (MuJoCo) joint order; the frame's column 7 + 14
LEGS_MJ = tuple(range(12))  # left / right hip pitch, roll, yaw, knee, ankle pitch, roll (MuJoCo order)
WAIST_MJ = (12, 13, 14)
ARM_MJ = {"left": tuple(range(15, 22)), "right": tuple(range(22, 29))}


def lock_cols(lock: str, side: str) -> tuple[list[int], bool]:
    """Frame columns held at the command frame's values, and whether the root (pelvis position and orientation) is held.
    waist: waist pitch only (revision 1); legs: hips, knees, ankles (revision 2); arm: everything but the reaching arm --
    legs, waist, the other arm and the root (revision 2)."""
    if lock == "waist":
        return [7 + WAIST_PITCH_MJ], False
    if lock == "legs":
        return [7 + j for j in LEGS_MJ], False
    if lock == "root":  # revision 3: the reference pelvis (position, height, orientation) only
        return [], True
    if lock == "waistlegs":  # revision 3: waist and legs together, the pelvis free
        return [7 + j for j in LEGS_MJ + WAIST_MJ], False
    if lock == "arm":
        other = "left" if side == "right" else "right"
        return [7 + j for j in LEGS_MJ + WAIST_MJ + ARM_MJ[other]], True
    return [], False


def run_ha(a) -> None:
    """--rep k: host seed SEED + 977 + k and the command k x 10 frames after the settle rule first holds (a different
    start state); --lock-waist: the reference's waist pitch held at its value at the command frame for the whole reach and
    hold (the rest of the reference unchanged) -- the coordinator's control after (b), 11-26."""
    D3 = load_d3(Path(a.code))
    L = D3.L
    z = np.load(a.ha_npz, allow_pickle=True)
    h = D3.make_host("heft", D3.SEED + 977 + a.rep)
    rig = D3.Rig(h)
    log = HLog(h, a.qpos)
    attach(h, log)
    out = Path(a.out)
    FPS = D3.FPS
    for c in [int(v) for v in a.chains.split(",")]:
        side = "left" if int(z["side"][c]) < 0 else "right"
        th = np.asarray(z["target"][c], float)
        att = dict(dyaw=float(z["dyaw"][c]), pitch=float(z["pitch"][c]), roll=float(z["roll"][c]))
        pz = None if float(z["pelvis_z"][c]) < 0 else float(z["pelvis_z"][c])
        reach_s = float(z["reach_s"][c])
        log.clear()
        t0 = time.time()
        h.reset()
        rig.clear()
        frames = h.start_frames()
        frames = np.concatenate([frames, np.repeat(frames[-1:], int(D3.HOLD_S * FPS), 0)])
        f, f_cmd, f_stop, rec, fell, delay, f_ok = 0, None, None, None, None, None, None
        while True:
            h.push(frames[min(f, len(frames) - 1)])
            if getattr(h, "buf", 0) is None:
                h.buf = h._find_buffer()
            h.step()
            if delay is None:
                delay = D3.ref_delay_f(h)
            x = h.sample()
            rig.record(x)
            if float(x["root"][2]) < L.FALL_Z:
                fell = f
                break
            if f_ok is None and ((f >= D3.FIRST_OPEN_F + D3.SETTLE_F and rig.quiet()) or f >= D3.FIRST_OPEN_F + D3.SETTLE_MAX_F):
                f_ok = f
            if f_cmd is None and f_ok is not None and f >= f_ok + 10 * a.rep:
                p, yaw = x["root"][:3].astype(float), L.yaw_of(L._qmat(x["root"][3:]))
                Rh = L.rot_z(yaw)
                tw = np.r_[p[:2] + (Rh @ np.r_[th[:2], 0.0])[:2], th[2]]
                Rw = Rh @ D3.R_of(att)
                pw, Rwr = x["wr"][side]
                length = float(np.linalg.norm(tw - (pw + Rwr @ L.HAND_OFF)))
                speed = length / reach_s
                cmd = dict(kind="rel", mode=side, pelvis_z=pz, speed=speed, pair_att=0,
                           hands={side: dict(start=None, vias=[], back=0.0, core=False, target_w=tw, rot_w=Rw)})  # fmt: skip
                blend, last, rec = D3.plan_reach_b(h, x, frames[f], cmd)
                frames = np.concatenate([frames[: f + 1], blend, L.hold(last, 12.0)])
                wp0 = float(frames[f, 7 + WAIST_PITCH_MJ])
                wp_end = float(frames[f + len(blend), 7 + WAIST_PITCH_MJ])
                lock = "waist" if a.lock_waist else a.lock
                cols_l, root_l = lock_cols(lock, side)
                for j in cols_l:
                    frames[f + 1 :, j] = frames[f, j]
                if root_l:
                    frames[f + 1 :, :7] = frames[f, :7]
                f_cmd = f
                start = dict(pelvis=x["root"].tolist(), dq_norm=float(np.linalg.norm(x["dq"])), f_settle=f_ok)
                f_stop = f + len(blend) + int(9.0 * FPS)
            if f_stop is not None and f >= f_stop:
                break
            f += 1
        hs = rec["hands"][side] if rec else {}
        meta = dict(mode="ha", c=h.c, rep=a.rep, lock_waist=a.lock_waist, lock=a.lock, waist_pitch_cmd=wp0 if rec else None,
                    waist_pitch_ref_end_unlocked=wp_end if rec else None, start=start if rec else None, target_w=tw if rec else None, ha_npz=a.ha_npz, chain=c, side=side, target_h=th, att=att, pelvis_z=pz, ha_reach_s=reach_s,
                    f_cmd=f_cmd, f_end=None if rec is None else f_cmd + int(round(rec["reach_s"] * FPS)), reach_s=None if rec is None else rec["reach_s"],
                    speed=None if rec is None else rec["speed"], ik_err=hs.get("ik_err"), cmd_target=hs.get("cmd_target"), d_start_cm=hs.get("d_start_cm"),
                    RT=None if rec is None else rec["RT"], RP=None if rec is None else rec["RP"], RQ=None if rec is None else rec["RQ"],
                    delay_f=delay, fell=fell, steps=len(log.rows), wall_s=time.time() - t0)  # fmt: skip
        name = f"ha_{Path(a.ha_npz).stem}_c{c}" + (f"_r{a.rep}" if a.rep else "") + ("_lockwp" if a.lock_waist else "") + (f"_lock{a.lock}" if a.lock else "")
        save(out, name, log, L, meta)
        print(f"fd_heft ha {name}: {side} target_h {np.round(th, 3).tolist()} dyaw {math.degrees(att['dyaw']):.1f} reach {meta['reach_s']} s "
              f"(HA {reach_s:.2f}), ik err {hs.get('ik_err')}, fell {fell}, {len(log.rows)} steps, {time.time() - t0:.1f} s", flush=True)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["d3", "ha"])
    ap.add_argument("--code", required=True, help="the directory holding selfmodel_data3_20261125/, locomanilab_20261117/, hardgrasp_20261125/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=30)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--js", default="", help="d3: these indices into the shard's reset list (in this order) instead of 0 .. n-1")
    ap.add_argument("--target-j", type=int, default=-1, help="d3, revision 3: run js in order, save only this one; --durs one value (or 'orig')")
    ap.add_argument("--chain", type=int, default=-1, help="d3, revision 3: the reach (chain) of the reset to retime")
    ap.add_argument("--durs", default="", help="d3, revision 3: after the reset as data3 ran it, again with that reach's reference lasting each of these (s)")
    ap.add_argument("--orig", default="/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/heft/data")
    ap.add_argument("--ha-npz", default="")
    ap.add_argument("--chains", default="0")
    ap.add_argument("--qpos", action="store_true")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--lock-waist", action="store_true")
    ap.add_argument("--lock", choices=["", "legs", "arm", "root", "waistlegs"], default="")
    a = ap.parse_args()
    run_d3(a) if a.mode == "d3" else run_ha(a)


if __name__ == "__main__":
    main()
