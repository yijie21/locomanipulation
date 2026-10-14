"""Measure the C1-C7 / clip runs of both controllers (TODO 25.4, plan §4.3) with one rule.

Per run (bfm_clip.py's or sonic_clip.py's npz + json: qpos at 50 Hz, clip time, the anchor), against the clip anchored
at the same start pose, over the clip proper (clip time 0 .. end):
* fell -- the run's own fall flag (pelvis below 0.40 m) any time after the settle;
* G-MPKPE -- mean over frames and ScaleBFM's 14 bodies of the world position error (the ScaleBFM paper's measure,
  its reported 0.09 m); L-MPKPE -- the same with each pose relative to its own pelvis, in its own pelvis heading;
* keyframes -- the jaw centre within 5 cm and 15 deg of the keyframe's pose for 0.5 s without a break, starting at
  most 0.25 s before the keyframe's window and ending at most 1.0 s after it (the reference is on the target for
  the whole window; the lateness allowance keeps SONIC's ~0.2 s streaming lag from deciding the result);
* success = not fallen and, for a clip with keyframes, every keyframe met; for a clip without, G-MPKPE <= 0.10 m.
Extras: torso pitch at its maximum, lowest pelvis, jaw heights reached.

Writes <out>/complex.{json,md}: per clip x (controller, mode) the counts and means, and the mode choice (plan §5: by
success rate, then G-MPKPE; how much the best mode beats mode 2).

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/controller_compare_20261013/clip_measure.py \\
        outputs/controller_compare_20261013/complex/bfm outputs/controller_compare_20261013/complex/sonic \\
        --out outputs/controller_compare_20261013/complex
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import clips as CL  # noqa: E402

GMPKPE_OK = 0.10
EARLY_S, LATE_S = 0.25, 1.0


class Fk:
    def __init__(self, scene):
        self.m = mujoco.MjModel.from_xml_path(str(REPO / "outputs" / "scene" / scene / "scene_locomani.xml"))
        self.d = mujoco.MjData(self.m)
        self.bids = [self.m.body(b).id for b in CL.KEYPOINTS]
        self.wrist = {s: self.m.body(CL.WRIST[s]).id for s in ("left", "right")}
        self.torso = self.m.body("torso_link").id
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in CL.MUJOCO_JOINT_NAMES])

    def run(self, qpos):
        """Per frame: 14 body positions, pelvis yaw, jaw poses, torso pitch."""
        n = len(qpos)
        kp, yaw = np.empty((n, len(self.bids), 3)), np.empty(n)
        jaw = {s: (np.empty((n, 3)), np.empty((n, 3, 3))) for s in ("left", "right")}
        pitch = np.empty(n)
        for i, q in enumerate(qpos):
            if len(q) == 36:  # a clip frame: pelvis + 29 joints
                self.d.qpos[:7] = q[:7]
                self.d.qpos[self.qadr] = q[7:]
            else:
                self.d.qpos[:] = q
            mujoco.mj_kinematics(self.m, self.d)
            kp[i] = self.d.xpos[self.bids]
            yaw[i] = CL.yaw_of(self.d.qpos[3:7])
            for s, b in self.wrist.items():
                Rm = self.d.xmat[b].reshape(3, 3)
                jaw[s][0][i] = self.d.xpos[b] + Rm @ CL.OFF
                jaw[s][1][i] = Rm
            Rt = self.d.xmat[self.torso].reshape(3, 3)
            pitch[i] = np.degrees(np.arctan2(Rt[0, 2], Rt[2, 2]))
        return kp, yaw, jaw, pitch


def local(kp, yaw):
    """Positions relative to the pelvis (body 0) in the pelvis heading frame."""
    rel = kp - kp[:, :1]
    c, s = np.cos(-yaw), np.sin(-yaw)
    x = c[:, None] * rel[..., 0] - s[:, None] * rel[..., 1]
    y = s[:, None] * rel[..., 0] + c[:, None] * rel[..., 1]
    return np.stack([x, y, rel[..., 2]], -1)


def longest_run(mask, t):
    best, start = 0.0, None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        if (not ok or i == len(mask) - 1) and start is not None:
            end = i if ok else i - 1
            best = max(best, t[end] - t[start] + 1.0 / CL.HZ)
            start = None
    return best


def measure(run_npz: Path, clip_dir: Path, fks: dict, clip_cache: dict) -> dict:
    f = json.loads(run_npz.with_suffix(".json").read_text())
    r = dict(np.load(run_npz))
    if f["clip"] not in clip_cache:
        z, cf = CL.load(clip_dir, f["clip"])
        ref_fk = fks.setdefault(cf["scene"], Fk(cf["scene"]))
        clip_cache[f["clip"]] = (z, cf, ref_fk.run(z["qpos"]))
    z, cf, (kp_c, yaw_c, jaw_c, _) = clip_cache[f["clip"]]
    fk = fks.setdefault(cf["scene"], Fk(cf["scene"]))
    out = dict(controller=f["controller"], mode=f.get("mode"), clip=f["clip"], item=cf["item"], rep=f["rep"])
    fell = f.get("fell_clip_t")
    out["fell"] = fell is not None
    out["fell_clip_t"] = fell
    if f.get("anchor_xy") is None:
        out.update(success=False, why="no anchor (fell before the clip)")
        return out
    xy, yaw0 = np.asarray(f["anchor_xy"]), float(f["anchor_yaw"])
    Rz = CL.rz(yaw0)
    T = len(z["qpos"])
    ct = r["clip_t"]
    sel = np.flatnonzero((ct >= -1e-6) & (ct < T / CL.HZ))
    idx = np.clip(np.round(ct[sel] * CL.HZ).astype(int), 0, T - 1)
    kp_m, yaw_m, jaw_m, pitch_m = fk.run(r["qpos"][sel])
    kp_r = np.einsum("ij,tkj->tki", Rz, kp_c[idx]) + np.array([xy[0], xy[1], 0.0])
    yaw_r = yaw_c[idx] + yaw0
    if len(sel):
        out["gmpkpe_m"] = float(np.linalg.norm(kp_m - kp_r, axis=-1).mean())
        out["lmpkpe_m"] = float(np.linalg.norm(local(kp_m, yaw_m) - local(kp_r, yaw_r), axis=-1).mean())
        out["torso_pitch_max_deg"] = float(pitch_m.max())
        out["pelvis_z_min_m"] = float(kp_m[:, 0, 2].min())
    out["clip_frames_seen"] = int(len(sel))
    kfs = []
    t_sel = ct[sel]
    for k in cf["keyframes"]:
        p_t = Rz @ np.asarray(k["pos"]) + np.array([xy[0], xy[1], 0.0])
        R_t = Rz @ np.asarray(k["R"])
        pj, Rj = jaw_m[k["side"]]
        e_pos = np.linalg.norm(pj - p_t, axis=1)
        e_rot = np.array([np.degrees(np.linalg.norm(CL.rot_err(R_t, Rm))) for Rm in Rj])
        win = (t_sel >= k["t0"] - EARLY_S) & (t_sel <= k["t1"] + LATE_S)
        ok = win & (e_pos <= CL.POS_TOL) & (e_rot <= CL.ROT_TOL_DEG)
        held = longest_run(ok, t_sel) if len(t_sel) else 0.0
        inside = win & np.isfinite(e_pos)
        kfs.append(
            dict(
                side=k["side"],
                what=k["what"],
                held_s=round(held, 2),
                ok=bool(held >= CL.HOLD_S - 1e-9),
                pos_err_min_cm=round(100 * float(e_pos[inside].min()), 2) if inside.any() else None,
                rot_err_at_min_deg=round(float(e_rot[inside][np.argmin(e_pos[inside])]), 1) if inside.any() else None,
                pos_err_median_cm=round(100 * float(np.median(e_pos[inside])), 2) if inside.any() else None,
            )
        )
    out["keyframes"] = kfs
    complete = len(sel) >= T - 2
    if cf["rule"] == "keyframes" and kfs:
        out["success"] = (not out["fell"]) and complete and all(k["ok"] for k in kfs)
    else:
        out["success"] = (not out["fell"]) and complete and out.get("gmpkpe_m", np.inf) <= GMPKPE_OK
    return out


def summarise(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["clip"], r["controller"], r["mode"])].append(r)
    table = []
    for (clip, ctl, mode), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
        g = [r["gmpkpe_m"] for r in rs if "gmpkpe_m" in r and not r["fell"]]
        lm = [r["lmpkpe_m"] for r in rs if "lmpkpe_m" in r and not r["fell"]]
        kmin = [k["pos_err_min_cm"] for r in rs for k in r.get("keyframes", []) if k["pos_err_min_cm"] is not None]
        table.append(
            dict(
                clip=clip,
                item=rs[0]["item"],
                controller=ctl,
                mode=mode,
                n=len(rs),
                falls=sum(r["fell"] for r in rs),
                success=sum(bool(r.get("success")) for r in rs),
                gmpkpe_cm=round(100 * float(np.mean(g)), 1) if g else None,
                lmpkpe_cm=round(100 * float(np.mean(lm)), 1) if lm else None,
                keyframes_met=sum(k["ok"] for r in rs for k in r.get("keyframes", [])),
                keyframes=sum(len(r.get("keyframes", [])) for r in rs),
                kf_err_min_cm_median=round(float(np.median(kmin)), 1) if kmin else None,
                torso_pitch_max_deg=round(
                    float(np.median([r["torso_pitch_max_deg"] for r in rs if "torso_pitch_max_deg" in r])), 1
                )
                if any("torso_pitch_max_deg" in r for r in rs)
                else None,
                pelvis_z_min_cm=round(
                    100 * float(np.median([r["pelvis_z_min_m"] for r in rs if "pelvis_z_min_m" in r])), 1
                )
                if any("pelvis_z_min_m" in r for r in rs)
                else None,
            )
        )
    choice = {}
    for clip in sorted({t["clip"] for t in table}):
        bfm = [t for t in table if t["clip"] == clip and t["controller"] == "scalebfm"]
        if not bfm:
            continue
        best = max(bfm, key=lambda t: (t["success"] / t["n"], -(t["gmpkpe_cm"] if t["gmpkpe_cm"] is not None else 1e9)))
        m2 = next((t for t in bfm if t["mode"] == 2), None)
        son = next((t for t in table if t["clip"] == clip and t["controller"] == "sonic"), None)
        choice[clip] = dict(
            best_mode=best["mode"],
            best=f"{best['success']}/{best['n']}",
            best_gmpkpe_cm=best["gmpkpe_cm"],
            mode2=f"{m2['success']}/{m2['n']}" if m2 else None,
            mode2_gmpkpe_cm=m2["gmpkpe_cm"] if m2 else None,
            sonic=f"{son['success']}/{son['n']}" if son else None,
            sonic_gmpkpe_cm=son["gmpkpe_cm"] if son else None,
            sonic_falls=son["falls"] if son else None,
            best_falls=best["falls"],
        )
    return table, choice


def write_md(path: Path, table, choice, clips_meta):
    lines = [
        "# C1–C7 复杂动作与片段跟踪（TODO 25.4）",
        "",
        "生成：`clip_measure.py`。成功 = 不摔且关键帧都达到（手 ≤ 5 cm、≤ 15°、连续 0.5 s）；无关键帧的片段为 G-MPKPE ≤ 10 cm。",
        "",
    ]
    lines += [
        "## 每个动作：ScaleBFM 最好的 mode、mode 2、SONIC",
        "",
        "| 片段 | 项 | 来源 | ScaleBFM 最好 mode | 成功 | G-MPKPE cm | mode 2 成功 | mode 2 G-MPKPE | SONIC 成功 | SONIC G-MPKPE | 摔倒 ScaleBFM 最好 / SONIC |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for clip, c in choice.items():
        m = clips_meta.get(clip, {})
        src = "分布内 LAFAN1" if m.get("in_distribution") else "分布外 IK"
        lines.append(
            f"| {clip} | {m.get('item', '')} | {src} | {c['best_mode']} | {c['best']} | {c['best_gmpkpe_cm']} | {c['mode2']} | "
            f"{c['mode2_gmpkpe_cm']} | {c['sonic']} | {c['sonic_gmpkpe_cm']} | {c['best_falls']} / {c['sonic_falls']} |"
        )
    lines += [
        "",
        "## 全表",
        "",
        "| 片段 | 控制器 | mode | 局 | 摔 | 成功 | G-MPKPE cm | L-MPKPE cm | 关键帧达成 | 关键帧最小误差中位 cm | 躯干最大俯仰 ° | 骨盆最低 cm |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in table:
        lines.append(
            f"| {t['clip']} | {t['controller']} | {t['mode']} | {t['n']} | {t['falls']} | {t['success']} | {t['gmpkpe_cm']} | "
            f"{t['lmpkpe_cm']} | {t['keyframes_met']}/{t['keyframes']} | {t['kf_err_min_cm_median']} | {t['torso_pitch_max_deg']} | {t['pelvis_z_min_cm']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="run dirs (<dir>/<clip>/*.npz)")
    ap.add_argument("--clips", type=Path, default=REPO / "outputs" / "controller_compare_20261013" / "clips")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    fks, cache, rows = {}, {}, []
    for d in a.runs:
        for p in sorted(d.glob("*/*.npz")):
            rows.append(measure(p, a.clips, fks, cache))
    table, choice = summarise(rows)
    meta = {c["name"]: c for c in json.loads((a.clips / "clips.json").read_text())["clips"]}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "complex.json").write_text(json.dumps(dict(rows=rows, table=table, choice=choice), indent=1))
    write_md(a.out / "complex.md", table, choice, meta)
    for clip, c in choice.items():
        print(clip, json.dumps(c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
