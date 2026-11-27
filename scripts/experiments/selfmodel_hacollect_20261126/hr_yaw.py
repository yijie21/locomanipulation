"""selfmodel_hadata_20261126 (dev diagnosis, 2026-11-26): does our ScaleBFM / HEFT path on HA (sx_ha.HAG1Host:
ha_g1ctrl's provider fed through lml_host's motion-buffer transform) walk wrong when the episode starts at a heading
other than ~0? In OpenDoor (spawn heading 89 deg) hr_real's walks came out 1.3-2.7x the commanded length; in PPBox and
DoubleDesk (-0.8 / -1.1 deg) they matched.

On the dev copy only (the frozen asrun is not read): PPBox, hc_ha's collection site (SITE, clear floor), no start
noise; after HA's reset and before the controller hears of it, the root is turned to each --yaws heading (world), then
the stance wait (data3's) and the same walks, each relative to the pose at its start (hg_walk.walk_frames from the
leg-odometry pose, as hr_real): 1.0 m ahead, 0.5 m back, 0.4 m left, then (0.6 ahead, 0.5 right). Per walk: asked and
walked (the simulator's base: this is a diagnosis), the ratio of lengths, the direction error, the tracking deviation.
--at spawn: the task's own spawn instead of the site (OpenDoor: its spawn heading is 89 deg).

    SX_HA_TASK=ppbox HC_SCRIPT=hr_yaw.py bash hc_ha.sh <gpu> --controller scalebfm --out DIR --yaws=-0.8,0,45,90,180
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hr_real as HR  # noqa: E402

HC, SXH, ha_sim = HR.HC, HR.SXH, HR.ha_sim
WALKS = (("ahead1.0", (1.0, 0.0)), ("back0.5", (-0.5, 0.0)), ("left0.4", (0.0, 0.4)), ("diag", (0.6, -0.5)))


def yaw_host(base):
    class YawHost(base):
        yaw_set = None

        def place(self) -> None:
            HC.HCMixin.place(self)  # the site (xy), sigma 0: no noise
            if self.yaw_set is None:
                return
            import torch

            rob = self.sim.robot
            rp = rob.data.root_link_pose_w.clone()
            y = math.radians(self.yaw_set)
            rp[:, 3:7] = torch.tensor([math.cos(y / 2), 0.0, 0.0, math.sin(y / 2)], dtype=rp.dtype, device=rp.device)
            rob.write_root_link_pose_to_sim(rp)
            rob.write_root_velocity_to_sim(torch.zeros((1, 6), device=rp.device))
            psv = getattr(self.sim.env.sim, "physics_sim_view", None)
            if psv is not None:
                psv.update_articulations_kinematic()

    return YawHost


def main() -> int:
    p = HR.parser()
    p.add_argument("--yaws", default="-0.8,0,45,90,180", help="deg; -0.8 = HA's own spawn heading (no turn)")
    p.add_argument("--at", default="site", choices=["site", "spawn"], help="hc_ha's SITE (ppbox) or the task's own spawn")
    a = p.parse_args()
    _orig = ha_sim.sev._configure_renderer_gpu

    def _cfg(args):
        _orig(args)
        SXH.kit_single_gpu(args)

    ha_sim.sev._configure_renderer_gpu = _cfg
    a = ha_sim.finish_args(a, a.ha_task)
    a.no_walker = True
    a.out.mkdir(parents=True, exist_ok=True)
    sim = ha_sim.Sim(a)
    SXH.setup_paths()
    sys.path.insert(0, str(HC.W_REPO / "src"))
    import sx_collect as SX

    import hg_walk as HW

    D3, L = SX.D3, SX.L
    D3.SETTLE_MAX_F = int(round(HC.SETTLE_MAX_S * D3.FPS))
    D3.HOLD_S = HC.HOLD_S
    D3.SETTLE_V = HC.SETTLE_V[a.controller]
    site = HC.SITE if a.at == "site" else None
    if a.controller == "sonic":
        h = yaw_host(HC.HCSonic)(sim, L, D3, SXH.SONIC_DELAY_F, "native", 0.0, site)
    else:
        h = yaw_host(HC.HCG1)(sim, L, D3, a.controller, "lml", None, 0.0, site)
    rig = SX.SRig(h)
    h.sample()
    out = []
    for yw in [float(v) for v in a.yaws.split(",")]:
        h.yaw_set = None if abs(yw + 0.8) < 1e-9 else yw
        h.ep_seed = ha_sim.episode_seed(a.task, 296, 0)
        t0 = time.time()
        ep = HR.Episode(h, rig, SX, D3, L, HW)
        rec = dict(controller=a.controller, yaw_set=yw, walks=[], end="done")
        try:
            ep.stand_still()
            rec["yaw_start"] = round(math.degrees(L.yaw_of(L._qmat(ep.x["root"][3:]))), 2)
            rec["xy_start"] = np.round(ep.x["root"][:2], 3).tolist()
            rec["buffer_transform"] = None
            for tag, rel in WALKS:
                b, by = ep.pose()
                goal = b + L.rot_z(by)[:2, :2] @ np.asarray(rel, float)
                pt0 = ep.x["root"][:2].copy()
                w = ep.walk_to(goal, by, tag)
                if rec["buffer_transform"] is None and getattr(h, "buf", None) is not None:
                    dyaw, src, dst = h.transform()
                    rec["buffer_transform"] = dict(dyaw_deg=round(math.degrees(dyaw), 2), src=np.round(src, 3).tolist(), dst=np.round(dst, 3).tolist())
                asked = np.asarray(w["asked"][:2])
                got = L.rot_z(by)[:2, :2].T @ (ep.x["root"][:2] - pt0)
                ang = math.degrees(math.atan2(asked[0] * got[1] - asked[1] * got[0], asked @ got))
                rec["walks"].append(dict(tag=tag, asked=np.round(asked, 3).tolist(), walked=np.round(got, 3).tolist(),
                                         ratio=round(float(np.linalg.norm(got) / np.linalg.norm(asked)), 2), dir_err_deg=round(ang, 1),
                                         yaw_err_deg=round(math.degrees(HR.B.wrap(L.yaw_of(L._qmat(ep.x["root"][3:])) - by)), 1),
                                         dev_cm=round(w["dev_cm"], 1), odo_err_cm=round(w["odo_err_cm"], 1), settled=w["settled"]))  # fmt: skip
        except HR.End as e:
            rec["end"] = str(e)
        rec["wall_s"] = round(time.time() - t0)
        out.append(rec)
        print(f"hr_yaw {a.controller} {a.ha_task} {a.at} yaw {yw}: start {rec.get('yaw_start')} deg, buffer {rec.get('buffer_transform')}, end {rec['end']}", flush=True)
        for w in rec["walks"]:
            print(f"  {w['tag']:9s} asked {w['asked']} walked {w['walked']} ratio {w['ratio']} dir {w['dir_err_deg']} deg "
                  f"yaw {w['yaw_err_deg']} deg dev {w['dev_cm']} cm odo {w['odo_err_cm']} cm settled {w['settled']}", flush=True)  # fmt: skip
        (a.out / f"{a.controller}_{a.ha_task}_{a.at}.json").write_text(json.dumps(out, indent=1))
    print(f"hr_yaw {a.controller} {a.ha_task} {a.at}: DONE", flush=True)
    return 0


if __name__ == "__main__":
    import os
    import traceback

    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    os._exit(rc)
