"""The LocoManiLab side of the HumanoidArena controller smoke: smoke_common's phases on lml_host.HostDriver (the benchmark
host, EmptyEnv, G1_BRIDGE with the given backend), the same code and references probe_ctrl.py runs in HA, 2026-11-25.

    # notebook: skip ha controller integration smoke
    PYTHONPATH=$G1B:$W/src $LML_PY lml_ref.py --backend heft --out DIR [--phases stand,lag,reach,walk]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import smoke_common as SC  # noqa: E402

import lml_host as LH  # noqa: E402


class LMLHost(LH.HostDriver):
    """HostDriver with the buffer fetched after the first push (as d3_collect does) and the PD clamp reported."""

    def __init__(self, backend: str, seed: int):
        super().__init__(env_name="EmptyEnv", backend=backend, seed=seed)
        import mujoco

        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        m = self.m
        self.act = []
        for n in MUJOCO_JOINT_NAMES:
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0_" + n)
            self.act.append(a)
        self.act = np.array(self.act)
        self.lim = m.actuator_ctrlrange[self.act, 1].copy()
        self.limited = m.actuator_ctrllimited[self.act].copy()
        self.steps = 0
        self.feet_b = {sd: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot0_{sd}_ankle_roll_link") for sd in ("left", "right")}

    def sample(self) -> dict:
        x = super().sample()
        x["feet"] = {sd: self.d.xpos[b].copy() for sd, b in self.feet_b.items()}
        return x

    @property
    def ref_delay_s(self) -> float:
        return float(self._find_buffer().delay_s)

    def push(self, fw):
        super().push(fw)
        if self.buf is None:
            self.buf = self._find_buffer()

    def step(self):
        super().step()
        self.steps += 1

    def extra(self) -> dict:
        tau = self.d.ctrl[self.act]  # the last substep's PD torque, before MuJoCo's ctrlrange clamp
        ratio = np.abs(tau) / np.maximum(self.lim, 1e-6)
        names = SC.mj_names()
        return dict(sat_joints=int((ratio >= 1.0).sum()), pd_over_limit=float(ratio.max()),
                    pd_over_limit_joint=names[int(ratio.argmax())] if ratio.max() >= 0.9 else None)  # fmt: skip


def mass_report(h: LMLHost) -> dict:
    """As probe_ctrl's HADriver.mass_report, on the LML host's MuJoCo model."""
    import mujoco

    m, d = h.m, h.d
    mujoco.mj_forward(m, d)
    pel = h.pelvis

    def under(b):
        while b > 0:
            if b == pel:
                return True
            b = m.body_parentid[b]
        return False

    ids = [b for b in range(m.nbody) if under(b) and m.body_mass[b] > 0]  # the pelvis's own 3.8 kg sits in manipulator_mount
    com = d.subtree_com[pel].copy()
    pp, Rp = d.xpos[pel], d.xmat[pel].reshape(3, 3)
    mid = 0.5 * (d.xpos[h.feet_b["left"]] + d.xpos[h.feet_b["right"]])
    Ry = SC.L.rot_z(SC.L.yaw_of(Rp))
    nm = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).replace("robot0_", "")  # noqa: E731
    return dict(total_kg=float(m.body_subtreemass[pel]), per_body_kg={nm(b): float(m.body_mass[b]) for b in ids},
                com_in_pelvis_fk_convention=(Rp.T @ (com - pp) - h.c).tolist(), com_minus_ankle_mid_heading=(Ry.T @ (com - mid)).tolist())  # fmt: skip


def body_report(h: LMLHost) -> dict:
    """The LML host's joint dynamics for the 29 body joints (for the HA comparison)."""
    m = h.m
    names = SC.mj_names()
    import mujoco

    out = {}
    for i, n in enumerate(names):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot0_" + n)
        dof = m.jnt_dofadr[j]
        out[n] = dict(armature=float(m.dof_armature[dof]), damping=float(m.dof_damping[dof]),
                      frictionloss=float(m.dof_frictionloss[dof]), ctrlrange=float(h.lim[i]), ctrllimited=int(h.limited[i]))  # fmt: skip
    return dict(timestep=float(m.opt.timestep), joints=out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("scalebfm", "heft"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phases", default="stand,lag,reach,walk")
    ap.add_argument("--stand_s", type=float, default=30.0)
    ap.add_argument("--reach_dist", type=float, default=0.30)
    ap.add_argument("--walk_dist", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    h = LMLHost(a.backend, a.seed)
    rep = dict(backend=a.backend, host="LocoManiLab G1_BRIDGE EmptyEnv", build_s=time.time() - t0, pelvis_offset_c=h.c.tolist(),
               body=body_report(h), ref_delay_s=None)  # fmt: skip
    if "mass" in a.phases:  # after 3 s on the stance reference, as probe_ctrl's kin phase
        h.reset()
        fr = h.start_frames()
        for f in range(150):
            h.push(fr[min(f, len(fr) - 1)])
            h.step()
        rep["mass_after_3s_stance"] = mass_report(h)
        print("[lml_ref] mass", json.dumps(rep["mass_after_3s_stance"]), flush=True)
    phases = [p for p in a.phases.split(",") if p and p != "mass"]
    fns = dict(stand=lambda: SC.phase_stand(h, a.stand_s), lag=lambda: SC.phase_lag(h),
               reach=lambda: SC.phase_reach(h, "right", a.reach_dist), walk=lambda: SC.phase_walk(h, a.walk_dist))  # fmt: skip
    for p in phases:
        t1, n1 = time.time(), h.steps
        try:
            rep[p] = fns[p]()
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            rep[p] = dict(error=f"{type(e).__name__}: {e}")
        if isinstance(rep[p], dict):
            tr = rep[p].pop("_trace", None)
            if tr is not None:
                json.dump(tr, open(a.out / f"{p}_trace.json", "w"))
            rep[p].update(wall_s=time.time() - t1, control_steps=h.steps - n1)
        rep["ref_delay_s"] = h.ref_delay_s
        print(f"[lml_ref] {p}", json.dumps(rep[p], default=str), flush=True)
        json.dump(rep, open(a.out / "report.json", "w"), indent=2, default=str)
    print("[lml_ref] DONE", a.out / "report.json", flush=True)


if __name__ == "__main__":
    main()
