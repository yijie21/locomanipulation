"""LocoManiLab's offline tracking runner (g1_backend sim2real/sim_env/integrated_sim2sim.py at 25d262a) on the benchmark's
physics model. Its IntegratedSimRuntime calls `load_sim_model(robot_cfg)`, which selects robot_fk.xml -- the hand-less
FK/reference model, whose motors are named without the `_joint` suffix, so the actuator lookup raises -- and applies no
per-controller joint dynamics. assets/g1_unified/README.md says the physics model is robot.xml (29 body + 14 Dex3) and
that armature/damping/frictionloss come from the controller's policy.yaml. Importing this module patches both (plus
`pelvis` as a root joint name, robot.xml's free joint); their checkout is not edited. Run as a script it is their
runner with these fixes:

    cd <g1_backend>; PYTHONPATH=. python lml_run_integrated.py --robot g1 --policy-config checkpoints/heft/pmg/policy.yaml ...
"""

import dataclasses
import os
import sys

import numpy as np
import yaml

import sim2real.sim_env.integrated_sim2sim as I

_DYNAMICS: dict = {}


def set_dynamics(policy_config: str) -> None:
    """The controller whose armature / damping / frictionloss the sim model gets."""
    _DYNAMICS.clear()
    _DYNAMICS.update(yaml.safe_load(open(policy_config)))
    _DYNAMICS["_path"] = policy_config


_orig_load = I.load_sim_model


def load_sim_model(robot_cfg, **kw):
    assert _DYNAMICS, "lml_run_integrated.set_dynamics(policy_config) first"
    kw.setdefault("with_dex3_hands", True)
    kw.setdefault("joint_dynamics", {k: v for k, v in _DYNAMICS.items() if k != "_path"})
    m = _orig_load(robot_cfg, **kw)
    print(f"[lml_run_integrated] sim model nq {m.nq} nu {m.nu} (Dex3 hands, {_DYNAMICS['_path']} joint dynamics)", flush=True)
    scale = os.environ.get("LML_HAND_MASS_SCALE")
    if scale is not None:  # diagnostic: the Dex3 bodies (everything below the wrist_yaw_links) made (almost) massless
        hand_mass_scale(m, float(scale))
    return m


def hand_mass_scale(m, scale: float) -> None:
    """Scale the mass and inertia of every body below left/right_wrist_yaw_link (kept >= 1e-4 kg / 1e-9 kg m^2: bodies
    with joints need mass)."""
    import mujoco

    roots = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_link") for s in ("left", "right")}
    below = [b for b in range(m.nbody) if b not in roots and _ancestor_in(m, b, roots)]
    before = float(m.body_mass[below].sum())
    m.body_mass[below] = np.maximum(m.body_mass[below] * scale, 1e-4)
    m.body_inertia[below] = np.maximum(m.body_inertia[below] * scale, 1e-9)
    print(f"[lml_run_integrated] LML_HAND_MASS_SCALE {scale}: {len(below)} hand bodies {before:.3f} -> "
          f"{float(m.body_mass[below].sum()):.3f} kg (robot {float(m.body_mass.sum()):.2f} kg)", flush=True)  # fmt: skip


def _ancestor_in(m, b: int, roots: set) -> bool:
    while b > 0:
        b = int(m.body_parentid[b])
        if b in roots:
            return True
    return False


I.load_sim_model = load_sim_model
_orig_rt_init = I.IntegratedSimRuntime.__init__


def _rt_init(self, robot_cfg, *a, **k):
    names = tuple(robot_cfg.root_joint_names)
    if "pelvis" not in names:  # robot.xml's free joint; RobotCfg only knows robot_fk.xml's floating_base_joint
        robot_cfg = dataclasses.replace(robot_cfg, root_joint_names=names + ("pelvis",))
    _orig_rt_init(self, robot_cfg, *a, **k)


I.IntegratedSimRuntime.__init__ = _rt_init

if __name__ == "__main__":
    import tyro

    set_dynamics(sys.argv[sys.argv.index("--policy-config") + 1])
    I.IntegratedSim2Sim(tyro.cli(I.IntegratedSim2SimArgs)).run()
