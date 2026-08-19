"""run_sim_loop.py, but on our two-surface scene and publishing ground truth.

Two deltas from the stock entry point, both minimal on purpose so the SONIC
tree stays unpatched:

  1. `ROBOT_SCENE` points at the composed scene from build_sim_scene.py.
     base_sim.init_scene does `GEAR_SONIC_ROOT / config["ROBOT_SCENE"]`, and
     pathlib lets an absolute right-hand side win, so an absolute path here
     replaces the default rather than being appended to it.

  2. `DefaultEnv.update_reward` also publishes privileged state. That hook is
     called every REWARD_DT (0.02 s -> 50 Hz) from BaseSimulator.start and its
     stock body does nothing useful, so wrapping it needs no change to the
     simulation loop itself.

Usage (same flags as run_sim_loop.py):
    python src/teleop_free/run_sim_locomani.py --interface lo \\
        --no-enable-onscreen --enable-offscreen --enable-image-publish
"""

from __future__ import annotations

import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tyro  # noqa: E402

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv  # noqa: E402
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig  # noqa: E402
from gear_sonic.scripts.run_sim_loop import SimWrapper  # noqa: E402
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory  # noqa: E402
from gear_sonic.data.robot_model.instantiation.g1 import (  # noqa: E402
    instantiate_g1_robot_model,
)

from privileged_pub import EpisodeControlServer, PrivilegedPublisher  # noqa: E402

# Set SIM_VIDEO_DIR to record a tracking-camera mp4 per episode. Off by default
# because rendering costs sim time.
VIDEO_DIR = os.environ.get("SIM_VIDEO_DIR")

SCENE_DIR = "/workspace/code/locomanipulation/outputs/scene/sim"


def install_privileged_publisher(meta: dict, default_dof_angles=None) -> None:
    object_bodies = {k: v["body"] for k, v in meta["objects"].items()}
    surface_bodies = [v["body"] for v in meta.get("surfaces", {}).values()]
    original = DefaultEnv.update_reward

    def update_reward(self):
        original(self)
        pub = getattr(self, "_privileged_pub", None)
        if pub is None:
            pub = PrivilegedPublisher(self.mj_model, self.mj_data,
                                      object_bodies, surfaces=surface_bodies)
            self._privileged_pub = pub
            recorder = None
            if VIDEO_DIR:
                from sim_video import VideoRecorder
                recorder = VideoRecorder(self.mj_model, VIDEO_DIR)
                self._recorder = recorder
            self._episode_ctl = EpisodeControlServer(
                self.mj_model, self.mj_data, meta, default_dof_angles,
                recorder=recorder)
        self._episode_ctl.poll()
        if getattr(self, "_recorder", None) is not None:
            self._recorder.frame(self.mj_data)
        pub.publish()

    DefaultEnv.update_reward = update_reward


def disable_fall_reset() -> None:
    """Report falls instead of teleporting out of them.

    `check_fall` calls `reset()` the moment the pelvis drops below 0.2 m
    (base_sim.py:617-623). The C++ deploy's state estimator has no idea that
    happened, so its odometry keeps integrating from the pre-teleport pose and
    the `facing` frame silently diverges from MuJoCo world — after a few falls
    the robot walks off in a direction unrelated to what was commanded. An
    episode that ends in a fall should end, not be rescued mid-flight; resets
    happen deliberately, between episodes, over :5582.
    """
    def check_fall(self):
        self.fall = bool(self.mj_data.qpos[2] < 0.2)
        if self.fall and not getattr(self, "_fall_reported", False):
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m "
                  "(not resetting; waiting for an episode reset)")
            self._fall_reported = True
        elif not self.fall:
            self._fall_reported = False

    DefaultEnv.check_fall = check_fall


def inject_reach_zone(scene_xml: str, meta: dict) -> str:
    """Add the FULL 3D graspable volume as a translucent cloud.

    Every right-arm reach-map voxel (pelvis frame) becomes a small green
    box on a mocap body that the privileged publisher keeps glued to the
    pelvis (position incl. HEIGHT + yaw). Standing, the cloud hangs where
    the arm can grasp; when the robot squats the whole cloud rides down
    with the body — the volume it sweeps is the total graspable region
    from standing to the deepest crouch. Group 4: debug camera only.
    """
    try:
        import reach_map
        z = np.load(reach_map.MAP_PATH)
    except Exception as e:  # a missing map must not kill the sim
        print(f"[sim] reach zone skipped ({e})")
        return scene_xml
    occ = z["right"]
    sweet = z["sweet_right"] if "sweet_right" in z else np.zeros_like(occ)
    vox, org = float(z["vox"]), z["origin"]
    if not occ.any():
        return scene_xml
    # Two tones: pale = full envelope (grasp possible), dark opaque core =
    # sweet zone (grasp comfortable, servo has correction room).
    parts = []
    for tag, grid, size, rgba in (
            ("env", occ & ~sweet, 0.0100, "0.2 0.9 0.35 0.10"),
            ("core", sweet, 0.0115, "0.02 0.45 0.12 0.55")):
        pts = np.argwhere(grid) * vox + org + vox / 2.0
        parts += [
            f'<geom name="reach_{tag}_{k}" type="box" '
            f'size="{size:.4f} {size:.4f} {size:.4f}" '
            f'pos="{x:.4f} {y:.4f} {zz:.4f}" contype="0" conaffinity="0" '
            f'mass="0" group="4" rgba="{rgba}"/>'
            for k, (x, y, zz) in enumerate(pts)]
    body = ('<body name="reach_zone" mocap="true" pos="0 0 0">'
            + "".join(parts) + "</body>")
    xml = open(scene_xml).read().replace("</worldbody>",
                                         body + "</worldbody>")
    out = os.path.join(SCENE_DIR, "scene_locomani_zone.xml")
    with open(out, "w") as f:
        f.write(xml)
    print(f"[sim] reach volume: {int(occ.sum())} voxels, "
          f"core {int(sweet.sum())} (pelvis frame)")
    return out


def main(config: SimLoopConfig) -> None:
    meta = json.load(open(f"{SCENE_DIR}/scene_meta.json"))
    scene_xml = meta["scene_xml"]
    if not os.path.isabs(scene_xml):
        scene_xml = os.path.abspath(scene_xml)
    if not os.path.exists(scene_xml):
        raise FileNotFoundError(
            f"{scene_xml} missing — run teleop_free/build_sim_scene.py first"
        )
    scene_xml = inject_reach_zone(scene_xml, meta)

    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = config.env_name
    wbc_config["ROBOT_SCENE"] = scene_xml
    # Parallel instances: each sim+deploy pair gets its own DDS domain so
    # their unitree channels never cross on lo. Pairs with the deploy's
    # SONIC_DDS_DOMAIN patch.
    wbc_config["DOMAIN_ID"] = int(os.environ.get(
        "SONIC_DDS_DOMAIN", wbc_config.get("DOMAIN_ID", 0)))

    if config.enable_image_publish:
        assert config.enable_offscreen, (
            "enable_offscreen must be True when enable_image_publish is True"
        )

    install_privileged_publisher(meta, wbc_config.get("DEFAULT_DOF_ANGLES"))
    disable_fall_reset()
    print(f"[sim] scene   {scene_xml}")
    print(f"[sim] objects {list(meta['objects'])}")

    robot_model = instantiate_g1_robot_model()
    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        env_name=config.env_name,
        config=wbc_config,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", True),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=config.enable_image_publish,
    )

    # The elastic band hauls the pelvis towards a fixed anchor at (0, 0, 1) with
    # kp 10000 (unitree_sdk2py_bridge.py:371-387) — a constant sideways yank on a
    # robot that starts 1.5 m away and then walks 4 m. It cannot be turned off
    # via ENABLE_ELASTIC_BAND: that flag also skips creating the attribute, and
    # sim_step/handle_keyboard_button both dereference it unconditionally
    # (base_sim.py:394, 498). Disabling the instance is the same thing the
    # viewer's '9' key does, and is the only way that does not crash.
    band = getattr(sim_wrapper.sim.sim_env, "elastic_band", None)
    if band is not None:
        band.enable = False
        print("[sim] elastic band disabled")

    # Arm pin: give the arm motors their own position loop, on demand. The
    # walking policy will not hold an elbow past ~1.5 rad no matter what
    # reference it is given (its gait prior wins), so the level-forearm
    # walking pose is applied here, at the actuator, the way real hardware
    # would run those joints in position mode. GetAction returns the 43
    # position targets [legs 12 | waist 3 | L arm 7 | L hand 7 | R arm 7 |
    # R hand 7]; the pin overwrites the two arm slices while set.
    from privileged_pub import ARM_PIN

    _orig_cbt = DefaultEnv.compute_body_torques
    PIN_KP, PIN_KD, PIN_TAU_MAX = 120.0, 3.0, 30.0
    ARM_MOTORS = {"left": list(range(15, 22)), "right": list(range(22, 29)),
                  # Waist trio: frozen-upper mode pins these through the
                  # deploy during walking anyway; during the streamed grasp
                  # the pin makes the IK's (limit-clamped) lean deliver
                  # exactly instead of at the policy's ~1/3.
                  "waist": [12, 13, 14]}

    def compute_body_torques(self):
        tau = _orig_cbt(self)
        br = self.unitree_bridge
        if br is None or not br.low_cmd:
            return tau
        for side, motors in ARM_MOTORS.items():
            pin = ARM_PIN[side]
            if pin is None:
                continue
            for k, i in enumerate(motors):
                if br.use_sensor:
                    q = self.mj_data.sensordata[i]
                    dq = self.mj_data.sensordata[i + br.num_body_motor]
                else:
                    q = self.mj_data.qpos[self.body_joint_index[i]
                                          + self.qpos_offset - 1]
                    dq = self.mj_data.qvel[self.body_joint_index[i]
                                           + self.qvel_offset - 1]
                t = PIN_KP * (pin[k] - q) - PIN_KD * dq
                tau[i] = max(-PIN_TAU_MAX, min(PIN_TAU_MAX, t))
        return tau

    DefaultEnv.compute_body_torques = compute_body_torques
    print("[sim] arm pin hook installed (compute_body_torques)")

    SimulatorFactory.start_simulator(
        sim_wrapper.sim,
        as_thread=False,
        enable_image_publish=config.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )


if __name__ == "__main__":
    main(tyro.cli(SimLoopConfig))
