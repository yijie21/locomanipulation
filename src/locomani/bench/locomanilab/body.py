"""The robot's belief on LocoManiLab (TODO 49.4): bench/simple/body.py's Belief with this benchmark's camera, and the bed's recording
version of it. Fixed infrastructure, not loop-editable (it was `LmlBelief` in the package's view_lml.py, locked there).

`LmlBelief` differs from SIMPLE's only in what the camera is: the mirrored model's "head_camera", 320x240, the renderer's exact
depth and a field of view that comes with every image (`frame`), and the finger encoders the skill's `Hand` reads.
`RecBelief` adds what the old bed wrapped round the instance: the record of where each walk really ended against its goal
(stop_true_cm, odo_end_cm, stop_err_cm, ...) and the camera-pose comparison of each scene call. Record only.
"""

import numpy as np

from locomani.bench.simple.body import Belief

CAM = "head_camera"  # the robot model's name of the head camera


class LmlBelief(Belief):
    cam = CAM

    def frame(self):
        """The head camera's colour image, depth (m, float64) and vertical field of view (deg) of this frame."""
        e = self.sim.camera_image()
        return e["rgb"], np.asarray(e["depth"], float), float(e["fovy"])


class RecBelief(LmlBelief):
    env = None  # the bed's Env: the clock's label, the record row, the parsed instruction (set by the bed)
    recorder = None  # record.Recorder (set by the bed)

    def scene(self):
        out = super().scene()
        if self.recorder is not None:
            self.recorder.on_scene()
        return out

    def walk(self, xy, yaw, **kw):
        env, sim = self.env, self.sim
        pick_walk = env.want["kind"] not in ("navigation", "articulate")
        first = env.clock.label == "walk" and "stop_err_cm" not in env.row and pick_walk
        xy0 = np.asarray(xy)
        second = env.clock.label == "walk2"
        goal_sim = (
            self.to_sim(np.r_[xy, 0.0])[:2] if first or second else None
        )  # record: the goal in the simulator's frame, before the tie moves
        out = super().walk(xy, yaw, **kw)
        if second:
            env.row["stop2_true_cm"] = round(100 * float(np.linalg.norm(sim.d.qpos[:2] - goal_sim)), 1)
            env.row["odo_end2_cm"] = round(float(out[0]), 1)
        if (
            first
        ):  # record: where the pelvis really stopped against the goal, and the odometry's own error at the end (cm)
            env.row["stop_true_cm"] = round(100 * float(np.linalg.norm(sim.d.qpos[:2] - goal_sim)), 1)
            env.row["odo_end_cm"] = round(float(out[0]), 1)
        if env.clock.label == "walk" and "stop_err_cm" not in env.row and pick_walk:  # the pick's stance walk
            d_ = self.pelvis()[0][:2] - xy0
            env.row["stop_err_cm"] = round(100 * float(np.linalg.norm(d_)), 1)
            env.row["stop_vec_cm"] = [
                round(100 * float(d_ @ [np.cos(yaw), np.sin(yaw)]), 1),
                round(100 * float(d_ @ [-np.sin(yaw), np.cos(yaw)]), 1),
            ]  # ahead, left
        return out
