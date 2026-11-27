"""selfmodel_hadata_20261126 (diagnosis for viewing only, 2026-11-26): hr_real.py --hrvid DIR (not --video: HA's second parser of sys.argv would take it for --video_fps). Imported by hr_real only
when the switch is given; hr_real's hooks (Episode.vid, the phase labels) do nothing without it. Nothing here is read
back by the episode (its own LegModel copy for the reference's feet; the camera is a prim of its own); these episodes
enter no statistic and no criterion (coordinator 2026-11-26: why HEFT stands badly and walks short in HA)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class Video:
    """A third-person RGB camera of our own (a USD camera prim and a replicator render product added after the scene is
    up) on the robot's left, 2.4 m out, 1.3 m up, looking at 0.5 m ahead of the pelvis, 0.55 m up: the feet and the
    table in one view. It follows the pelvis (and heading) only while walking (smoothed); otherwise it stays put, so a
    base slide shows against the still scene. A frame every 5th control step (10 Hz, played in real time) with the
    phase, the pelvis's planar speed (actual and reference), the feet's load, the reference's swing foot and a red
    banner while the base slides (> 8 cm/s outside a walk).
    Per control step (steps/<name>.npz): f, t, actual root x y z yaw, planar speed, the reference frame pushed this
    step (the controller's command: root x y yaw, both ankles' height from the reference joints), the actual ankles'
    height and xy (robot bodies; the reference's ankle xy too), both feet's vertical load (the ankle links' contact force, as the odometry reads it), walking.
    Events (steps/<name>_events.json): phases, every walk (asked in the start heading frame, the reference's step
    count, the reference's own root displacement, the actual true displacement and the belief's), foot lift-offs /
    touchdowns (< 15 N / > 60 N), base slides. Video: video/<name>.mp4 (H.264, crf 28)."""

    EVERY, W, H, SLIDE, OFF_N, ON_N = 5, 640, 360, 0.08, 15.0, 60.0

    def __init__(self, sim, out: Path):
        import omni.replicator.core as rep
        import omni.usd
        from pxr import UsdGeom

        self.sim, self.out = sim, out
        for k in ("video", "steps"):
            (out / k).mkdir(parents=True, exist_ok=True)
        stage = omni.usd.get_context().get_stage()
        cam = UsdGeom.Camera.Define(stage, "/World/hr_video_cam")
        cam.CreateFocalLengthAttr(14.0)  # with the default 20.955 mm aperture: 73 deg across
        cam.CreateClippingRangeAttr((0.05, 100.0))
        self.op = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()
        self.rp = rep.create.render_product("/World/hr_video_cam", (self.W, self.H))
        self.ann = rep.AnnotatorRegistry.get_annotator("rgb")
        self.ann.attach([self.rp])
        self.ep, self.lm = None, None

    def begin(self, name: str, ep) -> None:
        self.name, self.ep = name, ep
        self.writer, self.n_frames, self.rows, self.events = None, 0, [], []
        self.prev, self.sliding, self.label, self.cam = None, False, "stance", None
        self.n_walks, self.walk0, self.lifted = 0, None, {"left": False, "right": False}
        self.i_foot = self.i_body = None

    def phase(self, label: str) -> None:
        self.label = label
        self.event(label)

    def event(self, what: str, **kw) -> None:
        if self.ep is not None and self.ep.f >= 0:
            self.events.append(dict(t=round(self.ep.f / 50.0, 2), f=int(self.ep.f), what=what, **kw))

    def _look(self, xy, yaw) -> None:
        from pxr import Gf

        c, s_ = math.cos(yaw), math.sin(yaw)
        R2 = np.array([[c, -s_], [s_, c]])
        eye = np.r_[xy + R2 @ np.array([0.5, 2.4]), 1.3]
        at = np.r_[xy + R2 @ np.array([0.5, 0.0]), 0.55]
        self.op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*at), Gf.Vec3d(0, 0, 1)).GetInverse())

    def step(self, ep, f: int, x) -> None:
        h, L = ep.h, ep.L
        if self.i_foot is None:
            names = list(h.cs.body_names)
            self.i_foot = [names.index(f"{s}_ankle_roll_link") for s in ("left", "right")]
            bn = list(h.sim.robot.body_names)
            self.i_body = [bn.index(f"{s}_ankle_roll_link") for s in ("left", "right")]
        if self.lm is None:
            self.lm = ep.HW.LegModel(h.pl)  # our own copy (the episode's h._legs is not touched)
        F = h.cs.data.net_forces_w[0].detach().cpu().numpy().astype(np.float64)
        load = [abs(float(F[i, 2])) for i in self.i_foot]
        P = h.sim.robot.data.body_link_pos_w[0].detach().cpu().numpy().astype(np.float64)
        az = [float(P[i, 2]) for i in self.i_body]
        axy = [P[i, :2].copy() for i in self.i_body]
        r = np.asarray(x["root"][:3], float)
        yaw = L.yaw_of(L._qmat(x["root"][3:]))
        fr = np.asarray(ep.frames[min(f, len(ep.frames) - 1)], float)
        rf = self.lm.feet(fr[:3], fr[3:7], fr[7:])
        rz = [float(rf[s][0][2]) for s in ("left", "right")]
        rxy = [rf[s][0][:2].copy() for s in ("left", "right")]
        ryaw = L.yaw_of(L._qmat(fr[3:7]))
        v = 0.0 if self.prev is None else float(np.linalg.norm(r[:2] - self.prev[0][:2])) * 50.0
        vr = 0.0 if self.prev is None else float(np.linalg.norm(fr[:2] - self.prev[1][:2])) * 50.0
        self.prev = (r, fr)
        self.rows.append((f, f / 50.0, r[0], r[1], r[2], yaw, v, fr[0], fr[1], ryaw, vr, rz[0], rz[1], az[0], az[1],
                          load[0], load[1], float(ep.walking), axy[0][0], axy[0][1], axy[1][0], axy[1][1],
                          rxy[0][0], rxy[0][1], rxy[1][0], rxy[1][1]))  # fmt: skip
        # walks: the start, then the result once hr_real appends it
        if ep.walking and self.walk0 is None:
            self.walk0 = (f, r.copy(), fr.copy(), yaw)
        if len(ep.walks) > self.n_walks:
            w = ep.walks[-1]
            self.n_walks = len(ep.walks)
            if self.walk0 is not None:
                f0, r0, fr0, y0 = self.walk0
                Rt = L.rot_z(y0).T
                dref = (Rt @ np.r_[fr[:2] - fr0[:2], 0.0])[:2]
                self.event("walk result", tag=w["tag"], asked_m=[round(a, 3) for a in w["asked"][:2]],
                           asked_dist=round(float(np.hypot(*w["asked"][:2])), 3), ref_steps=w["plan"]["n_steps"],
                           ref_root_moved=[round(float(a), 3) for a in dref], actual_true=[round(a, 3) for a in w["real_true"]],
                           belief=[round(a, 3) for a in w["real"][:2]], dev_cm=round(w["dev_cm"], 1), wait_s=w["wait_s"])  # fmt: skip
            self.walk0 = None
        for k, s in enumerate(("left", "right")):
            if not self.lifted[s] and load[k] < self.OFF_N:
                self.lifted[s] = True
                self.event(f"{s} foot lifts off", walking=bool(ep.walking))
            elif self.lifted[s] and load[k] > self.ON_N:
                self.lifted[s] = False
                self.event(f"{s} foot touches down", walking=bool(ep.walking))
        sl = (not ep.walking) and v > self.SLIDE and self.label != "stance"
        if sl and not self.sliding:
            self.event(f"base sliding ({100 * v:.0f} cm/s) during: {self.label}")
        self.sliding = sl
        if ep.odo is None:  # the stance wait: no frames yet
            return
        if self.cam is None or (ep.walking and f % self.EVERY == 0) or float(np.linalg.norm(r[:2] - self.cam[0])) > 0.8:
            if self.cam is None:
                self.cam = (r[:2].copy(), yaw)
            else:
                a, ya = self.cam
                self.cam = (a + 0.15 * (r[:2] - a), ya + 0.15 * float(math.remainder(yaw - ya, 2 * math.pi)))
            self._look(*self.cam)
        if f % self.EVERY:
            return
        import cv2

        self.sim.env.sim.render()
        im = np.asarray(self.ann.get_data())
        if im.ndim != 3 or im.shape[0] != self.H:
            return
        im = np.ascontiguousarray(im[..., :3])
        gz = min(rz)
        sw = [s[0].upper() for k, s in enumerate(("left", "right")) if rz[k] > gz + 0.01]
        tx = [f"{self.name}  t = {f / 50.0:5.1f} s", f"{self.label}",
              f"pelvis {100 * v:3.0f} cm/s (reference {100 * vr:3.0f})   feet load L {load[0]:4.0f} N  R {load[1]:4.0f} N",
              f"reference swing: {'/'.join(sw) or '-'}   actual ankle lift L {100 * (az[0] - min(az)):3.1f} R {100 * (az[1] - min(az)):3.1f} cm"]  # fmt: skip
        for k, t in enumerate(tx):
            cv2.putText(im, t, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, t, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if sl:
            cv2.rectangle(im, (0, self.H - 32), (self.W, self.H), (200, 30, 30), -1)
            cv2.putText(im, f"BASE SLIDING {100 * v:.0f} cm/s", (10, self.H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if self.writer is None:
            import imageio.v2 as iio

            self.writer = iio.get_writer(str(self.out / "video" / f"{self.name}.mp4"), fps=50 // self.EVERY, codec="libx264",
                                         quality=None, ffmpeg_params=["-crf", "28", "-preset", "medium", "-pix_fmt", "yuv420p"])  # fmt: skip
        self.writer.append_data(im)
        self.n_frames += 1

    def end(self, end: str) -> None:
        self.event(f"end: {end}")
        if self.writer is not None:
            self.writer.close()
        cols = ("f", "t", "x", "y", "z", "yaw", "v_planar", "ref_x", "ref_y", "ref_yaw", "ref_v_planar", "ref_ankle_z_left",
                "ref_ankle_z_right", "ankle_z_left", "ankle_z_right", "load_left", "load_right", "walking",
                "ankle_x_left", "ankle_y_left", "ankle_x_right", "ankle_y_right",
                "ref_ankle_x_left", "ref_ankle_y_left", "ref_ankle_x_right", "ref_ankle_y_right")  # fmt: skip
        np.savez_compressed(self.out / "steps" / f"{self.name}.npz", **{c: np.array([r[i] for r in self.rows]) for i, c in enumerate(cols)})
        (self.out / "steps" / f"{self.name}_events.json").write_text(json.dumps(self.events, indent=1))
        self.ep = None
