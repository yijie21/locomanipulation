"""Render REFERENCE vs release-delivered vs v1_1-delivered as one side-by-side clip.

Kinematic replay of what was recorded, not a new simulation: the reference comes
from the official CSVs, the two deliveries from the privileged state traces. All
three are drawn in the same robot model, from the same camera, with the root's
yaw and xy removed, so the only thing that differs between panels is posture.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

from locomani.primitives.g1_joints import isaac_to_mujoco

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "outputs/diagnostics/sonic_native_phase3_20260910"
REF = ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example/squat_001__A359"
from locomani import paths

XML = os.environ.get("RENDER_XML", paths.SCENE_XML)
SPAWN = (-1.5, 0.0)  # where the robot actually stood; keeps furniture out of the panels
OUT = Path(os.environ.get("SONIC_RENDER_OUT", DIAG / "compare"))
W, H = int(os.environ.get("PANEL_W", 440)), int(os.environ.get("PANEL_H", 640))
BAR = 74
FPS = 50


def rot(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def deyaw(q):
    """Drop the heading, keep pitch and roll, so the panels are comparable."""
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(-yaw / 2), np.sin(-yaw / 2)
    return np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w])


BODY_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def body_qpos_index(model):
    """qpos address of each of the 29 motor joints, by NAME.

    A flat qpos[7:36] slice is wrong here: this scene's joint order is
    legs 12 | waist 3 | L arm 7 | L gripper 2 (SLIDE) | R arm 7 | R gripper 2,
    so a flat write puts a shoulder angle in metres onto a gripper slide and
    throws the jaw metres away, and shifts the whole right arm by two slots.
    """
    idx = []
    for name in BODY_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"{name} not in {XML}")
        idx.append(int(model.jnt_qposadr[jid]))
    return np.array(idx)


def torso_pitch(model, data):
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    return float(np.degrees(np.arcsin(np.clip(-data.xmat[tid].reshape(3, 3)[2, 0], -1, 1))))


def load_reference():
    ji = np.loadtxt(REF / "joint_pos.csv", delimiter=",", skiprows=1)
    bq = np.loadtxt(REF / "body_quat.csv", delimiter=",", skiprows=1)
    bp = np.loadtxt(REF / "body_pos.csv", delimiter=",", skiprows=1)
    return [(bp[f, 2], bq[f, 0:4], isaac_to_mujoco(ji[f])) for f in range(len(ji))]


def load_delivery(path, n_frames):
    """Resample a trace onto the reference's 50 Hz grid via play_elapsed."""
    tr = json.loads(Path(path).read_text())["trace"]
    t = np.array([v["play_elapsed"] for v in tr])
    out = []
    for f in range(n_frames):
        i = int(np.abs(t - f / FPS).argmin())
        st = tr[i]["state"]
        out.append((st["pelvis"]["pos"][2], np.array(st["pelvis"]["quat"]), np.array(st["joints"])))
    return out


def main():
    panels = [
        ("REFERENCE  (what the motion asks)", None, (235, 235, 235)),
        ("policy/release  (delivered)", DIAG / "retry_reset_02/native_release_1.json", (170, 170, 255)),
        ("policy/sonic_v1_1  (delivered)", DIAG / "v1_1/native_release_1.json", (170, 255, 170)),
    ]
    model = mujoco.MjModel.from_xml_path(XML)
    ref = load_reference()
    n = len(ref)
    series = [ref] + [load_delivery(p, n) for _, p, _ in panels[1:]]

    model.mat_reflectance[:] = 0.0  # the mirrored floor doubles every ghost outline
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.azimuth = float(os.environ.get("CAM_AZ", 270))
    cam.elevation = float(os.environ.get("CAM_EL", -6))
    cam.distance = float(os.environ.get("CAM_D", 2.55))
    cam.lookat[:] = [SPAWN[0] + float(os.environ.get("CAM_DX", 0.14)), SPAWN[1], float(os.environ.get("CAM_Z", 0.66))]
    opt = mujoco.MjvOption()
    data = mujoco.MjData(model)
    qidx = body_qpos_index(model)

    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.46

    # Curves first: the whole motion at a glance beats eyeballing one frame.
    def series_angles(seq):
        waist, torso = [], []
        for z, quat, q in seq:
            mujoco.mj_resetData(model, data)
            data.qpos[0], data.qpos[1] = SPAWN
            data.qpos[2] = z
            data.qpos[3:7] = deyaw(quat)
            data.qpos[qidx] = q
            mujoco.mj_kinematics(model, data)
            waist.append(np.degrees(q[14]))
            torso.append(torso_pitch(model, data))
        return np.array(waist), np.array(torso)

    curves = [series_angles(seq) for seq in series]
    PH, PW = 190, W * 3 // 2
    colours = [(235, 235, 235), (170, 170, 255), (170, 255, 170)]

    def plot(which, title, lo, hi):
        c = np.full((PH, PW, 3), 24, np.uint8)
        cv2.putText(c, title, (12, 20), font, 0.46, (210, 210, 210), 1, cv2.LINE_AA)
        for v in (0, 10, 20, 30, 40):
            if not lo <= v <= hi:
                continue
            y = int(PH - 18 - (v - lo) / (hi - lo) * (PH - 46))
            cv2.line(c, (44, y), (PW - 8, y), (52, 52, 52), 1)
            cv2.putText(c, f"{v:3d}", (8, y + 4), font, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
        for k in range(3):
            y = curves[k][which]
            pts = np.array(
                [
                    [
                        44 + int(i / (n - 1) * (PW - 52)),
                        int(PH - 18 - (val - lo) / (hi - lo) * (PH - 46)),
                    ]
                    for i, val in enumerate(y)
                ],
                np.int32,
            )
            cv2.polylines(c, [pts], False, colours[k], 1, cv2.LINE_AA)
        return c

    plots = np.hstack([plot(0, "waist pitch (deg)", -6, 34), plot(1, "torso pitch (deg)", -10, 44)])

    OUT.mkdir(parents=True, exist_ok=True)
    frames_dir = OUT / "frames"
    frames_dir.mkdir(exist_ok=True)
    only = int(os.environ.get("ONLY_FRAME", -1))
    todo = [only] if only >= 0 else range(n)

    for f in todo:
        tiles = []
        for k, (label, _, colour) in enumerate(panels):
            z, quat, q = series[k][f]
            mujoco.mj_resetData(model, data)
            data.qpos[0], data.qpos[1] = SPAWN
            data.qpos[2] = z
            data.qpos[3:7] = deyaw(quat)
            data.qpos[qidx] = q
            mujoco.mj_kinematics(model, data)
            renderer.update_scene(data, camera=cam, scene_option=opt)
            img = renderer.render()[:, :, ::-1].copy()
            if k == 0:
                ghost = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 60, 160) > 0
            else:
                # The panels share a camera and a de-yawed root, so the reference
                # outline lands where the reference limb would be. The gap between
                # the outline and the robot IS the undelivered motion.
                img[ghost] = (0.30 * img[ghost] + 0.70 * np.array([0, 170, 255])).astype(np.uint8)
            bar = np.full((BAR, W, 3), 28, np.uint8)
            cv2.putText(bar, label, (10, 24), font, fs, colour, 1, cv2.LINE_AA)
            cv2.putText(
                bar,
                f"torso {torso_pitch(model, data):5.1f}   waist {np.degrees(q[14]):5.1f}   pelvis z {z:.3f}",
                (10, 50),
                font,
                fs,
                (215, 215, 215),
                1,
                cv2.LINE_AA,
            )
            tile = np.vstack([bar, img])
            cv2.rectangle(tile, (0, 0), (W - 1, BAR + H - 1), (70, 70, 70), 1)
            tiles.append(tile)
        sheet = np.hstack(tiles)
        board = plots.copy()
        for half in (0, PW):
            x = half + 44 + int(f / (n - 1) * (PW - 52))
            cv2.line(board, (x, 26), (x, PH - 18), (90, 90, 90), 1)
        strip = np.full((30, sheet.shape[1], 3), 20, np.uint8)
        cv2.putText(
            strip,
            f"squat_001__A359  frame {f:3d}/{n - 1}  t={f / FPS:5.2f}s   "
            "orange outline = reference pose   |   curves: white = reference, "
            "pink = release, green = v1_1",
            (10, 20),
            font,
            0.42,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(frames_dir / f"{f:04d}.png"), np.vstack([sheet, board, strip]))

    if only >= 0:
        print(frames_dir / f"{only:04d}.png")
        return 0
    mp4 = OUT / "reference_vs_release_vs_v1_1.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(frames_dir / "%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(mp4),
        ],
        check=True,
    )
    print(mp4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
