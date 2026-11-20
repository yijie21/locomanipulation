"""The frames the pipeline works in, and the G1's (29 DoF, Dex3 hands) head camera and palm.

base frame    origin at the pelvis, x the robot's heading made level, z straight up (against gravity); metres.
              Every point, height and pose in and out of the pipeline is in this frame.
camera frame  OpenCV's optical frame of the colour image: x right, y down, z forward.

The numbers below are from Unitree's g1_29dof_with_hand.urdf. The URDF places the camera's body (d435_link), not its
colour sensor: check the result with tools/check_camera.py and correct it with `offset`.
"""

import numpy as np


def rot(axis, a: float) -> np.ndarray:
    """The rotation by `a` rad about the unit `axis`."""
    k = np.asarray(axis, float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def rpy(r: float, p: float, y: float) -> np.ndarray:
    """URDF's roll-pitch-yaw: about the fixed x, then y, then z."""
    return rot((0, 0, 1), y) @ rot((0, 1, 0), p) @ rot((1, 0, 0), r)


def pose(R=None, p=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def quat_wxyz(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def level(R: np.ndarray) -> np.ndarray:
    """R with its heading taken out: the pelvis's tilt in the base frame."""
    return rot((0, 0, 1), -np.arctan2(R[1, 0], R[0, 0])) @ R


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Each pixel's point in the camera frame, H x W x 3, from the depth along the optical axis (m)."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float64)
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=2)


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def project(T_base_cam: np.ndarray, K: np.ndarray, pts) -> np.ndarray:
    """Points in the base frame to pixels (u, v); NaN behind the camera."""
    c = apply(np.linalg.inv(T_base_cam), np.atleast_2d(np.asarray(pts, float)))
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2], K[1, 1] * c[:, 1] / c[:, 2] + K[1, 2]], axis=1)
    uv[c[:, 2] <= 0] = np.nan
    return uv


# ---- the G1

WAIST = (  # the joints from the pelvis to torso_link: name, origin in the parent (m), axis
    ("waist_yaw_joint", (0.0, 0.0, 0.0), (0, 0, 1)),
    ("waist_roll_joint", (-0.0039635, 0.0, 0.035), (1, 0, 0)),
    ("waist_pitch_joint", (0.0, 0.0, 0.019), (0, 1, 0)),
)
D435_IN_TORSO = ((0.0576235, 0.01753, 0.41987), (0.0, 0.8307767239493009, 0.0))  # d435_joint: xyz, rpy (47.6 deg down)
OPTICAL_IN_LINK = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # optical axes in a camera link (x fwd, z up)


def head_camera(waist=(0.0, 0.0, 0.0), imu_wxyz=None, offset=(0.0, 0.0, 0.0)) -> np.ndarray:
    """T_base_cam of the head D435's colour image.

    waist     the waist joints' readings (rad): yaw, roll, pitch (0 0 0 for a robot with a locked waist)
    imu_wxyz  the pelvis IMU's orientation; its heading is dropped (the base frame turns with the robot). None: upright
    offset    the colour sensor in d435_link (m; x forward, y left, z up), from your own calibration
    """
    T = pose(level(quat_wxyz(imu_wxyz)) if imu_wxyz is not None else None)
    for (_, xyz, axis), q in zip(WAIST, waist):
        T = T @ pose(rot(axis, q), xyz)
    return T @ pose(rpy(*D435_IN_TORSO[1]), D435_IN_TORSO[0]) @ pose(OPTICAL_IN_LINK, offset)


PALM_IN_WRIST = (0.0415, -0.003, 0.0)  # right_hand_palm_joint: right_hand_palm_link in right_wrist_yaw_link (fixed)


def wrist_of(T_base_palm: np.ndarray) -> np.ndarray:
    """right_wrist_yaw_link's pose for a right_hand_palm_link pose (the arm's IK usually takes the wrist)."""
    return T_base_palm @ np.linalg.inv(pose(p=PALM_IN_WRIST))


DEX3_RIGHT = (  # the right Dex3's joints and their limits (rad) in the URDF
    ("right_hand_thumb_0_joint", -1.04719755, 1.04719755),
    ("right_hand_thumb_1_joint", -1.04719755, 0.72431163),
    ("right_hand_thumb_2_joint", -1.74532925, 0.0),
    ("right_hand_index_0_joint", 0.0, 1.57079632),
    ("right_hand_index_1_joint", 0.0, 1.74532925),
    ("right_hand_middle_0_joint", 0.0, 1.57079632),
    ("right_hand_middle_1_joint", 0.0, 1.74532925),
)
