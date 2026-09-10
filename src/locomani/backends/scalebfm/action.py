"""Native 29-D ScaleBFM command in the current pelvis YAW frame.

Layout: pelvis pose9, right wrist pose9, pelvis linear velocity3/yaw rate1,
right wrist linear velocity3/angular velocity3, gripper1. Rot6d stores two
matrix columns consecutively. Velocities are native planned world velocities,
rotated into the row's yaw frame, not differences of moving-frame poses.

Both collection and deployment must call decode_action. It constructs the same
six-frame linear/angular forecast without future measured root/object states.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

ACTION_DIM = 29


def heading(quaternion):
    rotation = Rotation.from_quat(quaternion, scalar_first=True)
    matrix = rotation.as_matrix()
    return Rotation.from_euler("z", np.arctan2(matrix[1, 0], matrix[0, 0]))


def rotation6(rotation):
    return rotation.as_matrix()[:, :2].T.reshape(6)


def from_rotation6(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Rotation must be a finite 6-vector")
    x = values[:3]
    if np.linalg.norm(x) < 1e-8:
        raise ValueError("Degenerate rotation first column")
    x = x / np.linalg.norm(x)
    y = values[3:] - np.dot(x, values[3:]) * x
    if np.linalg.norm(y) < 1e-8:
        raise ValueError("Degenerate rotation second column")
    y = y / np.linalg.norm(y)
    return Rotation.from_matrix(np.column_stack([x, y, np.cross(x, y)]))


def encode_action(targets, root_position, root_quaternion, gripper, dt=0.02):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    root = np.asarray(root_position)
    frame = heading(root_quaternion)
    inverse = frame.inv()
    row = np.zeros(ACTION_DIM)
    for name, offset, velocity_offset in (("pelvis", 0, 18), ("right_wrist_yaw_link", 9, 22)):
        positions, quaternions = map(np.asarray, targets[name])
        if len(positions) < 2 or len(quaternions) < 2:
            raise ValueError("Native velocity requires at least two planned future targets")
        rotations = Rotation.from_quat(quaternions[:2], scalar_first=True)
        row[offset : offset + 3] = inverse.apply(positions[0] - root)
        row[offset + 3 : offset + 9] = rotation6(inverse * rotations[0])
        row[velocity_offset : velocity_offset + 3] = inverse.apply((positions[1] - positions[0]) / dt)
        omega = (rotations[1] * rotations[0].inv()).as_rotvec() / dt
        if name == "pelvis":
            row[21] = omega[2]
        else:
            row[25:28] = inverse.apply(omega)
    row[28] = gripper
    if not np.isfinite(row).all():
        raise ValueError("Nonfinite action")
    return row


def decode_action(row, root_position, root_quaternion, hand_rel, dt=0.02, future_count=6):
    row = np.asarray(row, dtype=float)
    if row.shape != (ACTION_DIM,) or not np.isfinite(row).all():
        raise ValueError("Action must be a finite 29-vector")
    if dt <= 0 or future_count < 1:
        raise ValueError("Invalid future grid")
    root = np.asarray(root_position)
    frame = heading(root_quaternion)
    tau = np.arange(future_count) * dt
    result = {}
    for name, offset, velocity_offset in (("pelvis", 0, 18), ("right_wrist_yaw_link", 9, 22)):
        p0 = root + frame.apply(row[offset : offset + 3])
        r0 = frame * from_rotation6(row[offset + 3 : offset + 9])
        velocity = frame.apply(row[velocity_offset : velocity_offset + 3])
        omega = np.array([0, 0, row[21]]) if name == "pelvis" else frame.apply(row[25:28])
        p = p0 + tau[:, None] * velocity
        rotations = Rotation.from_rotvec(tau[:, None] * omega) * r0
        result[name] = p, rotations.as_quat(scalar_first=True)
    p, q = result["pelvis"]
    left_pos, left_matrix = hand_rel["left"]
    rotations = Rotation.from_quat(q, scalar_first=True)
    result["left_wrist_yaw_link"] = (
        p + rotations.apply(left_pos),
        (rotations * Rotation.from_matrix(left_matrix)).as_quat(scalar_first=True),
    )
    return result, float(row[28])
