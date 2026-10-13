"""wbc_compare step 2 (TODO 24.2): the open-loop hand probe's cells, shared by both controllers.

Decision 2 of the comparison (docs/handoff/wbc-compare-20261012.md §1): each controller gets one feed-forward constant
bias measured by an open-loop probe at the 65-object line's stance, and the same SONIC-line servo on top. The probe
commands the right Dex1's tool (grip centre, OFFSET 0.1515 m along the palm x from the wrist) to a grid of world poses
around where the cached grasps put it, holds, and reads the delivered tool back. The grid is anchored on the stance
the hand program stands at (grasp_planned STAND_FORWARD 0.38 / STAND_RIGHT 0.12 along the heading) and spans the
chosen grasps of the 2026-10-10 gate batches (114 records: tool 0.35-0.38 m ahead of the pelvis, 0.12-0.16 m to its
right, z 0.65-0.77 m; approach forward in 83 %, jaw horizontal):

    forward 0.30 / 0.38 / 0.46 m from the stance point along the heading
    left   -0.20 / -0.12 / -0.04 m (the object sits at -0.12)
    z       0.67 / 0.74 / 0.81 m world (table top 0.62; the lowest cell leaves 5 cm under the tool)

27 cells; the nine at left -0.12 are the object's own lateral (CENTRAL), the decision's "about nine". The palm
attitude is the same in every cell: the approach (palm x) along the heading pitched PITCH_DOWN_DEG down, the jaw
(palm y) horizontal to the left, palm z up-forward. 30 deg down is the median approach elevation of the grasps the
gate batches chose (q25-q90 all 29-30 deg; a third are level); the first probe run commanded a level approach and
the program's IK (StreamedArm waist limits, shoulder roll cap) could not bring the tool down to the 0.67 row at
all (15/27 cells solvable offline, 25/27 at 30 deg down). Errors are reported in the heading frame (forward, left,
up) at the *commanded* stance yaw, so both controllers' numbers live in the frame the hand program plans in.
Between cells the tool rises to OVER_CLEAR above the table top, returns to HOME (over the central column), travels
and descends -- grasp_planned's staged path from one common start, so one unreachable cell cannot drag the next
(the first run lost 22 cells to one IK branch jump). The object is parked PARK_BEHIND behind the stance on the
floor for the whole probe: every cell overlaps where it stands.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OFFSET = np.array([0.1515, 0.0, 0.0])  # tool (grip centre) in the wrist frame, dex1 (grasp_cache.py OFFSET)
FORWARD = (0.30, 0.38, 0.46)
LEFT = (-0.20, -0.12, -0.04)
Z = (0.67, 0.74, 0.81)
CENTRAL_LEFT = -0.12
PITCH_DOWN_DEG = 30.0  # approach elevation below the horizontal (the chosen grasps' median)
OVER_CLEAR = 0.22  # grasp_planned _plan_ik: the travel height above the table top
HOME = (0.38, -0.12)  # forward, left of the start/return point between cells, at the travel height
PARK_BEHIND = 1.5  # m behind the stance, on the floor
TABLE_TOP = 0.62
IK_POS_TOL, IK_ROT_TOL_DEG = 0.015, 10.0  # grasp_planned: a path ending outside this is "no path"


def cells() -> list[dict]:
    out = []
    for z in Z:
        for left in LEFT:
            for fwd in FORWARD:
                out.append(dict(id=len(out), forward=fwd, left=left, z=z, central=left == CENTRAL_LEFT))
    return out


def heading_frame(yaw: float) -> np.ndarray:
    """Rows: forward, left, up unit vectors in the world for the stance yaw."""
    return np.array([[np.cos(yaw), np.sin(yaw), 0.0], [-np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])


def palm_rotation(yaw: float, pitch_down_deg: float = PITCH_DOWN_DEG) -> np.ndarray:
    """The probe attitude in the world: approach along the heading pitched down, jaw to the left (columns x, y, z)."""
    p = np.radians(pitch_down_deg)
    in_heading = np.array([[np.cos(p), 0.0, np.sin(p)], [0.0, 1.0, 0.0], [-np.sin(p), 0.0, np.cos(p)]])
    return heading_frame(yaw).T @ in_heading


def tool_pose(stance_xy, yaw: float, cell: dict) -> tuple[np.ndarray, np.ndarray]:
    """The cell's commanded tool position (world) and palm rotation (columns: approach, jaw, palm normal)."""
    H = heading_frame(yaw)
    pos = np.array([stance_xy[0], stance_xy[1], 0.0]) + cell["forward"] * H[0] + cell["left"] * H[1]
    pos[2] = cell["z"]
    return pos, palm_rotation(yaw)


def home_tool(stance_xy, yaw: float, table_top: float = TABLE_TOP) -> np.ndarray:
    H = heading_frame(yaw)
    pos = np.array([stance_xy[0], stance_xy[1], 0.0]) + HOME[0] * H[0] + HOME[1] * H[1]
    pos[2] = table_top + OVER_CLEAR
    return pos


def to_heading(vec, yaw: float) -> list[float]:
    return [float(v) for v in heading_frame(yaw) @ np.asarray(vec, dtype=float)]


def park_position(stance_xy, yaw: float) -> np.ndarray:
    H = heading_frame(yaw)
    return np.array([stance_xy[0], stance_xy[1], 0.08]) - PARK_BEHIND * H[0]


def legs(tool_from: np.ndarray, tool_to: np.ndarray, table_top: float = TABLE_TOP) -> list[tuple]:
    """The staged tool path between two tool positions: up, over, down (grasp_planned's legs)."""
    over_z = max(table_top + OVER_CLEAR, tool_from[2], tool_to[2])
    up = tool_from.copy()
    up[2] = over_z
    over = tool_to.copy()
    over[2] = over_z
    return [(tool_from, up), (up, over), (over, tool_to)]


def legs_via_home(tool_from: np.ndarray, tool_to: np.ndarray, home: np.ndarray) -> list[tuple]:
    """Up from the last cell, back to HOME, over the next cell, down: every cell starts from the same point."""
    up = tool_from.copy()
    up[2] = home[2]
    over = tool_to.copy()
    over[2] = home[2]
    out = [(tool_from, up)] if np.linalg.norm(up - tool_from) > 1e-6 else []
    out += [(up, home)] if np.linalg.norm(home - up) > 1e-6 else []
    return out + [(home, over), (over, tool_to)]


def rot_err_deg(R_cmd: np.ndarray, R_meas: np.ndarray) -> float:
    c = (np.trace(R_cmd.T @ R_meas) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def wrist_from_tool(tool: np.ndarray, R: np.ndarray) -> np.ndarray:
    return tool - R @ OFFSET


# --- the attitude sweep (TODO 24.6, bfm_attitude_probe.py / attitude_summary.py / bfm_grasp.py --envelope) ------------
ELEVATIONS = (0, 15, 30, 45, 60, 75, 90)  # approach pitched below the horizontal, deg
YAW_DEVS = (-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180)  # approach turned left of the heading, deg
ROLLS = (0, 180)  # about the approach: 0 = jaw to the left, palm normal up (the 24.2 probe attitude); 180 = jaw right
ATTITUDE_POSITIONS = (  # the 24.2 cells' central column at the two heights the cached grasps sit at
    dict(id=0, forward=0.38, left=CENTRAL_LEFT, z=0.67),
    dict(id=1, forward=0.38, left=CENTRAL_LEFT, z=0.74),
)
GRID_MAX_DEG = 35.0  # a candidate farther than this from every probed attitude counts as unprobed


def _rx(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def attitude_in_heading(elev_deg: float, yaw_dev_deg: float, roll_deg: float) -> np.ndarray:
    """Palm rotation in the heading frame (columns approach, jaw, normal): turned about up, pitched down, rolled about
    the approach. (30, 0, 0) is palm_rotation's attitude."""
    return _rz(yaw_dev_deg) @ _ry(elev_deg) @ _rx(roll_deg)


def palm_rotation_at(yaw: float, elev_deg: float, yaw_dev_deg: float, roll_deg: float) -> np.ndarray:
    return heading_frame(yaw).T @ attitude_in_heading(elev_deg, yaw_dev_deg, roll_deg)


def attitudes() -> list[dict]:
    """The probed grid. At 90 deg elevation (yaw, roll 180) is the same rotation as (yaw + 180, roll 0), so only roll 0
    is listed there."""
    out = []
    for e in ELEVATIONS:
        for y in YAW_DEVS:
            for r in ROLLS:
                if e == 90 and r == 180:
                    continue
                out.append(dict(id=len(out), elev=e, yaw_dev=y, roll=r))
    return out


def attitude_of(R_world, yaw: float) -> tuple[float, float, float]:
    """(elevation, yaw deviation, roll) in degrees of a world palm rotation at the stance yaw: palm_rotation_at's
    inverse (yaw deviation is arbitrary at +-90 deg elevation; the roll then absorbs it)."""
    Rh = heading_frame(yaw) @ np.asarray(R_world, dtype=float)
    a = Rh[:, 0]
    elev = float(np.degrees(np.arcsin(np.clip(-a[2], -1.0, 1.0))))
    yaw_dev = float(np.degrees(np.arctan2(a[1], a[0]))) if np.hypot(a[0], a[1]) > 1e-9 else 0.0
    res = (_rz(yaw_dev) @ _ry(elev)).T @ Rh
    return elev, yaw_dev, float(np.degrees(np.arctan2(res[2, 1], res[1, 1])))


class Envelope:
    """probe/envelope.json (attitude_summary.py): which probed attitudes ScaleBFM delivered. A candidate grasp is
    judged by the nearest grid attitude (rotation distance), unprobed beyond GRID_MAX_DEG."""

    def __init__(self, path):
        self.path = Path(path)
        d = json.loads(self.path.read_text())
        self.thresholds = d["thresholds"]
        self.rows = d["attitudes"]
        self.R = np.array([attitude_in_heading(r["elev"], r["yaw_dev"], r["roll"]) for r in self.rows])

    def nearest(self, R_world, yaw: float) -> tuple[dict, float]:
        Rh = heading_frame(yaw) @ np.asarray(R_world, dtype=float)
        c = np.clip((np.einsum("nij,ij->n", self.R, Rh) - 1.0) / 2.0, -1.0, 1.0)  # trace(R_grid^T Rh)
        d = np.degrees(np.arccos(c))
        i = int(np.argmin(d))
        return self.rows[i], float(d[i])

    def reject(self, R_world, yaw: float) -> str | None:
        row, dist = self.nearest(R_world, yaw)
        tag = f"elev {row['elev']} yaw {row['yaw_dev']} roll {row['roll']}"
        if dist > GRID_MAX_DEG:
            return f"attitude unprobed ({dist:.0f} deg from the grid, nearest {tag})"
        if not row["deliverable"]:
            return (
                f"attitude outside the envelope ({tag}: {row.get('err_mean_cm', float('nan')):.1f} cm, "
                f"{row.get('rot_mean_deg', float('nan')):.0f} deg)"
            )
        return None
