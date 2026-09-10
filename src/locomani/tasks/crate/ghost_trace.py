"""Per-tick record of the bimanual IK targets, for the ghost hands in rendered videos.

`BimanualArms.solve` asks the IK for a grip point `targets_w[side]` (pad centre,
world) with palm rotation `rots_w[side]`; the palm frame is `<side>_wrist_yaw_link`
and the grip point sits `OFFSET` in front of it (bimanual_carry.OFFSET). The wrist
pose the IK is aiming for is therefore (target - R @ OFFSET, R). One row per solve
call, keyed by the privileged `seq` of the observation it was solved against, so a
render can put the ghost on the exact recorded state (StateTrace rows carry the
same seq). Written next to the state trace as `<label>_ghost.npz`:

    seq  [N]        pos [N, 2, 3] (left, right; world)    quat [N, 2, 4] (wxyz)
"""

from __future__ import annotations

import os

import numpy as np

SIDES = ("left", "right")
PARK = np.array([0.0, 0.0, -5.0])


def _mat2quat(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


class GhostTargetTrace:
    def __init__(self, out_dir: str, offset):
        self.dir = out_dir
        self.offset = np.asarray(offset, dtype=float)
        self._label = None
        self._rows: list = []
        os.makedirs(out_dir, exist_ok=True)

    def start(self, label: str) -> None:
        self.stop()
        self._label, self._rows = label, []

    def row(self, seq, targets_w: dict, rots_w: dict) -> None:
        if self._label is None or seq is None:
            return
        pos = np.zeros((2, 3))
        quat = np.zeros((2, 4))
        for i, sd in enumerate(SIDES):
            R = np.asarray(rots_w[sd], dtype=float)
            pos[i] = np.asarray(targets_w[sd], dtype=float) - R @ self.offset
            quat[i] = _mat2quat(R)
        self._rows.append((int(seq), pos, quat))

    def stop(self) -> str | None:
        if self._label is None:
            return None
        path = os.path.join(self.dir, f"{self._label}_ghost.npz")
        rows = self._rows
        self._label, self._rows = None, []
        if not rows:
            return None
        np.savez_compressed(
            path,
            seq=np.asarray([r[0] for r in rows], dtype=np.int64),
            pos=np.stack([r[1] for r in rows]),
            quat=np.stack([r[2] for r in rows]),
        )
        print(f"[ghost] wrote {path} ({len(rows)} rows)")
        return path


def load_ghost_trace(path: str) -> dict:
    """{seq: (pos [2,3], quat [2,4])}; the last row wins for a repeated seq."""
    z = np.load(path)
    return {int(s): (z["pos"][i], z["quat"][i]) for i, s in enumerate(z["seq"])}


def ghost_poses(table: dict, seq: int, tolerance: int = 3):
    """Ghost wrist poses for a recorded state: the row at `seq`, else the nearest
    within `tolerance` sequence numbers (the expert solves at ~50 Hz against
    the latest state, so a frame or two apart is the same intent), else parked."""
    for d in range(tolerance + 1):
        for s in (seq - d, seq + d):
            if s in table:
                return table[s]
    park = np.stack([PARK, PARK])
    ident = np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    return park, ident
