"""plan_episode's spawn distance in a scene_discovery room: the body must fit, and the hands must clear every surface."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import plan_episode  # noqa: E402


class OpenFloor:
    """An occupancy map with nothing on it (the surfaces are checked by their edges, not the map)."""

    res = 0.05
    occupied = np.zeros((1, 1), dtype=bool)

    def is_free(self, xy):
        return True

    def cell_of(self, xy):
        return (0, 0)

    def in_bounds(self, c):
        return True


def meta(side_table_x):
    return {
        "surfaces": {
            "T": {"top": 0.70, "xy": [0.0, 0.0], "half_extent": [0.2, 0.15]},
            # a side table beside the spawn line, 1.05-1.55 m behind the stand
            "S": {"top": 0.52, "xy": [side_table_x, -1.75], "half_extent": [0.25, 0.25]},
        }
    }


def test_spawn_keeps_the_hands_clear_of_a_table_beside_it():
    stand, fv2 = np.array([0.0, -0.45]), np.array([0.0, 1.0])  # facing +y, spawn is behind (-y)
    far = meta(1.5)
    assert plan_episode.clear_spawn_off(far, stand, fv2, 1.2, _maps={id(far): OpenFloor()}) == 1.2
    # 0.30 m beside the table: 1.2 / 1.0 m behind the stand fit the body (0.26) but not the hands, 0.8 m is 0.39 m
    # from its corner; 0.6 m (0.54 m away) is the first that clears it
    near = meta(0.55)
    assert plan_episode.clear_spawn_off(near, stand, fv2, 1.2, _maps={id(near): OpenFloor()}) == 0.6
