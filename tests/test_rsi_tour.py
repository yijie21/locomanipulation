"""Vision RSI D2: the mapping tour's planner (locomani.tasks.scene.tour) -- every surface of the room is seen from its
viewing range with a clear line of sight, a surface behind a wall is not, and the route is a closed loop."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from locomani.primitives.navigator import build  # noqa: E402
from locomani.tasks.scene import tour as T  # noqa: E402


def surface(xy, he, top=0.75, room="r", yaw=0.0):
    return dict(xy=list(xy), half_extent=list(he), top=top, yaw=yaw, room=room, area=4 * he[0] * he[1])


# a 6 x 6 m room: a table near each of two corners, a wall boxing in a third table with no way around it, and a
# shelf in another room
META = dict(
    room=dict(bounds=dict(x=[0.0, 6.0], y=[0.0, 6.0]), walls=[[[4.0, 0.0], [4.0, 1.6]], [[4.0, 1.6], [6.0, 1.6]]]),
    surfaces=dict(
        a=surface((1.0, 4.8), (0.4, 0.3)),
        b=surface((4.8, 4.8), (0.3, 0.3), top=0.45),
        hidden=surface((5.0, 0.6), (0.3, 0.2)),
        other=surface((3.0, 3.0), (0.2, 0.2), room="elsewhere"),
    ),
)


class Plan(unittest.TestCase):
    def setUp(self):
        self.occ, self.nav = build(META, body_radius=0.35)
        self.start = np.array([1.5, 1.5])

    def test_every_visible_surface_is_covered_from_its_range(self):
        vps, s = T.plan(META, self.occ, self.nav, self.start, T.TourConfig(room="r", must=("a",)))
        covered = {k for v in vps for k in v.surfaces}
        self.assertEqual(covered, {"a", "b"})
        self.assertEqual(s["unseen"], ["hidden"])  # boxed in by the wall, out of range from every free cell
        self.assertEqual(s["must_unseen"], [])
        for v in vps:
            self.assertTrue(self.occ.free[tuple(self.occ.cell_of(v.xy))])
            for k in v.surfaces:
                d = np.linalg.norm(T.nearest_on_box(v.xy, META["surfaces"][k]) - v.xy)
                self.assertTrue(0.6 <= d <= 1.5, (k, d))

    def test_the_viewpoint_faces_its_surfaces(self):
        vps, _ = T.plan(META, self.occ, self.nav, self.start, T.TourConfig(room="r"))
        for v in vps:
            p = np.mean([T.nearest_on_box(v.xy, META["surfaces"][k]) for k in v.surfaces], axis=0)
            want = np.arctan2(*(p - v.xy)[::-1])
            self.assertAlmostEqual(np.cos(v.yaw - want), 1.0, places=6)

    def test_the_route_is_short_and_closed(self):
        pts = [np.array([5.0, 3.0]), np.array([1.0, 3.0]), np.array([3.0, 5.0])]
        route = T.order(self.nav, self.start, pts)
        self.assertEqual(sorted(route), [0, 1, 2])
        self.assertIn(route, ([1, 2, 0], [0, 2, 1]))  # around the room, not across it

    def test_nearest_point_on_a_turned_box(self):
        s = surface((0.0, 0.0), (1.0, 0.5), yaw=np.pi / 2)  # long along y
        np.testing.assert_allclose(T.nearest_on_box([0.0, 3.0], s), [0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(T.nearest_on_box([2.0, 0.0], s), [0.5, 0.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
