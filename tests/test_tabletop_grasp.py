"""Extracted tabletop strategy matches the previous candidate and live replan."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/teleop_free'))
import adaptive_grasp
from primitives import Primitives
from tabletop_grasp import TabletopTarget, TabletopGrasp


class TabletopTests(unittest.TestCase):
    def test_initial_and_live_plan_match_legacy_geometry_for_both_sides(self):
        st = json.loads((Path(__file__).parent/'fixtures/crouched_pregrasp.json').read_text())
        p = SimpleNamespace(obs=lambda: st,
            object_pos=lambda key, s: np.array(s['objects'][key]['pos']),
            pelvis_xy=lambda s: np.array(s['pelvis']['pos'][:2]),
            pelvis_yaw=lambda s: s['pelvis']['yaw'])
        for side in ['right', 'left']:
            target = TabletopTarget('water_bottle', side, .72, .6214,
                                    {'half_extent': [.02265, .02265, .08]})
            strategy = TabletopGrasp(p, target)
            # Reference formula from grasp_gym before extraction, using the
            # captured pose, then a changed live target position.
            for initial in [True, False]:
                if not initial:
                    st['objects']['water_bottle']['pos'][0] += .015
                grip = strategy.grip(st)
                d = grip[:2] - p.pelvis_xy(st)
                d /= max(np.linalg.norm(d), 1e-6)
                wrist = (grip - .25*np.array([*d, 0.])+[0., 0., .08]
                         if initial else np.array(st['palms'][side]['pos']))
                rot = Primitives._rot(st['palms'][side]['quat'])
                yaw = p.pelvis_yaw(st)
                lat = -.31 if side == 'right' else .31
                pos = st['pelvis']['pos']
                reference = adaptive_grasp.plan(grip, wrist, rot, target.surface_top,
                    box=target.box, obj_rot=Primitives._rot(st['objects'][target.key]['quat']),
                    azimuth_from=np.array([pos[0]-np.sin(yaw)*lat,
                                           pos[1]+np.cos(yaw)*lat, pos[2]+.44]),
                    avoid_dir=np.array([*d, 0.]), avoid_sign=1. if side == 'right' else -1.)
                actual = strategy.initial(st) if initial else strategy.replan(wrist, rot)
                self.assertEqual(actual.ok, reference.ok)
                self.assertEqual(actual.why, reference.why)
                if actual.ok:
                    np.testing.assert_allclose(actual.wrist, reference.wrist, atol=1e-12)
                    np.testing.assert_allclose(actual.rot, reference.rot, atol=1e-12)


if __name__ == '__main__': unittest.main()

class LocalGraspPointTests(unittest.TestCase):
    def test_lower_point_tracks_object_translation_and_rotation(self):
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_euler('xyz', [.1, .2, .6])
        state = {'objects': {'bottle': {'pos': [1., 2., .7],
                 'quat': rot.as_quat()[[3, 0, 1, 2]].tolist()}}}
        robot = SimpleNamespace(object_pos=lambda key, s: np.asarray(s['objects'][key]['pos']))
        task = TabletopGrasp(robot, TabletopTarget('bottle', 'right', .99, .6,
            {'half_extent': [.025, .025, .08], 'center': [.002, 0., .003]}, below_center=.01))
        expected = np.array([1., 2., .7]) + rot.apply([.002, 0., -.007])
        np.testing.assert_allclose(task.grip(state), expected)
        state['objects']['bottle']['pos'][2] += .05
        np.testing.assert_allclose(task.grip(state), expected + [0., 0., .05])

    def test_grip_cannot_be_below_object_bottom(self):
        for offset in (.08, -.01, float('nan')):
            with self.assertRaises(ValueError):
                TabletopTarget('bottle', 'right', .7, .6,
                    {'half_extent': [.02, .02, .08]}, below_center=offset)
