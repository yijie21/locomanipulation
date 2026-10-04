"""Pure-geometry checks for the place phase (tasks/bottle/place_phase.py)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from locomani.tasks.bottle.config import BottleTaskConfig, PlaceConfig
from locomani.tasks.bottle.place_phase import (
    Landing,
    _pulse_to_band,
    _SurfaceGeometry,
    band_residual,
    ego_visible,
    in_container,
    level_rotation,
    quat_matrix,
    resting_on,
    squat_drop,
    step_reference,
    surface_band,
    tilt_from,
    turn_command,
    yaw_of_quat,
)

IDENT = [1.0, 0.0, 0.0, 0.0]
YAW90 = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]  # the built basket's yaw


class InContainerTests(unittest.TestCase):
    half, floor, rim = (0.12, 0.07), 0.63, 0.772

    def test_centre_inside(self):
        ok, why = in_container([0.0, 0.0, 0.68], [0.0, 0.0, 0.7], IDENT, self.half, self.floor, self.rim)
        self.assertTrue(ok, why)

    def test_yaw_rotates_the_footprint(self):
        # 0.10 m along world y is inside a yaw-90 basket (its long axis lies on y) and outside an unyawed one.
        ok, _ = in_container([0.0, 0.10, 0.68], [0.0, 0.0, 0.7], YAW90, self.half, self.floor, self.rim)
        self.assertTrue(ok)
        ok, why = in_container([0.0, 0.10, 0.68], [0.0, 0.0, 0.7], IDENT, self.half, self.floor, self.rim)
        self.assertFalse(ok)
        self.assertIn("footprint", why)

    def test_resting_on_the_rim_is_not_inside(self):
        ok, why = in_container([0.0, 0.0, 0.80], [0.0, 0.0, 0.7], IDENT, self.half, self.floor, self.rim)
        self.assertFalse(ok)
        self.assertIn("rim", why)

    def test_yaw_of_quat(self):
        self.assertAlmostEqual(yaw_of_quat(YAW90), np.pi / 2, places=6)
        self.assertAlmostEqual(yaw_of_quat(IDENT), 0.0, places=6)


class TurnCommandTests(unittest.TestCase):
    band_x, band_y = (0.26, 0.44), (-0.16, -0.06)

    def test_in_band_stops(self):
        in_band, mv, face, speed = turn_command(0.30, -0.11, 0.3, self.band_x, self.band_y)
        self.assertTrue(in_band)
        self.assertEqual(mv, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(np.arctan2(face[1], face[0]), 0.3)
        self.assertEqual(speed, 0.0)

    def test_goal_to_the_left_turns_left_without_walking(self):
        # Goal ahead-left (x in band): the facing rotates left (positive yaw), no translation.
        in_band, mv, face, speed = turn_command(0.30, 0.20, 0.0, self.band_x, self.band_y)
        self.assertFalse(in_band)
        self.assertGreater(np.arctan2(face[1], face[0]), 0.0)
        self.assertEqual(mv, (0.0, 0.0, 0.0))
        self.assertGreater(speed, 0.0)

    def test_goal_too_far_walks_forward_along_the_facing(self):
        in_band, mv, face, _ = turn_command(0.55, -0.11, 0.0, self.band_x, self.band_y)
        self.assertFalse(in_band)
        self.assertGreater(mv[0] * face[0] + mv[1] * face[1], 0.9)

    def test_goal_too_close_backs_up(self):
        _, mv, face, _ = turn_command(0.20, -0.11, 0.0, self.band_x, self.band_y)
        self.assertLess(mv[0] * face[0] + mv[1] * face[1], -0.9)

    def test_facing_step_is_bounded(self):
        _, _, face, _ = turn_command(0.30, 0.60, 0.0, self.band_x, self.band_y)
        self.assertLessEqual(abs(np.arctan2(face[1], face[0])), 0.35 + 1e-9)


class ReferenceTests(unittest.TestCase):
    def test_bounded_speed_then_arrival(self):
        ref = np.zeros(3)
        steps = 0
        while not np.allclose(ref, [0.0, 0.0, 0.10]):
            ref = step_reference(ref, [0.0, 0.0, 0.10], 0.10, 0.02)
            steps += 1
            self.assertLessEqual(ref[2], 0.10 + 1e-12)
        self.assertEqual(steps, 50)  # 0.10 m at 0.10 m/s in 0.02 s ticks

    def test_ego_gate(self):
        self.assertTrue(ego_visible([0.5, -0.5]))
        self.assertFalse(ego_visible([0.9, 0.0]))
        self.assertFalse(ego_visible(None))


class SurfacePutDownTests(unittest.TestCase):
    """The surface variant's pure pieces (put_down_single, docs/handoff/put-down-single-20261003.md)."""

    def test_resting_on(self):
        on = dict(surf_xy=[0.0, 1.0], surf_half=[0.25, 0.25], surf_yaw=-np.pi / 2)
        self.assertEqual(resting_on([0.0, 0.87, 0.62], 0.618, 3.0, **on), (True, ""))
        self.assertIn("above its resting", resting_on([0.0, 0.87, 0.70], 0.618, 0.0, **on)[1])
        self.assertIn("fell off", resting_on([0.0, 0.87, 0.30], 0.618, 0.0, **on)[1])
        self.assertIn("outside the surface", resting_on([0.0, 0.70, 0.62], 0.618, 0.0, **on)[1])
        self.assertIn("tipped 40", resting_on([0.0, 0.87, 0.62], 0.618, 40.0, **on)[1])

    def test_tilt_from_the_resting_up_axis(self):
        self.assertAlmostEqual(tilt_from([1, 0, 0, 0], [0, 0, 1]), 0.0)
        q = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]  # 90 deg about x: body +y points up
        self.assertAlmostEqual(tilt_from(q, [0, 1, 0]), 0.0, places=6)
        self.assertAlmostEqual(tilt_from(q, [0, 0, 1]), 90.0, places=6)

    def test_squat_drop_only_when_the_release_is_below_the_standing_reach(self):
        # palm 4 cm over the object; release 3.2 cm lower puts the palm at 0.668, 12.2 cm under a 0.79 pelvis
        self.assertAlmostEqual(squat_drop(0.70, 0.66, 0.628, 0.79), 0.122 + 0.03)
        self.assertEqual(squat_drop(0.83, 0.80, 0.80, 0.79), 0.0)
        self.assertEqual(squat_drop(0.40, 0.66, 0.30, 0.79), 0.20)  # capped at the calibrated squat

    def test_landing(self):
        land = Landing("can", 0.618)

        def at(z, t):
            return {"t": t, "objects": {"can": {"pos": [0.0, 0.0, z]}}}

        self.assertFalse(land(at(0.70, 0.0)))
        self.assertTrue(land(at(0.627, 0.02)))
        self.assertEqual(land.how, "at_rest_height")
        land = Landing("can", 0.618)  # stopped 2 cm high: resting on the top higher than the box says
        self.assertFalse(land(at(0.640, 0.0)))
        self.assertFalse(land(at(0.6395, 0.2)))
        self.assertTrue(land(at(0.6392, 0.35)))
        self.assertEqual(land.how, "stopped_sinking")
        land = Landing("can", 0.618)  # still sinking 5 mm a tick: not landed
        self.assertFalse(any(land(at(0.70 - 0.005 * k, 0.02 * k)) for k in range(10)))

    def test_surface_geometry(self):
        def ctx(init_quat):
            box = {"half_extent": [0.041, 0.0615, 0.098], "centre": [0.0, 0.0, 0.0]}
            return SimpleNamespace(
                place_config=PlaceConfig(),
                obj_key="can",
                surface={"key": "round", "top": 0.52, "xy": [0.0, 1.0], "half_extent": [0.25, 0.25], "yaw": 0.0},
                spot=[0.0, 0.87],
                drop_m=0.01,
                meta={"objects": {"can": {"box": box, "height": 0.196, "init_quat": init_quat}}},
            )

        s = {"objects": {"can": {"pos": [0.1, 0.6, 0.90]}}}
        g = _SurfaceGeometry(ctx([1.0, 0.0, 0.0, 0.0]), s)
        self.assertAlmostEqual(g.rest_z, 0.618)
        self.assertAlmostEqual(g.release_z, 0.628)
        # carried at 0.90: no raise, over the spot at that height, then squat, then straight down
        self.assertEqual((g.raise_z, g.hover_z), (None, 0.90))
        self.assertAlmostEqual(g.descend_timeout(PlaceConfig(), 0.40), 0.40 / 0.05 + 2.0)
        low = _SurfaceGeometry(ctx([1.0, 0.0, 0.0, 0.0]), {"objects": {"can": {"pos": [0.1, 0.6, 0.62]}}})
        self.assertAlmostEqual(low.hover_z, 0.648)  # never below hover_clearance over the rest
        self.assertAlmostEqual(low.raise_z, 0.648)  # carried lower than that: raised first
        self.assertEqual(low.descend_timeout(PlaceConfig(), 0.05), PlaceConfig().descend_timeout)
        self.assertEqual(list(g.goal_pos), [0.0, 0.87, 0.52])
        lying = _SurfaceGeometry(ctx([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]), s)
        self.assertAlmostEqual(lying.rest_z, 0.52 + 0.0615)  # rested on its side: body y was up


class ServoLogTests(unittest.TestCase):
    """Every `_servo` call leaves its start / end / target in the pelvis frame in `place_servo`."""

    def ctx(self):
        from locomani.pipeline.execution_record import ExecutionRecord

        state = {
            "t": 0.0,
            "pelvis": {"pos": [0.0, 0.0, 0.79], "quat": [1.0, 0.0, 0.0, 0.0]},
            "palms": {"right": {"pos": [0.3, -0.1, 0.9], "quat": [1.0, 0.0, 0.0, 0.0]}},
            "objects": {"can": {"pos": [0.35, -0.1, 0.9], "quat": [1.0, 0.0, 0.0, 0.0]}},
        }
        p = SimpleNamespace(obs=lambda: state, world_to_root_full=lambda w, s: np.asarray(w, float) - [0, 0, 0.79])
        ex = ExecutionRecord()
        ex.enter("place_carry")
        return SimpleNamespace(p=p, side="right", obj_key="can", execution=ex, meta={"objects": {"can": {}}})

    def test_arrival_and_timeout_are_both_logged(self):
        from locomani.tasks.bottle.place_phase import _servo

        ctx = self.ctx()
        watch = lambda s, holding=True: None  # noqa: E731
        ok, why, _ = _servo(
            ctx, None, watch, lambda s: [0.35, -0.1, 0.9], control="object", v=0.1, tol=0.01, timeout=1.0
        )
        self.assertTrue(ok)
        ok, why, _ = _servo(
            ctx, None, watch, lambda s: [0.35, -0.1, 0.6], control="object", v=0.1, tol=0.01, timeout=0.0
        )
        self.assertFalse(ok)
        self.assertTrue(why.startswith("timed out 30.0 cm short"))
        rows = ctx.execution.metrics["place_servo"]
        self.assertEqual([r["ok"] for r in rows], [True, False])
        self.assertEqual(rows[1]["stage"], "place_carry")
        self.assertEqual(rows[1]["end"]["ctrl_pf"], [0.35, -0.1, 0.11])
        self.assertEqual(rows[1]["end"]["target_pf"], [0.35, -0.1, -0.19])
        self.assertEqual(rows[1]["end"]["obj_tilt_deg"], 0.0)


class LevelTests(unittest.TestCase):
    def test_level_rotation_turns_a_tilted_axis_upright(self):
        tilt = np.radians(20.0)
        up = np.array([0.0, -np.sin(tilt), np.cos(tilt)])  # 20 deg about +x
        self.assertTrue(np.allclose(level_rotation(up) @ up, [0.0, 0.0, 1.0]))
        r = level_rotation(up, max_angle=np.radians(5.0))  # capped: 15 deg of tilt left
        self.assertAlmostEqual(float(np.degrees(np.arccos((r @ up)[2]))), 15.0, places=6)
        self.assertTrue(np.allclose(level_rotation([0.0, 0.0, 1.0]), np.eye(3)))

    def test_quat_matrix_matches_tilt_from(self):
        q = [np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0]  # 45 deg about x
        self.assertAlmostEqual(float(np.degrees(np.arccos(quat_matrix(q)[2, 2]))), 45.0, places=6)
        self.assertAlmostEqual(tilt_from(q, [0, 0, 1]), 45.0, places=6)


class BandPulseTests(unittest.TestCase):
    """Surface put-down band correction: SLOW_WALK pulses on the arm's frames, brake, re-measure."""

    def test_surface_band_is_the_centre_of_the_place_band(self):
        bx, by = surface_band((0.26, 0.44), (-0.16, -0.06))
        self.assertTrue(np.allclose(bx, (0.30, 0.42)) and np.allclose(by, (-0.16, -0.06)))

    def test_band_residual(self):
        bx, by = (0.26, 0.44), (-0.16, -0.06)
        self.assertEqual(list(band_residual(0.35, -0.10, bx, by)), [0.0, 0.0])
        self.assertTrue(np.allclose(band_residual(0.60, -0.10, bx, by), [0.25, 0.01]))

    def run_pulses(self, step_m):
        """A base that moves `step_m` along the commanded direction per pulse; the spot starts 0.60 m ahead."""
        from locomani.backends.sonic.agent_server import LocomotionMode
        from locomani.pipeline.execution_record import ExecutionRecord

        world = {"pelvis": np.array([0.0, 0.0])}

        class Arm:
            q, mode, pushed = np.zeros(29), LocomotionMode.IDLE, []

            def _push(self, q, movement=(0.0, 0.0, 0.0), speed=-1.0):
                self.pushed.append((self.mode, speed))
                if self.mode == LocomotionMode.SLOW_WALK:
                    world["pelvis"] = world["pelvis"] + step_m / 25 * np.asarray(movement[:2])

        p = SimpleNamespace(
            obs=lambda: {"pelvis": {"vel": [0.0, 0.0, 0.0]}},
            world_to_root=lambda w, s: np.asarray(w, float) - [*world["pelvis"], 0.0],
            pelvis_yaw=lambda s: 0.0,
            pelvis_xy=lambda s: world["pelvis"],
            log=lambda text: None,
        )
        ctx = SimpleNamespace(p=p, execution=ExecutionRecord())
        arm = Arm()
        ok, why, goal = _pulse_to_band(ctx, arm, lambda s: None, [0.60, -0.10, 0.52], PlaceConfig())
        return ok, why, arm, ctx.execution.metrics["place_pulses"]

    def test_pulses_walk_the_spot_into_the_band(self):
        ok, why, arm, pulses = self.run_pulses(0.12)
        self.assertTrue(ok, why)
        self.assertEqual(len(pulses), 2)  # 0.60 -> 0.48 -> 0.36, inside the surface band 0.30-0.42
        self.assertEqual({p[2] for p in pulses}, {0.5})  # no short pulses
        self.assertEqual({sp for m, sp in arm.pushed if m.name == "SLOW_WALK"}, {0.35})
        self.assertEqual(arm.pushed[-1][0].name, "IDLE")

    def test_a_reachable_spot_needs_no_pulse(self):
        from locomani.backends.sonic.agent_server import LocomotionMode
        from locomani.pipeline.execution_record import ExecutionRecord

        p = SimpleNamespace(
            obs=lambda: {"pelvis": {"vel": [0.0, 0.0, 0.0]}},
            world_to_root=lambda w, s: np.asarray(w, float),
            pelvis_yaw=lambda s: 0.0,
            pelvis_xy=lambda s: np.zeros(2),
            log=lambda text: None,
        )
        arm = SimpleNamespace(q=np.zeros(29), mode=LocomotionMode.IDLE, _push=lambda q, **kw: None)
        asked = []

        def respot(state, aim):
            asked.append(np.asarray(aim))
            return np.asarray(aim) + [0.02, 0.0]  # the free point nearest the band centre

        ctx = SimpleNamespace(p=p, execution=ExecutionRecord(), respot=respot)
        ok, why, goal = _pulse_to_band(ctx, arm, lambda s: None, [0.60, -0.10, 0.52], PlaceConfig())
        self.assertTrue(ok, why)
        self.assertTrue(np.allclose(asked[0], [0.36, -0.11]))  # the surface band's centre, world = pelvis frame here
        self.assertTrue(np.allclose(goal, [0.38, -0.11, 0.52]))
        self.assertEqual(ctx.execution.metrics["place_pulses"], [])

    def test_a_base_that_does_not_move_fails_after_the_pulses(self):
        ok, why, _arm, pulses = self.run_pulses(0.0)
        self.assertFalse(ok)
        self.assertTrue(why.startswith("[turn] spot not in the placing band after 6 pulses"))
        self.assertEqual(len(pulses), 6)


class ConfigTests(unittest.TestCase):
    def test_place_config_round_trips_inside_the_task_config(self):
        cfg = BottleTaskConfig(goal="basket", place=PlaceConfig(band_x=(0.20, 0.28)))
        back = BottleTaskConfig.from_dict(cfg.to_dict())
        self.assertEqual(back, cfg)
        self.assertEqual(back.place.band_x, (0.20, 0.28))

    def test_pick_place_yaml_names_the_goal(self):
        from locomani import profiles

        cfg = profiles.load("task", "pick_place")
        self.assertEqual((cfg.target, cfg.goal), ("soup_can", "basket"))


if __name__ == "__main__":
    unittest.main()
