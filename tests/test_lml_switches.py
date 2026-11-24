"""TODO 49.4: the switches of the LocoManiLab bed: the capability table (SPEC) agrees with what the bed implements (config.check_scope),
the profile's defaults are the loops', and a value outside the scope is refused with its reason."""

import unittest

from locomani.bench import options, registry
from locomani.bench.locomanilab import config as cfgmod
from locomani.bench.spec import READY


def resolve(*sets, allow_planned=True):
    return options.resolve("locomanilab", list(sets), allow_planned=allow_planned)


class Switches(unittest.TestCase):
    def test_profile_is_the_loops_defaults(self):
        o = resolve().options
        self.assertEqual(
            (o.target, o.sensor, o.depth_noise, o.self_pose, o.controller_root, o.controller),
            ("perception", "rgbd", "none", "odometry", "odometry", "scalebfm"),
        )
        self.assertEqual((o.target, o.sensor, o.depth_noise), cfgmod.DEFAULT_VIEW)
        self.assertEqual((o.self_pose, o.controller_root), cfgmod.DEFAULT_CONTROL)

    def test_every_ready_value_is_in_the_beds_scope_and_the_other_way_round(self):
        spec = registry.spec("locomanilab")
        for axis, values in spec.supports.items():
            ready = {v for v, s in values.items() if s == READY}
            scope = set(cfgmod.SCOPE[axis]) - (
                {"none"} if axis == "sensor" else set()
            )  # sensor `none` is target=truth's, not a value of the table
            self.assertEqual(ready, scope, axis)

    def test_ready_values_resolve_without_allowing_planned(self):
        for sets in (
            (),
            ("depth_noise=realsense",),
            ("self_pose=truth",),
            ("controller_root=truth",),
            ("self_pose=truth", "controller_root=truth"),
            ("target=truth", "self_pose=truth"),
        ):
            r = resolve(*sets, allow_planned=False)
            cfgmod.check_scope(r.options)

    def test_values_outside_the_scope_are_refused_with_the_reason(self):
        for sets, why in (
            (("sensor=stereo",), "does not support"),
            (("controller=native",), "does not support"),
        ):
            with self.assertRaises(ValueError) as e:
                resolve(*sets)
            self.assertIn(why, str(e.exception))
        # target=truth on the odometry belief: resolves (the table allows both) but the bed refuses it, naming the frame
        r = resolve("target=truth")
        with self.assertRaises(cfgmod.ScopeError) as e:
            cfgmod.check_scope(r.options)
        self.assertIn("self_pose=truth", str(e.exception))

    def test_stereo_note_says_why(self):
        spec = registry.spec("locomanilab")
        self.assertIsNone(spec.status("sensor", "stereo"))
        self.assertIn("one head camera", spec.notes["sensor=stereo"])

    def test_realsense_note_says_there_are_no_shadows(self):
        self.assertIn("shadow", registry.spec("locomanilab").notes["depth_noise=realsense"])


if __name__ == "__main__":
    unittest.main()
