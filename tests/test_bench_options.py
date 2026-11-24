"""Run switches (ported from bench-options' TODO 49.1; selfwm TODO 38.11, user 2026-11-24): benchmark, truth or
perception targets, stereo or RGB-D, depth noise, self pose, the pelvis ScaleBFM is told, the whole-body controller --
resolved once and checked against each benchmark's capability table before anything starts."""

import io
import itertools
import unittest
from contextlib import redirect_stderr, redirect_stdout

from locomani import profiles
from locomani.bench import BENCHES, BenchOptions, registry, resolve
from locomani.bench import __main__ as cli
from locomani.bench import options as O
from locomani.bench import simple as S
from locomani.bench.spec import AXES, PLANNED, READY, BenchSpec


class Profiles(unittest.TestCase):
    def test_every_benchmark_has_a_profile_that_resolves_to_itself_within_its_table(self):
        self.assertEqual(profiles.names("bench"), sorted(BENCHES))
        for name in BENCHES:
            o = profiles.load("bench", name)
            self.assertIsInstance(o, BenchOptions)
            self.assertEqual(o.bench, name)
            self.assertEqual(O.consistency_errors(o), [], name)
            self.assertEqual(O.support_errors(o, allow_planned=True), [], name)
            r = resolve(name, allow_planned=True)
            self.assertEqual((r.options, r.explicit, r.normalized), (o, (), ()), name)

    def test_the_simple_profile_is_the_default_combination(self):
        # the user's default (2026-11-24): SIMPLE, ScaleBFM, one RGB-D camera -- perception column, the replica's
        # MuJoCo look, leg odometry for the belief and for ScaleBFM's root (BFM_ROOT=odo)
        s = resolve("simple").options
        self.assertEqual((s.target, s.sensor, s.depth_noise, s.self_pose, s.controller_root, s.controller),
                         ("perception", "rgbd", "none", "odometry", "odometry", "scalebfm"))  # fmt: skip
        self.assertEqual(s.signature(), "simple/perception-rgbd/pose-odometry/root-odometry/scalebfm")
        self.assertEqual(resolve("locomanilab", allow_planned=True).options.sensor, "rgbd")
        for name in ("bigym2", "humanoidarena"):
            self.assertEqual(resolve(name, allow_planned=True).options.controller, "native")


class Precedence(unittest.TestCase):
    def test_code_default_then_profile_then_override(self):
        self.assertEqual(BenchOptions().self_pose, "odometry")  # code default
        self.assertEqual(resolve("bigym2", allow_planned=True).options.self_pose, "truth")  # the profile's
        r = resolve("bigym2", ["self_pose=odometry"], allow_planned=True)
        self.assertEqual(r.options.self_pose, "odometry")  # the override's
        self.assertEqual(r.explicit, ("self_pose",))

    def test_an_explicit_value_equal_to_the_default_is_still_recorded_as_explicit(self):
        r = resolve("simple", ["sensor=rgbd"])
        self.assertEqual(r.explicit, ("sensor",))
        self.assertEqual(r.record()["explicit"], ["sensor"])

    def test_the_benchmark_can_come_from_an_override_but_not_twice(self):
        self.assertEqual(resolve(None, ["bench=locomanilab"], allow_planned=True).options.bench, "locomanilab")
        with self.assertRaisesRegex(ValueError, "two benchmarks"):
            resolve("simple", ["bench=bigym2"], allow_planned=True)
        with self.assertRaisesRegex(ValueError, "unknown benchmark"):
            resolve("bigym", allow_planned=True)


class InvalidInput(unittest.TestCase):
    def test_an_unknown_choice_or_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not in"):
            resolve("simple", ["target=xray"])
        with self.assertRaisesRegex(ValueError, "not in"):
            resolve("simple", ["controller=hover"])
        with self.assertRaisesRegex(ValueError, "unknown option"):
            resolve("simple", ["camera=stereo"])
        with self.assertRaisesRegex(ValueError, "unknown field"):
            BenchOptions.from_dict({"bench": "simple", "track": "official"})

    def test_truth_takes_no_sensor(self):
        r = resolve("simple", ["target=truth"])
        self.assertEqual(r.options.sensor, "none")
        self.assertIn("sensor", r.normalized)
        with self.assertRaisesRegex(ValueError, "takes no sensor"):
            resolve("simple", ["target=truth", "sensor=stereo"])

    def test_perception_needs_a_sensor(self):
        with self.assertRaisesRegex(ValueError, "needs sensor"):
            resolve("simple", ["sensor=none"])

    def test_depth_noise_belongs_to_rgbd(self):
        r = resolve("bigym2", ["depth_noise=realsense"], allow_planned=True)
        self.assertEqual(r.options.signature(), "bigym2/perception-rgbd-realsense/pose-truth/root-truth/native")
        with self.assertRaisesRegex(ValueError, "rgbd only"):
            resolve("simple", ["sensor=stereo", "depth_noise=realsense"])
        with self.assertRaisesRegex(ValueError, "does not support depth_noise=realsense.*not ported"):
            resolve("simple", ["depth_noise=realsense"])  # the RGB-D default, but no noise model on selfwm's bed


class Couplings(unittest.TestCase):
    def test_on_selfwms_simple_bed_the_self_pose_comes_with_the_target(self):
        r = resolve("simple", ["target=truth"])
        self.assertEqual((r.options.self_pose, r.normalized), ("truth", ("sensor", "self_pose")))
        self.assertEqual(r.options.signature(), "simple/truth/pose-truth/root-odometry/scalebfm")
        self.assertEqual(resolve("simple", ["sensor=stereo"]).options.self_pose, "odometry")

    def test_an_explicit_contradiction_is_refused_with_the_reason(self):
        with self.assertRaisesRegex(ValueError, "target=truth fixes self_pose=truth.*simulator"):
            resolve("simple", ["target=truth", "self_pose=odometry"])
        with self.assertRaisesRegex(ValueError, "target=perception fixes self_pose=odometry"):
            resolve("simple", ["self_pose=truth"])
        r = resolve("simple", ["target=truth", "self_pose=truth"])  # explicit and agreeing: fine, not normalised
        self.assertEqual(r.normalized, ("sensor",))

    def test_a_coupling_to_an_unsupported_value_is_refused(self):
        full = {a: {"x": PLANNED} for a in AXES}
        with self.assertRaisesRegex(ValueError, "coupling"):
            BenchSpec("b", "B", "", full, couples={"target=x": ("self_pose", "y")})


class CapabilityTables(unittest.TestCase):
    def test_a_value_the_benchmark_lacks_is_rejected_with_its_reason(self):
        with self.assertRaisesRegex(ValueError, "does not support sensor=stereo.*second rendered camera"):
            resolve("locomanilab", ["sensor=stereo"], allow_planned=True)
        with self.assertRaisesRegex(ValueError, "does not support controller=scalebfm"):
            resolve("bigym2", ["controller=scalebfm"], allow_planned=True)
        with self.assertRaisesRegex(ValueError, "does not support controller_root=odometry"):
            resolve("humanoidarena", ["controller_root=odometry"], allow_planned=True)
        with self.assertRaisesRegex(ValueError, "does not support controller_root=truth.*BFM_ROOT=odo"):
            resolve("simple", ["controller_root=truth"])
        with self.assertRaisesRegex(ValueError, "does not support controller=native"):
            resolve("simple", ["controller=native"])

    def test_a_planned_value_stops_a_run_unless_allowed(self):
        with self.assertRaisesRegex(ValueError, "planned, not implemented yet"):
            resolve("locomanilab")
        r = resolve("locomanilab", allow_planned=True)
        self.assertIn("sensor=rgbd", r.planned)
        with self.assertRaisesRegex(ValueError, "controller=amo is planned.*never run on this bed"):
            resolve("simple", ["controller=amo"])

    def test_simple_runs_both_controllers_with_every_view(self):
        # the combinations selfwm's SIMPLE bed has run: ScaleBFM and SONIC, truth, the RGB-D and the stereo look
        for ctrl, view in itertools.product(("scalebfm", "sonic"), ([], ["target=truth"], ["sensor=stereo"])):
            r = resolve("simple", [f"controller={ctrl}", *view])
            self.assertEqual(r.planned, (), (ctrl, view))

    def test_every_table_covers_every_axis_and_every_value_is_a_known_choice(self):
        choices = {f: BenchOptions.__dataclass_fields__[f].metadata["choices"] for f in AXES}
        for name in BENCHES:
            spec = registry.spec(name)
            for axis, value, status in spec.table():
                self.assertIn(value, choices[axis], f"{name}.{axis}")
                self.assertIn(status, (READY, PLANNED))
                if axis == "sensor":
                    self.assertNotEqual(value, "none", f"{name}: sensor=none is implied by target=truth")

    def test_a_table_with_a_missing_axis_or_a_bad_status_is_refused(self):
        full = {a: {"x": PLANNED} for a in AXES}
        with self.assertRaisesRegex(ValueError, "missing"):
            BenchSpec("b", "B", "", {a: v for a, v in full.items() if a != "controller"})
        with self.assertRaisesRegex(ValueError, "statuses"):
            BenchSpec("b", "B", "", {**full, "target": {"truth": "soon"}})


class SimpleBed(unittest.TestCase):
    def test_the_switches_map_onto_simple_benchs_own_settings(self):
        bed = S.bed_settings(resolve("simple").options)
        self.assertEqual(bed, dict(column="perception", cloud="cloud_blind", look="look_dump_blind", controller="scalebfm",
                                   root="odo"))  # fmt: skip
        bed = S.bed_settings(resolve("simple", ["sensor=stereo", "controller=sonic"]).options)
        self.assertEqual((bed["column"], bed["cloud"], bed["look"], bed["controller"]),
                         ("perception", "cloud_eyes_final", "look_dump_eyes", "sonic"))  # fmt: skip
        bed = S.bed_settings(resolve("simple", ["target=truth"]).options)
        self.assertEqual((bed["column"], bed["cloud"]), ("truth", "cloud_blind"))  # simple_bench's old default
        with self.assertRaisesRegex(ValueError, "runs bench=simple"):
            S.bed_settings(resolve("bigym2", allow_planned=True).options)

    def test_the_shell_helper_prints_the_bed_settings_or_the_reason(self):
        from locomani.bench.simple import __main__ as bed_cli

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(bed_cli.main(["--bench-set", "controller=sonic", "--bench-set", "sensor=stereo"]), 0)
        self.assertIn("BED_CLOUD=cloud_eyes_final\n", out.getvalue())
        self.assertIn("BED_CONTROLLER=sonic\n", out.getvalue())
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            self.assertEqual(bed_cli.main(["--bench-set", "controller=amo"]), 2)
        self.assertIn("planned", err.getvalue())


class Signatures(unittest.TestCase):
    def test_different_switches_never_share_a_signature(self):
        seen = {}
        grid = itertools.product(BENCHES, O.TARGETS, O.SENSORS, O.DEPTH_NOISE, O.SELF_POSE, O.CONTROLLER_ROOT,
                                 O.CONTROLLERS)  # fmt: skip
        for b, t, s, n, p, r, c in grid:
            o = BenchOptions(b, t, s, n, p, r, c)
            if O.consistency_errors(o):
                continue
            sig = o.signature()
            self.assertNotIn(sig, seen, f"{o} and {seen.get(sig)}")
            seen[sig] = o


class Entry(unittest.TestCase):
    def test_show_prints_the_resolved_record_and_table_prints_every_benchmark(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["show", "--bench", "simple", "--bench-set", "target=truth"]), 0)
        self.assertIn('"signature": "simple/truth/pose-truth/', out.getvalue())
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli.main(["table"]), 0)
        for name in BENCHES:
            self.assertIn(f"({name})", out.getvalue())
        self.assertIn("couples: target=truth fixes self_pose=truth", out.getvalue())

    def test_show_reports_an_impossible_combination_as_an_error(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["show", "--bench", "locomanilab", "--bench-set", "sensor=stereo"]), 2)


if __name__ == "__main__":
    unittest.main()
