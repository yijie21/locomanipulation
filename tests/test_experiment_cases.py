import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free"))
from experiment_cases import load_cases, save_cases


class CaseBankTests(unittest.TestCase):
    def test_roundtrip_and_reject_wrong_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene.json"
            scene.write_text('{"height":0.6}')
            bank = Path(tmp) / "cases.json"
            c = dict(
                case_id="a",
                target="water_bottle",
                appearance=42,
                can_xy=[0.1, 0.2],
                stand=[0.3, 0.4],
                rel=[0.2, -0.2],
                psi=1.234567891,
                spawn_off=1.2,
                spawn_lat=0.01,
                dpsi=0.1,
                squat_hint=0.12,
            )
            save_cases(bank, [c], scene)
            self.assertEqual(load_cases(bank, scene), [c])
            with self.assertRaises(FileExistsError):
                save_cases(bank, [c], scene)
            scene.write_text('{"height":0.7}')
            with self.assertRaises(ValueError):
                load_cases(bank, scene)

    def test_duplicate_and_nonfinite_rejected(self):
        from experiment_cases import validate

        c = dict(
            case_id="a",
            target="water_bottle",
            appearance=42,
            can_xy=[0.1, 0.2],
            stand=[0.3, 0.4],
            rel=[0.2, -0.2],
            psi=1.0,
            spawn_off=1.2,
            spawn_lat=0.01,
            dpsi=0.1,
            squat_hint=0.12,
        )
        with self.assertRaises(ValueError):
            validate([c, c])
        with self.assertRaises(ValueError):
            validate([dict(c, psi=float("nan"))])


if __name__ == "__main__":
    unittest.main()
