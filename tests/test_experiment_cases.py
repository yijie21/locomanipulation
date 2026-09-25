import json
import sys
import tempfile
import unittest
from pathlib import Path

from locomani.pipeline.experiment_cases import load_cases, save_cases


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

    def test_the_same_scene_in_another_checkout_loads_the_bank(self):
        """Scene metas carry absolute paths; a bank written in one checkout must load in another."""
        from locomani.pipeline import experiment_cases as E

        with tempfile.TemporaryDirectory() as tmp:
            metas = []
            for checkout in ("alice/locomanipulation", "bob/code/locomanipulation"):
                root = Path(tmp) / checkout
                meta = root / "outputs/scene/sim_lowcom/scene_meta.json"
                meta.parent.mkdir(parents=True)
                meta.write_text(
                    json.dumps(
                        {
                            "scene_xml": str(root / "outputs/scene/sim_lowcom/scene_locomani.xml"),
                            "objects": {"water_bottle": {"mesh": str(root / "assets/robocasa/b/model.xml")}},
                            "height": 0.6214,
                        }
                    )
                )
                metas.append((root, meta))
            (a_root, a), (b_root, b) = metas
            self.assertNotEqual(E.scene_hash(a), E.scene_hash(b))  # schema 1: the bytes differ
            self.assertEqual(E.portable_scene_hash(a, a_root), E.portable_scene_hash(b, b_root))
            b.write_text(b.read_text().replace("0.6214", "0.65"))  # a different table is still a different scene
            self.assertNotEqual(E.portable_scene_hash(a, a_root), E.portable_scene_hash(b, b_root))

    def test_a_schema_1_bank_still_loads_against_its_exact_meta(self):
        from locomani.pipeline import experiment_cases as E

        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene_meta.json"
            scene.write_text('{"height":0.6}')
            c = dict(
                case_id="a", target="water_bottle", appearance=42, can_xy=[0.1, 0.2], stand=[0.3, 0.4],
                rel=[0.2, -0.2], psi=1.0, spawn_off=1.2, spawn_lat=0.01, dpsi=0.1, squat_hint=0.12,
            )  # fmt: skip
            old = Path(tmp) / "old.json"
            old.write_text(json.dumps(dict(schema_version=1, scene_sha256=E.scene_hash(scene), cases=[c])))
            self.assertEqual(load_cases(old, scene), [c])
            new = Path(tmp) / "new.json"
            save_cases(new, [c], scene)
            self.assertEqual(json.loads(new.read_text())["schema_version"], 2)

    def test_duplicate_and_nonfinite_rejected(self):
        from locomani.pipeline.experiment_cases import validate

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
