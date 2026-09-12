"""Capture effective reach settings and real sampling outputs from either worktree.

Run with the simulation Python and the same LOCOMANI_SIM_DIR for both revisions:
    python <script> --repo <worktree> --out <capture.json>
Compare the `profiles` mappings. This checks configuration/sampling compatibility,
not physical success rates or asynchronous controller behavior.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo.resolve()
    sys.path.insert(0, str(root / "src"))
    import numpy as np

    from locomani import paths
    from locomani.tasks.bottle.config import BottleTaskConfig
    from locomani.tasks.bottle.reach_config import ReachConfig
    from locomani.tasks.bottle.ready_grasp_policy import sample_stand

    files = sorted((root / "configs/collection").glob("*.json"))
    files += [
        root / "configs/tasks" / name
        for name in ("bottle_visible.json", "bottle_visible_teleop.json", "pick_place.json")
    ]
    files += [root / "tests/fixtures/reach_config_ready07d_200.json"]
    profiles = {}
    for path in files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = BottleTaskConfig.load(path).reach if path.parent.name == "tasks" else ReachConfig.load(path)
        effective = cfg.to_dict()
        effective.pop("schema_version")
        effective.pop("strategy", None)
        for name in ("ready_grasp_baseline", "ready_grasp_recovery", "coupled_squat", "image_servo", "teleop_arm"):
            effective[name] = getattr(cfg, name)
        samples = []
        for height in (0.683, 0.723):
            for seed in range(8):
                try:
                    rel, drop = sample_stand(SimpleNamespace(grip_z=height), np.random.default_rng(seed), cfg)
                except ValueError as error:
                    samples.append(dict(height=height, seed=seed, rejected=str(error)))
                else:
                    samples.append(dict(height=height, seed=seed, rel=rel.tolist(), drop=float(drop)))
        profiles[path.relative_to(root).as_posix()] = dict(effective=effective, samples=samples)
    source_hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "src/locomani/tasks/bottle").glob("*.py"))
    }
    result = dict(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_sha256=source_hashes,
        scene_sha256=hashlib.sha256(Path(paths.SCENE_XML).read_bytes()).hexdigest(),
        profiles=profiles,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    count = sum(len(p["samples"]) for p in profiles.values())
    accepted = sum("rejected" not in sample for p in profiles.values() for sample in p["samples"])
    print(
        f"captured {len(profiles)} profiles: {count} samples, {accepted} accepted, {count - accepted} rejected: {args.out}"
    )


if __name__ == "__main__":
    main()
