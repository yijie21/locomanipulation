"""Driver: one (task, version) of the simple_art story videos through the adapter's session (2026-11-25).

    record_story.py <version 0|1> <Task> <label,label,...>      (taskset -c 24-31; PYTHONPATH=<pinned worktree>/src; sim venv python)
Takes one slot (flock), starts the Isaac server (GPU 0) and the depth service (GPU 1) of that slot, expects SAM 3 on 18832, runs
bed_record.py in the place of the adapter's `batch` (same arguments, same cwd = the worktree), stops what it started by pid, releases the
slot. Practice pool of the loop's sandbox (read only copy), lazy render off, run dir of its own; nothing is written into the worktree.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import story_lib as SL  # noqa: E402

from locomani.bench.simple import config as cfgmod  # noqa: E402
from locomani.bench.simple.session import Session  # noqa: E402

BED = Path(__file__).resolve().parent / "bed_record.py"


class FilmSession(Session):
    def bed_cmd(self, sub, *args):
        if sub != "batch":
            return super().bed_cmd(sub, *args)  # cachedir: the adapter's own command
        return [self.cfg.sim_python, str(BED), *args, *self.forward]

    def bed_env(self):
        e = super().bed_env()
        e.update(
            RSV_VERSION=os.environ["RSV_VERSION"],
            RSV_TRIES=os.environ.get("RSV_TRIES", "3"),
            PYTHONDONTWRITEBYTECODE="1",
        )
        if os.environ.get("RSV_CAM"):
            e["RSV_CAM"] = os.environ["RSV_CAM"]
        return e


def build(ver: int, task: str, labels: list[str], stamp: str | None = None):
    run = Path("/workspace/tmp/storyvid_art") / f"run_{task}_v{ver}_{stamp or time.strftime('%H%M%S')}"
    sets = [f"practice_dir={SL.S['practice']}", f"run_dir={run}", "lazy_render=false"]
    forward = ["--bench-set", "target=perception", "--bench-set", "sensor=stereo"]
    for kv in sets:
        forward += ["--set", kv]
    forward += ["--skills", str(SL.version_dir(ver))]
    cfg = cfgmod.SimpleConfig().with_overrides(sets)
    settings = {json.dumps(SL.diag_record(ver, lb)["simple"]["setting"], sort_keys=True) for lb in labels}
    assert len(settings) == 1, f"labels of one session must share the setting: {settings}"
    return cfg, forward, run, settings.pop()


def main():
    ver, task, labels = int(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
    os.environ["RSV_VERSION"] = str(ver)
    cfg, forward, run, setting = build(ver, task, labels)
    run.mkdir(parents=True, exist_ok=True)
    ids = run / "ids.txt"
    ids.write_text("".join(SL.episode_of(lb) + "\n" for lb in labels))
    out = SL.OUT / "run" / f"{task}_v{ver}.jsonl"
    s = FilmSession(cfg, task, ids, out, setting, None, forward, None, stereo=True, look=True)
    rc = s.run_round()
    print("session rc", rc, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
