"""Phase 1 offline check (no simulator, no GPU process): packages and hashes, the loop's records, the outcome rule on all of them,
the argument building of the session, the adapter's own argument parsing, the episodes' layout, the encoder on synthetic frames.
Run under the sim venv python with PYTHONPATH=<worktree>/src."""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("RSV_VERSION", "0")
import numpy as np  # noqa: E402
import record_story as RS  # noqa: E402
import story_lib as SL  # noqa: E402

from locomani.bench import options  # noqa: E402
from locomani.bench.simple import __main__ as M  # noqa: E402
from locomani.bench.simple import episodes as EP  # noqa: E402


def skills_hash(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*.py")):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


ok = True
for v in (0, 1):
    d = SL.version_dir(v)
    h = skills_hash(d)
    print(
        "v%d package %s hash %s expect %s" % (v, d, h, SL.S["versions"][str(v)]["skills"]),
        "OK" if h == SL.S["versions"][str(v)]["skills"] else "MISMATCH",
    )
    ok &= h == SL.S["versions"][str(v)]["skills"]
labels = [s["label"] for s in SL.S["stories"]]
n = bad = 0
for v in (0, 1):
    for lb in labels:
        d = SL.diag_record(v, lb)
        o = SL.outcome(d["simple"])
        n += 1
        if (o["ok"], o["failure_class"]) != (d["ok"], d["failure_class"] or "") or d["skills"] != SL.S["versions"][
            str(v)
        ]["skills"]:
            bad += 1
            print("outcome rule differs", v, lb, o, d["ok"], d["failure_class"])
    print("v%d settings" % v, {json.dumps(SL.diag_record(v, lb)["simple"]["setting"], sort_keys=True) for lb in labels})
print("outcome rule reproduces the loop's ok/class on %d/%d records" % (n - bad, n))
ok &= bad == 0
for st in SL.S["stories"]:
    cfg, forward, run, setting = RS.build(0, st["task"], [st["label"]], "check")
    EP.check_id(SL.episode_of(st["label"]), cfg)
    ep = EP.load(SL.episode_of(st["label"]), cfg)
cfg, forward, run, setting = RS.build(
    1, "OpenFaucetTeleop", [s["label"] for s in SL.S["stories"] if s["task"] == "OpenFaucetTeleop"], "check"
)
print("forward", forward)
print("setting", setting, "lazy_render", cfg.lazy_render, "practice", cfg.practice_dir, "run_dir", cfg.run_dir)
s = (
    RS.FilmSession(
        cfg,
        "OpenFaucetTeleop",
        run / "ids.txt",
        SL.OUT / "run" / "x.jsonl",
        setting,
        None,
        forward,
        None,
        stereo=True,
        look=True,
    )
    if False
    else None
)


class Dummy(RS.FilmSession):
    def __init__(self, cfg, forward, setting):
        self.cfg, self.forward, self.setting, self.slot = cfg, forward, setting, 1


d = Dummy(cfg, forward, setting)
cmd = d.bed_cmd("batch", "--slot", "1", "--setting", setting, "--ids", "IDS", "--out", "OUT")
print("bed cmd:", " ".join(cmd))
print("cachedir cmd:", " ".join(d.bed_cmd("cachedir", "OpenFaucetTeleop", "--setting", setting)))
a = M.build_parser().parse_args(["batch", *cmd[2:]])
r = options.from_args(a, allow_planned=True)
print("adapter parse: slot", a.slot, "skills", a.skills, "sets", a.set, "bench signature", r.record()["signature"])
ok &= r.record()["signature"].endswith("scalebfm")
# the encoder on synthetic moving frames (CPU, ffmpeg)
fr = [np.full((360, 640, 3), 90, np.uint8) + (np.random.rand(360, 640, 3) * 30).astype(np.uint8) for _ in range(300)]
with tempfile.TemporaryDirectory() as t:
    p = Path(t) / "x.mp4"
    size, nfr, pad, _ = SL.encode(fr, "SUCCESS 12.6 s", p, "v1 第 1 轮后  OpenOvenTeleop  OpenOvenTeleop_s2")
    print("encode: bytes", size, "frames", nfr, "padded", pad)
    ok &= size <= 500_000
    size, nfr, pad, _ = SL.encode(fr[:2], "FAIL perceive", p, "v0 种子  x  y")
    print("short: bytes", size, "frames", nfr, "padded", pad)
print("PHASE1", "OK" if ok else "PROBLEM")
