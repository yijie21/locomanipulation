"""The bed process of the story recorder: `batch` of the SIMPLE adapter with filming (2026-11-25, RSI story videos of simple_art).

Started by record_story.py's session (slot lock, Isaac server, depth service already up) under the sim venv with PYTHONPATH=<pinned
worktree>/src, in the place of `python -m locomani.bench.simple batch`; it takes that command's arguments (--slot --setting --ids --out
--skills --set --bench-set). Extra inputs by environment: RSV_VERSION (0 / 1), RSV_TRIES (attempts per episode), RSV_CAM (json, optional).
For each id <Task>/Q/<n> (label <Task>_s<n>) it runs up to RSV_TRIES attempts with the exact setting of that version's diagnostic record
(simple.setting), through the adapter's own run.run (lazy render off by the config), filming a third-person camera (server tp_setup / tp_frame)
with the head camera as an inset; stops at the first attempt whose ok / failure_class equal the loop's record. Per attempt:
  attempts/<label>_v<k>.a<j>.mp4 + .jpg, rows/<label>_v<k>.a<j>.json. One line per finished episode in --out (the session's row check).
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import story_lib as SL  # noqa: E402

from locomani.bench import options  # noqa: E402
from locomani.bench.simple import __main__ as M  # noqa: E402
from locomani.bench.simple import config as cfgmod  # noqa: E402
from locomani.bench.simple import run as bed_run  # noqa: E402

VER = int(os.environ["RSV_VERSION"])
TRIES = int(os.environ.get("RSV_TRIES", "3"))
CAM = json.loads(
    os.environ.get("RSV_CAM", "[2.4, 0.5, 1.5, 0.0]")
)  # left, forward (of the robot), height, weight of the robot start in the aim
STATE: dict = {}


class Rec:
    def __init__(self, env):
        self.env, self.frames, self.last, self.tp_seen, self.n = env, [], -1, 0, 0
        self.camera = "third-person (Isaac, fixed) + head inset"
        info = env.info
        c = np.mean(
            [np.asarray(info["start"]["xy"], float)]
            + [np.asarray(t["pos"][:2], float) for t in info["tables"].values()],
            axis=0,
        )
        yaw = info["start"]["yaw"]
        fwd, left = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
        ground = float(env.sim.first["qpos"][2]) - 0.79
        xy = c + CAM[0] * left + CAM[1] * fwd
        w = CAM[3] if len(CAM) > 3 else 0.0
        aim = (1 - w) * c + w * np.asarray(info["start"]["xy"], float)
        self.limit_s = env.limit_eff * 0.02
        env.sim.link.send(("tp_setup", [xy[0], xy[1], ground + CAM[2]], [aim[0], aim[1], ground + 0.8], 1.4))
        env.sim.link.recv()

    def grab(self):
        env = self.env
        env.sim.link.send(("tp_frame", True))
        r = env.sim.link.recv()
        tp, head = r["tp"], r["head"]
        blank = tp is None or tp.size == 0 or float(np.std(tp)) < 2.0
        self.n += 1
        if (
            blank and self.tp_seen == 0 and self.n <= 12
        ):  # the camera renders from the first step on: skip the first frames
            return
        if blank:
            if head is None:
                return
            self.camera = "head camera, left image (the third-person frame was empty)"
            img = cv2.resize(cv2.cvtColor(head, cv2.COLOR_RGB2BGR), (SL.W, SL.H))
        else:
            self.tp_seen += 1
            img = cv2.cvtColor(tp, cv2.COLOR_RGB2BGR)
            if head is not None:
                ins = cv2.resize(cv2.cvtColor(head, cv2.COLOR_RGB2BGR), (192, 108))
                img[360 - 108 - 6 : 360 - 6, 640 - 192 - 6 : 640 - 6] = ins
                cv2.rectangle(img, (640 - 192 - 6, 360 - 108 - 6), (640 - 6, 360 - 6), (255, 255, 255), 1)
        t = env.clock.steps * 0.02
        txt = f"{t:4.1f} s / {self.limit_s:.0f} s"
        cv2.putText(img, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        self.frames.append(img)
        self.last = env.clock.steps


_setup, _free = bed_run.setup, bed_run.free


def setup(bed, episode_id, K, stance, cloud_dir):
    env = _setup(bed, episode_id, K, stance, cloud_dir)
    rec = Rec(env)
    STATE["rec"] = rec
    tick0 = env.mo.tick

    def tick():
        ok = tick0()
        n = env.clock.steps
        if env.clock.counting and n % SL.EVERY == 0 and n != rec.last:
            rec.grab()
        return ok

    env.mo.tick = tick
    rec.grab()
    return env


def free(env):
    rec = STATE.get("rec")
    if rec is not None and env.sim is not None:
        rec.grab()  # the last frame (the step the episode ended in)
    _free(env)


bed_run.setup, bed_run.free = setup, free  # run.run looks both up in its module globals


def attempt(bed, label, eid, setting, old, k):
    stem = f"{label}_v{VER}.a{k}"
    rowf = SL.OUT / "rows" / f"{stem}.json"
    if rowf.exists():  # a restart after a lost server: the attempts already made stay
        return json.loads(rowf.read_text())
    STATE.clear()
    t0 = time.time()
    err = None
    try:
        rec = bed_run.run(bed, eid, dict(setting), None)
    except (EOFError, ConnectionError, BrokenPipeError):
        print("server lost at", eid, flush=True)
        sys.exit(3)
    except Exception as e:  # a broken episode is a row with loss error, as in the loop's bed
        err = f"{type(e).__name__}: {e}"
        rec = dict(
            episode_id=eid, success=False, loss="error", error=err, t_end=None, comply=dict(ok=False, why="not_success")
        )
    wall = round(time.time() - t0, 1)
    new = SL.outcome(rec)
    match = bool(new["ok"] == old["ok"] and new["failure_class"] == (old["failure_class"] or ""))
    r = STATE.get("rec")
    frames = r.frames if r else []
    res = dict(label=label, version=VER, hash=SL.S["versions"][str(VER)]["hash"], attempt=k, episode=eid, setting=setting, match=match,
               new_ok=new["ok"], new_class=new["failure_class"], new_t_end=new["t_end"], new_why=rec.get("why") or rec.get("error"),
               old_ok=old["ok"], old_class=old["failure_class"] or "", old_why=old["why"], old_t_end=old["t_end"], error=err,
               phases=rec.get("phases"), comply=rec.get("comply"), fell=rec.get("fell"), wall_s=wall, rec_wall_s=rec.get("wall_s"),
               cam=CAM, camera=getattr(r, "camera", None), lazy_render=bed.cfg.lazy_render)  # fmt: skip
    if frames:
        tag = "SUCCESS %.1f s" % new["t_end"] if new["ok"] else f"FAIL {new['failure_class']}"
        path = SL.OUT / "attempts" / f"{stem}.mp4"
        size, nfr, padded, outf = SL.encode(
            frames, tag, path, f"{SL.S['versions'][str(VER)]['caption']}  {eid.split('/')[0]}  {label}"
        )
        bad = new["failure_class"] and new["t_end"]
        i = min(
            len(frames) - 1,
            max(0, int((((new["t_end"] or 0) - 1.0) if bad else (new["t_end"] or 0) * 0.6) / (SL.EVERY * 0.02))),
        )
        cv2.imwrite(str(path.with_suffix(".jpg")), outf[i], [cv2.IMWRITE_JPEG_QUALITY, 85])
        res.update(bytes=size, frames=nfr, padded_frames=padded, sim_s=round(len(frames) * SL.EVERY * 0.02, 1), decisive_frame=i,
                   motion=SL.motion(frames))  # fmt: skip
    rowf.write_text(json.dumps(res, indent=1, default=str))
    print(
        json.dumps(
            {
                x: res.get(x)
                for x in (
                    "label",
                    "version",
                    "attempt",
                    "match",
                    "new_ok",
                    "new_class",
                    "new_t_end",
                    "old_class",
                    "bytes",
                    "wall_s",
                    "motion",
                )
            },
            default=str,
        ),
        flush=True,
    )
    return res


def main():
    a = M.build_parser().parse_args(["batch", *sys.argv[1:]])
    resolved = options.from_args(a, allow_planned=True)
    cfg = cfgmod.SimpleConfig().with_overrides(a.set)
    cfgmod.check_scope(resolved.options)
    assert not cfg.lazy_render, "lazy render must be off for filming"
    bed = bed_run.Bed(
        cfg,
        bed_run.load_package(a.skills),
        a.slot,
        dict(bench=resolved.record(), simple_config=cfg.to_dict()),
        resolved.options,
    )
    bed.record.update(view="camera", depth_source=bed.depth_source)
    out = Path(a.out)
    done = {json.loads(x)["episode_id"] for x in open(out)} if out.exists() else set()
    for eid in [s.strip() for s in open(a.ids) if s.strip()]:
        if eid in done:
            continue
        task, _, n = eid.split("/")
        label = f"{task}_s{n}"
        old = SL.diag_record(VER, label)
        setting = old["simple"]["setting"]
        rows = []
        for k in range(1, TRIES + 1):
            r = attempt(bed, label, eid, setting, old, k)
            rows.append(r)
            if r["match"]:
                break
        with open(out, "a") as f:
            f.write(json.dumps(dict(episode_id=eid, attempts=len(rows), match=rows[-1]["match"])) + "\n")


if __name__ == "__main__":
    main()
