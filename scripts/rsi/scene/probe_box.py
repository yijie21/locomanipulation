"""The two-hand feasibility probe on the logistics room (TODO 38.7, docs/handoff/box-bimanual-rsi-plan-20261023.md
D5): before the loop runs on `box_bimanual`, show that the motor layer's two-hand actions lift the carton. The robot
starts at the scene's default start (`robot_start`, 1.1 m in front of the platform's -y edge); the box takes `--n`
random placements drawn as the family's pools draw them (its two tasks in turn, `placement` included); the
hand-written pick in scripts/rsi/probe_box_skills/ runs each start through scene_round.py, so the records carry the
same fields as the loop's (steps, failure_class, events) and the 50 Hz traces; a video per start replays the trace.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src $PY scripts/rsi/probe_box.py \\
        [--n 10] [--seed 700000] [--set '{"INWARD_M": 0.04}'] [--out <dir>] [--no-video]

Writes <out>/pool/ (the starts), <out>/rounds/ (records and traces), <out>/videos/ (one mp4 and a <= 500 KB preview
per start) and <out>/summary.json: successes per task, the failure classes, each round's steps. Rule D5's line: 5 / 10
or more opens the loop; 1-4 tunes the squeeze and the grasp height once; 0 goes back to the controller.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "report"), str(REPO / "src")]

import family as F  # noqa: E402
import scene_round as SR  # noqa: E402

from locomani.tasks.scene.start import Start, place_on_surface  # noqa: E402

FAMILY = "box_bimanual"
SKILLS = HERE / "probe_box_skills"
OUT = REPO / "outputs" / "scene_rsi_20261016" / "box_probe"
SPEED = 2  # replay speed of the videos (25 fps from a 50 Hz trace)


def draw_pool(fam: dict, out: Path, n: int, seed: int) -> list[str]:
    """`n` starts: the default robot start, the box placed as the family's tasks ask, tasks in turn."""
    meta = fam["meta"]
    rs = meta["robot_start"]
    robot_xy, yaw = tuple(float(x) for x in rs["xy"]), float(np.radians(rs["yaw_deg"]))
    out.mkdir(parents=True, exist_ok=True)
    labels = []
    for i in range(n):
        t = F.task(fam, fam["tasks"][i % len(fam["tasks"])])
        key = list(t.objects)[0]
        placement = {**(fam.get("placement") or {}), **(fam["rows"][t.id].get("placement") or {})} or None
        s = seed + i
        spot = place_on_surface(meta, key, np.random.default_rng(s), [], placement)
        if spot is None:
            raise SystemExit(f"no spot on the platform meets {placement} (seed {s})")
        st = Start(s, robot_xy, yaw, {key: spot})
        label = f"{t.id}_s{s}"
        (out / f"{label}.json").write_text(json.dumps(dict(task=t.id, start=st.to_dict()), indent=1))
        labels.append(label)
    (out / "_pool.json").write_text(json.dumps(dict(family=fam["name"], pool="probe", written=n, skipped=[],
                                                      unavailable={}, next_seed=seed + n), indent=1))  # fmt: skip
    return labels


def video(scene_dir: Path, rec: dict, npz: Path, out: Path) -> None:
    """One start's replay on the full scene from the robot's side of the box, written like the report videos."""
    import visualize as V

    dv = V.dv
    z = dict(np.load(npz, allow_pickle=True))
    key = next(k for k in z if k.startswith("obj_"))

    class Rd:
        pass

    r = Rd()
    r.z, r.key = z, key
    sc = V.Scene(scene_dir)
    n = len(z["t"])
    box0 = z[key][0, :3]
    pel0 = z["robot"][0, :2]
    base = float(np.degrees(np.arctan2(box0[1] - pel0[1], box0[0] - pel0[0])))
    look = [float(box0[0]), float(box0[1]), float(box0[2]) + 0.25]
    sc.set(r, 0)  # the kinematics the ray casts below need
    cam = dict(lookat=look, distance=2.6, azimuth=base + 150.0, elevation=-22.0)
    # the view the racks block least: the first azimuth with room for the full distance, else the roomiest
    clear = {
        turn: sc.clear_distance(look, base + turn, cam["elevation"])
        for turn in (150.0, 210.0, 120.0, 240.0, 90.0, 270.0, 180.0)
    }
    turn = next((t for t, c in clear.items() if c >= cam["distance"]), max(clear, key=clear.get))
    cam["azimuth"], cam["distance"] = base + turn, min(cam["distance"], clear[turn])
    w = dv.Writer(out, V.PANEL[0], V.PANEL[1])
    stride = max(1, int(round(SPEED * 50 / dv.FPS)))
    frame = None
    for i in range(0, n, stride):
        sc.set(r, i)
        frame = V.label(sc.render(cam, V.PANEL), f"{rec['label']}  t = {float(z['t'][i]):5.1f} s ({SPEED}x)", V.DIM, 18)
        w.write(frame)
    verdict = "held" if rec["ok"] else f"failed: {rec.get('failure_class')}  {rec.get('why') or ''}"
    w.write(V.label(frame, verdict, V.FG, 22, (12, 40)), int(2 * dv.FPS))
    w.close()
    dv.preview(out, out.parent / "previews" / out.stem)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=700000, help="first start seed (clear of every pool's range)")
    ap.add_argument("--set", default="", help="probe_box_skills parameter overrides: inline JSON or a JSON file")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--labels", default="", help="comma-separated start labels to (re)run; default all")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args(argv)
    fam = F.load(FAMILY)
    pool, rounds, videos = a.out / "pool", a.out / "rounds", a.out / "videos"
    labels = draw_pool(fam, pool, a.n, a.seed)
    if a.labels:
        labels = [x for x in a.labels.split(",") if x]
    overrides = SR.parse_set(a.set)
    SR.run(FAMILY, pool, SKILLS, rounds, labels, overrides, True)
    recs = {x: json.loads((rounds / f"{x}.json").read_text()) for x in labels}
    by_task: dict = {}
    for x, r in recs.items():
        by_task.setdefault(r["task"], []).append(bool(r["ok"]))
    summary = dict(
        family=FAMILY,
        skills=str(SKILLS),
        set=overrides,
        n=len(labels),
        ok=sum(bool(r["ok"]) for r in recs.values()),
        by_task={t: f"{sum(v)} / {len(v)}" for t, v in sorted(by_task.items())},
        classes={
            c: sum(1 for r in recs.values() if r.get("failure_class") == c)
            for c in sorted({r.get("failure_class") for r in recs.values()})
        },  # fmt: skip
        rounds={
            x: dict(
                ok=r["ok"],
                why=r.get("why"),
                failure_class=r.get("failure_class"),
                fell=r.get("fell"),
                steps=[
                    dict(
                        step=s.get("step"),
                        ok=s.get("ok"),
                        why=s.get("why"),
                        **{
                            k: s[k]
                            for k in (
                                "stop_err_cm",
                                "jaw_err_cm",
                                "common",
                                "lead_cm",
                                "rise_cm",
                                "touching",
                                "pelvis_z",
                                "edge",
                            )
                            if k in s
                        },
                    )
                    for s in r.get("steps") or []
                ],
            )
            for x, r in recs.items()
        },  # fmt: skip
    )
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("n", "ok", "by_task", "classes")}, ensure_ascii=False))
    for x, r in recs.items():
        print(
            f"  {x}: {'ok' if r['ok'] else r.get('failure_class')}  "
            + " > ".join(f"{s.get('step')}{'' if s.get('ok') else '!'}" for s in r.get("steps") or [])
        )
    if not a.no_video:
        videos.mkdir(parents=True, exist_ok=True)
        for x, r in recs.items():
            npz = rounds / f"{x}.npz"
            if npz.exists():
                video(fam["scene_dir"], dict(r, label=x), npz, videos / f"{x}.mp4")
        print(f"videos in {videos} (previews/ <= 500 KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
