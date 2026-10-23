"""TODO 31 (user 2026-10-20: "我想要看到这整个 RSI 的进化过程"): one empty-seed family's evolution as videos and one
picture, from its loop sandbox. Reads the ledger's accepted gate calls in order; for each, the start's round under the
incumbent before it (`diag/incumbent_<hash>/`) and under the accepted candidate (`diag/candidate_<hash>/`), both with
their 50 Hz traces. Writes under --out:
* step_NN.mp4 -- accepted change NN on one diagnostic start it turned from a failure into a success, before and
  after side by side, after a card; a change over the empty seed shows the start's first frame held on the left (the
  seed has no skill). Each gets a <= 500 KB preview (previews/) and a poster JPG;
* overview.png -- one start per task (rows) under every version (columns), each cell the round's deciding moment;
* task_NN_<task>.mp4 -- every task of the family (user 2026-10-23: 对每一个任务都生成对应的视频): the seed and the
  final version on one diagnostic start side by side -- the final version's shortest success, else its shortest
  failure -- with every version's successes on that task's diagnostic starts on the card; tasks.json indexes them.
  A step video exists only for an accepted change, so a family with two acceptances has two; this gives one per task.
`--cards` (JSON, keyed by the accepted hash) gives a step's card lines {"title", "what"} and optionally its "label";
without it the card is the Manager's summary of that iteration. Its "_names" entry maps object keys to row names.

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/rsi/report/evolution_media.py --sandbox /workspace/code/rsi-scene/grasp-atomic-a --family grasp_atomic \\
        --out outputs/scene_rsi_20261016/grasp_atomic_media [--cards cards.json] [--only step,overview,task]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import report_media as RM  # noqa: E402
import visualize as V  # noqa: E402

SPEED = 3
CELL = (360, 240)


def ledger(loop: Path) -> list[dict]:
    return [json.loads(x) for x in (loop / "loop" / "ledger.jsonl").read_text().splitlines() if x.strip()]


def versions(loop: Path) -> list[dict]:
    """The seed and every accepted candidate, in order: {name, hash, iter, diag (rounds dir), summary}."""
    es = ledger(loop)
    seed = sorted((loop / "incumbents").iterdir(), key=lambda p: p.stat().st_mtime)[0].name
    out = [dict(name="种子", hash=seed, iter=None, diag=loop / "diag" / f"incumbent_{seed}", summary="")]
    for e in es:
        if e.get("step") == "gate" and e.get("accepted") and not e.get("void"):
            mgr = next((m for m in es if m.get("step") == "manager" and m.get("iter") == e["iter"]), {})
            d = loop / "diag" / f"candidate_{e['hash']}"
            if not d.exists():
                d = loop / "diag" / f"incumbent_{e['hash']}"
            out.append(dict(name=f"第 {len(out)} 版", hash=e["hash"], iter=e["iter"], diag=d,
                            summary=mgr.get("summary_zh", ""), result=e.get("result", {})))  # fmt: skip
    return out


def rec(d: Path, label: str) -> dict | None:
    f = d / f"{label}.json"
    return json.loads(f.read_text()) if f.exists() else None


def has_trace(d: Path, label: str) -> bool:
    return (d / f"{label}.npz").exists()


def flipped(before: Path, after: Path) -> list[str]:
    out = []
    for f in sorted(after.glob("*.json")):
        if f.name.startswith("_"):
            continue
        a, b = json.loads(f.read_text()), rec(before, f.stem)
        if a.get("ok") and not (b or {}).get("ok") and has_trace(after, f.stem):
            out.append(f.stem)
    return out


def task_of(label: str) -> str:
    return label.rsplit("_s", 1)[0]


def step_video(sc: V.Scene, k: int, prev: dict, cur: dict, card: dict, used: set, out_dir: Path) -> Path | None:
    labels = flipped(prev["diag"], cur["diag"])
    if card.get("label"):
        labels = [card["label"]]
    if not labels:
        return None
    fresh = [x for x in labels if task_of(x) not in used] or labels
    label = min(fresh, key=lambda x: len(np.load(cur["diag"] / f"{x}.npz")["t"]))  # the shortest: a short video
    used.add(task_of(label))
    a = RM.Rd(cur["diag"] / label)
    b = RM.Rd(prev["diag"] / label) if has_trace(prev["diag"], label) else None
    title = card.get("title") or f"第 {k} 次采纳（第 {cur['iter']} 轮）"
    what = card.get("what") or cur["summary"]
    lines = [title, what, f"{label} · 左：{prev['name']} · 右：{cur['name']}"]
    rounds = [x for x in (b, a) if x is not None]
    ps = RM.plan(sc, rounds, None)
    right = (RM.frames(sc, a, ps[-1], SPEED), [("新版本", V.FG), outcome(a)])
    if b is None:
        sc.set(a, 0)
        p0 = ps[0]
        lk = V.follow_lookat(a.z["robot"][0, :3], p0["obj0"])
        cam = dict(lookat=lk, distance=min(V.follow_distance(a.z["robot"][0, :3], p0["obj0"], 1.3),
                                           sc.clear_distance(lk, p0["az1"], -26)), azimuth=p0["az1"], elevation=-26)  # fmt: skip
        why = (rec(prev["diag"], label) or {}).get("why") or ""
        note = "种子没有技能：规划直接拒绝，机器人原地不动" if "no skill" in why else "失败：" + V.why_zh(why)
        left = (RM.held(sc.render(cam, V.PANEL, ghosts=[a.z["ref_wrists"][0, j] for j in range(2)]), note),
                [("旧版本", V.FG), ("失败", (255, 110, 100))])  # fmt: skip
    else:
        left = (RM.frames(sc, b, ps[0], SPEED), [("旧版本", V.FG), outcome(b)])
    return RM.compose(sc, out_dir / f"step_{k:02d}.mp4", [left[0], right[0]], [left[1], right[1]], lines, SPEED)


def outcome(r: RM.Rd) -> tuple[str, tuple]:
    if r.rec["ok"]:
        return "成功", (90, 220, 120)
    return "失败：" + V.why_zh(r.rec.get("plan_why") or r.rec.get("why") or ""), (255, 110, 100)


def overview(sc: V.Scene, vs: list[dict], out: Path, names: dict | None = None) -> Path:
    """Rows: the first diagnostic start of every task; columns: the versions. A cell is the deciding moment."""
    labels = sorted({p.stem for p in vs[-1]["diag"].glob("*.json") if not p.name.startswith("_")})
    rows = []
    for t in sorted({task_of(x) for x in labels}):
        lab = next((x for x in labels if task_of(x) == t and rec(vs[-1]["diag"], x).get("ok")), None)
        rows.append(lab or next(x for x in labels if task_of(x) == t))
    head_w, head_h = 220, 40
    W, H = head_w + CELL[0] * len(vs), head_h + CELL[1] * len(rows)
    img = Image.new("RGB", (W, H), (24, 24, 28))
    for j, v in enumerate(vs):
        img.paste(Image.fromarray(V.label(np.zeros((head_h, CELL[0], 3), np.uint8) + 24, v["name"], V.FG, 20, (10, 8))),
                  (head_w + j * CELL[0], 0))  # fmt: skip
    for i, lab in enumerate(rows):
        body = RM.Rd(vs[-1]["diag"] / lab).key.removeprefix("obj_") if has_trace(vs[-1]["diag"], lab) else None
        obj = next((o for o in sc.meta["objects"].values() if o.get("body") == body), {})
        key = next((k for k, o in sc.meta["objects"].items() if o.get("body") == body), "")
        name = (names or {}).get(key) or (obj.get("semantic") or {}).get("phrase", task_of(lab))
        tag = np.zeros((CELL[1], head_w, 3), np.uint8) + 24
        img.paste(
            Image.fromarray(V.label(tag, name.removeprefix("the "), V.FG, 18, (10, 10))), (0, head_h + i * CELL[1])
        )
        for j, v in enumerate(vs):
            box = (head_w + j * CELL[0], head_h + i * CELL[1])
            if not has_trace(v["diag"], lab):
                why = (rec(v["diag"], lab) or {}).get("why") or ""
                text = "没有技能，原地不动" if "no skill" in why else "规划拒绝：" + V.why_zh(why)
                cell = np.zeros((CELL[1], CELL[0], 3), np.uint8) + 40
                cell = V.label(cell, text, V.DIM, 16, (10, CELL[1] // 2 - 10))
            else:
                r = RM.Rd(v["diag"] / lab)
                i_d = r.lift() if r.rec["ok"] and r.lift() is not None else r.n - 1
                i_d = min(r.n - 1, i_d + 50)
                p = RM.plan(sc, [r], None)[0]
                sc.set(r, i_d)
                lk = V.follow_lookat(r.z["robot"][i_d, :3], p["obj0"])
                cam = dict(lookat=lk, distance=min(V.follow_distance(r.z["robot"][i_d, :3], p["obj0"], 1.3),
                                                   sc.clear_distance(lk, p["az1"], -26)), azimuth=p["az1"], elevation=-26)  # fmt: skip
                cell = sc.render(cam, CELL, ghosts=[r.z["ref_wrists"][i_d, k] for k in range(2)])
                ok = r.rec["ok"]
                cell = V.label(cell, "成功" if ok else "失败", (90, 220, 120) if ok else (255, 110, 100), 20, (8, 6))
            img.paste(Image.fromarray(cell), box)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    img.convert("RGB").save(out.with_suffix(".jpg"), quality=85)
    return out


def object_name(sc: V.Scene, r: RM.Rd, names: dict | None) -> str:
    """The round's object as the scene file phrases it (or `--cards` "_names"), without a leading "the"."""
    body = r.key.removeprefix("obj_")
    key = next((k for k, o in sc.meta["objects"].items() if o.get("body") == body), "")
    phrase = ((sc.meta["objects"].get(key) or {}).get("semantic") or {}).get("phrase")
    return ((names or {}).get(key) or phrase or task_of(r.label)).removeprefix("the ")


def pick_start(final: Path, labels: list[str]) -> str:
    """The start a task's video shows: the final version's shortest success, else its shortest round. A plan refusal
    is a round too (the robot stands through the settle and the trace is there), so every task has one."""
    ok = [x for x in labels if (rec(final, x) or {}).get("ok") and has_trace(final, x)]
    pool = ok or [x for x in labels if has_trace(final, x)]
    return min(pool, key=lambda x: len(np.load(final / f"{x}.npz")["t"]))


def task_videos(sc: V.Scene, vs: list[dict], out_dir: Path, names: dict | None) -> list[dict]:
    """One video per task of the family: the seed and the final version on the start `pick_start` chooses, side by
    side after a card that lists every version's successes on the task's diagnostic starts. Returns tasks.json's rows."""
    seed, final = vs[0], vs[-1]
    shown = [seed, final] if len(vs) > 1 else [final]
    labels = sorted(p.stem for p in final["diag"].glob("*.json") if not p.name.startswith("_"))
    index = []
    for k, t in enumerate(sorted({task_of(x) for x in labels}), 1):
        mine = [x for x in labels if task_of(x) == t]
        label = pick_start(final["diag"], mine)
        rounds = [RM.Rd(v["diag"] / label) for v in shown]
        counts = [(v["name"], sum(1 for x in mine if (rec(v["diag"], x) or {}).get("ok")), len(mine)) for v in vs]
        name = object_name(sc, rounds[-1], names)
        lines = [f"{k}. {name}", "诊断池成功数：" + " → ".join(f"{n} {o}/{m}" for n, o, m in counts),
                 f"{label} · " + " · ".join(f"{side}：{v['name']}" for side, v in zip(("左", "右"), shown))]  # fmt: skip
        ps = RM.plan(sc, rounds, None)
        panels = [RM.frames(sc, r, p, SPEED) for r, p in zip(rounds, ps)]
        caps = [[(v["name"], V.FG), outcome(r)] for v, r in zip(shown, rounds)]
        out = RM.compose(sc, out_dir / f"task_{k:02d}_{t}.mp4", panels, caps, lines, SPEED)
        index.append(dict(task=t, object=name, label=label, video=str(out), final_ok=bool(rounds[-1].rec["ok"]),
                          versions=[dict(name=n, ok=o, n=m) for n, o, m in counts]))  # fmt: skip
        print("task", k, t, label, out, flush=True)
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sandbox", type=Path, required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cards", type=Path)
    ap.add_argument("--only", default="step,overview,task")
    a = ap.parse_args(argv)
    loop = a.sandbox / "outputs" / "scene_rsi_20261016" / a.family
    vs = versions(loop)
    cards = json.loads(a.cards.read_text()) if a.cards else {}
    a.out.mkdir(parents=True, exist_ok=True)
    sc = V.Scene()
    only = set(a.only.split(","))
    if "step" in only:
        used: set = set()
        for k in range(1, len(vs)):
            p = step_video(sc, k, vs[k - 1], vs[k], cards.get(vs[k]["hash"], {}), used, a.out)
            print("step", k, vs[k]["hash"], p, flush=True)
    if "overview" in only and len(vs) > 1:
        print("overview", overview(sc, vs, a.out / "overview.png", cards.get("_names")), flush=True)
    if "task" in only:
        rows = task_videos(sc, vs, a.out, cards.get("_names"))
        (a.out / "tasks.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    (a.out / "versions.json").write_text(json.dumps([{k: str(v) if isinstance(v, Path) else v for k, v in x.items()}
                                                     for x in vs], indent=1, ensure_ascii=False))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
