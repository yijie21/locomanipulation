"""TODO 33.12 (2026-10-23): the chooser against what the skills actually did -- an offline check, no simulation. For
every recorded round of a set of starts (the sealed and acceptance pools and the redraw's fresh starts of a run), the
chooser's first option for that start is set beside the round's own stance (the pre-grasp's forward distance, hand,
pinch or side, its measured jaw error and outcome), and both get the body model's odds of a miss. The questions:
on the rounds that failed by not delivering the reach (side_reach, pinch_reach, knocked, too_wide), would the chooser
have stood somewhere the body model likes better; on the rounds that succeeded, does it leave them alone.

    PYTHONPATH=src .venv_sim/bin/python scripts/rsi/choose_check.py --sandbox DIR [--out DIR]
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from locomani.tasks.scene import body as B  # noqa: E402
from locomani.tasks.scene import choose as C  # noqa: E402
from locomani.tasks.scene import graspability as G  # noqa: E402

SANDBOX = Path("/workspace/code/rsi-scene/ga-cont-c-20261022")
OUT = REPO / "outputs" / "scene_rsi_20261016" / "choose_20261023"
REACH_FAILS = ("side_reach", "pinch_reach", "knocked", "too_wide", "miss_near")


def rounds_of(sandbox: Path) -> list[tuple[Path, Path]]:
    """(start file, round record) pairs: the sealed pool, the acceptance pool and the redraw's fresh starts."""
    fam = sandbox / "outputs" / "scene_rsi_20261016" / "grasp_atomic"
    hid = sandbox / "outputs" / "scene_rsi_20261016_hidden" / "grasp_atomic"
    inc = json.loads((hid / "gate" / "incumbent.json").read_text())
    gate_dir = Path(inc["rounds"]).parent if inc.get("rounds") else None  # gate/NN_<hash>/candidate -> gate/NN_<hash>
    sealed = sorted(p for p in (hid / "sealed").glob("[0-9][0-9]_*") if p.is_dir())  # the latest sealed version
    gate = [gate_dir] if gate_dir is not None and gate_dir.exists() else []
    pairs = []
    if sealed:
        for p in sorted(sealed[-1].glob("pick_*.json")):
            pairs.append((hid / "pool_sealed" / p.name, p))
    if gate:
        for p in sorted((gate[-1] / "candidate").glob("pick_*.json")):
            pairs.append((hid / "pool_accept" / p.name, p))
    for d in sorted((fam / "diag_redraw").glob("*/draw_*")):
        rounds = d.parent / d.name.replace("draw_", "rounds_")
        for p in sorted(d.glob("pick_*.json")):
            if (rounds / p.name).exists():
                pairs.append((p, rounds / p.name))
    return pairs


def own_option(r: dict) -> dict | None:
    """The round's own first reach as the body model sees it: forward distance, hand, attitude, jaw height."""
    steps = r.get("steps") or []
    geom = next((s for s in steps if s.get("step") == "geom"), None) or {}
    pg = next((s for s in steps if s.get("step") in ("pregrasp", "pre")), None)
    if pg is None or geom.get("shoulder_to_obj") is None:
        return None
    ev = r.get("events") or {}
    hand = ev.get("grip_side") or "right"
    pinch = bool(ev.get("pinch")) or any(s.get("step") == "descend" for s in steps)
    z = (pg.get("grasp") or [None, None, None])[2]
    if z is None:
        z = float(r.get("object_z", 0.0)) or None
    return dict(d=float(geom["shoulder_to_obj"]), hand=hand, pinch=pinch, z=z, jaw_err_cm=pg.get("jaw_err_cm"),
                att_err=pg.get("attitude_err_deg"))  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sandbox", type=Path, default=SANDBOX)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    fam = json.loads((a.sandbox / "scripts" / "rsi" / "families" / "grasp_atomic.json").read_text())
    scene = G.StaticScene(a.sandbox / fam["scene"])
    pairs = rounds_of(a.sandbox)
    print(f"{len(pairs)} rounds", flush=True)
    rows, t0 = [], time.time()
    for start_p, round_p in pairs:
        st = json.loads(start_p.read_text())["start"]
        st = ast.literal_eval(st) if isinstance(st, str) else st
        r = json.loads(round_p.read_text())
        key = r["task"].replace("pick_", "")
        scene.place(st.get("placements") or {})
        best = C.choose(scene, key, st["robot_xy"])
        own = own_option(r)
        own_p = None
        if own and own["z"] is not None:
            att = "down" if own["pinch"] else "level"  # this line's pinches point the wrist straight down
            own_p = B.reach(own["d"], C.SHOULDER_Y_M, float(own["z"]), att, None, own["hand"])["p_miss"]
        rows.append(dict(label=r["label"], task=key, ok=bool(r["ok"]), cls=r.get("failure_class") or "",
                         own=own, own_p_miss=own_p,
                         chosen=best.as_dict() if best else None))  # fmt: skip
    print(f"chooser ran on {len(rows)} starts in {time.time() - t0:.0f} s", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "check.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    # the summary
    by_cls = collections.defaultdict(list)
    for x in rows:
        by_cls["ok" if x["ok"] else (x["cls"] if x["cls"] in REACH_FAILS else "other_fail")].append(x)
    lines = [f"# chooser check: {len(rows)} rounds of {a.sandbox.name}", ""]
    lines.append(
        "| rounds | n | own p_miss (median) | chooser p_miss (median) | chooser lower by >= 0.2 | changes hand | changes grasp kind | needs prep | none |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cls in ["ok", *REACH_FAILS, "other_fail"]:
        xs = by_cls.get(cls) or []
        if not xs:
            continue
        own_p = [x["own_p_miss"] for x in xs if x["own_p_miss"] is not None]
        ch_p = [x["chosen"]["p_miss"] for x in xs if x["chosen"]]
        lower = sum(
            1
            for x in xs
            if x["chosen"] and x["own_p_miss"] is not None and x["own_p_miss"] - x["chosen"]["p_miss"] >= 0.2
        )
        hand = sum(1 for x in xs if x["chosen"] and x["own"] and x["chosen"]["hand"] != x["own"]["hand"])
        kind = sum(
            1 for x in xs if x["chosen"] and x["own"] and (x["chosen"]["kind"] != "side") != bool(x["own"]["pinch"])
        )
        prep = sum(1 for x in xs if x["chosen"] and x["chosen"]["needs_prep"])
        none = sum(1 for x in xs if not x["chosen"])
        med = lambda v: f"{np.median(v):.2f}" if v else "-"  # noqa: E731
        lines.append(
            f"| {cls} | {len(xs)} | {med(own_p)} | {med(ch_p)} | {lower} | {hand} | {kind} | {prep} | {none} |"
        )
    lines += [
        "",
        "## per task",
        "",
        "| task | n | ok | own p_miss | chooser p_miss | chooser kinds | hands |",
        "|---|---|---|---|---|---|---|",
    ]
    by_task = collections.defaultdict(list)
    for x in rows:
        by_task[x["task"]].append(x)
    for t, xs in sorted(by_task.items()):
        own_p = [x["own_p_miss"] for x in xs if x["own_p_miss"] is not None]
        ch_p = [x["chosen"]["p_miss"] for x in xs if x["chosen"]]
        kinds = collections.Counter(f"{x['chosen']['kind']}/{x['chosen']['att']}" for x in xs if x["chosen"])
        hands = collections.Counter(x["chosen"]["hand"] for x in xs if x["chosen"])
        lines.append(f"| {t} | {len(xs)} | {sum(x['ok'] for x in xs)} | {np.median(own_p) if own_p else float('nan'):.2f} | "
                     f"{np.median(ch_p) if ch_p else float('nan'):.2f} | {dict(kinds)} | {dict(hands)} |")  # fmt: skip
    lines += ["", "## the reach failures, one by one", ""]
    for x in rows:
        if x["ok"] or x["cls"] not in REACH_FAILS:
            continue
        o, c = x["own"], x["chosen"]
        own_s = (f"own: {o['d']:.2f} m ahead, {o['hand']}, {'pinch' if o['pinch'] else 'side'}, jaw err {o['jaw_err_cm']} cm, "
                 f"p_miss {x['own_p_miss']}") if o else "own: no pre-grasp recorded"  # fmt: skip
        ch_s = (f"chooser: {c['kind']}/{c['att']}, {c['hand']}, {c['distance_m']:.2f} m ahead, approach {c['approach_deg']:+.0f} deg, "
                f"pelvis {c['pelvis_z']}, p_miss {c['p_miss']}{' NEEDS PREP' if c['needs_prep'] else ''}") if c else "chooser: none"  # fmt: skip
        lines.append(f"- {x['label']} [{x['cls']}] -- {own_s} | {ch_s}")
    (a.out / "check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
