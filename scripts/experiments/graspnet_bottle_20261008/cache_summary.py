"""TODO 23.2.5: the grasp-cache acceptance -- per object, the cache's yield and every cache round's outcome, against
the 21.3.4 graspnet rounds (summary.json: IK 16/65, one round per object).

Reads <scene>/grasp_cache_summary.json, outputs/runs/gnb_obj_XXX<tag>/grasp_planned/round_*/record.json (the
`ik` rounds of run_grasp.py --grasp-source cache --planners ik) and the graspnet summary. Each cache round is
classed: `lifted`; `no_candidate` (nothing validated fits at this pose); `no_path` (the IK planner reached none);
`execution` (a validated grasp was reached and the expert's entry / close / lift still failed -- the pose is cleared,
the execution is not) with the expert's failure text. Works for runs of either grasp source (graspnet rounds have no
cache columns). Writes outputs/graspnet_bottle_20261008/cache/summary.json and summary.md for the acceptance tag
_cache, summary<tag>.json / .md for any other tag.

    python3 scripts/experiments/graspnet_bottle_20261008/cache_summary.py [tag _cache]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "outputs" / "graspnet_bottle_20261008"


def classify(rec: dict | None, manifest_row: dict | None) -> tuple[str, str]:
    if rec is None:
        why = (manifest_row or {}).get("why") or (manifest_row or {}).get("reason") or "no record"
        return "crash", str(why)
    if rec.get("ok"):
        return "lifted", ""
    why = rec.get("why", "")
    if "no cached grasp fits" in why or "no graspnet candidate" in why:
        return "no_candidate", why
    if "found no path" in why or "no path" in why:
        return "no_path", why
    if "perception/planning error" in why:
        return "crash", why
    return "execution", why


def main(tag: str = "_cache") -> int:
    objects = json.loads((ROOT / "objects.json").read_text())["objects"]
    graspnet = {s["id"]: s for s in json.loads((ROOT / "summary.json").read_text())}
    rows = []
    for o in objects:
        oid = o["id"]
        scene = ROOT / "scenes" / f"obj_{oid:03d}"
        cache = (
            json.loads((scene / "grasp_cache_summary.json").read_text())
            if (scene / "grasp_cache_summary.json").exists()
            else {}
        )
        run = REPO / "outputs" / "runs" / f"gnb_obj_{oid:03d}{tag}"
        manifest = {}
        mf = run / "skills" / "manifest.jsonl"
        if mf.exists():
            for k, line in enumerate(mf.read_text().splitlines()):
                try:
                    manifest[k] = json.loads(line)
                except json.JSONDecodeError:
                    pass
        rounds = []
        for rd in sorted((run / "grasp_planned").glob("round_*")) if (run / "grasp_planned").exists() else []:
            rec = json.loads((rd / "record.json").read_text()) if (rd / "record.json").exists() else None
            k = int(rd.name.split("_")[1])
            kind, why = classify(rec, manifest.get(k))
            g = (rec or {}).get("grasp") or {}
            rounds.append(
                dict(
                    round=k, kind=kind, why=why, tier=g.get("tier"), score=g.get("score"), width=g.get("width"),
                    row=g.get("row"), candidates=(rec or {}).get("candidates"),
                    fit=((rec or {}).get("cache") or {}).get("fit"), rise_m=(rec or {}).get("rise_m"),
                    plan_seconds=((rec or {}).get("plan") or {}).get("plan_seconds"),
                    pregrasp_error_m=(rec or {}).get("pregrasp_error_open_loop_m"),
                )
            )  # fmt: skip
        gn = graspnet.get(oid, {}).get("ik", {})
        rows.append(
            dict(
                id=oid, height=o["height"], width=o["width"], mass=o["mass"],
                cache=dict(sampled=cache.get("sampled"), closed=cache.get("closed"), held=cache.get("held"), why=cache.get("why")),
                rounds=rounds,
                lifted=sum(r["kind"] == "lifted" for r in rounds),
                graspnet_ik=dict(ok=gn.get("ok"), why=gn.get("why"), tier=gn.get("tier")),
            )
        )  # fmt: skip
    out = ROOT / "cache"
    out.mkdir(exist_ok=True)
    stem = "summary" if tag == "_cache" else f"summary{tag}"
    (out / f"{stem}.json").write_text(json.dumps(rows, indent=1))
    n = len(rows)
    with_cache = [r for r in rows if r["cache"]["held"]]
    ran = [r for r in rows if r["rounds"]]
    kinds = Counter(x["kind"] for r in rows for x in r["rounds"])
    exec_whys = Counter(x["why"].split(" (")[0] for r in rows for x in r["rounds"] if x["kind"] == "execution")
    any_lift = sum(1 for r in rows if r["lifted"])
    total_rounds = sum(len(r["rounds"]) for r in rows)
    gn_ok = sum(1 for r in rows if r["graspnet_ik"]["ok"])
    lines = [
        f"# grasp_planned rounds ({tag})",
        "",
        f"- objects {n}; with a validated grasp in the cache {len(with_cache)}; ran {len(ran)}; rounds {total_rounds}",
        f"- cache yield: sampled {sum(r['cache']['sampled'] or 0 for r in rows)}, closed {sum(r['cache']['closed'] or 0 for r in rows)}, held {sum(r['cache']['held'] or 0 for r in rows)}",
        f"- rounds by class: {dict(kinds)}",
        f"- lifted in at least one round: {any_lift}/{n} objects; round 0 (the 21.3.4 yaw) lifted {sum(1 for r in rows if r['rounds'] and r['rounds'][0]['kind'] == 'lifted')}/{len(ran)}; graspnet IK (21.3.4) lifted {gn_ok}/{n}",
        f"- execution failures by text: {dict(exec_whys)}",
        "",
        "| id | h cm | w cm | kg | cache held/sampled | rounds (kind:tier) | graspnet IK |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rr = " ".join(f"{x['kind']}:{x['tier']}" for x in r["rounds"]) or "-"
        g = r["graspnet_ik"]
        lines.append(
            f"| {r['id']:03d} | {100 * r['height']:.1f} | {100 * r['width']:.1f} | {r['mass']:.2f} | "
            f"{r['cache']['held']}/{r['cache']['sampled']} | {rr} | {'ok' if g['ok'] else (g['why'] or '-')[:40]} |"
        )
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "_cache"))
