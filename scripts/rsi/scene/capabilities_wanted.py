#!/usr/bin/env python3
"""TODO 33.4 (plan docs/handoff/scene-rsi-strategy-plan-20261021.md section 5): what the inner loops said they could not
do, gathered across every task family. Two sources in each loop sandbox under --root:
  - the "做不到" claims' exits (`loop/iter_*/hypotheses.json`, `infeasible[].exits[].capability`): the way out the
    Reviewer named for a claimed-infeasible task, with the starts it blocks;
  - the Manager's gap proposals (`loop/gaps.jsonl`: name, capability, tasks).
A family's same capability on the same task counts once (replays and A/B sandboxes of a family repeat its claims).
The names are free text ("slide to the edge", "pull it to the edge", "drag it out first"); one Claude call groups them
into classes of the same physical action (`--no-llm`: each normalised name is its own class). Per class: the names,
how many families and which, the (family, start) pairs it would unblock, the tasks, one example of how. Writes
outputs/scene_rsi_20261016/outer/capabilities_wanted.json; outer.py select --wanted <it> weights by it.

    python3 scripts/rsi/capabilities_wanted.py [--root /workspace/code/rsi-scene] [--no-llm] [--out <json>]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path("/workspace/code/rsi-scene")
OUT = REPO / "outputs" / "scene_rsi_20261016" / "outer" / "capabilities_wanted.json"
MODEL = "claude-opus-5-5"
SCHEMA = {
    "type": "object",
    "required": ["classes"],
    "properties": {
        "classes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "members"],
                "properties": {"label": {"type": "string"}, "members": {"type": "array", "items": {"type": "string"}}},
            },
        }
    },
}
PROMPT = """Below are capability names that robot-skill improvement loops proposed as ways out of tasks they could not
do (a humanoid with two-finger grippers picking and placing objects in an apartment), each with one example of how it
would be done. Group them into classes of the same physical action: "slide to the edge", "drag it to the edge" and
"pull it out first" are one class; "tilt the wrist" and "move the neighbour first" are not. Every name goes in exactly
one class, written exactly as given. Give each class a short English label.

"""


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 +-]", " ", name.lower())).strip()


def collect(root: Path) -> list[dict]:
    """One row per (family, task, capability name): source, how, starts."""
    rows: dict = {}
    for f in sorted(root.glob("*/outputs/scene_rsi_20261016/*/loop/iter_*/hypotheses.json")):
        fam = f.parents[2].name
        try:
            h = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for c in h.get("infeasible") or []:
            if not isinstance(c, dict):
                continue
            for e in c.get("exits") or []:
                name = e.get("capability") if isinstance(e, dict) else None
                if not name:
                    continue
                r = rows.setdefault((fam, c.get("task", ""), norm(name)), dict(
                    family=fam, task=c.get("task", ""), name=name, source="exit", how=e.get("how", ""), starts=set()))  # fmt: skip
                r["starts"] |= set(c.get("starts") or [])
    for f in sorted(root.glob("*/outputs/scene_rsi_20261016/*/loop/gaps.jsonl")):
        fam = f.parents[1].name
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            for t in p.get("tasks") or [""]:
                rows.setdefault((fam, t, norm(p["name"])), dict(family=fam, task=t, name=p["name"], source="gap",
                                                                how=p.get("capability", ""), starts=set()))  # fmt: skip
    return list(rows.values())


def group(names: dict[str, str], use_llm: bool) -> dict[str, str]:
    """name -> class label. `names`: one name -> an example `how`."""
    if not use_llm:
        return {n: norm(n) for n in names}
    listing = "\n".join(f"- {n}: {h[:200]}" for n, h in sorted(names.items()))
    cmd = ["claude", "-p", "--model", MODEL, "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
           "--settings", json.dumps({"disableAllHooks": True}), "--strict-mcp-config", "--effort", "medium",
           "--disallowedTools", "Bash,Edit,Write,NotebookEdit,Agent,WebFetch,WebSearch,Read,Glob,Grep"]  # fmt: skip
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_CODE_SSE")}
    env.update(CLAUDE_CODE_DISABLE_AUTO_MEMORY="1", CLAUDE_CODE_DISABLE_CLAUDE_MDS="1")
    r = subprocess.run(cmd, input=PROMPT + listing, capture_output=True, text=True, env=env, cwd="/tmp")
    so = json.loads(r.stdout).get("structured_output")
    if not so:
        raise SystemExit(f"no grouping from claude: {r.stdout[-300:]} {r.stderr[-300:]}")
    out = {}
    for c in so["classes"]:
        for m in c["members"]:
            if m in names:
                out[m] = c["label"]
    for n in names:  # a name the call dropped stays its own class
        out.setdefault(n, norm(n))
    return out


def aggregate(rows: list[dict], label: dict[str, str]) -> list[dict]:
    by = collections.defaultdict(lambda: dict(names=set(), families=set(), starts=set(), tasks=set(), how=""))
    for r in rows:
        c = by[label[r["name"]]]
        c["names"].add(r["name"])
        c["families"].add(r["family"])
        c["starts"] |= {(r["family"], s) for s in r["starts"]}
        if r["task"]:
            c["tasks"].add(r["task"])
        c["how"] = c["how"] or r["how"]
    out = [dict(label=k, names=sorted(v["names"]), families=sorted(v["families"]), n_families=len(v["families"]),
                starts_blocked=len(v["starts"]), tasks=sorted(v["tasks"]), example_how=v["how"][:300])
           for k, v in by.items()]  # fmt: skip
    return sorted(out, key=lambda c: (-c["n_families"], -c["starts_blocked"], c["label"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-llm", action="store_true", help="no Claude grouping: each normalised name its own class")
    a = ap.parse_args(argv)
    rows = collect(a.root)
    names = {}
    for r in rows:
        names.setdefault(r["name"], r["how"])
    label = group(names, not a.no_llm)
    classes = aggregate(rows, label)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(root=str(a.root), rows=len(rows), names=len(names), classes=classes), indent=1))
    print(f"{len(rows)} (family, task, name) rows, {len(names)} names -> {len(classes)} classes ({a.out})")
    for c in classes[:15]:
        print(f"  {c['n_families']:2d} families {c['starts_blocked']:3d} starts  {c['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
