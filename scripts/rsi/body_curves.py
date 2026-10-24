"""TODO 33.1.7 (plan section 1.2, D17): gather the body model's measured curves and constraints of finished sandboxes
into the origin's body directory, which every new sandbox copies (sandbox.py) and skills read at run time
(locomani.tasks.scene.body.curves / best_level). Run by a person over finished sandboxes, like records.py build; the
loop never writes the origin's files (D1).

    python3 scripts/rsi/body_curves.py build $LOCOMANI_WORKSPACE/code/rsi-scene/<sandbox> [...]
    python3 scripts/rsi/body_curves.py show [--name pick.STANCE_MIN_D_M]

`build` reads each sandbox's outputs/scene_rsi_20261016/body/: curves.jsonl (the snapshot it was built with, plus
the design probes of runs before TODO 33.1.7, which wrote there), curves_run.jsonl (its own run's probes) and
constraints.jsonl (layer attribution's `motor` verdicts); adds the sandbox's name as `source`, the machine this runs on
where a row has none (every sandbox so far ran here; the A800's probes were never written as curves), the scope
(`skill` for a parameter, `start` for a start factor) and the controller from the probe's own round records; drops
rows seen before (same family, incumbent, iteration, hypothesis or start label, probe name and factor); and appends
the new ones to the origin's files. Constraints are for the roles (they name diagnostic starts); skills read curves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import records as RC  # noqa: E402

BODY = REPO / "outputs" / "scene_rsi_20261016" / "body"
SUB = Path("outputs") / "scene_rsi_20261016" / "body"


def jl(p: Path) -> list[dict]:
    out = []
    for line in p.read_text().splitlines() if p.exists() else []:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def key(r: dict) -> tuple:
    return (r.get("family"), r.get("incumbent"), r.get("iter"), r.get("hypothesis") or r.get("label"), r.get("name"),
            r.get("factor") or r.get("kind"))  # fmt: skip


def controller_of(r: dict) -> str | None:
    """The controller the probe's rounds ran under, from their records (None when they are gone)."""
    for p in sorted(Path(r.get("dir") or "/nonexistent").glob("level_*/*.json")):
        try:
            c = json.loads(p.read_text()).get("controller")
        except (ValueError, OSError):
            continue
        if c:
            return c
    return None


def build(sandboxes: list[Path], out: Path = BODY) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    added = {}
    for name, sources in (
        ("curves.jsonl", ("curves.jsonl", "curves_run.jsonl")),
        ("constraints.jsonl", ("constraints.jsonl",)),
    ):
        have = jl(out / name)
        seen = {key(r) for r in have}
        new = []
        for sb in sandboxes:
            for src in sources:
                for r in jl(sb / SUB / src):
                    if key(r) in seen:
                        continue
                    seen.add(key(r))
                    r = dict(r, source=r.get("source") or sb.name, machine=r.get("machine") or RC.machine())
                    if name == "curves.jsonl":
                        r.setdefault("scope", "skill" if r.get("factor") == "param" else "start")
                        r["controller"] = r.get("controller") or controller_of(r)
                    new.append(r)
        with (out / name).open("a") as fh:
            for r in new:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        added[name] = len(new)
    return added


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("sandboxes", type=Path, nargs="+")
    b.add_argument("--out", type=Path, default=BODY)
    s = sub.add_parser("show")
    s.add_argument("--name", default=None)
    s.add_argument("--curves", type=Path, default=BODY / "curves.jsonl")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(json.dumps(build([p.resolve() for p in a.sandboxes], a.out)))
        return 0
    from locomani.tasks.scene import body

    cs = body.curves(name=a.name, path=str(a.curves))
    for c in cs:
        lv = ", ".join(f"{x['level']:g}: {x['ok']}/{x['n']}" for x in c.get("levels") or [])
        print(f"{c.get('source')} iter {c.get('iter')} {c.get('factor')} {c.get('name') or ''} {c.get('tasks')} "
              f"[{c.get('scope')}, {c.get('machine')}, {c.get('controller')}]: {lv}")  # fmt: skip
    for name in sorted({c["name"] for c in cs if c.get("factor") == "param" and c.get("name")}):
        print(f"best {name}: {json.dumps(body.best_level(name, path=str(a.curves)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
