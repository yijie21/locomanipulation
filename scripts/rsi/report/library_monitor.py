"""TODO 33.11 (plan section 3.3, from Dream-RSI): does the experience library narrow the loop's search? Two numbers per
run, read from the records only -- no new runs, no model calls:

* borrow, two ways. `cites`: the share of the Reviewer's hypotheses whose claim, instead, evidence or needs_code name
  the library or one of its entries (the Reviewers write "the library's r2_on_single cue", "pick_single's lesson"),
  and which entries. `worded`: the share whose text (claim + instead) is closer to an entry's cue + lesson than
  library-less hypotheses get to any entry -- TF-IDF cosine above the CHANCE_Q quantile of their best match -- in a
  session that read the index (it carries every cue) or opened that entry. On the grasp_atomic runs to 2026-10-23
  the first is 17 / 41 and the second about chance: the Reviewers cite entries as evidence and paraphrase them.
* mechanisms: hypotheses per iteration (the Reviewer's are distinct within an iteration), and over the run the
  distinct mechanisms by the Manager's `repeats` links (TODO 33.14: this iteration's hypothesis is iteration N's
  mechanism), i.e. how many hypotheses are a mechanism already proposed. Runs before 33.14 carry no links: `n/a`.
  Text similarity cannot stand in for the links: the Manager-linked pairs' cosines (0.04-0.48, median about 0.12)
  overlap the unlinked pairs' (median 0.04, 99th percentile 0.24).

`run` reads sandboxes (each holds one family's loop; a replay's copied iterations are left out, only the run's own
count); `ab` compares the library A/B replays of TODO 32.10 (same starting state with and without the library),
paired by state and repetition. The chance threshold comes from every no-library sandbox given to the command.

    python3 scripts/rsi/report/library_monitor.py run <sandbox> [<sandbox> ...] [--baseline <sandbox> ...] [--out f.json]
    python3 scripts/rsi/report/library_monitor.py ab outputs/rsi_speed_20261021/ab/plan_library.json [--out f.json]

`--baseline` adds no-library sandboxes for the chance level only; without any, `run` uses the library A/B replays'
no-library arms when they exist on this machine (AB_ROOT), else it reports similarities without a borrow verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
AB_PLAN = REPO / "outputs" / "rsi_speed_20261021" / "ab" / "plan_library.json"
AB_ROOT = Path("/workspace/code/rsi-scene")  # where the A/B replays were built (TODO 32.10, 2026-10-21)
CHANCE_Q = 0.95  # a borrowed hypothesis is closer to an entry than 95 % of library-less hypotheses get
CITE = re.compile(r"experience|librar(?:y|ies)", re.I)  # plus the entries' names, per sandbox (cite_pattern)
WORD = re.compile(r"[a-z][a-z0-9_]{2,}")
STOP = set(
    "the and for that this with from into onto than then when where which while what have has had are was were "
    "not but its it's can could would should will does did done any all one two each per also only more most less "
    "very so such them they their there here over under about after before because between both same other these "
    "those how why our out off via use used using get gets got set sets may might must just like".split()
)


def words(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


def loop_dir(sandbox: Path) -> Path:
    loops = sorted(sandbox.glob("outputs/scene_rsi_20261016/*/loop/ledger.jsonl"))
    if len(loops) != 1:
        raise SystemExit(f"{sandbox}: expected one family loop, found {len(loops)}")
    return loops[0].parent


def transcript(sandbox: Path, session: str) -> Path | None:
    p = Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(sandbox)) / f"{session}.jsonl"
    return p if p.exists() else None


def seen(sandbox: Path, it_dir: Path) -> dict:
    """What the iteration's Reviewer session(s) read of the library: the index, and which entries' own files."""
    out = {"index": False, "entries": set(), "sessions": 0}
    for raw in sorted(it_dir.glob("reviewer*_raw.json")):
        session = json.loads(raw.read_text()).get("session_id")
        p = transcript(sandbox, session) if session else None
        if p is None:
            continue
        out["sessions"] += 1
        for line in p.open():
            if '"tool_use"' not in line or "experience" not in line:
                continue
            try:
                content = (json.loads(line).get("message") or {}).get("content")
            except ValueError:
                continue
            for c in content if isinstance(content, list) else []:
                if not (isinstance(c, dict) and c.get("type") == "tool_use"):
                    continue
                s = json.dumps(c.get("input"))
                out["index"] |= "experience/INDEX.md" in s
                out["entries"] |= set(re.findall(r"scripts/rsi/experience/([A-Za-z0-9_]+)/", s))
    return out


def library(sandbox: Path) -> dict[str, str]:
    """The entries a sandbox holds: name -> cue + lesson."""
    out = {}
    for f in sorted((sandbox / "scripts" / "rsi" / "experience").glob("*/entry.json")):
        x = json.loads(f.read_text()).get("experience") or {}
        out[f.parent.name] = f"{x.get('cue', '')} {x.get('lesson', '')}"
    return out


def run_iterations(sandbox: Path) -> list[Path]:
    """The run's own iterations with hypotheses: a replay's copied earlier iterations are another run's."""
    loop = loop_dir(sandbox)
    meta = json.loads((loop / "meta.json").read_text()) if (loop / "meta.json").exists() else {}
    first = (meta.get("replay") or {}).get("iteration", 0)
    its = [d for d in sorted(loop.glob("iter_*")) if (d / "hypotheses.json").exists()]
    return [d for d in its if int(d.name.split("_")[1]) >= first]


def hypotheses(it_dir: Path) -> list[dict]:
    h = json.loads((it_dir / "hypotheses.json").read_text())
    return [x for x in h.get("hypotheses", []) if isinstance(x, dict)]


def text(h: dict) -> str:
    return f"{h.get('claim', '')} {h.get('instead', '')}"


class Space:
    """TF-IDF over every hypothesis and entry text the command reads, so generic vocabulary (stance, grasp, hand)
    weighs little and shared specifics weigh much."""

    def __init__(self, docs: list[str]):
        self.df = Counter(w for d in docs for w in set(words(d)))
        self.n = max(1, len(docs))

    def vec(self, doc: str) -> dict[str, float]:
        tf = Counter(words(doc))
        v = {w: c * math.log((1 + self.n) / (1 + self.df[w])) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    @staticmethod
    def cos(a: dict, b: dict) -> float:
        return sum(x * b.get(w, 0.0) for w, x in a.items())


def cited(h: dict, names: list[str]) -> list[str]:
    """The library words and entry names a hypothesis's own fields mention ("library" for the library as such)."""
    blob = " ".join(str(h.get(k) or "") for k in ("claim", "instead", "evidence", "needs_code"))
    out = {n for n in names if re.search(rf"\b{re.escape(n)}\b", blob)}
    return sorted(out) or (["library"] if CITE.search(blob) else [])


def links(it_dir: Path) -> list[tuple[str, int, str]]:
    """The Manager's repeats for this iteration: (this hypothesis id, earlier iteration, its id)."""
    m = it_dir / "manager.json"
    out = []
    for r in (json.loads(m.read_text()).get("repeats") or []) if m.exists() else []:
        for x in r.get("same_as") or []:
            k, _, hid = str(x).partition(":")
            if k.isdigit() and hid:
                out.append((r.get("now"), int(k), hid))
    return out


def collect(sandboxes: list[Path]) -> list[dict]:
    rows = []
    for sb in sandboxes:
        lib = library(sb)
        for it in run_iterations(sb):
            rows.append(
                dict(sandbox=sb, it=it, hyps=hypotheses(it), lib=lib, seen=seen(sb, it) if lib else None,
                     links=links(it))
            )  # fmt: skip
    return rows


def chance(space: Space, base_rows: list[dict], entries: dict[str, str]) -> float | None:
    """The CHANCE_Q quantile of library-less hypotheses' best cosine to any entry."""
    ev = [space.vec(t) for t in entries.values()]
    best = sorted(max(Space.cos(space.vec(text(h)), e) for e in ev) for r in base_rows for h in r["hyps"] if ev)
    return best[min(len(best) - 1, int(CHANCE_Q * len(best)))] if best else None


def summarise(space: Space, rows: list[dict], threshold: float | None) -> dict:
    """Per sandbox: per iteration and over the run, hypotheses, citations, worded borrows, mechanism repeats."""
    out = {}
    for sb in dict.fromkeys(r["sandbox"] for r in rows):
        mine = [x for x in rows if x["sandbox"] == sb]
        names = sorted(mine[0]["lib"]) + ["grasp_atomic"] if mine[0]["lib"] else []
        its, entries_cited, linked_any = [], Counter(), False
        for r in mine:
            ev = {k: space.vec(v) for k, v in r["lib"].items()}
            cites, worded, sims = [], [], []
            for h in r["hyps"]:
                c = cited(h, names) if r["lib"] else []
                if isinstance(h, dict) and h.get("knowledge_refs"):  # an explicit knowledge-library citation
                    c = sorted(set(c) | set(map(str, h["knowledge_refs"])))
                if c:
                    cites.append(dict(hypothesis=h.get("id"), names=c))
                    entries_cited.update(c)
                if not ev:
                    continue
                v = space.vec(text(h))
                name, sim = max(((k, Space.cos(v, e)) for k, e in ev.items()), key=lambda kv: kv[1])
                sims.append(round(sim, 3))
                saw = r["seen"] and (r["seen"]["index"] or name in r["seen"]["entries"])
                if threshold is not None and saw and sim > threshold:
                    worded.append(dict(hypothesis=h.get("id"), entry=name, cos=round(sim, 3)))
            repeats = sorted({now for now, _, _ in r["links"]})
            linked_any |= bool(r["links"]) or (r["it"] / "manager.json").exists() and _has_repeats_field(r["it"])
            its.append(
                dict(
                    iteration=r["it"].name,
                    hypotheses=len(r["hyps"]),
                    repeats_earlier=len(repeats),
                    read_index=bool(r["seen"] and r["seen"]["index"]),
                    opened=sorted(r["seen"]["entries"]) if r["seen"] else [],
                    cites=cites,
                    worded=worded,
                    best_cos=sims,
                )
            )
        n_h = sum(i["hypotheses"] for i in its)
        out[sb.name] = dict(
            library=bool(mine[0]["lib"]),
            iterations=its,
            hypotheses=n_h,
            hypotheses_per_iteration=round(n_h / max(1, len(its)), 2),
            mechanisms_run=(n_h - sum(i["repeats_earlier"] for i in its)) if linked_any else "n/a",
            cites=round(sum(len(i["cites"]) for i in its) / max(1, n_h), 3) if mine[0]["lib"] else None,
            entries_cited=dict(entries_cited.most_common()),
            worded=None if threshold is None or not mine[0]["lib"] else round(
                sum(len(i["worded"]) for i in its) / max(1, n_h), 3),
            read_index=sum(i["read_index"] for i in its),
            opened_entries=sum(bool(i["opened"]) for i in its),
        )  # fmt: skip
    return out


def _has_repeats_field(it_dir: Path) -> bool:
    """A Manager decision written under 33.14 or later carries `repeats` (possibly empty)."""
    return "repeats" in json.loads((it_dir / "manager.json").read_text())


def ab_sandboxes(plan: Path) -> list[tuple[str, str, str, Path]]:
    """(state, repetition, arm, sandbox) of every library A/B replay present on this machine."""
    out = []
    for p in json.loads(plan.read_text()):
        _, state, arm, rep = p["name"].split("-")
        sb = AB_ROOT / f"ab-{p['name']}"
        if (sb / "outputs").exists():
            out.append((state, rep, arm, sb))
    return out


def corpus(rows: list[dict]) -> list[str]:
    docs = [text(h) for r in rows for h in r["hyps"]]
    return docs + [t for r in rows for t in r["lib"].values()]


def cmd_run(a) -> dict:
    sbs = [Path(s).resolve() for s in a.sandboxes]
    base = [Path(s).resolve() for s in a.baseline] or [
        sb for state, rep, arm, sb in (ab_sandboxes(AB_PLAN) if AB_PLAN.exists() else []) if arm == "none"
    ]
    rows, base_rows = collect(sbs), collect([b for b in base if b not in sbs])
    base_rows += [r for r in rows if not r["lib"]]
    space = Space(corpus(rows + base_rows))
    entries = next((r["lib"] for r in rows if r["lib"]), {})
    threshold = chance(space, base_rows, entries)
    return dict(chance_cos=threshold, chance_q=CHANCE_Q, runs=summarise(space, rows, threshold))


def cmd_ab(a) -> dict:
    arms = ab_sandboxes(Path(a.plan))
    rows = collect([sb for *_, sb in arms])
    space = Space(corpus(rows))
    entries = next((r["lib"] for r in rows if r["lib"]), {})
    threshold = chance(space, [r for r in rows if not r["lib"]], entries)
    runs = summarise(space, rows, threshold)
    pairs = []
    for state, rep in sorted({(s, r) for s, r, *_ in arms}):
        arm = {x: runs.get(sb.name) for s, r, x, sb in arms if (s, r) == (state, rep)}
        if arm.get("lib") and arm.get("none"):
            pairs.append(
                dict(
                    state=state,
                    rep=rep,
                    **{
                        f"{x}_{k}": arm[x][k]
                        for x in ("lib", "none")
                        for k in ("hypotheses", "cites", "worded", "read_index", "opened_entries")
                    },
                )
            )
    tot = {k: sum(p[k] for p in pairs) for k in ("lib_hypotheses", "none_hypotheses")}
    tot["lib_cites"] = sum(len(i["cites"]) for x in runs.values() if x["library"] for i in x["iterations"])
    tot["lib_worded"] = sum(len(i["worded"]) for x in runs.values() if x["library"] for i in x["iterations"])
    return dict(chance_cos=threshold, chance_q=CHANCE_Q, pairs=pairs, totals=tot, runs=runs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("sandboxes", nargs="+")
    r.add_argument("--baseline", nargs="*", default=[])
    r.add_argument("--out", type=Path)
    b = sub.add_parser("ab")
    b.add_argument("plan", nargs="?", default=str(AB_PLAN))
    b.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    rep = cmd_run(a) if a.cmd == "run" else cmd_ab(a)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=1, default=str))
    print(f"chance cos (q {CHANCE_Q} of library-less hypotheses' best match): {rep['chance_cos']}")
    for name, x in rep["runs"].items():
        print(f"{name}: library {x['library']}, {len(x['iterations'])} iterations, {x['hypotheses']} hypotheses "
              f"({x['hypotheses_per_iteration']} / iteration), distinct mechanisms over the run {x['mechanisms_run']}; "
              f"read index {x['read_index']}, opened entries {x['opened_entries']}; cites {x['cites']} "
              f"{x['entries_cited']}, worded {x['worded']}")  # fmt: skip
    for p in rep.get("pairs", []):
        print(f"  {p['state']} {p['rep']}: lib {p['lib_hypotheses']} hypotheses, cites {p['lib_cites']}, worded "
              f"{p['lib_worded']}; none {p['none_hypotheses']} hypotheses")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
