"""Failure injection at a phase boundary (agentic-plan Q11 / milestone M3).

An `Injection` is one world event the runner fires once per episode, right after the first successful
call of a named primitive: the object teleports through the sim's episode channel (`pin_object` for
`settle_s`, then release), so the plan being executed is now wrong. Both arms of the paired comparison
get the same event at the same phase on the same seed: the fixed plan (`scripts/plan_episode.py --plan`,
which cannot react) and the agent (`--agent`, which reads the state after every call). Nothing tells the
agent an injection happened; it has the privileged-state text and the session snapshot, and the
snapshot still says what the last primitive left there (`held` after a `drop_held`), because that is
what proprioception would say too.

Kinds:
- `move_target`: the object moves by `offset_m` (world xy) from where it is now, keeping its height and
  orientation. Fired after `approach`: the stance is now wrong for the grasp.
- `drop_held`: the object goes back onto its surface at its scene `init_pos` plus `offset_m`, upright.
  Fired after `grasp_single` / `grasp_bimanual`: the hand closes on nothing; `held` is stale.

The two need only what the sim already serves (`EpisodeControlClient.pin_object`); pushing the robot
(`走偏`) or a force on the load (`施力`) would need a sim-side command that does not exist yet.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

KINDS = ("move_target", "drop_held")


@dataclass(frozen=True)
class Injection:
    kind: str  # one of KINDS
    after: str  # primitive whose first successful completion fires the event
    target: str | None = None  # object key; None = the fired call's `target` arg, else the held object
    offset_m: tuple[float, float] = (0.0, 0.0)  # world xy, metres
    settle_s: float = 0.5  # how long the pin holds before release

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"injection kind {self.kind!r}; known: {KINDS}")
        if len(self.offset_m) != 2 or not np.isfinite(self.offset_m).all():
            raise ValueError("injection offset_m must be two finite metres")


def load_injection(path: str | Path) -> Injection:
    raw = json.loads(Path(path).read_text())
    if "offset_m" in raw:
        raw["offset_m"] = tuple(float(v) for v in raw["offset_m"])
    return Injection(**raw)


class Injector:
    """Fires one `Injection` per episode; `after(entry, executor)` is called with every trace entry."""

    def __init__(self, injection: Injection, *, log=print, sleep=time.sleep):
        self.injection, self.log, self.sleep = injection, log, sleep
        self.fired: dict | None = None
        self.calls = 0

    def after(self, entry: dict, executor) -> dict | None:
        """Returns the event when this entry fires the injection, else None."""
        self.calls += 1
        if self.fired is not None or "outcome" not in entry or not entry["outcome"]["ok"]:
            return None
        if entry.get("primitive") != self.injection.after:
            return None
        session = executor.session
        key = self._key(entry, session)
        if key is None:
            return None
        event = self._fire(key, session)
        event["after_call"] = self.calls
        self.fired = event
        executor.note(event)
        self.log(
            f"[inject] {event['kind']} {key}: moved {event['moved_m']:.2f} m (z {event['moved_z_m']:+.2f}) after call {self.calls}"
        )
        return event

    def _key(self, entry: dict, session) -> str | None:
        inj = self.injection
        if inj.target is not None:
            if inj.kind == "move_target" and entry.get("args", {}).get("target", inj.target) != inj.target:
                return None  # an approach to something else (the goal) is not the trigger
            return inj.target
        if inj.kind == "drop_held":
            return session.held.key if session.held is not None else None
        return entry.get("args", {}).get("target")

    def _fire(self, key: str, session) -> dict:
        inj = self.injection
        state = session.stack.obs()
        obj = state["objects"][key]
        before = np.asarray(obj["pos"], dtype=float)
        if inj.kind == "move_target":
            dest = before + np.array([inj.offset_m[0], inj.offset_m[1], 0.0])
            quat = obj.get("quat")
        else:
            meta = session.scene.meta["objects"][key]
            init = np.asarray(meta["init_pos"], dtype=float)
            dest = init + np.array([inj.offset_m[0], inj.offset_m[1], 0.0])
            quat = meta.get("init_quat")
        dest = dest.round(4)
        session.reset.pin_object(key, dest.tolist(), None if quat is None else [float(v) for v in quat])
        self.sleep(inj.settle_s)
        session.reset.pin_object(key)  # release: the object is free again, at rest
        self.sleep(0.2)
        after = np.asarray(session.stack.obs()["objects"][key]["pos"], dtype=float)
        return {
            "event": "injection",
            "kind": inj.kind,
            "target": key,
            "injection": asdict(inj),
            "before": before.round(4).tolist(),
            "commanded": dest.round(4).tolist(),
            "after": after.round(4).tolist(),
            "moved_m": float(np.linalg.norm(after[:2] - before[:2])),
            "moved_z_m": float(after[2] - before[2]),  # drop_held: the fall from the hand to the surface
            "landed_err_m": float(np.linalg.norm(after[:2] - dest[:2])),
        }


# ---- paired comparison (scripts/injection_report.py) ---------------------------------------------------------


def recovery_segment(trace: list[dict]) -> list[str] | None:
    """The calls after the injection event in one round's trace: `name` for a completed call, `name!`
    for a failed one, `name?` for a rejected one. None when nothing was injected."""
    seen = False
    out: list[str] = []
    for e in trace:
        if e.get("event") == "injection":
            seen = True
            continue
        if not seen or "primitive" not in e:
            continue
        name = e["primitive"]
        if name == "abort":
            out.append("abort")
        elif "rejected" in e:
            out.append(name + "?")
        elif not e["outcome"]["ok"]:
            out.append(name + "!")
        else:
            out.append(name)
    return out if seen else None


def regrasped(segment: list[str] | None) -> bool:
    """Did a grasp complete after the event? That is the injected failure recovered, whatever the task did next."""
    return bool(segment) and any(s in ("grasp_single", "grasp_bimanual") for s in segment)


def sign_test_p(b: int, c: int) -> float:
    """Two-sided exact binomial (McNemar without the normal approximation) on the discordant pairs:
    `b` rounds only the agent passed, `c` rounds only the plan passed."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_summary(plan_rows: list[dict], agent_rows: list[dict], agent_traces: dict[int, list[dict]] | None = None):
    """Pairs manifest rows by seed. Returns (pairs, summary): one dict per seed both arms ran, and the counts."""
    by_seed = {r["seed"]: r for r in plan_rows if "seed" in r}
    pairs = []
    for a in agent_rows:
        s = by_seed.get(a.get("seed"))
        if s is None:
            continue
        seg = recovery_segment((agent_traces or {}).get(a["round"], []))
        pairs.append(
            {
                "seed": a["seed"],
                "plan_ok": bool(s["ok"]),
                "agent_ok": bool(a["ok"]),
                "plan_injected": bool(s.get("injected")),
                "agent_injected": bool(a.get("injected")),
                "plan_why": s.get("why", ""),
                "agent_why": a.get("why", ""),
                "agent_calls": a.get("agent_calls"),
                "recovery": seg,
                "regrasped": regrasped(seg),
            }
        )
    both = [p for p in pairs if p["plan_injected"] and p["agent_injected"]]
    b = sum(1 for p in both if p["agent_ok"] and not p["plan_ok"])
    c = sum(1 for p in both if p["plan_ok"] and not p["agent_ok"])
    summary = {
        "pairs": len(pairs),
        "pairs_both_injected": len(both),
        "plan_ok": sum(p["plan_ok"] for p in both),
        "agent_ok": sum(p["agent_ok"] for p in both),
        "agent_only_ok": b,
        "plan_only_ok": c,
        "sign_test_p": sign_test_p(b, c),
        "agent_regrasped": sum(1 for p in both if p["regrasped"]),
        "recovery_classes": _classes(both),
    }
    return pairs, summary


def _classes(pairs: list[dict]) -> dict[str, dict]:
    """Recovery segments grouped by their first call (what the agent did first after the event)."""
    out: dict[str, dict] = {}
    for p in pairs:
        seg = p["recovery"] or []
        first = seg[0] if seg else "(nothing)"
        row = out.setdefault(first, {"n": 0, "ok": 0, "segments": []})
        row["n"] += 1
        row["ok"] += int(p["agent_ok"])
        row["segments"].append(" ".join(seg))
    return out


def render_report(pairs: list[dict], summary: dict, *, plan_run: str, agent_run: str, injection: dict | None) -> str:
    lines = [
        "# Paired failure-injection comparison",
        "",
        f"- plan run: `{plan_run}`",
        f"- agent run: `{agent_run}`",
        f"- injection: `{json.dumps(injection)}`" if injection else "- injection: (none recorded)",
        "",
        "| | plan | agent |",
        "|---|---|---|",
        f"| OK / rounds both injected | {summary['plan_ok']}/{summary['pairs_both_injected']} "
        f"| {summary['agent_ok']}/{summary['pairs_both_injected']} |",
        "",
        f"Discordant pairs: agent-only OK {summary['agent_only_ok']}, plan-only OK {summary['plan_only_ok']}; "
        f"two-sided exact sign test p = {summary['sign_test_p']:.3f}. "
        f"Pairs seen {summary['pairs']}, both arms injected {summary['pairs_both_injected']}. "
        f"Agent re-grasped the target after the event in {summary['agent_regrasped']}/{summary['pairs_both_injected']} "
        "(the injected failure recovered; the task may still fail later).",
        "",
        "## Recovery segments (agent calls after the event; `!` failed, `?` rejected)",
        "",
        "| first call | n | OK | segments |",
        "|---|---|---|---|",
    ]
    for first, row in sorted(summary["recovery_classes"].items(), key=lambda kv: -kv[1]["n"]):
        segs = "<br>".join(row["segments"])
        lines.append(f"| {first} | {row['n']} | {row['ok']} | {segs} |")
    lines += [
        "",
        "## Rounds",
        "",
        "| seed | inj | plan | agent | regrasp | calls | plan why | agent why |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in pairs:
        inj = ("P" if p["plan_injected"] else "-") + ("A" if p["agent_injected"] else "-")
        lines.append(
            f"| {p['seed']} | {inj} | {'OK' if p['plan_ok'] else 'FAIL'} | {'OK' if p['agent_ok'] else 'FAIL'} "
            f"| {'Y' if p['regrasped'] else '-'} | {p['agent_calls']} | {p['plan_why'][:70]} | {p['agent_why'][:70]} |"
        )
    return "\n".join(lines) + "\n"
