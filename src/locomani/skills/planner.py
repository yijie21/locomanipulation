"""The agent side of the executor loop: `claude -p` as a pure JSON planner (agentic-plan Q8/Q9).

One `ClaudePlanner` serves one episode. It never holds control flow: `scripts/plan_episode.py` asks it for
the next primitive, runs that through `skills.Executor`, and asks again with the executor's trace entry.
Each ask is one `claude -p` call with `--output-format json --json-schema ... --tools ""` (no tools, no
files, no shell), the first with `--session-id`, the rest with `--resume` so the conversation carries the
episode; every request and reply is appended to `log_path` (`agent_io_NN.jsonl`), the Q9 "everything on
disk" rule. The planner sees text only: the task, the scene's task menu, the catalogue with each
primitive's arguments, the privileged-state view (`scene_view`) and the session snapshot. It decides; the
executor validates and the sim judges.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

SCHEMA = {
    "type": "object",
    "properties": {
        "next": {"type": "string", "description": "a catalogue primitive name, or done, or abort"},
        "args": {"type": "object", "description": "keyword arguments for the primitive (JSON values)"},
        "rationale": {"type": "string"},
    },
    "required": ["next", "rationale"],
}

SYSTEM = """You plan one robot manipulation episode for a Unitree G1 humanoid in simulation.
You are a planner, not a controller: you name the next primitive and its arguments; a runtime executes
it, checks its preconditions, and reports back. You never judge success yourself: the runtime evaluates
the task criterion from the simulator when you answer "done".

Rules:
- Answer only with the JSON object the schema asks for. `next` is a primitive name from the catalogue,
  or "done" when you believe the task is complete, or "abort" when no primitive can help.
- Use only arguments listed for that primitive; omit an argument to take its default.
- Arm ownership: walking primitives need planner_swing or planner_frozen; grasps and places declare what
  they start from and what they yield. A rejected call tells you which check failed; change the plan
  instead of repeating the call.
- Profiles bind approach and grasp parameters: use the profile the task menu names for the target.
- Objects: positions are given in the pelvis frame (x forward, y left, z up, metres) and in the world.
- With something held, walk with carry_to / navigate_to (arms frozen), not the free walk.
"""


def catalogue_manual(catalogue) -> list[dict]:
    """What the planner may call: name, owners it starts from / yields, arguments with defaults, one line of doc."""
    rows = []
    for name, prim in sorted(catalogue.items()):
        params = inspect.signature(prim.run).parameters
        args = {}
        for pname, par in params.items():
            if pname == "session" or par.kind is inspect.Parameter.VAR_KEYWORD:
                continue
            args[pname] = "required" if par.default is inspect.Parameter.empty else _plain(par.default)
        yields = getattr(prim, "yields", None)
        rows.append(
            {
                "name": name,
                "starts_from": sorted(o.value for o in prim.requires),
                "yields": (
                    "unchanged"
                    if yields is None
                    else sorted(o.value for o in yields)
                    if isinstance(yields, frozenset)
                    else yields.value
                ),
                "args": args,
                "doc": (inspect.getdoc(prim) or "").splitlines()[0] if inspect.getdoc(prim) else "",
            }
        )
    return rows


def scene_view(stack, scene, state: dict) -> dict:
    """The privileged-state text the agent reads (Q4): where things are, relative to the pelvis and in the world."""
    pelvis_xy = np.asarray(stack.pelvis_xy(state), dtype=float)
    yaw = float(stack.pelvis_yaw(state))
    objects = {}
    for key, obj in (state.get("objects") or {}).items():
        pos = np.asarray(obj["pos"], dtype=float)
        rel = np.asarray(stack.world_to_root(pos, state), dtype=float)
        info = scene.objects.get(key) if scene is not None else None
        objects[key] = {
            "pelvis_frame_xyz": [round(float(v), 3) for v in rel[:3]],
            "world_xyz": [round(float(v), 3) for v in pos[:3]],
            "distance_m": round(float(np.linalg.norm(pos[:2] - pelvis_xy)), 3),
            "touching": sorted(obj.get("touching") or ()),
            "in_ego_view": obj.get("cam") is not None,
            **(
                {"bimanual": info.bimanual, "container": info.container, "on_surface": info.on_surface}
                if info is not None
                else {}
            ),
        }
    surfaces = (
        {
            k: {"top_z": s.top, "world_xy": list(s.xy), "half_extent": list(s.half_extent)}
            for k, s in scene.surfaces.items()
        }
        if scene is not None
        else {}
    )
    return {
        "robot": {
            "world_xy": [round(float(v), 3) for v in pelvis_xy],
            "yaw_rad": round(yaw, 3),
            "fallen": bool(state.get("fallen")),
        },
        "objects": objects,
        "surfaces": surfaces,
    }


class ClaudePlanner:
    def __init__(
        self, *, command=("claude",), model: str = "sonnet", log_path: Path | None = None, timeout_s: float = 240.0
    ):
        self.command, self.model, self.timeout_s = tuple(command), model, timeout_s
        self.log_path = Path(log_path) if log_path is not None else None
        self.session_id: str | None = None
        self.calls = 0
        self.context: dict = {}

    def start_episode(self, *, task: str, menu: list[str], catalogue, profiles=(), hints: str = "") -> None:
        """`profiles` are the profile names a primitive's `profile` argument accepts (the first smoke's agent
        guessed "pick_place" and "simple" and burned two calls on the rejection)."""
        self.session_id, self.calls = str(uuid.uuid4()), 0
        self.context = {
            "task": task,
            "menu": menu,
            "profiles": sorted(profiles),
            "catalogue": catalogue_manual(catalogue),
            "hints": hints,
        }

    def next_step(self, *, session: dict, scene: dict, last: dict | None) -> dict:
        """One ask. `last` is the executor's previous trace entry (or None on the first ask)."""
        request = {"session": session, "scene": scene, "last_call": last}
        if self.calls == 0:
            request = {**self.context, **request, "instruction": "Give the first primitive call."}
        else:
            request["instruction"] = "Given the last call's result and the state now, give the next primitive call."
        argv = [
            *self.command,
            "-p",
            json.dumps(request, default=_plain),
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(SCHEMA),
            "--tools",
            "",
            "--model",
            self.model,
            "--system-prompt",
            SYSTEM,
            "--resume" if self.calls else "--session-id",
            self.session_id,
        ]
        t0 = time.time()
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout_s, check=False)
            raw = proc.stdout
            reply = json.loads(raw) if raw.strip() else {}
            decision = reply.get("structured_output") or _parse_result(reply.get("result", ""))
            error = "" if proc.returncode == 0 and decision else f"claude exit {proc.returncode}: {proc.stderr[-400:]}"
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:  # the planner failing is a decision too
            reply, decision, error = {}, {}, f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - t0
        self.calls += 1
        decision = {
            "next": str(decision.get("next", "abort")) if decision else "abort",
            "args": dict(decision.get("args") or {}) if decision else {},
            "rationale": str(decision.get("rationale", "")) if decision else error,
        }
        self._log(
            {
                "call": self.calls,
                "session_id": self.session_id,
                "elapsed_s": round(elapsed, 3),
                "request": request,
                "decision": decision,
                "error": error,
                "usage": reply.get("usage"),
                "cost_usd": reply.get("total_cost_usd"),
                "model": self.model,
            }
        )
        return decision

    def _log(self, entry: dict) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry, default=_plain) + "\n")


def _parse_result(text: str) -> dict:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _plain(value):
    from locomani.skills.contract import _plain as plain

    return plain(value)
