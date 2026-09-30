#!/usr/bin/env python3
"""Where does the runner stop commanding for > 1 s? (TODO 8.3.5.5)

    $VENV_SIM_PY -u scripts/experiments/command_gap_probe_20260929.py configs/experiments/skills_pick_place_20260928.yaml \\
        --gpu 1 --instance 1 --rounds 4 --seed 500 --plan configs/plans/command_gap_probe_20260929.json --name <run>

`scripts/plan_episode.py` with three things added for the diagnosis, nothing changed in the stack:
every `AgentCommander` send (planner frame, pose chunk, token frame) stamps the time; a watchdog thread samples
the main thread's Python stack every 0.1 s while nothing has been sent for more than `SILENCE_S`, and writes one
line per silence to `skills/command_gaps.jsonl` (primitive running, length, the stacks it saw); the fixed plan
runs every step whatever the previous one returned, so the agent's failure paths are exercised too. The deploy
drops the frozen upper body and the hands after 1 s without a planner message (pick_place ledger §14).
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import plan_episode  # noqa: E402

SILENCE_S = 0.3  # the deploy times out at 1.0 s; a path silent this long normally is one contention away from it

LAST = {"t": time.monotonic(), "kind": "start"}
CURRENT = {"step": "(before)", "out": None}


def stamp(kind: str):
    def wrap(fn):
        def inner(self, *a, **kw):
            out = fn(self, *a, **kw)
            LAST["t"], LAST["kind"] = time.monotonic(), kind
            return out

        return inner

    return wrap


def install_send_stamps() -> None:
    from locomani.backends.sonic.agent_server import AgentCommander

    for name in ("frame", "pose_chunk", "token_frame"):
        setattr(AgentCommander, name, stamp(name)(getattr(AgentCommander, name)))


def install_step_names() -> None:
    from locomani.skills.executor import Executor

    step = Executor.step

    def named(self, name, args=None):
        CURRENT["step"] = name
        CURRENT["out"] = CURRENT["out"] or (self.trace_path.parent if self.trace_path else None)
        try:
            return step(self, name, args)
        finally:
            CURRENT["step"] = f"(after {name})"

    Executor.step = named


def watchdog(main_ident: int) -> None:
    episode = None
    while True:
        time.sleep(0.1)
        silence = time.monotonic() - LAST["t"]
        if silence > SILENCE_S:
            frame = sys._current_frames().get(main_ident)
            stack = [
                f"{Path(f.filename).name}:{f.lineno} {f.name}"
                for f in traceback.extract_stack(frame)
                if "locomani" in f.filename or "scripts" in f.filename
            ][-8:]
            if episode is None:
                episode = {
                    "step": CURRENT["step"],
                    "last_send": LAST["kind"],
                    "started": round(time.time() - silence, 2),
                    "stacks": [],
                }
            if not episode["stacks"] or episode["stacks"][-1] != stack:
                episode["stacks"].append(stack)
            episode["silence_s"] = round(silence, 2)
        elif episode is not None:
            out = CURRENT["out"]
            if out is not None:
                with (Path(out) / "command_gaps.jsonl").open("a") as f:
                    f.write(json.dumps(episode) + "\n")
            print(
                f"[gap] {episode['silence_s']:.2f} s during {episode['step']}: {episode['stacks'][0][-3:]}", flush=True
            )
            episode = None


def every_step(executor, steps, injector=None, think_s=0.0, capture=None, keepalive=True):
    """plan_episode.scripted_round without the early stop: the failure paths are what is being probed."""
    entries = []
    for step in steps:
        if executor.session.stack.obs().get("fallen"):
            break
        entries.append(executor.step(step["primitive"], step.get("args")))
    return entries


def pin_instance_ports(argv) -> None:
    """What plan_episode.main does first, done before install_send_stamps imports agent_server: every port default
    is read at import time, so importing it first bound the commander to instance 0 (memory
    paths-sh-exports-instance0-ports; the first launch of this probe: "no body_q_target", never stood up)."""
    import os

    from locomani.sim.ports import instance_ports

    instance = int(argv[argv.index("--instance") + 1]) if "--instance" in argv else 1
    os.environ["INSTANCE"] = str(instance)
    os.environ.update({k: str(v) for k, v in instance_ports(instance).items()})


def main() -> int:
    pin_instance_ports(sys.argv[1:])
    install_send_stamps()
    install_step_names()
    plan_episode.scripted_round = every_step
    threading.Thread(target=watchdog, args=(threading.main_thread().ident,), daemon=True).start()
    return plan_episode.main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
