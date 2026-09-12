"""The crate line's round loop, shared by both experts.

`bimanual_wall.main` and `bimanual_carry.main` each carried their own copy of
the same ~45 lines: the port block, the scene check, the Primitives/reset
priming, the per-round timing + manifest append, and the printout. The copies
had already drifted -- only the wall one had a deploy self-heal, and only the
wall one wrote a ghost trace -- which is the usual cost of a duplicated loop:
the fix goes into whichever copy the author had open.

This module is that loop, once. Each expert supplies its own `run_round`; the
differences that are real (ghost trace, label prefix) are parameters.

`--record` lives here too. Until now the crate line could not produce a
LeRobot dataset at all: `docs/crate_collection.md` §11 listed "drive the
exporter from the expert, the way grasp_gym.py does" as the missing step. The
protocol is the bottle line's, exactly (`zmq_driver.SonicCommander.key`):

    'c'  IDLE -> RECORDING     `start_recording`, called by the expert's
                               run_round once the reset phase has settled, so
                               frame 0 is the task and not a robot standing still
    'c'  RECORDING -> SAVE     sent here when the round succeeded
    'x'  RECORDING -> DISCARD  sent here when it failed

and the manifest row carries `ok` and `produced`, which is what
`convert_to_hf_format.py --manifest` joins on (positionally, per session).
Recording still needs INSTANCE=0: `start_stack.sh` starts the exporter only
there (memory only-instance-0-can-record), which is why `--record` is off by
default and every pilot to date ran without it.
"""

from __future__ import annotations

import json
import time

from locomani.backends.sonic import deploy
from locomani.backends.sonic.agent_server import LocomotionMode


def start_recording(p, m: dict) -> None:
    """`'c'`: the exporter goes IDLE -> RECORDING, and this round will produce
    an episode. Called by each expert's `run_round` once the reset phase has
    settled -- the runner cannot know that moment, and the phases should not
    have to know about the exporter."""
    p.cmd.key("c")
    m["produced"] = True


def prime(p, reset) -> None:
    """Bring the robot to the IDLE stance both experts start every session from."""
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset()
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(2.5, mode=LocomotionMode.IDLE)


def run_rounds(
    p,
    reset,
    meta,
    *,
    run_round,
    rng,
    rounds: int,
    label: str,
    tag: str,
    manifest: str = "",
    record: bool = False,
    self_heal_rounds: int = 2,
    instance: int | None = None,
    ghost=None,
) -> int:
    """Run `rounds` rounds; return how many succeeded.

    `run_round(p, reset, meta, rnd, rng, log, video_label, record=...)` is the
    expert's own entry. `tag` is the log prefix (`wall` / `bimanual`).
    """
    healer = deploy.SelfHeal(self_heal_rounds)
    wins = 0
    for rnd in range(rounds):
        t0 = time.time()
        video_label = f"{label}_{rnd:02d}"
        if ghost is not None:
            ghost.start(video_label)
        try:
            m = run_round(p, reset, meta, rnd, rng, p.log, video_label, record=record)
        finally:
            if ghost is not None:
                ghost.stop()
            try:
                reset.end_video()
            except Exception:
                pass
        m["seconds"] = round(time.time() - t0, 1)
        m["video"] = video_label
        m["deploy_restarts"] = healer.restarts
        m.setdefault("produced", False)
        wins += int(m["ok"])
        if record and m["produced"]:
            # 'c' saves, 'x' discards -- the exporter writes the episode from
            # its own loop. A round that never reached the reset's settle never
            # started recording, so `produced` is False and neither key is due.
            p.cmd.key("c" if m["ok"] else "x")
            p.cmd.hold(2.0, mode=LocomotionMode.IDLE)
        if healer.observe(m["why"]):
            print(f"[{tag}] deploy looks dead ({healer.consecutive} rounds); restarting instance {instance} deploy")
            print(f"[{tag}] deploy restart {'ok' if deploy.restart(instance) else 'FAILED'}")
            healer.fired()
            prime(p, reset)
        print(f"[{tag}] round {rnd}: {'OK' if m['ok'] else 'FAIL'} {m['why']} ({m['seconds']} s)")
        if manifest:
            with open(manifest, "a") as f:
                f.write(json.dumps(m, default=float) + "\n")
    print(f"[{tag}] {wins}/{rounds} OK")
    return wins
