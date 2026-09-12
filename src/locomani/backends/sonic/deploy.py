"""Restarting this instance's SONIC deploy: one implementation, two policies.

`scripts/restart_deploy.sh` was called from three places, each with its own
subprocess call and its own idea of what success looks like:

  * `grasp_gym.fresh_deploy` -- capture_output, 360 s timeout, success is the
    string "deploy restarted" in stdout, then the bottle line's own re-priming
    (heading_offset back to 0, a 4 s wait for the TensorRT engines);
  * `bimanual_wall.main` -- an inline `subprocess.run` with no timeout and NO
    success check at all (it printed the exit code and carried on), fired after
    two consecutive "[stand] did not stabilise" rounds;
  * `bimanual_carry.main` -- nothing. The same deploy, the same failure mode,
    no self-heal, so a dead deploy turned every remaining round into a
    "did not stabilise" (memory sonic-deploy-lowstate-trip).

`restart()` here is the common primitive -- the subprocess, the timeout and the
success check. Each line keeps its OWN re-priming afterwards, because they are
genuinely different: the bottle line zeroes the commander's heading offset to
match a fresh deploy's clean anchor, while the crate line re-primes through
`reset.pin_arms()` and re-anchors per round anyway (wall_phases `reanchor`).

`SelfHeal` is the crate line's trigger, extracted so both crate experts get it
and the threshold is a config field instead of a literal `>= 2`.
"""

from __future__ import annotations

import os
import subprocess

from locomani import paths

OK_MARKER = "deploy restarted"  # restart_deploy.sh's own success line


def restart(instance: int | None = None, timeout: float = 360.0) -> bool:
    """Kill and relaunch this instance's deploy. True only when it came back.

    `instance` is passed as INSTANCE in the child's environment (the script
    derives its port block from it); None inherits whatever this process has,
    which is what the bottle line wants -- it runs inside the gym, already in
    the right instance's environment.
    """
    env = dict(os.environ)
    if instance is not None:
        env["INSTANCE"] = str(instance)
    try:
        r = subprocess.run(
            ["bash", paths.RESTART_DEPLOY],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=paths.REPO,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print(f"[deploy] restart timed out after {timeout:.0f}s")
        return False
    if OK_MARKER not in r.stdout:
        print(f"[deploy] restart failed (exit {r.returncode}): {r.stdout[-300:]}")
        return False
    return True


class SelfHeal:
    """Count consecutive rounds whose `why` says the deploy is gone.

    The SONIC deploy stops itself on a one-loop LowState timeout -- another
    session's per-round deploy restart is enough to blip the sim clock (rail
    pilot 12) -- and after that EVERY reset reports "did not stabilise". One
    such round is ordinary bad luck; two in a row is the deploy.
    """

    TRIGGER_PREFIX = "[stand] did not stabilise"

    def __init__(self, threshold: int = 2):
        self.threshold = threshold
        self.consecutive = 0
        self.restarts = 0

    def observe(self, why: str) -> bool:
        """Record a round's `why`; True when a restart is now due."""
        if not self.threshold:
            return False
        self.consecutive = self.consecutive + 1 if (why or "").startswith(self.TRIGGER_PREFIX) else 0
        return self.consecutive >= self.threshold

    def fired(self) -> None:
        self.consecutive = 0
        self.restarts += 1
