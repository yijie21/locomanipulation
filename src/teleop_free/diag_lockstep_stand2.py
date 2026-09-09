"""Which bring-up order stands the robot under lockstep?

    LOCKSTEP=1 TF_SONIC_PUB_CONNECT=5584 python diag_lockstep_stand2.py

  X  reset (default pose) BEFORE the first session of a fresh deploy, then
     prime + 300 standing-token ticks            -> token from default pose?
  Y  reset, THEN restart the deploy (its 3 s INIT ramp PDs the robot to the
     default pose, auto-start engages the policy), prime, 300 ticks
                                                 -> wall-clock's fresh_deploy order
Pelvis z every 25 ticks. Assumes the stack is up and the deploy is fresh.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LOCKSTEP", "1")
os.environ.setdefault("TF_SONIC_PUB_CONNECT", str(int(os.environ.get("TF_SONIC_PORT", 5556)) + 28))

import agent_server  # noqa: E402
import vla_round  # noqa: E402
from privileged_pub import EpisodeControlClient, PrivilegedSubscriber  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ls = vla_round.LockstepLink()
assert ls.ping(), "no lockstep sim"
cmd = agent_server.AgentCommander()
clock = vla_round.LockstepClock(ls, cmd)
priv = PrivilegedSubscriber()
reset = EpisodeControlClient()


def z(tag):
    s = priv.wait(2.0)
    print(f"  {tag}: pelvis z {s['pelvis']['pos'][2]:.3f}  fallen {s.get('fallen')}", flush=True)


def restart_deploy():
    env = dict(os.environ, LOCKSTEP="1", NO_ENCODER="1", INSTANCE="0")
    r = subprocess.run(
        ["bash", f"{REPO}/scripts/restart_deploy.sh"], env=env, capture_output=True, text=True, timeout=360
    )
    print("  restart:", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:])


def watch(n_blocks=12):
    for i in range(n_blocks):
        clock.hold_frames(25)
        z(f"t={(i + 1) * 25}")


print("X: reset before first session, then prime + 300 ticks", flush=True)
reset.reset()
time.sleep(0.5)
assert clock.ensure_primed(120), "prime failed"
z("X primed")
watch()

print("Y: reset, then restart deploy (INIT ramp), prime + 300 ticks", flush=True)
reset.reset()
time.sleep(0.5)
restart_deploy()
clock.session = None
clock.primed = False
assert clock.ensure_primed(240), "prime failed"
z("Y primed")
watch()
clock.stop()
