"""Does the standing token hold / raise the robot under lockstep?

    LOCKSTEP=1 TF_SONIC_PUB_CONNECT=5584 python diag_lockstep_stand.py

Against a running lockstep stack (no eval). Phases, pelvis z printed:
  A  prime the session, 100 ticks of the standing token          (hold?)
  B  reset (teleport to qpos0 = the crouched keyframe), watch 300 ticks
     while the clock pump re-primes                              (stand up?)
  C  send command(start=True, planner=False) explicitly, 200 more ticks
The answer decides what the lockstep bring-up must do after every reset.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LOCKSTEP", "1")
os.environ.setdefault("TF_SONIC_PUB_CONNECT", str(int(os.environ.get("TF_SONIC_PORT", 5556)) + 28))

import vla_round  # noqa: E402
import agent_server  # noqa: E402
from privileged_pub import EpisodeControlClient, PrivilegedSubscriber as PrivilegedClient  # noqa: E402

ls = vla_round.LockstepLink()
assert ls.ping(), "no lockstep sim"
cmd = agent_server.AgentCommander()
clock = vla_round.LockstepClock(ls, cmd)
priv = PrivilegedClient()
reset = EpisodeControlClient()


def z(tag):
    s = priv.wait(2.0)
    print(f"  {tag}: pelvis z {s['pelvis']['pos'][2]:.3f}  fallen {s.get('fallen')}")


print("A: prime + hold 100 ticks")
assert clock.ensure_primed(60), "prime failed"
for i in range(4):
    clock.hold_frames(25)
    z(f"A t={(i+1)*25}")

print("B: reset (crouched keyframe), pump re-primes, 300 ticks")
reset.reset()
time.sleep(0.3)
clock.resume()
for i in range(12):
    time.sleep(0.5)
    z(f"B ~{(i+1)*25} ticks  primed={clock.primed} session={clock.session}")
clock.pause()

print("C: explicit start (planner=False) then 200 ticks")
cmd.command(start=True, stop=False, planner=False)
for i in range(8):
    clock.hold_frames(25)
    z(f"C t={(i+1)*25}")
clock.stop()
