"""After a completed hold: hand the arms back to the planner as they are and walk backward carrying the load.

The streamed arm is released with `preserve_reference=True`, which primes a FROZEN_UPPER_BODY planner packet
holding the arm reference and both closed hands (patches/sonic_hand_ownership.patch keeps the deploy from
dropping the hand command across the switch). The planner then walks SLOW_WALK backward along the approach
heading, facing unchanged, until the pelvis has moved `retreat_distance`, and IDLEs for a second.

The load counts as held while its height stays at least `0.6 * lift_height` above where it started and both
hands touch it; a contact gap under 0.3 s is tolerated (contacts flicker on a moving grasp).
"""

from __future__ import annotations

import time

import numpy as np

from locomani.backends.sonic.primitives import DT, LocomotionMode

CONTACT_GAP = 0.3  # simulated seconds a hand may lose contact before the load counts as dropped
SETTLE = 1.0  # simulated seconds of IDLE after the retreat, still checking the load


def retreat(p, arm, cfg, key, heading, start_height):
    """Returns (reason, diagnostics, final_state); reason is "completed" when the load is still held."""
    p.cmd.set_hands(left=arm.hands["left"], right=arm.hands["right"])  # stop() re-sends only its own side
    arm.stop(preserve_reference=True)
    h = np.array([np.cos(heading), np.sin(heading)])
    facing = (float(h[0]), float(h[1]), 0.0)
    backward = (float(-h[0]), float(-h[1]), 0.0)
    state = p.obs()
    start_xy, t0 = np.asarray(state["pelvis"]["pos"][:2], dtype=float), float(state["t"])
    floor = start_height + 0.6 * cfg.lift_height
    lost_since, walked, settle_from = None, 0.0, None
    diag = dict(distance_target=cfg.retreat_distance, min_load_height=np.inf)
    while True:
        state = p.obs()
        now = float(state["t"])
        obj = state["objects"][key]
        diag["min_load_height"] = min(diag["min_load_height"], float(obj["pos"][2]))
        walked = float((start_xy - np.asarray(state["pelvis"]["pos"][:2])) @ h)
        held = float(obj["pos"][2]) >= floor and {"left", "right"} <= set(obj.get("touching", []))
        lost_since = None if held else (now if lost_since is None else lost_since)
        reason = None
        if state["fallen"]:
            reason = "retreat_fallen"
        elif lost_since is not None and now - lost_since >= CONTACT_GAP:
            reason = "retreat_dropped"
        elif settle_from is None and now - t0 > cfg.retreat_timeout:
            reason = "retreat_timeout"
        if reason is not None:
            break
        if settle_from is None and walked >= cfg.retreat_distance:
            settle_from = now
        if settle_from is not None and now - settle_from >= SETTLE:
            reason = "completed"
            break
        if settle_from is None:
            p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=cfg.retreat_speed, movement=backward, facing=facing)
        else:
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=facing)
        time.sleep(DT)
    diag.update(walked=walked, seconds=float(state["t"]) - t0, reason=reason)
    return reason, diag, state
