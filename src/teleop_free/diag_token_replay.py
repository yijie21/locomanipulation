"""Does the recorded 64-d token actually cause the motion we recorded?

The dataset's premise is that `action.motion_token` IS the action: a VLA
predicts it, protocol v4 streams it, and the policy turns it into the robot's
behaviour with the encoder bypassed
(zmq_endpoint_interface.hpp:678, "Token-only mode - no motion data, just
tokens for the policy"). That premise is why the sim-side arm pin had to go --
with the pin the visible grasp was produced by a position loop, not by the
latent. It has never been tested.

This tests it end to end:

  CAPTURE  run one gym episode while subscribing to the deploy's `g1_debug`
           topic, logging token_state at the control rate along with the
           joints it produced. The gym writes its scene sample to JSON.
  REPLAY   reset to the IDENTICAL sample, then stream the captured tokens
           back as protocol-v4 pose messages and log what the robot does.
  COMPARE  joint trajectories, pelvis path, and whether the can was lifted.

Read the result with the right expectation: the replay is OPEN LOOP. The
original episode ran a 50 Hz servo that re-aimed on the measured can, so a
frame-exact match is not the bar and small differences compound. The question
this answers is whether the token sequence is the CAUSE of the behaviour --
if the robot walks to the table and reaches, the premise holds; if it does
something unrelated, the label does not describe the outcome and no amount of
data collection fixes that.

    python src/teleop_free/diag_token_replay.py [--seed 21]
Output: outputs/diag/token_replay.npz  + a printed comparison
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

REPO = paths.REPO
PY_SIM = f"{REPO}/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python"
SCENE_META = f"{REPO}/outputs/scene/sim/scene_meta.json"
OUT = f"{REPO}/outputs/diag/token_replay.npz"
SAMPLE = "/tmp/token_replay_sample.json"
STATE_PORT = 5557
DT = 0.02


class TokenTap:
    """Subscribe to g1_debug and log (t, token_state, body_q) at the tick rate."""

    def __init__(self, port: int = STATE_PORT):
        sys.path.insert(0, f"{REPO}/third_party/GR00T-WholeBodyControl")
        from gear_sonic.utils.data_collection.zmq_state_subscriber import (
            ZMQStateSubscriber)
        self.sub = ZMQStateSubscriber(host="localhost", port=port)
        self.rows: list = []
        self._run = False
        self._th: threading.Thread | None = None

    def _loop(self):
        last = None
        while self._run:
            m = self.sub.get_msg(clear=True)
            if m is not None and "token_state" in m:
                tok = np.asarray(m["token_state"], dtype=np.float32).ravel()
                # g1_debug is CONFLATE: the same frame can be read twice.
                if last is None or not np.array_equal(tok, last):
                    self.rows.append(dict(
                        t=time.time(),
                        token=tok,
                        body_q=np.asarray(m.get("body_q", []), dtype=np.float32),
                        lh=np.asarray(m.get("left_hand_q", []), dtype=np.float32),
                        rh=np.asarray(m.get("right_hand_q", []), dtype=np.float32)))
                    last = tok
            time.sleep(0.004)

    def start(self):
        self._run = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._run = False
        if self._th:
            self._th.join(timeout=2.0)


def capture(seed: int, tap: TokenTap) -> tuple[dict, list]:
    """Run one recorded gym episode and return (scene sample, token rows)."""
    if os.path.exists(SAMPLE):
        os.remove(SAMPLE)
    tap.start()
    t0 = time.time()
    r = subprocess.run(
        [PY_SIM, "-u", f"{REPO}/src/teleop_free/grasp_gym.py",
         "--rounds", "1", "--seed", str(seed), "--dump-sample", SAMPLE],
        cwd=REPO, capture_output=True, text=True, timeout=600)
    tap.stop()
    ok = "EPISODE OK" in r.stdout
    for line in r.stdout.splitlines():
        if line.startswith("EPISODE") or "approach: exit" in line:
            print("   ", line.strip())
    if not ok:
        sys.exit("capture episode failed; rerun (the replay needs a success)")
    if not os.path.exists(SAMPLE):
        sys.exit("no scene sample written")
    rows = list(tap.rows)
    print(f"[replay] captured {len(rows)} token frames over "
          f"{time.time() - t0:.0f} s")
    return json.load(open(SAMPLE)), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--phase", choices=("capture", "replay"), required=True,
                    help="capture: run an episode on the NORMAL stack and save "
                         "its token stream. replay: feed that stream back on a "
                         "stack launched with NO_ENCODER=1 (the token-only "
                         "configuration a VLA deploys into).")
    ap.add_argument("--store", default="/tmp/token_capture.npz")
    args = ap.parse_args()

    import stream_reach  # noqa: E402
    from primitives import LocomotionMode, Primitives, StreamMode  # noqa: E402
    from privileged_pub import EpisodeControlClient  # noqa: E402

    meta = json.load(open(SCENE_META))
    # NOT meta["target_object"]. The gym draws its target from a pool now, so
    # the captured round may be about any of them; the sample records which,
    # and every predicate below has to be about that same object. Falls back
    # to the scene default for samples captured before the pool existed.
    obj_key = meta["target_object"]

    if args.phase == "capture":
        print("[replay] === CAPTURE (normal stack, encoder ON) ===")
        tap = TokenTap()
        sample, rows = capture(args.seed, tap)
        obj_key = sample.get("target", obj_key)
        print(f"[replay] captured round's target: {obj_key}")
        if len(rows) < 200:
            sys.exit(f"only {len(rows)} token frames captured — is the deploy "
                     f"publishing g1_debug on :{STATE_PORT}?")
        np.savez_compressed(
            args.store,
            tokens=np.stack([r["token"] for r in rows]),
            body_q=np.stack([r["body_q"] for r in rows if len(r["body_q"])]),
            rh=np.stack([r["rh"] for r in rows if len(r["rh"])]),
            ts=np.array([r["t"] for r in rows], dtype=float),
            sample=json.dumps(sample))
        span = rows[-1]["t"] - rows[0]["t"]
        print(f"[replay] wrote {args.store}: {len(rows)} frames over "
              f"{span:.1f} s ({len(rows) / span:.1f} Hz)")
        print("[replay] now relaunch the stack with NO_ENCODER=1 and run "
              "--phase replay")
        return

    cap = np.load(args.store, allow_pickle=True)
    tokens, caps_q, rh = cap["tokens"], cap["body_q"], cap["rh"]
    ts_cap = cap["ts"]
    sample = json.loads(str(cap["sample"]))
    obj_key = sample.get("target", obj_key)
    rows = [{"t": float(t)} for t in ts_cap]
    span = ts_cap[-1] - ts_cap[0]
    print(f"[replay] === REPLAY (stack must be NO_ENCODER=1) ===")
    print(f"[replay] token stream {tokens.shape}, joints {caps_q.shape}, "
          f"right hand {rh.shape}; captured over {span:.1f} s "
          f"({len(ts_cap) / span:.1f} Hz), replayed on those timestamps")
    p = Primitives(meta=meta, verbose=False)
    reset = EpisodeControlClient()
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.reset_pose()
    reset.pin_arms()
    # d_robot in the sample is the STAND; run_round subtracts the spawn
    # offset itself, so the replay has to do the same or it starts 1.8 m
    # too far in -- against the table, replaying a "walk forward" token
    # stream straight into it (first run: 3.07 m travelled, robot fell).
    fv = np.array([np.cos(sample["psi"]), np.sin(sample["psi"])])
    d_robot = np.asarray(sample["d_robot"], dtype=float) - sample["spawn_off"] * fv
    reset.reset(objects={obj_key: {"dxy": sample["d_can"]}},
                robot={"dxy": [float(d_robot[0]), float(d_robot[1])],
                       "yaw": sample["psi"]})
    time.sleep(0.4)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    stream_reach.reanchor(p, side=sample.get("side", "right"))
    s0 = p.obs()
    start_xy = np.asarray(p.pelvis_xy(s0))
    can0 = np.asarray(p.object_pos(obj_key, s0))
    print(f"[replay] start pelvis {np.round(start_xy, 2)} "
          f"can z {can0[2]:.3f}")

    # ORDER MATTERS, and it is the same order the v1 channel needs: PRIME
    # THE BUFFER BEFORE THE SWITCH. StreamedArm.start() pushes pose frames
    # for 0.3 s and only then calls enter_streamed(), because "the deploy
    # stops control if streamed mode goes active over an empty motion
    # buffer". Doing it the other way round dropped the robot during the
    # handover twice: flagged fallen 5 tokens in on the first attempt, and
    # pelvis z 0.238 before control even came up on the second.
    #
    # The first v4 frame is also what flips is_using_encoder_ off at runtime
    # (g1_deploy_onnx_ref.cpp:3038), so the stack needs no relaunch, and
    # TOKEN_TIMEOUT_MS is 200 (a 5 Hz floor that only warns), so the ~30 Hz
    # capture rate sits comfortably inside it.
    for _ in range(30):                       # 0.6 s of the opening token
        p.cmd.token_frame(tokens[0], right_hand=rh[0] if len(rh) else None,
                          frame_index=0)
        time.sleep(0.02)
    p.cmd.enter_streamed()
    for _ in range(30):                       # hold it across the switch
        p.cmd.token_frame(tokens[0], right_hand=rh[0] if len(rh) else None,
                          frame_index=0)
        time.sleep(0.02)
    st = p.obs()
    print(f"[replay] streamed mode up, fallen={st['fallen']}, "
          f"pelvis z {st['pelvis']['pos'][2]:.3f}")

    # Replay on the CAPTURED timestamps, not a nominal 50 Hz. g1_debug is a
    # CONFLATE socket and repeated tokens are dropped, so the capture lands
    # near 18 Hz -- pushing it out at 50 Hz ran the whole motion 2.8x fast.
    ts = np.array([r["t"] for r in rows], dtype=float)
    ts -= ts[0]
    rep, t0 = [], time.time()
    for i in range(len(tokens)):
        p.cmd.token_frame(tokens[i],
                          right_hand=rh[i] if i < len(rh) else None,
                          frame_index=i)
        lag = (t0 + ts[i]) - time.time()
        if lag > 0:
            time.sleep(lag)
        if i % 5 == 0:
            st = p.obs()
            rep.append(dict(i=i, q=np.array(st["joints"]),
                            pelvis=np.asarray(p.pelvis_xy(st)),
                            can=np.asarray(p.object_pos(obj_key, st)),
                            fallen=bool(st["fallen"])))
    p.cmd.exit_streamed()
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)

    st = p.obs()
    end_xy = np.asarray(p.pelvis_xy(st))
    can1 = np.asarray(p.object_pos(obj_key, st))
    fell = any(r["fallen"] for r in rep)
    print()
    print("=== replay vs capture ===")
    print(f"  pelvis travelled   {np.linalg.norm(end_xy - start_xy):.2f} m")
    print(f"  can moved          {np.linalg.norm(can1 - can0) * 100:.1f} cm "
          f"(z {can0[2]:.3f} -> {can1[2]:.3f})")
    print(f"  can lifted         {'YES' if can1[2] - can0[2] > 0.06 else 'no'}")
    print(f"  robot fell         {'YES' if fell else 'no'}")
    qs = np.stack([r["q"] for r in rep])
    n = min(len(qs), len(caps_q[::5]))
    d = np.abs(qs[:n] - caps_q[::5][:n])
    print(f"  joint |diff| vs capture: median {np.median(d):.3f} rad, "
          f"p90 {np.percentile(d, 90):.3f}, max {d.max():.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, tokens=tokens, capture_q=caps_q,
                        replay=np.array(rep, dtype=object),
                        sample=json.dumps(sample))
    print(f"[replay] wrote {OUT}")


if __name__ == "__main__":
    main()
