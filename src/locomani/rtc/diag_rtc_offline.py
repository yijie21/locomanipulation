"""Wire-level validation of serve_policy's RTC primitive. No sim needed.

    # server already up (any checkpoint):
    python diag_rtc_offline.py --port 5590

Sends the SAME synthetic observation three times and asserts the inpainting
contract end-to-end (client wire format -> RTCPolicy -> model primitive ->
decode), which is the part a unit test on the primitive alone would miss:

  1. rtc.reset + rtc.elapsed 0   -> chunk A, no rtc.applied (nothing cached
     yet), but the server caches A's normalized form;
  2. rtc.elapsed E, rtc.frozen F -> chunk B with rtc.applied == [40-E, F];
     the frozen contract: B[0:F] must equal A[E:E+F] (same obs -> identical
     decode of identical normalized rows, so tolerance is float32 noise);
     the ramp region B[F:40-E] should sit BETWEEN free prediction and A's
     rows -- reported, not asserted (it is a soft blend);
  3. plain request (no rtc.*)    -> no rtc.applied, and the server must have
     dropped its cache (a following elapsed>0 request gets no rtc.applied
     either): plain clients keep plain semantics.

Content of the obs is random -- the contract is about WHICH rows the denoiser
was allowed to move, which does not depend on the scene making sense.
"""

from __future__ import annotations

import argparse
import sys

import msgpack
import msgpack_numpy as mnp
import numpy as np


def synth_obs(seed: int = 0) -> dict:
    r = np.random.default_rng(seed)
    f32 = lambda n: r.uniform(-0.5, 0.5, (1, n)).astype(np.float32)  # noqa: E731
    obs = {
        "state.left_leg": f32(6),
        "state.right_leg": f32(6),
        "state.waist": f32(3),
        "state.left_arm": f32(7),
        "state.right_arm": f32(7),
        "state.left_hand": f32(7),
        "state.right_hand": f32(7),
        "state.projected_gravity": np.asarray([[0, 0, -1]], dtype=np.float32),
        "state.right_wrist_pos": f32(3),
        "state.right_wrist_abs_quat": np.asarray([[1, 0, 0, 0]], dtype=np.float32),
        "annotation.human.task_description": "pick up the water bottle",
        "video.ego_view": r.integers(0, 255, (1, 256, 256, 3), dtype=np.uint8),
        "video.right_wrist": r.integers(0, 255, (1, 256, 256, 3), dtype=np.uint8),
    }
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5590)
    ap.add_argument("--elapsed", type=int, default=10)
    ap.add_argument("--frozen", type=int, default=10)
    args = ap.parse_args()

    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 120000)
    sock.connect(f"tcp://localhost:{args.port}")

    def call(extra):
        obs = synth_obs()
        obs.update(extra)
        sock.send(msgpack.packb({"endpoint": "get_action_raw", "data": {"obs": obs}}, default=mnp.encode))
        rep = msgpack.unpackb(sock.recv(), object_hook=mnp.decode)
        if isinstance(rep, dict) and "error" in rep:
            raise SystemExit(f"server error: {rep['error']}")
        return rep

    E, F = args.elapsed, args.frozen
    fails = []

    a = call({"rtc.reset": 1, "rtc.elapsed": 0, "rtc.frozen": 0})
    if "rtc.applied" in a:
        fails.append("request 1 (elapsed 0) must not apply RTC")
    ta = np.asarray(a["action.motion_token"])
    ga = np.asarray(a["action.right_gripper"]).reshape(-1)
    print(f"[1] chunk A: token {ta.shape}, no rtc.applied {'OK' if 'rtc.applied' not in a else 'FAIL'}")

    b = call({"rtc.elapsed": E, "rtc.frozen": F})
    app = b.get("rtc.applied")
    if app is None:
        fails.append("request 2 got no rtc.applied -- cache/injection dead")
        print("[2] FAIL: no rtc.applied")
    else:
        ov, fr = int(app[0]), int(app[1])
        print(f"[2] rtc.applied = overlap {ov}, frozen {fr} (expected {ta.shape[0] - E}, {F})")
        if ov != ta.shape[0] - E or fr != F:
            fails.append(f"applied ({ov},{fr}) != expected ({ta.shape[0] - E},{F})")
        tb = np.asarray(b["action.motion_token"])
        gb = np.asarray(b["action.right_gripper"]).reshape(-1)
        d_frozen = float(np.abs(tb[:F] - ta[E : E + F]).max())
        d_frozen_g = float(np.abs(gb[:F] - ga[E : E + F]).max())
        d_ramp = float(np.abs(tb[F:ov] - ta[E + F : E + ov]).max())
        d_free = (
            float(np.abs(tb[ov:] - ta[E + ov : E + ov + tb.shape[0] - ov]).max())
            if ta.shape[0] > E + ov
            else float("nan")
        )
        print(f"    frozen  max|B[0:{F}] - A[{E}:{E + F}]|   = {d_frozen:.2e} (token) {d_frozen_g:.2e} (grip)")
        print(f"    ramp    max|B[{F}:{ov}] - A[...]|  = {d_ramp:.3f} (soft blend, should be small but nonzero)")
        print(f"    tail    max diff beyond overlap = {d_free:.3f} (free, should be the largest)")
        if d_frozen > 1e-3 or d_frozen_g > 1e-3:
            fails.append(f"frozen prefix moved: {d_frozen:.3e}")

    c = call({})
    if "rtc.applied" in c:
        fails.append("plain request must not apply RTC")
    d = call({"rtc.elapsed": E, "rtc.frozen": F})
    if "rtc.applied" in d:
        fails.append("plain request must DROP the cache; the follow-up elapsed>0 request still got rtc.applied")
    print(f"[3] plain request clears cache: {'OK' if 'rtc.applied' not in c and 'rtc.applied' not in d else 'FAIL'}")

    if fails:
        print("\nFAILED:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("\nRTC offline validation PASSED")


if __name__ == "__main__":
    main()
