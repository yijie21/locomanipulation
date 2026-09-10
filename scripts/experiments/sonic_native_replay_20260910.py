"""Private instance-1 native CSV replay, original release model/gains.

No Python pose packets; C++ reads the official motion directly. Task root-height
fall flag is recorded, not used to abort intentional deep squat references.
"""

import json
import os
import pty
import subprocess
import time
import tty
from pathlib import Path

import numpy as np

from locomani.backends.sonic.primitives import Primitives
from locomani.primitives.arm_ik import ArmIK
from locomani.sim.privileged_pub import EpisodeControlClient, PrivilegedSubscriber

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("SONIC_NATIVE_OUT", ROOT / "outputs/diagnostics/sonic_native_phase3_20260910/retry_reset"))
PRIVATE = ROOT / "outputs/diagnostics/sonic_lean_phase2_20260910/deploy"
# Single-factor knob: swap the whole checkpoint set (decoder + encoder + its own
# observation config) without touching the protocol. Default is the release set
# the first valid replay was measured on.
POLICY = os.environ.get("SONIC_NATIVE_POLICY", "policy/release")
INIT_TIMEOUT = float(os.environ.get("SONIC_NATIVE_INIT_TIMEOUT", "300"))


def main():
    assert os.environ.get("SONIC_DDS_DOMAIN") == "1"
    from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

    OUT.mkdir(parents=True, exist_ok=True)
    # Stop only the instance-1 bootstrap deploy, keep its simulation alive.
    for p in Path("/proc").glob("[0-9]*"):
        try:
            args = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if args and args[0].endswith(b"g1_deploy_onnx_ref") and b"--zmq-port" in args:
            if args[args.index(b"--zmq-port") + 1] == b"5566":
                os.kill(int(p.name), 15)
    time.sleep(1)
    sub = PrivilegedSubscriber(port=5591)
    reset = EpisodeControlClient(port=5592)
    ik = ArmIK()
    torso = ik.model.body("torso_link").id
    errors = []
    for repeat in range(1, 4):
        path = OUT / f"native_release_{repeat}.json"
        if path.exists():
            raise RuntimeError(f"refuse overwrite {path}")
        reset.pin_arms()
        reset.reset(label=f"native_{repeat}")
        time.sleep(0.5)
        master, slave = pty.openpty()
        tty.setraw(slave)
        env = dict(os.environ, SONIC_ENCODER_AUDIT_FILE=str(OUT / f"encoder_{repeat}.bin"))
        command = [
            str(PRIVATE / "target/release/g1_deploy_onnx_ref"),
            "lo",
            f"{POLICY}/model_decoder.onnx",
            "reference/example/",
            "--obs-config",
            f"{POLICY}/observation_config.yaml",
            "--encoder-file",
            f"{POLICY}/model_encoder.onnx",
            "--input-type",
            "zmq",
            "--output-type",
            "all",
            "--zmq-host",
            "localhost",
            "--zmq-port",
            "5566",
            "--zmq-out-port",
            "5567",
            "--disable-crc-check",
            "--default-motion",
            "squat_001__A359",
        ]
        logpath = OUT / f"deploy_{repeat}.log"
        with logpath.open("w") as log:
            proc = subprocess.Popen(command, cwd=PRIVATE, env=env, stdin=slave, stdout=log, stderr=subprocess.STDOUT)
        os.close(slave)
        debug = ZMQStateSubscriber(port=5567)
        rows = []
        result = dict(repeat=repeat, command=command, policy=POLICY, complete=False, trace=rows)
        try:
            end = time.monotonic() + INIT_TIMEOUT
            while "Init Done" not in logpath.read_text(errors="replace"):
                if proc.poll() is not None:
                    raise RuntimeError("deploy failed before ready")
                if time.monotonic() > end:
                    raise TimeoutError("deploy init timeout")
                time.sleep(0.5)
            # Loading the ONNX models leaves the simulator uncontrolled. Reset
            # AFTER readiness; never count that interval as reference tracking.
            reset_wall = time.time()
            reset.reset(label=f"native_ready_{repeat}")
            deadline = time.monotonic() + 2
            while True:
                result["preplay"] = sub.latest(40)
                if result["preplay"] is not None and result["preplay"].get("wall", 0) > reset_wall:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("no fresh post-reset state")
                time.sleep(0.02)
            result["startup_protocol"] = "reset_after_init__engage_1s_settle__then_play"
            if result["preplay"] is None or result["preplay"]["pelvis"]["pos"][2] < 0.6:
                raise RuntimeError("invalid preplay state")
            print("START", repeat, flush=True)
            result["start_wall"] = time.time()
            os.write(master, b"]")
            settle = []
            settle_end = time.monotonic() + 1.0
            while time.monotonic() < settle_end:
                st = sub.latest(40)
                if st is not None:
                    settle.append(dict(wall=time.time(), state=st))
                time.sleep(0.02)
            result["settle"] = settle
            if settle and settle[-1]["state"]["pelvis"]["pos"][2] < 0.6:
                raise RuntimeError("collapsed during control engage")
            print("PLAY", repeat, flush=True)
            if os.environ.get("SONIC_NATIVE_VIDEO"):
                reset.start_video(f"{POLICY.split('/')[-1]}_{repeat}")
            result["play_wall"] = time.time()
            start = time.monotonic()
            os.write(master, b"t")
            while time.monotonic() - start < 12:
                state = sub.latest(40)
                msg = debug.get_msg()
                if state is None:
                    continue
                ik.fk(np.array(state["joints"]), "right")
                mat = Primitives._rot(state["pelvis"]["quat"]) @ ik.data.xmat[torso].reshape(3, 3)
                rows.append(
                    dict(
                        wall=time.time(),
                        play_elapsed=time.monotonic() - start,
                        state=state,
                        torso_pitch=float(np.arcsin(np.clip(-mat[2, 0], -1, 1))),
                        debug=None
                        if msg is None
                        else {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in msg.items()},
                    )
                )
                # Height0.4 is crossed in the official reference itself. Distinguish that from collapse.
                if state["pelvis"]["pos"][2] < 0.15:
                    raise RuntimeError("physical collapse below15cm")
                if proc.poll() is not None:
                    raise RuntimeError("deploy exited during replay")
            result["complete"] = True
        except Exception as exc:
            result["error"] = repr(exc)
            errors.append(dict(repeat=repeat, error=repr(exc)))
        finally:
            if os.environ.get("SONIC_NATIVE_VIDEO"):
                reset.end_video()
                time.sleep(1.5)
            path.write_text(json.dumps(result))
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            debug.close()
            os.close(master)
        print("DONE", repeat, flush=True)
    sub.close()
    (OUT / "complete.json").write_text(
        json.dumps(dict(cases=3, errors=errors, status="complete" if not errors else "partial"))
    )


if __name__ == "__main__":
    main()
