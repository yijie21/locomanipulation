"""Instance-1 static coordinated-reference ablation with real encoder capture.

A pose from official squat clip frame85 is a coordinated reference, NOT a replay
of the collaborator's deployment. Existing pose channel's zero dq is retained
in every condition; assess only final steady holds, not dynamic tracking.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.sonic import stream_reach
from locomani.backends.sonic.primitives import LocomotionMode, Primitives, StreamMode
from locomani.primitives.arm_ik import ArmIK
from locomani.primitives.g1_joints import isaac_to_mujoco
from locomani.sim.privileged_pub import EpisodeControlClient

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/diagnostics/sonic_lean_phase2_20260910"
REF = ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example/squat_001__A359"
CONDITIONS = ["legacy_waist_052", "waist_only", "legs_waist", "legs_waist_root", "full_pose"]


def save(path, data):
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def rotq(mat):
    return Rotation.from_matrix(mat).as_quat()[[3, 0, 1, 2]]


def rest(p, reset, label):
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.set_upper_body_vel(None)
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label=label)
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3, mode=LocomotionMode.IDLE)
    stream_reach.reanchor(p)
    return p.obs()


def resource_gate():
    path = Path("/sys/fs/cgroup/cpu.stat")

    def read():
        return {k: int(v) for k, v in (x.split() for x in path.read_text().splitlines())}

    quota, period = map(int, Path("/sys/fs/cgroup/cpu.max").read_text().split())
    streak = 0
    for _ in range(120):
        a = read()
        t = time.monotonic()
        time.sleep(5)
        b = read()
        free = quota / period - (b["usage_usec"] - a["usage_usec"]) / 1e6 / (time.monotonic() - t)
        good = free >= 6 and b["throttled_usec"] == a["throttled_usec"]
        streak = streak + 1 if good else 0
        with (OUT / "resources.jsonl").open("a") as f:
            f.write(json.dumps(dict(wall=time.time(), free=free, streak=streak)) + "\n")
        if streak >= 3:
            return
    raise RuntimeError("resource gate timed out")


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    assert os.environ.get("TF_SONIC_PORT") == "5566" and os.environ.get("TF_EPISODE_PORT") == "5592"
    from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

    OUT.mkdir(parents=True, exist_ok=True)
    qclip = np.loadtxt(REF / "joint_pos.csv", delimiter=",", skiprows=1)
    roots = np.loadtxt(REF / "body_quat.csv", delimiter=",", skiprows=1).reshape(-1, 14, 4)
    positions = np.loadtxt(REF / "body_pos.csv", delimiter=",", skiprows=1).reshape(-1, 14, 3)
    qref = isaac_to_mujoco(qclip[85])
    rootref = Primitives._rot(roots[85, 0])
    yaw = np.arctan2(rootref[1, 0], rootref[0, 0])
    rootref = Rotation.from_euler("z", -yaw).as_matrix() @ rootref
    ik = ArmIK()
    torso = ik.model.body("torso_link").id
    ik.fk(qref, "right")
    reference_torso = rootref @ ik.data.xmat[torso].reshape(3, 3)
    save(
        OUT / "reference.json",
        dict(
            clip=str(REF),
            frame=85,
            joints_mj=qref.tolist(),
            root_rotation=rootref.tolist(),
            original_pelvis_z=float(positions[85, 0, 2]),
            torso_pitch_deg=float(np.degrees(np.arcsin(-reference_torso[2, 0]))),
            protocol="static pose ablation, 3s ramp/4s hold/3s return, zero reference dq in all conditions",
        ),
    )
    p = Primitives(verbose=False)
    reset = EpisodeControlClient()
    debug = ZMQStateSubscriber(port=5567)
    completed = []
    try:
        for rep in range(args.repeats):
            for name in CONDITIONS if rep % 2 == 0 else list(reversed(CONDITIONS)):
                dest = OUT / f"{name}_{rep + 1}.json"
                if dest.exists():
                    old = json.loads(dest.read_text())
                    if old.get("complete"):
                        completed.append(str(dest))
                        continue
                    raise RuntimeError(f"incomplete case needs inspection: {dest}")
                resource_gate()
                print("RUN", name, rep + 1, flush=True)
                st = rest(p, reset, f"phase2_{name}_{rep + 1}")
                if st["fallen"]:
                    raise RuntimeError("reset failed to stand")
                arm = stream_reach.StreamedArm(p, "right")
                arm.start()
                q0 = arm.q.copy()
                root0 = Primitives._rot(arm.root_quat)
                target = q0.copy()
                target[14] = 0.52 if name == "legacy_waist_052" else qref[14]
                if name in ["legs_waist", "legs_waist_root", "full_pose"]:
                    target[:15] = qref[:15]
                if name == "full_pose":
                    target = qref.copy()
                target_root = root0 @ rootref if name in ["legs_waist_root", "full_pose"] else root0
                ik.fk(target, "right")
                desired_torso = target_root @ ik.data.xmat[torso].reshape(3, 3)
                record = dict(
                    condition=name,
                    repeat=rep + 1,
                    complete=False,
                    requested=target.tolist(),
                    desired_torso_deg=float(np.degrees(np.arcsin(-desired_torso[2, 0]))),
                    trace=[],
                )
                original = arm._push
                phase = "ramp"

                def push(q):
                    original(q)
                    state = p.obs()
                    msg = debug.get_msg()
                    ik.fk(np.array(state["joints"]), "right")
                    mat = Primitives._rot(state["pelvis"]["quat"]) @ ik.data.xmat[torso].reshape(3, 3)
                    record["trace"].append(
                        dict(
                            wall=time.time(),
                            sim_t=state["t"],
                            phase=phase,
                            request=np.asarray(q).tolist(),
                            root_request=arm.root_quat.tolist(),
                            joints=state["joints"],
                            pelvis=state["pelvis"],
                            fallen=state["fallen"],
                            observation_age=time.time() - state["wall"],
                            torso_pitch=float(np.arcsin(np.clip(-mat[2, 0], -1, 1))),
                            debug=None
                            if msg is None
                            else {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in msg.items()},
                        )
                    )
                    if state["fallen"] or time.time() - state["wall"] > 0.2:
                        raise RuntimeError("fallen/stale observation")

                arm._push = push

                def ramp(qa, qb, ra, rb):
                    delta = Rotation.from_matrix(ra.T @ rb).as_rotvec()
                    for k in range(1, 151):
                        u = k / 150
                        u = 3 * u * u - 2 * u * u * u
                        arm.root_quat = rotq(ra @ Rotation.from_rotvec(delta * u).as_matrix())
                        arm._push(qa + (qb - qa) * u)
                    arm.q = qb.copy()

                try:
                    ramp(q0, target, root0, target_root)
                    phase = "hold"
                    arm.hold(4)
                    phase = "return"
                    ramp(target, q0, target_root, root0)
                    record["complete"] = True
                except Exception as exc:
                    record["error"] = repr(exc)
                    raise
                finally:
                    save(dest, record)
                arm.stop()
                completed.append(str(dest))
                save(OUT / "progress.json", dict(status="running", completed=completed))
                print("DONE", name, rep + 1, flush=True)
        save(OUT / "progress.json", dict(status="complete", completed=completed))
    finally:
        debug.close()
        p.close()


if __name__ == "__main__":
    main()
