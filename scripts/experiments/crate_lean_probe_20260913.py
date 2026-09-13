"""Compare frozen joints, VR three-point and full-pose lean delivery away from contact.

The same legal upper-body pose is used; VR/pose additionally request a 20 degree
pelvis hinge. All references ramp from the measured pose; no table contact.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from locomani import paths
from locomani.pipeline import stack
from locomani.pipeline.provenance import digest, file_digests, write_config_snapshot
from locomani.pipeline.run_crate_pilot import backend_config
from locomani.pipeline.stages import Stage
from locomani.tasks.crate.config import CrateTaskConfig


def worker(out):
    from locomani.backends.sonic.agent_server import StreamMode
    from locomani.backends.sonic.primitives import Primitives, upper_body_from_mj
    from locomani.backends.sonic.streamed_arm import PlannerArm, StreamedArm
    from locomani.backends.sonic.teleop_arm import NECK_OFFSET, VR_OFFSET, _mat2quat_wxyz
    from locomani.primitives.arm_ik import ArmIK
    from locomani.sim.privileged_pub import EpisodeControlClient
    from locomani.tasks.crate import round_runner
    from locomani.tasks.crate.carry_phases import CarryRoundContext, _phase_reset

    meta = json.loads(Path(paths.SCENE_META).read_text())
    p, reset, ik = Primitives(meta=meta), EpisodeControlClient(), ArmIK()
    round_runner.prime(p, reset)
    rng = np.random.default_rng(11)
    for index, channel in enumerate(("frozen", "vr", "pose")):
        label = f"lean_{channel}"
        ctx = CarryRoundContext(p, reset, meta, index, rng, p.log, label, dict(ok=False, why="", produced=False))
        samples = []
        arm = None
        try:
            if _phase_reset(ctx) is not None:
                continue
            s0 = p.obs()
            q0 = np.array(s0["joints"])
            q1 = q0.copy()
            q1[[0, 6]] -= 0.35
            q1[12:15] = [0, 0, 0.5]
            q1[15:22] = [-0.35, 0.2, 0, 0.4, 0, 0, 0]
            q1[22:29] = [-0.35, -0.2, 0, 0.4, 0, 0, 0]
            ranges = ik.model.jnt_range[ik.jids]
            q1 = np.clip(q1, ranges[:, 0], ranges[:, 1])
            base0 = Primitives._rot(s0["pelvis"]["quat"])
            base1 = base0 @ Rotation.from_euler("y", 0.35).as_matrix()
            base_slerp = Slerp([0, 1], Rotation.from_matrix([base0, base1]))
            bodies = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link", "neck": "torso_link"}
            offsets = dict(VR_OFFSET, neck=NECK_OFFSET)
            endpoints = {}
            for name, body in bodies.items():
                endpoints[name] = []
                for q, base in ((q0, base0), (q1, base1)):
                    point, rot = ik.body_pose(q, body)
                    endpoints[name].append(
                        (np.array(s0["pelvis"]["pos"]) + base @ (point + rot @ offsets[name]), base @ rot)
                    )
            rotations = {
                name: Slerp([0, 1], Rotation.from_matrix([r for _, r in pair])) for name, pair in endpoints.items()
            }
            if channel == "frozen":
                arm = PlannerArm(p, "right", send_vel=True, left_guard=False)
                arm.start(upper_reference=upper_body_from_mj(q0), settle_seconds=0)
            elif channel == "pose":
                arm = StreamedArm(p, "right", left_guard=False)
                arm.start()
            else:
                p.cmd.upper_body = None
                p.cmd.set_upper_body_vel(None)
                p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
                p.cmd.vr_targets = True
            start = time.time()
            while time.time() - start < 12:
                t = time.time() - start
                u = min(t / 6, 1)
                u = u**3 * (10 - 15 * u + 6 * u**2)
                q = q0 * (1 - u) + q1 * u
                state = p.obs()
                if state["fallen"]:
                    ctx.m["why"] = "fell while leaning"
                    break
                if channel == "vr":
                    root = Primitives._rot(state["pelvis"]["quat"])
                    for name, ((a, _), (b, _)) in endpoints.items():
                        pos = root.T @ (a * (1 - u) + b * u - state["pelvis"]["pos"])
                        rot = root.T @ rotations[name](u).as_matrix()
                        if name == "neck":
                            p.cmd.set_neck(pos, _mat2quat_wxyz(rot))
                        else:
                            p.cmd.set_wrists(**{f"{name}_pos": pos, f"{name}_quat": _mat2quat_wxyz(rot)})
                    p.cmd.frame(facing=p._facing(s0))
                    time.sleep(0.02)
                else:
                    if channel == "pose":
                        arm.root_quat = _mat2quat_wxyz(base_slerp(u).as_matrix())
                    arm._push(q)
                state = p.obs()
                torso_p, torso_r = ik.body_pose(np.array(state["joints"]), "torso_link")
                root = Primitives._rot(state["pelvis"]["quat"])
                samples.append(
                    dict(
                        t=t,
                        seq=state["seq"],
                        pelvis=state["pelvis"],
                        joints=state["joints"],
                        torso_deg=float(np.degrees(np.arccos(np.clip((root @ torso_r)[2, 2], -1, 1)))),
                        command_u=u,
                        command_joints=q.tolist(),
                        command_root_quat=_mat2quat_wxyz(base_slerp(u).as_matrix()).tolist(),
                    )
                )
            tail = [s for s in samples if s["t"] >= 10]
            ctx.m.update(
                ok=bool(tail) and not ctx.m["why"],
                meaning="diagnostic completion, not grasp success",
                torso_deg=float(np.median([s["torso_deg"] for s in tail])) if tail else None,
                pelvis_z=float(np.median([s["pelvis"]["pos"][2] for s in tail])) if tail else None,
            )
        finally:
            reset.end_video()
            (out / f"{label}.json").write_text(json.dumps(samples, indent=2) + "\n")
            with (out / "manifest.jsonl").open("a") as f:
                f.write(json.dumps(dict(ctx.m, video=label)) + "\n")
            print(label, json.dumps(ctx.m), flush=True)
            # Diagnostic only: release before the next reset, outside the video.
            if arm is not None:
                arm.stop()
            p.cmd.reset_pose()
            p.cmd.hold(2, facing=p._facing(p.obs()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.worker:
        worker(out)
        return
    out.mkdir(parents=True, exist_ok=False)
    scene = out / args.scene.name
    shutil.copytree(args.scene, scene)
    (out / "trace").mkdir()
    (out / "videos").mkdir()
    task, backend = CrateTaskConfig(), backend_config()
    write_config_snapshot(
        out,
        task_cfg=task,
        backend_cfg=backend,
        started=time.time(),
        inputs=dict(scene=file_digests(scene), script={Path(__file__).name: digest(__file__)}),
    )
    stage = Stage(out)
    env = stack.start_stack(
        backend,
        task,
        instance=1,
        gpu="0",
        video_dir=str(out / "videos"),
        extra_env=dict(LOCOMANI_SIM_DIR=str(scene), SIM_STATE_TRACE_DIR=str(out / "trace")),
        runner=stage.run,
        receipt_path=out / "deploy_receipts.jsonl",
    )
    try:
        rc = stage.run(
            "lean_probe",
            [paths.VENV_SIM_PY, "-u", str(Path(__file__).resolve()), "--worker", "--out", str(out)],
            env=env,
        )
    finally:
        stack.stop_instance(1, runner=stage.run)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
