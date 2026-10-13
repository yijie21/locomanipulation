"""wbc_compare step 6 (TODO 24.6): ScaleBFM's attitude delivery envelope.
The 24.2 probe (bfm_probe.py) walked 27 positions at one palm attitude (approach 30 deg down, jaw left); the free
arm then showed ScaleBFM failing on the tilted / yawed grasps the validation score prefers. Here the hand sits at
probe_cells.ATTITUDE_POSITIONS (the central column, two heights) and the palm attitude walks the grid
probe_cells.attitudes() (elevation x yaw deviation x roll about the approach, the same frame attitude_of reads a
cached grasp in): from HOME the palm turns in place to the attitude (at most ROT_SPEED deg/s), goes over and down
to the position, holds HOLD_S (the last MEASURE_S averaged, as bfm_probe), rises and returns to HOME. A fall ends
that cell (fell_at_s), the sim is reset and the hand brought back to HOME before the next one. Cells are sharded
round-robin over --shards processes (each writes attitude_<shard>.json after every cell); attitude_summary.py merges
them into envelope.json, which bfm_grasp.py --envelope uses to reject the attitudes ScaleBFM did not deliver.
    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=osmesa PYTHONPATH=src OMP_NUM_THREADS=2 third_party/Isaac-GR00T/.venv/bin/python \\
        scripts/experiments/wbc_compare_20261012/bfm_attitude_probe.py --object 40 --device cuda --shard 0 --shards 4 \\
        [--video] --out outputs/wbc_compare_20261012/probe/scalebfm_attitude
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import probe_cells as C  # noqa: E402
from bfm_object_scene import assert_matches_grasp_cache, make_object_sim, stance  # noqa: E402
from bfm_probe import HOLD_S, MEASURE_S, SPEED, Schedule, open_writer, preview, slerp  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402

ROT_SPEED = 90.0  # deg/s: the palm turns in place at HOME before each cell
MIN_TURN_S = 0.5


def turn(sched: Schedule, rot: np.ndarray, rot_speed: float = ROT_SPEED) -> None:
    t, p, r = sched.knots[-1]
    sched.knots.append((t + max(MIN_TURN_S, C.rot_err_deg(r, rot) / rot_speed), p.copy(), np.asarray(rot, float)))


class Probe:
    def __init__(self, pol: R.Policy, obj_id: int, hold_s: float, speed: float, video: Path | None):
        self.pol, self.hold_s, self.speed = pol, hold_s, speed
        self.sim = make_object_sim(pol.meta, obj_id, video=video is not None)
        self.xy, self.yaw, self.edge = stance(self.sim)
        self.home = C.home_tool(self.xy, self.yaw, self.sim.table_top)
        self.R0 = C.palm_rotation(self.yaw)
        self.writer = open_writer(video) if video else None
        self.k = 0  # policy steps so far (video every 4th)
        self.sim_s = 0.0
        self.resets = 0
        self.start()

    def start(self) -> None:
        """Reset, park the object, settle, bring the hand from the side up to HOME at the 24.2 attitude."""
        sim = self.sim
        sim.reset(self.xy, self.yaw, 0.0)
        sim.d.qpos[sim.crate_q : sim.crate_q + 3] = C.park_position(self.xy, self.yaw)
        mujoco.mj_forward(sim.m, sim.d)
        self.refs = R.Refs(self.pol, sim)
        self.pol.reset(sim.state())
        self.pel = self.refs.pelvis(self.xy, self.yaw)
        self.side = self.refs.hands_at_side(np.array([*self.xy, R.PELVIS_Z]), self.yaw)
        wp, wq = self.side["right_wrist_yaw_link"]
        R_side = R.mat(wq)
        sched = Schedule(wp + R_side @ C.OFFSET, R_side)
        sched.hold(R.SETTLE_S)
        for li, (_, b) in enumerate(C.legs(sched.knots[-1][1], self.home, sim.table_top)[:2]):
            sched.move(b, slerp(R_side, self.R0, 0.5 * (li + 1)), self.speed)
        self.pose = (self.home.copy(), self.R0.copy())
        fell = self.run(sched, None)[2]
        if fell is not None:
            raise RuntimeError(f"fell at {fell} s on the way to HOME")

    def run(self, sched: Schedule, window):
        """Step the schedule to its end. `window` = (t0, t1, tool_c, R_c): the hold to trace, its last MEASURE_S
        sampled (tool, palm R, pelvis pos, pelvis quat). Returns (samples, trace_cm, fell_at_s)."""
        sim = self.sim

        def ref(t):
            pos, rot = sched.at(t)
            return dict(
                self.pel,
                left_wrist_yaw_link=self.side["left_wrist_yaw_link"],
                right_wrist_yaw_link=(C.wrist_from_tool(pos, rot), R.quat_wxyz(rot)),
            )

        samples, trace = [], []
        n = int(round(sched.end / R.DT)) + 1
        for k in range(n):
            t = k * R.DT
            sim.step(self.pol(sim.state(), self.refs.futures(ref, t)))
            if self.writer and self.k % 4 == 0:
                self.writer.stdin.write(np.ascontiguousarray(sim.frame()).tobytes())
            self.k += 1
            self.sim_s += R.DT
            if sim.fallen():
                return samples, trace, round(t, 2)
            if window and window[0] <= t <= window[1]:
                t0, t1, tool_c, R_c = window
                tool_m = sim.pad("right")
                trace.append(round(100 * float(np.linalg.norm(tool_m - tool_c)), 2))
                if t >= t1 - MEASURE_S:
                    pp, pq = sim.body("pelvis")
                    _, wq_m = sim.body("right_wrist_yaw_link")
                    samples.append((tool_m, R.mat(wq_m), pp, pq))
        return samples, trace, None

    def cell(self, pos: dict, att: dict) -> dict:
        tool_c, _ = C.tool_pose(self.xy, self.yaw, pos)
        R_c = C.palm_rotation_at(self.yaw, att["elev"], att["yaw_dev"], att["roll"])
        p0, r0 = self.pose
        sched = Schedule(p0, r0)
        turn(sched, R_c)
        over = tool_c.copy()
        over[2] = self.home[2]
        sched.move(over, R_c, self.speed)
        sched.move(tool_c, R_c, self.speed)
        t0 = sched.end
        sched.hold(self.hold_s)
        t1 = sched.end
        sched.move(over, R_c, self.speed)
        sched.move(self.home, R_c, self.speed)
        samples, trace, fell = self.run(sched, (t0, t1, tool_c, R_c))
        row = dict(
            position=pos["id"], forward=pos["forward"], left=pos["left"], z=pos["z"],
            attitude=att["id"], elev=att["elev"], yaw_dev=att["yaw_dev"], roll=att["roll"],
            command_tool=[round(float(v), 4) for v in tool_c], fell_at_s=fell, hold_trace_cm=trace,
        )  # fmt: skip
        if samples:
            tool_m = np.mean([v[0] for v in samples], axis=0)
            R_m = samples[-1][1]
            pp = np.mean([v[2] for v in samples], axis=0)
            pyaw = float(np.mean([R.wrap(R.yaw_of(v[3]) - self.yaw) for v in samples]))
            err = tool_m - tool_c
            row.update(
                measured_tool=[round(float(v), 4) for v in tool_m],
                err_heading_cm=[round(100 * v, 2) for v in C.to_heading(err, self.yaw)],
                err_norm_cm=round(100 * float(np.linalg.norm(err)), 2),
                rot_err_deg=round(C.rot_err_deg(R_c, R_m), 1),
                measured_attitude_deg=[round(v, 1) for v in C.attitude_of(R_m, self.yaw)],
                pelvis_err_cm=[round(100 * v, 2) for v in C.to_heading(pp - np.array([*self.xy, 0.0]), self.yaw)[:2]],
                pelvis_yaw_err_deg=round(float(np.degrees(pyaw)), 2),
                pelvis_z=round(float(pp[2]), 3),
                hold_max_cm=max(trace) if trace else None,
            )
        if fell is not None:
            self.resets += 1
            self.start()
        else:
            self.pose = (self.home.copy(), R_c)
        return row

    def close(self) -> None:
        if self.writer:
            self.writer.stdin.close()
            self.writer.wait()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", type=int, default=40)
    ap.add_argument("--hold", type=float, default=HOLD_S)
    ap.add_argument("--speed", type=float, default=SPEED)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--video", action="store_true", help="attitude_<shard>.mp4 (12.5 fps) + a <=500 KB preview")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    assert_matches_grasp_cache()
    args.out.mkdir(parents=True, exist_ok=True)
    cells = [(pos, att) for pos in C.ATTITUDE_POSITIONS for att in C.attitudes()][args.shard :: args.shards]
    pol = R.Policy(device=args.device, backend=ScaleBfmBackendConfig())
    video = args.out / f"attitude_{args.shard}.mp4" if args.video else None
    t_wall = time.time()
    probe = Probe(pol, args.object, args.hold, args.speed, video)
    out = args.out / f"attitude_{args.shard}.json"
    rows = []
    for i, (pos, att) in enumerate(cells):
        row = probe.cell(pos, att)
        rows.append(row)
        print(
            f"[{i + 1}/{len(cells)}] pos {pos['id']} z {pos['z']:.2f} elev {att['elev']:3d} yaw {att['yaw_dev']:4d} "
            f"roll {att['roll']:3d}: err f/l/u {row.get('err_heading_cm', '-')} cm |{row.get('err_norm_cm', '-')}| "
            f"rot {row.get('rot_err_deg', '-')} deg fell {row['fell_at_s']} ({time.time() - t_wall:.0f} s)",
            flush=True,
        )
        out.write_text(
            json.dumps(
                dict(
                    controller="scalebfm",
                    object=args.object,
                    edge=probe.edge,
                    stance_xy=[round(float(v), 4) for v in probe.xy],
                    stance_yaw_deg=round(float(np.degrees(probe.yaw)), 2),
                    speed_m_s=args.speed,
                    hold_s=args.hold,
                    measure_s=MEASURE_S,
                    rot_speed_deg_s=ROT_SPEED,
                    positions=list(C.ATTITUDE_POSITIONS),
                    shard=args.shard,
                    shards=args.shards,
                    done=i + 1,
                    total=len(cells),
                    resets=probe.resets,
                    sim_s=round(probe.sim_s, 2),
                    wall_s=round(time.time() - t_wall, 1),
                    finger_law=dict(kp=probe.sim.cfg.grip_kp, kd=probe.sim.cfg.grip_kd, clip=probe.sim.grip_clip),
                    cells=rows,
                ),  # fmt: skip
                indent=1,
            )
        )
    probe.close()
    print(
        f"{len(rows)} cells, {probe.resets} resets, {probe.sim_s:.0f} sim s in {time.time() - t_wall:.0f} s; wrote {out}"
    )
    if video:
        print(f"preview {preview(video, args.out / f'attitude_{args.shard}_preview.mp4') / 1024:.0f} KB")


if __name__ == "__main__":
    main()
