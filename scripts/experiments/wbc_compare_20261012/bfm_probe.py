"""wbc_compare step 2 (TODO 24.2): the open-loop right-hand probe on the ScaleBFM runtime.

The robot is teleported to the 65-object line's stance on an object scene (bfm_object_scene.stance), the object is
parked behind it (probe_cells.park_position), and after the R.SETTLE_S settle the right wrist reference walks the
probe cells (probe_cells.cells): to each cell by grasp_planned's staged legs (up to OVER_CLEAR above the table,
over, down) at SPEED m/s with the palm attitude slerping from the hanging wrist to the cells' forward-facing attitude
during the first cell's rise, then a HOLD_S hold. Mode 2 sees the pelvis at the stance, the left wrist at the side
and the right wrist reference six frames ahead (R.Refs.futures). Over the hold's last MEASURE_S the delivered tool
(sim.pad("right") = wrist + OFFSET) is averaged against the command and reported in the heading frame, with the palm
rotation error and the pelvis drift. probe_summary.py turns both controllers' probe.json into the feed-forward bias
the hand program (24.3) adds to its commands.

    CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa PYTHONPATH=src third_party/Isaac-GR00T/.venv/bin/python \
        scripts/experiments/wbc_compare_20261012/bfm_probe.py --object 40 --video \
        --out outputs/wbc_compare_20261012/probe/scalebfm
"""

from __future__ import annotations

import argparse
import json
import subprocess
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

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402

SPEED = 0.15  # m/s along the tool reference between knots
HOLD_S = 2.0
MEASURE_S = 0.5  # the hold's last half second is averaged
FPS = 12.5  # video frames per second (every 4th policy step)
MIN_LEG_S = 0.2


def slerp(Ra: np.ndarray, Rb: np.ndarray, u: float) -> np.ndarray:
    qa, qb = R.quat_wxyz(Ra), R.quat_wxyz(Rb)
    d = float(qa @ qb)
    if d < 0:
        qb, d = -qb, -d
    if d > 0.9995:
        q = qa + u * (qb - qa)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - u) * th) * qa + np.sin(u * th) * qb) / np.sin(th)
    return R.mat(q / np.linalg.norm(q))


class Schedule:
    """Knots (t, tool position, palm rotation); linear position and slerp rotation between them; the last held."""

    def __init__(self, pos: np.ndarray, rot: np.ndarray):
        self.knots = [(0.0, np.asarray(pos, dtype=float).copy(), np.asarray(rot, dtype=float).copy())]

    @property
    def end(self) -> float:
        return self.knots[-1][0]

    def hold(self, seconds: float) -> None:
        t, p, r = self.knots[-1]
        self.knots.append((t + seconds, p.copy(), r.copy()))

    def move(self, pos: np.ndarray, rot: np.ndarray, speed: float = SPEED) -> None:
        t, p, _ = self.knots[-1]
        self.knots.append((t + max(MIN_LEG_S, float(np.linalg.norm(pos - p)) / speed), np.asarray(pos, float), rot))

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if t >= self.end:
            return self.knots[-1][1], self.knots[-1][2]
        for (ta, pa, ra), (tb, pb, rb) in zip(self.knots, self.knots[1:]):
            if ta <= t < tb:
                u = (t - ta) / (tb - ta)
                return pa + u * (pb - pa), slerp(ra, rb, u)
        return self.knots[0][1], self.knots[0][2]


def open_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)],
        stdin=subprocess.PIPE,
    )  # fmt: skip


def preview(full: Path, small: Path) -> int:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(full), "-vf", "scale=480:-2", "-c:v", "libx264",
         "-crf", "34", "-an", str(small)],
        check=True,
    )  # fmt: skip
    return small.stat().st_size


def probe(pol: R.Policy, obj_id: int, hold_s: float, speed: float, video: Path | None) -> dict:
    t_wall = time.time()
    sim = make_object_sim(pol.meta, obj_id, video=video is not None)
    xy, yaw, edge = stance(sim)
    sim.reset(xy, yaw, 0.0)
    sim.d.qpos[sim.crate_q : sim.crate_q + 3] = C.park_position(xy, yaw)
    mujoco.mj_forward(sim.m, sim.d)
    refs = R.Refs(pol, sim)
    pol.reset(sim.state())
    pel = refs.pelvis(xy, yaw)
    side = refs.hands_at_side(np.array([*xy, R.PELVIS_Z]), yaw)
    wp, wq = side["right_wrist_yaw_link"]
    R_side = R.mat(wq)
    sched = Schedule(wp + R_side @ C.OFFSET, R_side)
    sched.hold(R.SETTLE_S)
    windows = []
    home = C.home_tool(xy, yaw, sim.table_top)
    R_c = C.palm_rotation(yaw)
    # from the hanging hand: up, then to HOME, the attitude turning to the cells' on the way (half each leg)
    for li, (_, b) in enumerate(C.legs(sched.knots[-1][1], home, sim.table_top)[:2]):
        sched.move(b, slerp(R_side, R_c, 0.5 * (li + 1)), speed)
    tool_prev = home
    for cell in C.cells():
        tool_c, _ = C.tool_pose(xy, yaw, cell)
        for _, b in C.legs_via_home(tool_prev, tool_c, home):
            sched.move(b, R_c, speed)
        windows.append((cell, tool_c, R_c, sched.end, sched.end + hold_s))
        sched.hold(hold_s)
        tool_prev = tool_c

    def ref(t):
        pos, rot = sched.at(t)
        return dict(
            pel,
            left_wrist_yaw_link=side["left_wrist_yaw_link"],
            right_wrist_yaw_link=(C.wrist_from_tool(pos, rot), R.quat_wxyz(rot)),
        )

    writer = open_writer(video) if video else None
    n = int(round(sched.end / R.DT)) + 1
    samples: dict[int, list] = {w[0]["id"]: [] for w in windows}
    trace: dict[int, list] = {w[0]["id"]: [] for w in windows}
    fell_at = None
    wi = 0
    for k in range(n):
        t = k * R.DT
        sim.step(pol(sim.state(), refs.futures(ref, t)))
        if writer and k % 4 == 0:
            writer.stdin.write(np.ascontiguousarray(sim.frame()).tobytes())
        if sim.fallen():
            fell_at = round(t, 2)
            break
        while wi < len(windows) and t > windows[wi][4] + 1e-9:
            wi += 1
        if wi < len(windows) and windows[wi][3] <= t <= windows[wi][4]:
            cell, tool_c, R_c, t0, t1 = windows[wi]
            tool_m = sim.pad("right")
            trace[cell["id"]].append(round(100 * float(np.linalg.norm(tool_m - tool_c)), 2))
            if t >= t1 - MEASURE_S:
                pp, pq = sim.body("pelvis")
                _, wq_m = sim.body("right_wrist_yaw_link")
                samples[cell["id"]].append((tool_m, R.mat(wq_m), pp, pq))
    if writer:
        writer.stdin.close()
        writer.wait()
    rows = []
    for cell, tool_c, R_c, t0, t1 in windows:
        s = samples[cell["id"]]
        row = dict(cell, command_tool=[round(float(v), 4) for v in tool_c], hold_trace_cm=trace[cell["id"]])
        if s:
            tool_m = np.mean([v[0] for v in s], axis=0)
            R_m = s[-1][1]
            pp = np.mean([v[2] for v in s], axis=0)
            pyaw = float(np.mean([R.wrap(R.yaw_of(v[3]) - yaw) for v in s]))
            err = tool_m - tool_c
            row.update(
                measured_tool=[round(float(v), 4) for v in tool_m],
                err_heading_cm=[round(100 * v, 2) for v in C.to_heading(err, yaw)],
                err_norm_cm=round(100 * float(np.linalg.norm(err)), 2),
                rot_err_deg=round(C.rot_err_deg(R_c, R_m), 1),
                palm_x_heading=[round(v, 3) for v in C.to_heading(R_m[:, 0], yaw)],
                pelvis_err_cm=[round(100 * v, 2) for v in C.to_heading(pp - np.array([*xy, 0.0]), yaw)[:2]],
                pelvis_yaw_err_deg=round(float(np.degrees(pyaw)), 2),
                pelvis_z=round(float(pp[2]), 3),
                hold_max_cm=max(trace[cell["id"]]),
            )
        rows.append(row)
    return dict(
        controller="scalebfm",
        object=obj_id,
        edge=edge,
        stance_xy=[round(float(v), 4) for v in xy],
        stance_yaw_deg=round(float(np.degrees(yaw)), 2),
        speed_m_s=speed,
        hold_s=hold_s,
        measure_s=MEASURE_S,
        settle_s=R.SETTLE_S,
        fell_at_s=fell_at,
        sim_s=round(sched.end, 2),
        wall_s=round(time.time() - t_wall, 1),
        finger_law=dict(kp=sim.cfg.grip_kp, kd=sim.cfg.grip_kd, clip=sim.grip_clip),
        cells=rows,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", type=int, default=40)
    ap.add_argument("--hold", type=float, default=HOLD_S)
    ap.add_argument("--speed", type=float, default=SPEED)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--video", action="store_true", help="write probe.mp4 (12.5 fps) and a <=500 KB probe_preview.mp4")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    assert_matches_grasp_cache()
    args.out.mkdir(parents=True, exist_ok=True)
    pol = R.Policy(device=args.device, backend=ScaleBfmBackendConfig())
    video = args.out / "probe.mp4" if args.video else None
    result = probe(pol, args.object, args.hold, args.speed, video)
    (args.out / "probe.json").write_text(json.dumps(result, indent=1))
    for r in result["cells"]:
        print(
            f"cell {r['id']:2d} fwd {r['forward']:.2f} left {r['left']:+.2f} z {r['z']:.2f}: err f/l/u "
            f"{r.get('err_heading_cm', '-')} cm |{r.get('err_norm_cm', '-')}| rot {r.get('rot_err_deg', '-')} deg "
            f"pelvis {r.get('pelvis_err_cm', '-')} cm",
            flush=True,
        )
    print(
        f"fell {result['fell_at_s']}; {result['sim_s']} sim s in {result['wall_s']} s; wrote {args.out / 'probe.json'}"
    )
    if video:
        print(f"preview {preview(video, args.out / 'probe_preview.mp4') / 1024:.0f} KB")


if __name__ == "__main__":
    main()
