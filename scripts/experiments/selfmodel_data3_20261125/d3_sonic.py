"""SONIC v1.1 in LocoManiLab's benchmark host behind lml_host.Host's reference interface (selfmodel_data3_20261125).

SonicHost(env_name="EmptyEnv", seed=...) is lml_host.Host with backend "sonic": the same EmptyEnv / G1FloatingBody /
G1_BRIDGE composite, the same methods (sample, push, step, reset, transform, pelvis_pose, to_frame, start_frames,
plan, run_reset) and the same return values, so a collector written against Host(backend="scalebfm") drives it
unchanged. Only the backend underneath differs. Nothing in g1_backend, humanoid_benchmark or lml_host is edited; the
adaptation is three in-process substitutions made before the env is built:

1. Checkpoint: g1_backend's sonic family default (checkpoints/sonic/low_latency/g1) ships only the kPlanner ONNX, no
   encoder / decoder, so it cannot run. The v1.1 split encoder / decoder (checkpoints/sonic/v1_1/g1/policy.yaml) is
   passed as the policy_config. PD gains, default pose, action scale and observation layout are that yaml's, unchanged.
2. Command space: the host's SonicBackend declares vla_action_space "motion_token" (a VLA predicting the 64-D latent),
   which our planner cannot produce. SONIC's own reference path is its g1 sub-encoder (mode 0): future joint positions,
   joint velocities and root (anchor) orientation read from the RealtimeMotionBuffer -- the path the backend itself
   uses to feed kPlanner's qpos trajectory (SonicBackend._step_planner_family pushes a 36-D qpos frame, encoder mode
   "g1"). A subclass registered under the same name "sonic" declares "ref_qpos" instead and delivers the frame with
   sim2real's RefQposAdapter and the FK payload builder GenericTrackingBackend gives ScaleBFM / HEFT
   (default_reference_frame(robot_cfg).payload_from_qpos), so the buffer sees the same payload layout, the same
   first-frame stamp and the same episode transform as under ScaleBFM; the encoder is pinned to g1 mode every tick.
   The encoder runs (no token is injected); the decoder is untouched.
3. kPlanner: planner_path is dropped from the composite config. PLANNER mode is never entered under a VLA action, and
   the 739 MB planner ONNX would otherwise be loaded by every process.

What the g1 sub-encoder reads, and what it does not: joint angles / velocities of 10 future frames ([0, 5, .., 45]
steps) and the reference root orientation relative to the robot's heading. It does not read the reference root
position or height: squat depth comes from the leg joints of our IK reference, walking from the leg joints of hg_walk's
footstep reference. The buffer serves frame t - delay_s with delay_s = 45 * 0.02 + 0.04 = 0.94 s (ScaleBFM 0.14 s), so
SONIC executes a pushed frame ~0.94 s after it is pushed (with a 0.9 s preview). The methods keep Host's push-index
semantics; `ref_delay_s` / `ref_delay_f` tell a caller how far behind the robot is (Host: host.buf.delay_s).

Inference: onnx-cpu by default (~16 ms of a ~21 ms step on one core). D3_SONIC_INFER=onnx-gpu with
CUDA_VISIBLE_DEVICES=k runs the same encoder / decoder graphs on CUDA through the onnxruntime-gpu overlay (ORT_GPU
below; policy ~3 ms, step ~7 ms; same seeds give the same falls / losses as onnx-cpu, reach errors within 0.04 cm
median).

    # smoke (selfmodel_data3_20261125), one process per worker; cwd g1_backend, PYTHONPATH=$G1B:$LOCO/src
    [D3_SONIC_INFER=onnx-gpu CUDA_VISIBLE_DEVICES=k] python d3_sonic.py smoke --out OUT --inplace 6 --walk 3 --worker K
    python d3_sonic.py report --out OUT [--also OUT2 ...]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LML = HERE.parent / "locomanilab_20261117"
HG = HERE.parent / "hardgrasp_20261125"
sys.path[:0] = [str(LML), str(HG)]
import lml_host as LH  # noqa: E402
import lml_selfmodel as L  # noqa: E402

SONIC_POLICY = L.G1B / "checkpoints" / "sonic" / "v1_1" / "g1" / "policy.yaml"
# GPU inference: the env's onnxruntime is the CPU-only wheel, and swapping it there would change every other process
# using the env (ScaleBFM / HEFT). onnxruntime-gpu 1.25.0 (same graphs; max |cpu - gpu| 4e-6 on the decoder) sits in its
# own directory and is put ahead of the env only for a process that asks for D3_SONIC_INFER=onnx-gpu.
ORT_GPU = Path(os.environ.get("D3_ORT_GPU", "/wuji-vefps-D/wuji-il/yijie/envs/ort_gpu_overlay_py311"))
if os.environ.get("D3_SONIC_INFER") == "onnx-gpu" and ORT_GPU.is_dir():
    if "onnxruntime" in sys.modules:
        assert "CUDAExecutionProvider" in sys.modules["onnxruntime"].get_available_providers(), (
            "onnxruntime (CPU build) was imported before d3_sonic; import d3_sonic first or put "
            f"{ORT_GPU} first on PYTHONPATH")
    else:
        sys.path.insert(0, str(ORT_GPU))


def _install_sonic_ref_backend() -> None:
    """Register (in this process) a SonicBackend that takes ref_qpos through SONIC's g1 sub-encoder."""
    import g1_bridge.backends as B
    from g1_bridge.backends.sonic import SonicBackend
    from g1_bridge.reference.default import default_reference_frame

    if getattr(B._CACHE.get("sonic"), "_d3_ref_qpos", False):
        return

    class SonicRefQposBackend(SonicBackend):
        name = "sonic"
        vla_action_space = "ref_qpos"
        _d3_ref_qpos = True

        def make_vla_adapter(self, ctx):
            from sim2real.vla.adapters import get_adapter

            return get_adapter("ref_qpos", frame_builder=default_reference_frame(ctx.tracking.robot_cfg).payload_from_qpos)

        def vla_step(self, ctx, tick, action):
            enc = self._sonic_encoder_obs(ctx)
            if enc is None:
                raise RuntimeError("no split-onnx g1_input encoder observation: not a SONIC g1 checkpoint")
            enc.set_encoder_mode("g1")
            return super().vla_step(ctx, tick, action)

    B._CACHE["sonic"] = SonicRefQposBackend


@contextlib.contextmanager
def _sonic_config(policy_config: Path, inference_backend: str):
    """Host.__init__ builds its controller config with robocasa.g1_host.bridge.load_sim2real_controller_config(backend,
    pico_source="mock", inference_backend="onnx-cpu"); for its duration that call gets the v1.1 policy_config, the
    chosen inference backend and no kPlanner."""
    import robocasa.g1_host.bridge as BR

    orig = BR.load_sim2real_controller_config

    def wrapped(backend, pico_source, **kw):
        kw.update(policy_config=str(policy_config), inference_backend=inference_backend)
        cfg = orig(backend, pico_source, **kw)
        cfg["composite_controller_specific_configs"].pop("planner_path", None)
        return cfg

    BR.load_sim2real_controller_config = wrapped
    try:
        yield
    finally:
        BR.load_sim2real_controller_config = orig


class SonicHost(LH.HostDriver):
    """lml_host.HostDriver (the "Host") with the SONIC v1.1 backend (see the module docstring); everything else inherited."""

    def __init__(self, env_name: str = "EmptyEnv", seed: int = 0, inference_backend: str | None = None, **env_kw):
        sys.path[:0] = [str(LH.HB), str(L.G1B)]
        import robocasa  # noqa: F401

        assert SONIC_POLICY.exists(), SONIC_POLICY
        _install_sonic_ref_backend()
        self.inference_backend = inference_backend or os.environ.get("D3_SONIC_INFER", "onnx-cpu")
        with _sonic_config(SONIC_POLICY, self.inference_backend):
            super().__init__(env_name=env_name, backend="sonic", seed=seed, **env_kw)
        sess = self.comp._session
        assert type(sess.backend).__name__ == "SonicRefQposBackend", type(sess.backend)
        assert Path(sess.tracking.args.policy_config).resolve() == SONIC_POLICY.resolve()
        self.providers = sess.tracking._decoder_module.ort_session.get_providers()
        buf = sess.tracking.motion_buffer
        self.ref_delay_s = float(buf.delay_s)
        self.ref_delay_f = int(round(self.ref_delay_s * L.FPS))

    def reset(self) -> None:
        super().reset()
        assert type(self.comp._session.backend).__name__ == "SonicRefQposBackend"



# -- smoke ----------------------------------------------------------------------------------------------------------
SETTLE_V, SETTLE_S, SETTLE_MAX_S = 0.02, 0.5, 3.0  # the entry's settle rule: pelvis and both hands < 2 cm/s for 0.5 s
WALK_D, WALK_TURN = (0.3, 1.5), 90.0  # walk-first starts: 0.3-1.5 m, turn 0-90 deg
SQUAT_P, SQUAT_Z = 0.55, (0.52, 0.76)


def smoke_command(rng, side: str) -> dict:
    """lml_selfmodel.sample_command's target box (CORE 70 % / WIDE 30 %, y mirrored for the left hand) with the entry's
    wider wrist range (yaw +-90, pitch -30..90; roll is not in Host.plan's command) and 55 % squats at 0.52-0.76 m."""
    lo, hi = L.CORE if rng.random() < 0.7 else L.WIDE
    t = rng.uniform(lo, hi)
    if side == "left":
        t[1] = -t[1]
    return dict(side=side, target=t, dyaw=math.radians(rng.uniform(-90, 90)), pitch=math.radians(rng.uniform(-30, 90)),
                pelvis_z=float(rng.uniform(*SQUAT_Z)) if rng.random() < SQUAT_P else None)  # fmt: skip


def _points(h, x):
    p = x["root"][:3]
    hands = [pw + Rw @ L.HAND_OFF for pw, Rw in x["wr"].values()]
    return np.stack([p, *hands])


def run_smoke_reset(h: SonicHost, rng, walk: bool) -> dict:
    """One reset in the entry's protocol: stance; [walk 0.3-1.5 m / turn 0-90 deg]; then two reaches, each commanded
    only once the robot has settled, each labelled once it has settled again (3 s cap each). Settle and lost checks
    are taken in robot time (from the pushed frame's end + ref_delay_f)."""
    import hg_walk as W

    D = h.ref_delay_f
    nq, nmax = int(round(SETTLE_S * L.FPS)), int(round(SETTLE_MAX_S * L.FPS))
    t0 = time.time()
    h.reset()
    frames = h.start_frames()
    sides = [("right" if rng.random() < 0.5 else "left") for _ in range(2)]
    cmds = [smoke_command(rng, s) for s in sides]
    res = dict(walk=walk, reaches=[], fell=None, lost=None, why="")
    todo = (["walk"] if walk else []) + ["reach0", "reach1"]
    f, prev, quiet, settle_from, waited = 0, None, 0, D, 0
    anchor, cur, rec, plan_t = None, None, None, 0.0
    while True:
        if f + 2 >= len(frames):
            frames = np.concatenate([frames, np.repeat(frames[-1:], 50, 0)])
        h.push(frames[f])
        if h.buf is None:
            h.buf = h._find_buffer()
        h.step()
        if f == 1:
            assert not h.comp._settling
        x = h.sample()
        stop = None
        if float(x["root"][2]) < L.FALL_Z:
            stop = dict(fell=f, why="height")
        elif anchor is not None:  # reference displacement taken ref_delay_f behind, where the robot is
            fr = frames[max(f - D, anchor[2])]
            dev = (x["root"][:2] - anchor[0]) - (fr[:2] - anchor[1])
            if float(np.linalg.norm(dev)) > L.LOST_M:
                stop = dict(fell=f, lost=f, why="lost")
        if stop is not None:  # what was playing: the walk, or the reach (its command and IK reference)
            act = dict(phase=cur, f_rel=None if anchor is None else f - anchor[2])
            if cur is not None and cur != "walk":
                act.update({k: rec[k] for k in ("side", "pelvis_z", "dyaw", "pitch", "cmd_target", "ref_pel", "reach_s", "ik_err")})
            res.update(stop, active=act)
            break
        pts = _points(h, x)
        v = np.inf if prev is None else float(np.linalg.norm(pts - prev, axis=1).max()) * L.FPS
        prev = pts
        if f >= settle_from:
            quiet = quiet + 1 if v < SETTLE_V else 0
            waited += 1
        settled, timeout = f >= settle_from and quiet >= nq, f >= settle_from and waited >= nmax
        ready = False
        if cur is not None and (settled or timeout):  # label the finished motion
            if cur == "walk":
                g = np.array(res["walk_goal"])
                yaw = L.yaw_of(L._qmat(x["root"][3:]))
                res.update(walk_settled=bool(settled), walk_end=[*x["root"][:2].tolist(), yaw],
                           walk_pos_err_cm=100 * float(np.linalg.norm(x["root"][:2] - g[:2])),
                           walk_yaw_err_deg=math.degrees(W.wrap(yaw - g[2])), walk_settle_s=waited / L.FPS)  # fmt: skip
            else:
                pw, Rw = x["wr"][rec["side"]]
                land = pw + Rw @ L.HAND_OFF
                rw = np.array(rec["rot_w"])
                res["reaches"].append(dict(
                    side=rec["side"], settled=bool(settled), settle_s=waited / L.FPS, skipped=False,
                    err_cm=100 * float(np.linalg.norm(land - np.array(rec["target_w"]))),
                    att_deg=math.degrees(math.acos(np.clip((np.trace(rw.T @ Rw) - 1) / 2, -1, 1))),
                    pelvis_z=float(x["root"][2]), ref_pelvis_z=float(rec["ref_pel"][2]), pelvis_cmd=rec["pelvis_z"], ref_pel=rec["ref_pel"],
                    ik_err=rec["ik_err"], reach_s=rec["reach_s"], dyaw=rec["dyaw"], pitch=rec["pitch"],
                    cmd_target=rec["cmd_target"], pre_settled=rec["pre_settled"], pre_settle_s=rec["pre_settle_s"]))  # fmt: skip
            cur, anchor = None, None
            if not todo:
                break
            if settled:  # already quiet: the next command goes out now
                settle_from, waited, ready = f, quiet, True
            else:  # a fresh settle window for the next one
                settle_from, waited, quiet = f + 1, 0, 0
        elif cur is None and (settled or timeout):
            ready = True
        if ready and todo and todo[0] != "walk" and not settled:  # 3 s without settling: this reach is not taken
            c = cmds[int(todo.pop(0)[-1])]
            res["reaches"].append(dict(side=c["side"], skipped=True, pre_settled=False, pelvis_cmd=c["pelvis_z"]))
            settle_from, waited, quiet = f + 1, 0, 0
            if not todo:
                break
        elif ready and todo:
            nxt = todo.pop(0)
            if nxt == "walk":
                d, a = rng.uniform(*WALK_D), math.radians(rng.uniform(-90, 90))
                turn = math.radians(rng.uniform(0, WALK_TURN)) * (1 if rng.random() < 0.5 else -1)
                p, yaw = x["root"][:2], L.yaw_of(L._qmat(x["root"][3:]))
                g_xy = p + (L.rot_z(yaw) @ np.r_[d * math.cos(a), d * math.sin(a), 0.0])[:2]
                tic = time.time()
                wf = W.walk_frames(h, x, frames[f], g_xy, yaw + turn, info=(wi := {}))
                plan_t += time.time() - tic
                res.update(walk_goal=[*g_xy.tolist(), yaw + turn], walk_d=d, walk_turn_deg=math.degrees(turn),
                           walk_pre_settled=bool(settled), walk_plan=wi)  # fmt: skip
                frames = np.concatenate([frames[: f + 1], wf])
                f_end = f + len(wf)
            else:
                c = cmds[int(nxt[-1])]
                try:
                    blend, last, rec = h.plan(x, frames[f], c)
                except (np.linalg.LinAlgError, ValueError) as e:
                    res.update(why=f"ik {type(e).__name__}: {e}")
                    break
                rec.update(pre_settled=True, pre_settle_s=waited / L.FPS)
                frames = np.concatenate([frames[: f + 1], blend, L.hold(last, 0.2)])
                f_end = f + len(blend)
            cur = nxt
            anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
            settle_from, waited, quiet = f_end + D, 0, 0
        f += 1
    res.update(steps=f + 1, wall_s=time.time() - t0, plan_s=plan_t + h.plan_s, sim_s=(f + 1) / L.FPS)
    h.plan_s = 0.0
    return res


def smoke(out: Path, n_inplace: int, n_walk: int, worker: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fn = out / f"smoke_w{worker:02d}.jsonl"
    done = sum(1 for _ in fn.open()) if fn.exists() else 0
    t0, c0 = time.time(), time.process_time()
    h = SonicHost("EmptyEnv", seed=seed + worker)
    init_s = time.time() - t0
    print(f"worker {worker}: SonicHost up in {init_s:.1f} s, inference {h.inference_backend} {h.providers[0]}, ref delay "
          f"{h.ref_delay_s:.2f} s ({h.ref_delay_f} frames), pelvis offset c {np.round(h.c, 4).tolist()}", flush=True)  # fmt: skip
    plan = [False] * n_inplace + [True] * n_walk
    with fn.open("a") as fo:
        for i in range(done, len(plan)):
            rng = np.random.default_rng(seed + 1000 * worker + i)
            r = run_smoke_reset(h, rng, plan[i])
            ru = resource.getrusage(resource.RUSAGE_SELF)
            r.update(worker=worker, i=i, seed0=seed, infer=h.inference_backend, rss_mb=ru.ru_maxrss / 1024, cpu_s=time.process_time() - c0,
                     wall_total_s=time.time() - t0, init_s=init_s)  # fmt: skip
            fo.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)) + "\n")
            fo.flush()
            print(f"worker {worker} reset {i} walk={plan[i]}: fell {r['fell']} lost {r['lost']} {r['why']} "
                  f"reaches {[(q['side'][0], 'skip') if q['skipped'] else (q['side'][0], round(q['err_cm'], 1), q['settled']) for q in r['reaches']]} "
                  f"{r['wall_s']:.1f} s wall / {r['sim_s']:.1f} s sim", flush=True)  # fmt: skip


def report(outs: list[Path]) -> None:
    rs = [json.loads(s) for out in outs for fn in sorted(out.glob("smoke_w*.jsonl")) for s in fn.open()]
    med = lambda a: float(np.median(a)) if len(a) else float("nan")  # noqa: E731
    pct = lambda a, n: f"{100 * a / max(n, 1):.1f} % ({a}/{n})"  # noqa: E731
    print(f"{len(rs)} resets from {len(outs)} run(s) x {len({r['worker'] for r in rs})} workers")
    for walk in (False, True):
        g = [r for r in rs if r["walk"] == walk]
        n = len(g)
        fell = sum(r["fell"] is not None and r["lost"] is None for r in g)
        lost = sum(r["lost"] is not None for r in g)
        ik = sum(r["why"].startswith("ik") for r in g)
        rall = [q for r in g for q in r["reaches"]]
        rc = [q for q in rall if not q["skipped"]]
        pre = [q["pre_settled"] for q in rall]
        post = [q["settled"] for q in rc]
        print(f"\n== {'walk-first' if walk else 'in place'}: {n} resets, {len(rc)} reaches labelled")
        print(f"   fall {pct(fell, n)} | lost (>0.5 m off the reference displacement) {pct(lost, n)} | ik failures {ik}")
        ends = [r["active"] for r in g if r.get("active")]
        if ends:
            def ref_pitch(a):
                R = L._qmat(np.array(a["ref_pel"][3:7]))
                return math.degrees(math.atan2(-R[2, 0], R[2, 2]))
            ra = [a for a in ends if a["phase"] not in (None, "walk")]
            print(f"   falls / lost by phase: walk {sum(a['phase'] == 'walk' for a in ends)}, reach {len(ra)}, idle {sum(a['phase'] is None for a in ends)}; "
                  f"in reaches: squat {sum(a['pelvis_z'] is not None for a in ra)}/{len(ra)}, left {sum(a['side'] == 'left' for a in ra)}, "
                  f"IK pelvis pitch median {med([ref_pitch(a) for a in ra]):.0f} deg, target x median {med([a['cmd_target'][0] for a in ra]):.2f} m, "
                  f"z median {med([a['cmd_target'][2] for a in ra]):.2f} m")  # fmt: skip
        print(f"   reaches skipped (not settled within 3 s before the command) {pct(len(rall) - len(rc), len(rall))}")
        print(f"   settled before the command {pct(sum(pre), len(pre))} | settled after the reach {pct(sum(post), len(post))}"
              f" | settle time after the reference end (settled) median {med([q['settle_s'] for q in rc if q['settled']]):.2f} s")  # fmt: skip
        for side in ("right", "left"):
            s = [q for q in rc if q["side"] == side]
            ss = [q for q in s if q["settled"]]
            print(f"   {side:5s}: n {len(s)}, end error median {med([q['err_cm'] for q in s]):.1f} cm (all) / "
                  f"{med([q['err_cm'] for q in ss]):.1f} cm (settled, n {len(ss)}), attitude {med([q['att_deg'] for q in ss]):.1f} deg")  # fmt: skip
        if rc and "ref_pel" in rc[0]:
            pq = [math.degrees(math.atan2(-L._qmat(np.array(q["ref_pel"][3:7]))[2, 0], L._qmat(np.array(q["ref_pel"][3:7]))[2, 2])) for q in rc]
            print(f"   labelled reaches: IK pelvis pitch median {med(pq):.0f} deg, target x median {med([q['cmd_target'][0] for q in rc]):.2f} m")
        sq = [q for q in rc if q["pelvis_cmd"] is not None]
        if sq:
            dz = [q["pelvis_z"] - q["ref_pelvis_z"] for q in sq]
            print(f"   squats n {len(sq)}: reference pelvis {min(q['ref_pelvis_z'] for q in sq):.2f}-{max(q['ref_pelvis_z'] for q in sq):.2f} m, "
                  f"reached {min(q['pelvis_z'] for q in sq):.2f}-{max(q['pelvis_z'] for q in sq):.2f} m, measured - reference median "
                  f"{100 * med(dz):+.1f} cm, |.| p90 {100 * float(np.percentile(np.abs(dz), 90)):.1f} cm")  # fmt: skip
        st = [q for q in rc if q["pelvis_cmd"] is None]
        if st:
            print(f"   standing reaches n {len(st)}: pelvis {med([q['pelvis_z'] for q in st]):.3f} m median")
        if walk:
            w = [r for r in g if "walk_pos_err_cm" in r]
            print(f"   walks finished {len(w)}/{n}: settled {sum(r['walk_settled'] for r in w)}, end position error median "
                  f"{med([r['walk_pos_err_cm'] for r in w]):.1f} cm, |yaw error| median {med([abs(r['walk_yaw_err_deg']) for r in w]):.1f} deg")  # fmt: skip
        print(f"   per reset: wall {med([r['wall_s'] for r in g]):.1f} s median ({np.mean([r['wall_s'] for r in g]):.1f} mean), "
              f"sim {med([r['sim_s'] for r in g]):.1f} s, planning {med([r['plan_s'] for r in g]):.2f} s")  # fmt: skip
    last = {}
    for r in rs:
        last[(r["seed0"] if "seed0" in r else 0, r["worker"])] = r
    print(f"\nper process: RSS {med([r['rss_mb'] for r in last.values()]):.0f} MB, CPU / wall "
          f"{med([r['cpu_s'] / r['wall_total_s'] for r in last.values()]):.2f}, init {med([r['init_s'] for r in last.values()]):.1f} s")  # fmt: skip
    w = sum(r["wall_s"] for r in rs) / max(len(rs), 1)
    print(f"mean wall per reset (all) {w:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["smoke", "report"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--also", type=Path, nargs="*", default=[], help="report: more smoke directories pooled with --out")
    ap.add_argument("--inplace", type=int, default=6)
    ap.add_argument("--walk", type=int, default=3)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--seed", type=int, default=91000000)
    a = ap.parse_args()
    if a.phase == "smoke":
        smoke(a.out, a.inplace, a.walk, a.worker, a.seed)
    else:
        report([a.out, *a.also])
