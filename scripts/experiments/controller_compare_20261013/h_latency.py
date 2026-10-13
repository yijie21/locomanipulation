"""h veto (TODO 25.1, docs/handoff/controller-compare-20261013.md §2): ScaleBFM's per-step latency.

One step = what the deployed controller does every 20 ms: Policy.__call__ (history push, world -> base frame of the
future keypoints, the network with its internal FK, copy back to the host). Timed inside a live standing loop on the
mars warehouse scene, so the inputs are real. Three back ends:

* `eager`  -- runtime.Policy as every ScaleBFM expert runs it (torch, --device).
* `trt`    -- the same wrapper exported to ONNX (opset 17) and built with TensorRT (the venv's `tensorrt`), FP32 and
  FP16; outputs checked against eager on the same inputs before timing.
* `cpu`    -- eager on the CPU (an onboard computer without a usable GPU), fewer steps.

Veto: p99 > 20 ms. The onboard computer is unknown (TODO 25.11.1); this machine is the lower bound.

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 third_party/Isaac-GR00T/.venv/bin/python \\
        scripts/experiments/controller_compare_20261013/h_latency.py --out outputs/controller_compare_20261013/h
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from w0_admission import seat  # noqa: E402

SCENE = REPO / "outputs" / "scene" / "warehouse_box_45x32x33_mars"
INPUTS = ("root_quat", "ang_vel", "dof_pos", "dof_vel", "action", "target_pos", "target_quat", "mode", "time_offsets")


def stats(ms):
    ms = np.asarray(ms)
    return dict(
        n=int(ms.size),
        mean_ms=float(ms.mean()),
        p50_ms=float(np.percentile(ms, 50)),
        p99_ms=float(np.percentile(ms, 99)),
        max_ms=float(ms.max()),
    )


def standing_loop(pol, sim, steps, timed_call):
    """Run `steps` policy steps standing at the scene's start; timed_call(st, fut) returns the 29 targets."""
    xy = np.asarray(sim.smeta["robot_start"]["xy"], float)
    yaw = float(np.radians(sim.smeta["robot_start"]["yaw_deg"]))
    sim.reset(xy, yaw, 0.0)
    seat(sim)
    refs = R.Refs(pol, sim)
    pol.reset(sim.state())
    pel = np.array([xy[0], xy[1], R.PELVIS_Z])
    q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

    def ref(t):
        out = {"pelvis": (pel, q)}
        out.update(refs.hands_at_side(pel, yaw))
        return out

    ms = []
    for k in range(steps):
        fut = refs.futures(ref, k * R.DT)
        st = sim.state()
        t0 = time.perf_counter()
        tgt = timed_call(st, fut)
        ms.append(1e3 * (time.perf_counter() - t0))
        sim.step(tgt)
        if sim.fallen():
            raise RuntimeError(f"fell at step {k}")
    return ms


def capture_inputs(pol, st, fut):
    """The exact tensors Policy.__call__ hands the network (a copy of its body up to the network call)."""
    pol.push(st)
    pw = np.zeros((pol.F, pol.N, 3))
    qw = np.zeros((pol.F, pol.N, 4))
    qw[..., 0] = 1
    for n, (p, qq) in fut.items():
        pw[:, pol.isel[n]] = p
        qw[:, pol.isel[n]] = qq
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=pol.dev)
    rq = t(pol.buf["root_quat"])[None]
    root_q = rq[:, -1]
    tp = R.quat_apply_inverse(root_q, t(pw)[None] - t(st["root_pos"])[None, None, None, :])
    tq = R.quat_mul_inverse_left(root_q[:, None, None, :].expand(-1, pol.F, pol.N, -1), t(qw)[None])
    return (
        rq,
        t(pol.buf["ang_vel"])[None],
        t(pol.buf["dof_pos"])[None],
        t(pol.buf["dof_vel"])[None],
        t(pol.buf["action"])[None],
        tp,
        tq,
        pol.mode,
        pol.toff,
    )


class TrtPolicy:
    """A TensorRT engine of the wrapper, called like the network inside Policy.__call__ (same input tensors)."""

    def __init__(self, onnx_path, engine_path, fp16):
        import tensorrt as trt

        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        if not engine_path.exists():
            builder = trt.Builder(logger)
            net = builder.create_network(0)
            parser = trt.OnnxParser(net, logger)
            if not parser.parse(onnx_path.read_bytes()):
                raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            cfg = builder.create_builder_config()
            if fp16:
                cfg.set_flag(trt.BuilderFlag.FP16)
            engine_path.write_bytes(bytes(builder.build_serialized_network(net, cfg)))
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.outs = {}
        for n in self.names:
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.ctx.get_tensor_shape(n))
                dt = torch.float32 if self.engine.get_tensor_dtype(n) == trt.DataType.FLOAT else torch.int64
                self.outs[n] = torch.empty(shape, dtype=dt, device="cuda")
                self.ctx.set_tensor_address(n, self.outs[n].data_ptr())

    def __call__(self, *inputs):
        keep = []
        for n, x in zip(INPUTS, inputs):
            if n not in self.names:  # constant-folded away (an input named like an output would be renamed: refuse)
                assert not any(m.startswith(n + ".") for m in self.names), f"ONNX renamed input {n}"
                continue
            want = torch.int64 if self.engine.get_tensor_dtype(n) == self.trt.DataType.INT64 else torch.float32
            x = x.to(want).contiguous()
            keep.append(x)
            self.ctx.set_tensor_address(n, x.data_ptr())
        # torch's current stream: the inputs were written there (async H2D copies, quaternion ops), so the engine
        # must queue behind them; a stream of its own read them before they landed
        stream = torch.cuda.current_stream()
        self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        tgt, act = (self.outs[n] for n in self.names if n in self.outs)
        return tgt, act


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--cpu-steps", type=int, default=500)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = ScaleBfmBackendConfig(scene_dir=str(SCENE), object_name="nor_cal_box")
    res = dict(
        machine=dict(
            gpu=torch.cuda.get_device_name(0),
            cpu=platform.processor() or platform.machine(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
        ),
        veto_p99_ms=20.0,
    )

    pol = R.Policy(device="cuda", backend=cfg)
    sim = R.Sim(pol.meta, backend=cfg)
    standing_loop(pol, sim, 200, pol)  # warm-up
    ms = standing_loop(pol, sim, a.steps, lambda st, fut: (pol(st, fut), torch.cuda.synchronize())[0])
    res["eager_cuda"] = stats(ms)
    print("eager cuda", res["eager_cuda"], flush=True)

    # ONNX export on real inputs, then TensorRT FP32 / FP16. The export traces a CPU copy: on CUDA the exporter's
    # constant folding meets CPU-side constants (mixed devices); the ONNX graph is device-free either way.
    pol_cpu = R.Policy(device="cpu", backend=cfg)
    st0 = sim.state()
    xy = np.asarray(sim.smeta["robot_start"]["xy"], float)
    pel = np.array([xy[0], xy[1], R.PELVIS_Z])
    yaw = float(np.radians(sim.smeta["robot_start"]["yaw_deg"]))
    fut0 = R.Refs(pol, sim).futures(
        lambda t: {"pelvis": (pel, np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))}, 0.0
    )
    pol_cpu.reset(st0)
    ex_cpu = capture_inputs(pol_cpu, st0, fut0)
    ex = tuple(x.to("cuda") for x in ex_cpu)
    onnx_path = a.out / "scalebfm_m22200.onnx"
    if not onnx_path.exists():
        # load_policy builds the wrapper under inference_mode; tracing outside no_grad tries to save those tensors
        with torch.no_grad():
            torch.onnx.export(
                pol_cpu.w,
                ex_cpu,
                str(onnx_path),
                input_names=list(INPUTS),
                output_names=["pd_target", "action_out"],
                opset_version=17,
                dynamo=False,
            )
    with torch.inference_mode():
        ref_tgt, ref_act = pol.w(*ex)
    eager_net = pol.w
    for fp16 in (False, True):
        key = "trt_fp16" if fp16 else "trt_fp32"
        try:
            trt_pol = TrtPolicy(onnx_path, a.out / f"scalebfm_m22200_{key}.engine", fp16)
            tgt, act = trt_pol(*ex)
            err = float((tgt - ref_tgt).abs().max())
            res[key] = dict(max_abs_err_vs_eager_rad=err)
            pol.w = trt_pol
            standing_loop(pol, sim, 200, pol)
            ms = standing_loop(pol, sim, a.steps, pol)
            res[key] = dict(stats(ms), max_abs_err_vs_eager_rad=err)
        except Exception as e:  # noqa: BLE001 -- a failed build is a result, recorded and reported
            res[key] = dict(res.get(key, {}), error=repr(e)[:500])
        finally:
            pol.w = eager_net
        print(key, res[key], flush=True)

    standing_loop(pol_cpu, sim, 50, pol_cpu)
    res["eager_cpu"] = dict(stats(standing_loop(pol_cpu, sim, a.cpu_steps, pol_cpu)), threads=torch.get_num_threads())
    print("eager cpu", res["eager_cpu"], flush=True)
    (a.out / "latency.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
