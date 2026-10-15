"""TODO 26.3.1 (user 2026-10-15): the bottle expert on ScaleBFM, one round per start state, under a candidate spec -- the
program every role of the bottle RSI loop runs (Gate, Prober, the diagnostic re-runs). The bottle counterpart of
scalebfm_warehouse_rsi_20261014/bfm_round.py, with the same interface.

The round is L2's (controller_compare_20261013/bfm_expert_bottle.Round: walk-in to the guard line, the lifted ready, the
simple expert's grasp, the production tail of backing off and holding), with:
* a candidate spec (YAML or JSON) instead of flags --
      mode: 4                 # ScaleBFM mode 0-7 (L2 ran mode 4)
      set: {simple.walk_speed: 0.25}   # task.reach overrides by dotted path: simple.*, whole_body.*, or a top-level
                                       # reach field (the frozen ones in task.py are refused by the gate)
      reference: {}           # transforms of the expert's whole-body reference before FK (REFERENCE below)
      extensions: {}          # extensions.py hooks by name -> params (new behaviour the loop's Engineer adds)
* a fork: `--fork PHASE:DELTA` runs the base spec until PHASE (walk | ready | grasp) starts and the base merged with DELTA
  from there on (re-running the prefix: a round is decided by (start, spec));
* more recording: qpos (the whole scene state, for rendering), cmd_joints (the reference joints ScaleBFM's keypoints are
  FK'd from), phase (index into PHASES);
* field accounting: facts["fields_read"] lists every task.reach field the round read (the gate rejects a candidate that
  sets a field this path never reads);
* warm_up before the first round of a process (TorchScript's first call differs by ~6e-8, bfm_round.warm_up).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src OMP_NUM_THREADS=2 third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/scalebfm_bottle_rsi_20261015/bottle_round.py <states dir> --spec incumbent.yaml --out /tmp/x \\
        [--shard 0/4] [--only sonic_r00a0] [--fork grasp:delta.yaml]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CC = REPO / "scripts" / "experiments" / "controller_compare_20261013"
sys.path[:0] = [str(HERE), str(CC), str(REPO / "scripts" / "experiments" / "wbc_compare_20261012"), str(REPO / "src")]

import bfm_expert_bottle as X  # noqa: E402
import extensions  # noqa: E402
from bfm_bottle import HAND_FORCE, HAND_KD, HAND_KP, Recorder  # noqa: E402
from bfm_expert_warehouse import Body  # noqa: E402
from bottle_states import SCENE, save  # noqa: E402
from w0_admission import seat  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.primitives.arm_ik import ArmIK  # noqa: E402
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402
from locomani.sim.privileged_pub import PrivilegedPublisher  # noqa: E402
from locomani.tasks.bottle.reach_config import ReachConfig  # noqa: E402

PHASES = ("walk", "ready", "grasp", "verify")
FORKS = ("walk", "ready", "grasp")
SPEC_KEYS = {"mode", "set", "reference", "extensions"}
BASE_SPEC = dict(mode=4, set={}, reference={}, extensions={})
CONFIG = REPO / "configs" / "experiments" / "controller_compare_l2_bottle_sonic_20261013.yaml"
# Reference transforms (spec["reference"]), applied to the expert's whole-body reference before FK (ready, grasp, verify):
#   waist_gain: {yaw|roll|pitch: g}         waist joint reference times g (0 = neutral)
#   waist_clamp: {yaw|roll|pitch: [lo, hi]} waist joint reference clipped (rad)
#   root_z_min: z                           the reference pelvis no lower than z (m, world)
REFERENCE = ("waist_gain", "waist_clamp", "root_z_min")
WAIST_IDX = {n: MUJOCO_JOINT_NAMES.index(f"waist_{n}_joint") for n in ("yaw", "roll", "pitch")}


def load_delta(path_or_text) -> dict:
    p = Path(str(path_or_text))
    raw = yaml.safe_load(p.read_text()) if p.exists() else yaml.safe_load(str(path_or_text))
    raw = dict(raw or {})
    raw.pop("name", None)
    raw.pop("note", None)
    bad = set(raw) - SPEC_KEYS
    if bad:
        raise ValueError(f"unknown spec keys {sorted(bad)}; allowed {sorted(SPEC_KEYS)}")
    bad = set(raw.get("reference") or {}) - set(REFERENCE)
    if bad:
        raise ValueError(f"unknown reference transforms {sorted(bad)}; allowed {REFERENCE}")
    bad = set(raw.get("extensions") or {}) - set(extensions.HOOKS)
    if bad:
        raise ValueError(f"unknown extensions {sorted(bad)}; registered {sorted(extensions.HOOKS)}")
    return raw


def load_spec(path_or_text) -> dict:
    out = json.loads(json.dumps(BASE_SPEC))
    out.update(load_delta(path_or_text))
    return out


def merge(base: dict, delta: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in delta.items():
        if k in ("set", "reference", "extensions"):
            out[k] = {**out.get(k, {}), **(v or {})}
        else:
            out[k] = v
    return out


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


class FieldLog:
    """Which task.reach fields a round reads: each config dataclass instance gets a subclass that notes reads."""

    def __init__(self):
        self.read = set()

    def watch(self, obj, prefix=""):
        base, log = type(obj), self.read
        names = {f.name for f in dataclasses.fields(obj)}

        def getattribute(self_, name, _base=base):
            if name in names:
                log.add(prefix + name)
            return _base.__getattribute__(self_, name)

        object.__setattr__(obj, "__class__", type(base.__name__, (base,), {"__getattribute__": getattribute}))
        return obj


def reach_config(sets: dict, log: FieldLog) -> ReachConfig:
    reach = json.loads(json.dumps(yaml.safe_load(CONFIG.read_text())["task"]["reach"]))
    for key, value in sets.items():
        node, parts = reach, key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    rc = ReachConfig.from_dict(reach, source=str(CONFIG))
    log.watch(rc)
    log.watch(rc.simple, "simple.")
    log.watch(rc.whole_body, "whole_body.")
    return rc


class Context:
    """Everything a spec decides: the reach config, an IK with its waist limit, ScaleBFM's mode, the hooks."""

    _iks = {}

    def __init__(self, spec, log, xml):
        self.spec = spec
        self.rc = reach_config(spec["set"], log)
        key = float(self.rc.whole_body.waist_pitch_max)
        if key not in Context._iks:
            ik = ArmIK(xml)
            physical = ik.model.jnt_range[ik.jids]
            ik.lo, ik.hi = physical[:, 0].copy(), physical[:, 1].copy()
            ik.hi[14] = min(ik.hi[14], key)
            Context._iks[key] = ik
        self.ik = Context._iks[key]
        self.mode = int(spec["mode"])

    def reference(self, joints, root, **state):
        ref, ext = self.spec["reference"], self.spec["extensions"]
        if not ref and not ext:
            return joints, root
        q, r = np.array(joints, float), np.array(root, float)
        for n, g in (ref.get("waist_gain") or {}).items():
            q[WAIST_IDX[n]] *= float(g)
        for n, (lo, hi) in (ref.get("waist_clamp") or {}).items():
            q[WAIST_IDX[n]] = np.clip(q[WAIST_IDX[n]], lo, hi)
        if ref.get("root_z_min") is not None:
            r[2] = max(r[2], float(ref["root_z_min"]))
        for name, params in ext.items():
            hook = extensions.HOOKS[name].get("reference")
            if hook is not None:
                q, r = hook(q, r, params or {}, **state)
        return q, r

    def futures(self, fut, **state):
        for name, params in self.spec["extensions"].items():
            hook = extensions.HOOKS[name].get("futures")
            if hook is not None:
                fut = hook(fut, params or {}, **state)
        return fut


class Rec(Recorder):
    def __init__(self, sim):
        super().__init__(sim)
        self.rows.update(qpos=[], cmd_joints=[], phase=[])
        self.cur = dict(joints=None, phase="walk")

    def add(self, t, targets):
        super().add(t, targets)
        self.rows["qpos"].append(self.sim.d.qpos.copy())
        j = self.cur["joints"]
        self.rows["cmd_joints"].append(np.full(29, np.nan) if j is None else np.asarray(j, float))
        self.rows["phase"].append(PHASES.index(self.cur["phase"]))


class Round(X.Round):
    """bfm_expert_bottle.Round under a spec: per-phase context (the fork), the reference transforms and hooks wrapped
    around every keypoint FK the expert makes, the extra recording."""

    def __init__(self, pol, sim, z, f, ctxs, fork, body, pub):
        self.ctxs, self.fork_phase = ctxs, fork
        ctx = ctxs["fork"] if fork == "walk" else ctxs["base"]
        a = argparse.Namespace(mode=ctx.mode, tag="", odo=None)
        super().__init__(pol, sim, z, f, ctx.rc, a, body, ctx.ik, pub)
        self.rec = Rec(sim)
        self.st8 = {}
        self.ctx = ctx
        self.phase = "walk"

    def use(self, phase):
        """Switch to the fork's context when its phase starts."""
        self.phase = phase
        self.rec.cur["phase"] = phase
        if self.fork_phase is not None and phase == self.fork_phase and self.ctx is not self.ctxs["fork"]:
            self.ctx = self.ctxs["fork"]
            self.rc, self.simple, self.wb, self.ik = (
                self.ctx.rc,
                self.ctx.rc.simple,
                self.ctx.rc.whole_body,
                self.ctx.ik,
            )
            self.a.mode = self.ctx.mode
        self.pol.mode = torch.tensor([self.ctx.mode], dtype=torch.long, device=self.pol.mode.device)

    def run(self):
        t_wall = time.time()
        ok, why = False, ""
        real = X.futures

        def wrapped(body, pos, quat, joints, velocity, fidx):
            q, root = self.ctx.reference(joints, pos, t=self.t, phase=self.phase, sim=self.sim, st8=self.st8)
            self.rec.cur["joints"] = q
            fut = real(body, root, quat, q, velocity, fidx)
            return self.ctx.futures(fut, t=self.t, phase=self.phase, sim=self.sim, st8=self.st8)

        X.futures = wrapped
        try:
            for phase, step in zip(PHASES, (self.walk, self.ready, self.grasp, self.carry_and_verify)):
                self.use(phase)
                ok, why = step()
                if not ok:
                    break
        finally:
            X.futures = real
        self.events["end"] = self.t
        facts = dict(
            {k: v for k, v in self.f.items() if k not in ("ok", "why", "events", "rise", "program", "controller")},
            controller="scalebfm",
            program="rsi_bottle",
            mode=self.ctx.mode,
            events=self.events,
            stats=self.stats,
            sonic_ok=self.f.get("ok"),
            sonic_why=self.f.get("why"),
            ok=bool(ok and self.fell is None),
            why="" if ok and self.fell is None else (why or "fell"),
            wall_s=round(time.time() - t_wall, 1),
        )
        return self.rec.arrays(), facts


def warm_up(pol, sim, body, calls=3):
    """bfm_round.warm_up: the network's first TorchScript call per process differs; spend it before the first round."""
    sim.reset(np.zeros(2), 0.0, 0.0)
    seat(sim)
    st = sim.state()
    fut = X.futures(
        body, sim.d.qpos[:3].copy(), sim.d.qpos[3:7].copy(), sim.d.qpos[body.qadr].copy(), np.zeros(29), pol.fidx
    )
    for _ in range(calls):
        pol.reset(st)
        pol(st, fut)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("starts", type=Path, help="bottle_states.py files of SONIC's rounds (their start states)")
    ap.add_argument("--spec", required=True, help="candidate spec file or inline YAML")
    ap.add_argument("--fork", default="", help="PHASE:DELTA -- the delta spec (file or inline YAML) from PHASE on")
    ap.add_argument("--only", default="", help="comma-separated start labels")
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th start from the i-th, in label order")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    base = load_spec(a.spec)
    fork_phase, delta = None, None
    if a.fork:
        fork_phase, _, text = a.fork.partition(":")
        if fork_phase not in FORKS:
            raise SystemExit(f"--fork phase must be one of {FORKS}")
        delta = merge(base, load_delta(text))
    meta = json.loads((SCENE / "scene_meta.json").read_text())
    xml = str(SCENE / "scene_locomani.xml")
    log = FieldLog()
    ctxs = dict(base=Context(base, log, xml), fork=Context(delta, log, xml) if delta else None)
    bcfg = ScaleBfmBackendConfig(
        scene_dir=str(SCENE), object_name=X.KEY, grip_kp=HAND_KP, grip_kd=HAND_KD, grip_clip=HAND_FORCE
    )
    pol = R.Policy(device=a.device, backend=bcfg)
    pol.mode = torch.tensor([base["mode"]], dtype=torch.long, device=a.device)
    sim = R.Sim(pol.meta, backend=bcfg, perturb=None)
    body = Body(sim.m, pol.sel)
    warm_up(pol, sim, body)
    pub = PrivilegedPublisher(
        sim.m, sim.d, {X.KEY: meta["objects"][X.KEY]["body"]}, surfaces=[meta["surfaces"]["A"]["body"]],
        port=59000 + os.getpid() % 3000,
    )  # fmt: skip
    only = set(a.only.split(",")) if a.only else None
    i, n = (int(x) for x in a.shard.split("/"))
    starts = [p for p in sorted(a.starts.glob("*.npz")) if not only or p.stem in only][i::n]
    for npz in starts:
        z = dict(np.load(npz))
        f = json.loads(npz.with_suffix(".json").read_text())
        label = npz.stem.replace("sonic", "bfm", 1)
        if (a.out / f"{label}.json").exists():
            continue
        log.read.clear()
        arrays, facts = Round(pol, sim, z, f, ctxs, fork_phase, body, pub).run()
        facts.update(
            label=label,
            spec=base,
            spec_hash=spec_hash(base),
            fork=None if fork_phase is None else dict(phase=fork_phase, spec=delta, spec_hash=spec_hash(delta)),
            fields_read=sorted(log.read),
        )
        save(a.out, label, arrays, facts)
        print(f"{label}: ok {facts['ok']} ({facts['why'] or 'completed'}) {facts['wall_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
