"""Round-based x-boundary search over the (y, h, d, yaw) task grid.

    python -m locomani.tasks.crate.reach_envelope.boundary_search plan   <run_dir> <n>
    python -m locomani.tasks.crate.reach_envelope.boundary_search status <run_dir>
    python -m locomani.tasks.crate.reach_envelope.boundary_search table  <run_dir>

<run_dir>/search.json holds the grid, the envelope, the scene and the shard list (see DEFAULT_SEARCH).
plan n : write round_<n>/bank.json and one ghost-reach request per shard from the previous rounds'
         results; round 0 starts 10 cm inside the offline IK bound. Each line advances +5 cm on a pass
         and -5 cm on a fail until its boundary is bracketed. status/table summarise every line.
"""

import json
import sys
from pathlib import Path

import numpy as np

from locomani import paths
from locomani.tasks.crate.joint_reach_sampling import generate_joint_bank

STEP, X_MIN, X_MAX = 0.05, 0.20, 0.80
POS_TOL = 0.02  # bracket on best hand position error; orientation is recorded, not gated
DEFAULT_SEARCH = {
    "grid": {
        "y": [-0.12, 0.0, 0.12],
        "h": [0.55, 0.70, 0.85],
        "d": [0.32, 0.44, 0.52],
        "yaw": [-30.0, -15.0, 0.0, 15.0, 30.0],
    },
    "bodyward_tilt_deg": 30.0,
    "envelope": {
        "base_anchor": "measured",
        "com_shift_max": 0.11,
        "com_free_radius": 0.10,
        "pelvis_pitch_max": 0.85,
        "pelvis_height_min": 0.55,
        "knee_max": 1.5,
        "pelvis_forward_max": 0.35,
        "tracking_cap": 0.15,
        "tracking_gain": 1.0,
    },
    # Lines whose h is below `h_below` get these envelope fields on top of `envelope`. Measured
    # 2026-09-21: without the crouch h=0.30 is unreachable (pelvis stays at 0.73 m, shoulders hit
    # the torso); pulling the solver's pelvis to 0.65 m delivers a 0.57 m pelvis and 0.48-0.53 m
    # of reach at h=0.30, +10 cm at h=0.40. Not measured at h>=0.55, so it is not the default there.
    "envelope_overrides": [
        {"h_below": 0.50, "envelope": {"base_anchor": "measured_xy", "pelvis_height_target": 0.65}},
    ],
    "criteria": {
        "target_timeout_sim_s": 12.0,
        "success_hold_sim_s": 0.5,
        "position_tolerance_m": 0.02,
        "orientation_tolerance_deg": 15.0,
        "foot_horizontal_drift_m": 0.03,
        "foot_vertical_lift_m": 0.03,
        "wall_watchdog_s": 15.0,
    },
    "scene_dir": "outputs/scene/g1_floor",
    "shards": [{"gpu": "1", "instance": 0}, {"gpu": "0", "instance": 1}],
    "seed": 20260921,
}
_state = {}


def setup(run_dir: Path):
    run_dir = Path(run_dir).resolve()
    cfg_path = run_dir / "search.json"
    if not cfg_path.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(DEFAULT_SEARCH, indent=1))
    cfg = json.loads(cfg_path.read_text())
    scene = Path(cfg["scene_dir"])
    if not scene.is_absolute():
        scene = Path(paths.REPO) / scene
    _state.update(
        SEARCH=run_dir,
        GRID={k: tuple(v) for k, v in cfg["grid"].items()},
        ENVELOPE=cfg["envelope"],
        OVERRIDES=cfg.get("envelope_overrides", []),
        CRITERIA=cfg["criteria"],
        TILT=cfg["bodyward_tilt_deg"],
        SCENE=scene.resolve(),
        SHARDS=cfg["shards"],
        SEED=cfg["seed"],
    )
    return cfg


def envelope_for(line):
    """The envelope a line runs under: the base one plus the first override whose h_below exceeds its h."""
    env = dict(_state["ENVELOPE"])
    for rule in _state["OVERRIDES"]:
        if line["h"] < rule["h_below"]:
            env.update(rule["envelope"])
            break
    return env


def group_of(line):
    return next(
        (i for i, rule in enumerate(_state["OVERRIDES"]) if line["h"] < rule["h_below"]), len(_state["OVERRIDES"])
    )


def lines():
    G = _state["GRID"]
    return [dict(y=y, h=h, d=d, yaw=yaw) for y in G["y"] for h in G["h"] for d in G["d"] for yaw in G["yaw"]]


def key(l):
    return (round(l["y"], 3), round(l["h"], 3), round(l["d"], 3), round(l["yaw"], 1))


def ik_upper_bound():
    """Largest x per line the offline whole-body IK admits with the sweep envelope (an over-estimate)."""
    SEARCH = _state["SEARCH"]
    cache = SEARCH / "ik_upper_bound.json"
    if cache.exists():
        return {tuple(json.loads(k)): v for k, v in json.load(open(cache)).items()}
    from scipy.spatial.transform import Rotation as R

    from locomani.primitives.arm_ik import ArmIK
    from locomani.primitives.whole_body_ik import WholeBodyIK
    from locomani.tasks.crate.carry_config import CrateCarryConfig

    ik = ArmIK(str(_state["SCENE"] / "scene_locomani.xml"))
    # Nominal standing pose for the offline bound (hips -0.1, knees 0.2, elbows 0.8); the live
    # trial starts from the expert's own prepared stance, so this is only a starting-x heuristic.
    q = np.zeros(29)
    q[[0, 6]] = -0.1
    q[[3, 9]] = 0.2
    q[[4, 10]] = -0.1
    q[15:22] = [-0.35, 0.2, 0, 0.8, 0, 0, 0]
    q[22:29] = [-0.35, -0.2, 0, 0.8, 0, 0, 0]
    state = dict(joints=q, pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]))
    out = {}
    for l in lines():
        env = {k: v for k, v in envelope_for(l).items() if k not in ("tracking_cap", "tracking_gain", "base_anchor")}
        cfg = CrateCarryConfig(reach_control="whole_body", waist_dofs="all", waist_pitch_max=0.52, **env)
        A = (
            R.from_euler("z", l["yaw"], degrees=True).as_matrix()
            @ R.from_euler("y", 90 - _state["TILT"], degrees=True).as_matrix()
        )
        half = 0.5 * l["d"] * np.array([-np.sin(np.radians(l["yaw"])), np.cos(np.radians(l["yaw"])), 0])
        best = None
        body = WholeBodyIK(
            ik,
            state,
            cfg,
            tool_offset=[0.1515, 0, 0],
            com_shift_max=cfg.com_shift_max,
            com_free_radius=cfg.com_free_radius,
            knee_max=cfg.knee_max,
            pelvis_forward_max=cfg.pelvis_forward_max,
            pelvis_height_target=cfg.pelvis_height_target,
        )
        seed = body.seed.copy()
        for x in np.arange(X_MAX, X_MIN - 1e-9, -STEP):  # downward: the first admitted x is the (over-estimated) bound
            pose = body.solve(
                np.array([[x, l["y"], l["h"]] + half, [x, l["y"], l["h"]] - half]), [A, A], seed=seed, iterations=150
            )
            if pose.reachable:
                best = float(round(x, 3))
                break
        out[key(l)] = best
    json.dump({json.dumps(list(k)): v for k, v in out.items()}, open(cache, "w"), indent=1)
    return out


def observations():
    """{line key: {x: record}} from every finished round."""
    obs = {}
    for rd in sorted(_state["SEARCH"].glob("round_*")):
        for shard in list(rd.glob("shard_*")) + list(rd.glob("group_*/shard_*")):
            for cf in (shard / "result.cases").glob("*.json") if (shard / "result.cases").is_dir() else []:
                c = json.load(open(cf))["case"]
                src = c.get("target", {}).get("source_centres_robot_m")
                if src is None:
                    continue
                src = np.asarray(src)
                line = src[0] - src[1]
                k = (
                    round(float(src[:, 1].mean()), 3),
                    round(float(src[:, 2].mean()), 3),
                    round(float(np.linalg.norm(line)), 3),
                    round(float(np.degrees(np.arctan2(-line[0], line[1]))), 1),
                )
                x = round(float(src[:, 0].mean()), 3)
                tr = [t for t in c["traces"] if t["controller_phase"] == "reach"]
                best = min((max(t["position_error_m"]) for t in tr), default=None)
                rec = dict(
                    x=x,
                    status=c["status"],
                    reason=c["reason"],
                    best_pos=best,
                    best_orient=min((max(t["orientation_error_deg"]) for t in tr), default=None),
                    drift=max((max(t["foot_horizontal_drift_m"]) for t in c["traces"]), default=None),
                    case=str(cf.relative_to(_state["SEARCH"])),
                    round=rd.name,
                )
                obs.setdefault(k, {})[x] = rec
    return obs


def verdict(rec):
    if rec["reason"].startswith("controller_unsupported"):
        return "reject"
    if rec["reason"] in ("quiet_ready_state_lost_before_task_recording", "runtime_error"):
        return "retry"
    if rec["best_pos"] is None:
        return "retry"
    return "pass" if rec["best_pos"] <= POS_TOL else "fail"


def next_x(k, upper, obs):
    """One x to try for this line, or None when bracketed."""
    recs = obs.get(k, {})
    if not recs:
        return None if upper is None else max(X_MIN, round(upper - 0.10, 3))
    v = {x: verdict(r) for x, r in recs.items()}
    retry = [x for x, s in v.items() if s == "retry"]
    if retry:
        return retry[0]
    passes = [x for x, s in v.items() if s == "pass"]
    fails = [x for x, s in v.items() if s in ("fail", "reject")]
    if passes and fails and min(fails) > max(passes):
        return None
    if passes and not fails:
        x = round(max(passes) + STEP, 3)
        return x if x <= X_MAX and (upper is None or x <= upper + 1e-9) else None
    if fails and not passes:
        x = round(min(fails) - STEP, 3)
        return x if x >= X_MIN else None
    return None  # inconsistent (pass above a fail): leave for manual review


def plan(n):
    upper, obs = ik_upper_bound(), observations()
    rd = _state["SEARCH"] / f"round_{n}"
    rd.mkdir(parents=True, exist_ok=True)
    groups = {}
    for l in lines():
        x = next_x(key(l), upper.get(key(l)), obs)
        if x is not None:
            groups.setdefault(group_of(l), []).append([x, l["y"], l["h"], l["d"], l["yaw"]])
    if not groups:
        print("nothing left to sample")
        return
    total = 0
    for g, anchors in sorted(groups.items()):
        gd = rd / f"group_{g}"
        gd.mkdir(exist_ok=True)
        _write_group(n, g, gd, anchors, envelope_for({"h": anchors[0][2]}))
        total += len(anchors)
    print(f"round {n}: {total} trials, {len(groups)} envelope groups x {len(_state['SHARDS'])} shards")


def _write_group(n, g, gd, anchors, envelope):
    G = _state["GRID"]
    sampling = {
        "schema_version": 1,
        "method": "grid",
        "seed": _state["SEED"] + n,
        "count": 0,
        "bodyward_tilt_deg": _state["TILT"],
        "bounds": {  # the bank validator needs bounds; they only need to contain the grid
            "x_m": [X_MIN, X_MAX],
            "y_m": [min(G["y"]) - 0.01, max(G["y"]) + 0.01],
            "height_m": [min(G["h"]) - 0.01, max(G["h"]) + 0.01],
            "separation_m": [min(G["d"]) - 0.01, max(G["d"]) + 0.01],
            "yaw_deg": [min(G["yaw"]) - 1, max(G["yaw"]) + 1],
        },
        "axis_values": {
            "x_m": [anchors[0][0]],
            "y_m": [anchors[0][1]],
            "height_m": [anchors[0][2]],
            "separation_m": [anchors[0][3]],
            "yaw_deg": [anchors[0][4]],
        },
        "anchors": anchors[1:],
    }
    json.dump(sampling, open(gd / "sampling.json", "w"), indent=1)
    with open(gd / "bank.json", "w") as stream:  # what scripts/generate_joint_reach_bank.py writes for a grid
        json.dump(generate_joint_bank(sampling), stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    for shard, spec in enumerate(_state["SHARDS"]):
        d = gd / f"shard_{shard}"
        d.mkdir(exist_ok=True)
        req = {
            "schema_version": 1,
            "source_targets": str(gd / "bank.json"),
            "request_identity": f"{_state['SEARCH'].name}_r{n}_g{g}_s{shard}",
            "seed": _state["SEED"],
            "runtime": {
                "backend": {"no_encoder": False},
                "gpu": str(spec["gpu"]),
                "instance": int(spec["instance"]),
                "scene_dir": str(_state["SCENE"]),
                "video_label_prefix": f"bnd_r{n}_g{g}_s{shard}",
            },
            "experiment": _state["CRITERIA"],
            "envelope": envelope,
        }
        json.dump(req, open(d / "request.json", "w"), indent=1)


def status():
    upper, obs = ik_upper_bound(), observations()
    done = open_ = 0
    for l in lines():
        k = key(l)
        recs = obs.get(k, {})
        nx = next_x(k, upper.get(k), obs)
        done += nx is None
        open_ += nx is not None
        cells = "  ".join(
            f"{x:.2f}:{verdict(r)[0]}" + ("" if r["best_pos"] is None else f"{r['best_pos'] * 100:.1f}")
            for x, r in sorted(recs.items())
        )
        print(
            f"y={l['y']:+.2f} h={l['h']:.2f} d={l['d']:.2f} yaw={l['yaw']:+3.0f}  ik≤{upper.get(k)}  {cells}  {'' if nx is None else f'-> {nx:.2f}'}"
        )
    print(f"bracketed {done} / open {open_} / trials {sum(len(v) for v in obs.values())}")


def table():
    """boundary_table.json: per line the bracket [x_pass, x_fail] and every sample's continuous measurements."""
    upper, obs = ik_upper_bound(), observations()
    rows = []
    for l in lines():
        k = key(l)
        recs = obs.get(k, {})
        v = {x: verdict(r) for x, r in recs.items()}
        passes = [x for x, s in v.items() if s == "pass"]
        fails = [x for x, s in v.items() if s in ("fail", "reject")]
        rows.append(
            dict(
                **l,
                ik_upper=upper.get(k),
                x_pass=max(passes) if passes else None,
                x_fail=min(fails) if fails else None,
                trials=len(recs),
                samples=[
                    dict(
                        x=x,
                        pos=r["best_pos"],
                        orient=r["best_orient"],
                        drift=r["drift"],
                        reason=r["reason"],
                        case=r["case"],
                    )
                    for x, r in sorted(recs.items())
                ],
            )
        )
    out = _state["SEARCH"] / "boundary_table.json"
    json.dump(rows, open(out, "w"), indent=1)
    print(out, f"{sum(r['x_pass'] is not None and r['x_fail'] is not None for r in rows)}/{len(rows)} bracketed")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[0] not in ("plan", "status", "table"):
        print(__doc__)
        return 2
    setup(Path(argv[1]))
    if argv[0] == "plan":
        plan(int(argv[2]))
    elif argv[0] == "status":
        status()
    else:
        table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
