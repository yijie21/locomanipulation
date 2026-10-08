"""Render one fixed set of trace poses in URLab under a named setting, and compare settings: image A/B for Unreal.

render   applies --cvar console commands and --uepy editor scripts, waits --settle-s, renders --rows trace rows
         (evenly spread, the same rows for every tag) from the three dataset cameras in "fresh" mode, and writes
         <out>/<tag>.npz (per camera: rows x H x W x 3) and <out>/<tag>.json (settings, per-request ms).
         --repeats sets urlab.FreshCaptureRepeats for this render and restores the previous value afterwards.
compare  per camera: mean abs difference (0-255), PSNR and sharpness (mean |Laplacian|) for each pair of tags;
         with --ref, every tag against the reference. Writes <out>/compare_<name>.json.

Tags from one editor session are comparable. Render the same setting twice (a, b) for the noise floor.

    . /workspace/ue_57/ue_env.sh
    "$BRIDGE_PY" scripts/experiments/unreal_pose_set_20260928.py render --trace <npz> --trace-joints <json> \
        --out <dir> --tag base_a
    "$BRIDGE_PY" scripts/experiments/unreal_pose_set_20260928.py render ... --tag off_a --cvar 'ShowFlag.Rendering 0'
    "$BRIDGE_PY" scripts/experiments/unreal_pose_set_20260928.py compare --out <dir> --tags base_a base_b off_a
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unreal_smoke_20260928 import CAMERAS, apply_row, connect, push, qpos_map, uecmd  # noqa: E402


def cvar_int(name: str) -> int:
    return int(uecmd(f"print(unreal.SystemLibrary.get_console_variable_int_value({name!r}))").strip())


def console(cmd: str) -> None:
    uecmd(f"unreal.SystemLibrary.execute_console_command(None, {cmd!r})")


def cmd_render(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    for c in args.cvar:
        console(c)
    for f in args.uepy:
        print(uecmd(Path(f).read_text()).strip(), flush=True)
    previous = None
    if args.repeats:
        previous = cvar_int("urlab.FreshCaptureRepeats")
        console(f"urlab.FreshCaptureRepeats {args.repeats}")
    trace = np.load(args.trace)
    client = connect(args.host, local_model=False)
    if not client.manager_present:
        res = client.sim.start(raise_on_failure=False, timeout_s=300)
        print(f"[pose-set] PIE {res.state} {res.compile_error[:200]!r}", flush=True)
    client.local_model = True
    client.connect()
    client.runtime.set_mode("puppet")
    names = client.camera_names()
    ue = {c: next(n for n in names if n.split("/")[-1].endswith(c)) for c in CAMERAS}
    mapping = qpos_map(json.loads(Path(args.trace_joints).read_text()), client.model)
    n = len(trace["qpos"])
    rows = [int(r) for r in np.linspace(n * 0.05, n * 0.95, args.rows)]
    apply_row(client, mapping, trace["qpos"][rows[0]])
    push(client)
    push(client, {nm: "fresh" for nm in names})  # every camera once: all switch to on-demand
    time.sleep(args.settle_s)

    imgs = {c: [] for c in CAMERAS}
    ms = []
    for r in rows:
        t0 = time.perf_counter()
        apply_row(client, mapping, trace["qpos"][r])
        push(client)
        reply = push(client, {ue[c]: "fresh" for c in CAMERAS})
        ms.append((time.perf_counter() - t0) * 1e3)
        for c in CAMERAS:
            blk = reply["cameras"][ue[c]]
            px = np.frombuffer(blk["data"], np.uint8).reshape(int(blk["height"]), int(blk["width"]), 4)
            imgs[c].append(np.ascontiguousarray(px[..., [2, 1, 0]]))
    client.close()
    if previous is not None:
        console(f"urlab.FreshCaptureRepeats {previous}")
    np.savez_compressed(args.out / f"{args.tag}.npz", **{c: np.stack(v) for c, v in imgs.items()})
    meta = dict(
        tag=args.tag,
        rows=rows,
        cvars=args.cvar,
        uepy=args.uepy,
        repeats=args.repeats or cvar_int("urlab.FreshCaptureRepeats"),
        request_ms_p50=float(np.median(ms)),
        request_ms=[round(v, 1) for v in ms],
    )
    (args.out / f"{args.tag}.json").write_text(json.dumps(meta, indent=1))
    print(f"[pose-set] {args.tag}: {len(rows)} rows, request p50 {np.median(ms):.0f} ms -> {args.out}", flush=True)


def sharpness(img: np.ndarray) -> float:
    g = img.astype(np.float32).mean(-1)
    lap = 4 * g[..., 1:-1, 1:-1] - g[..., :-2, 1:-1] - g[..., 2:, 1:-1] - g[..., 1:-1, :-2] - g[..., 1:-1, 2:]
    return float(np.abs(lap).mean())


def pair_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = a.astype(np.float32) - b.astype(np.float32)
    mse = float((d**2).mean())
    return {
        "mean_abs": float(np.abs(d).mean()),
        "worst_row_mean_abs": float(np.abs(d).mean(axis=(1, 2, 3)).max()),
        "psnr_db": float(10 * np.log10(255.0**2 / max(mse, 1e-9))),
        "signed_mean": float(d.mean()),
    }


def cmd_compare(args) -> None:
    tags = list(dict.fromkeys(args.tags + ([args.ref] if args.ref else [])))
    data = {t: np.load(args.out / f"{t}.npz") for t in tags}
    rows = {t: json.loads((args.out / f"{t}.json").read_text())["rows"] for t in tags}
    if len({tuple(r) for r in rows.values()}) != 1:
        raise SystemExit(f"[compare] tags rendered different rows: {rows}")
    pairs = (
        [(t, args.ref) for t in args.tags if t != args.ref] if args.ref else list(itertools.combinations(args.tags, 2))
    )
    report = {"sharpness": {}, "pairs": {}}
    for c in CAMERAS:
        report["sharpness"][c] = {t: sharpness(data[t][c]) for t in tags}
    for a, b in pairs:
        report["pairs"][f"{a}|{b}"] = {c: pair_stats(data[a][c], data[b][c]) for c in CAMERAS}
        line = "  ".join(
            f"{c.split('_')[0][:5]} {s['mean_abs']:.2f}/{s['psnr_db']:.1f}dB/{s['signed_mean']:+.2f}"
            for c, s in report["pairs"][f"{a}|{b}"].items()
        )
        print(f"[compare] {a:>14s} vs {b:<14s} {line}", flush=True)
    for c in CAMERAS:
        print(f"[compare] sharpness {c:12s} " + " ".join(f"{t} {v:.2f}" for t, v in report["sharpness"][c].items()))
    (args.out / f"compare_{args.name}.json").write_text(json.dumps(report, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--trace", required=True)
    r.add_argument("--trace-joints", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--rows", type=int, default=12)
    r.add_argument("--cvar", nargs="*", default=[], help="console commands applied first")
    r.add_argument("--uepy", nargs="*", default=[], help="editor Python files run first (uepy protocol)")
    r.add_argument("--repeats", type=int, help="urlab.FreshCaptureRepeats for this render only")
    r.add_argument("--settle-s", type=float, default=5.0)
    r.add_argument("--host", default="tcp://127.0.0.1")
    c = sub.add_parser("compare")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--tags", nargs="+", required=True)
    c.add_argument("--ref", help="compare every tag against this one instead of all pairs")
    c.add_argument("--name", default="all")
    args = ap.parse_args()
    {"render": cmd_render, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
