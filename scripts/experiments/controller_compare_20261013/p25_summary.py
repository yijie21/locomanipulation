"""TODO 25.5 headline table: per perturbation cell, both controllers, from the round files directly.

* production cells (b20 b40 c d04 f e050 e150): the expert's own verdict (SONIC: its production expert under the
  perturbation, seeds 40000.. / 7000..; ScaleBFM: the 25.7 port on the same L1 seeds), falls, and the task's end check
  (bottle >= 3 cm above where it stood; box >= 5 cm up and <= 15 deg tilted);
* carries (nominal c g e050 e150): bottle back 2 m, box turn 180 deg + 1 m; held at the end (object >= 5 cm above its
  start) out of the rounds that lifted it;
* brake (b20 b40 d04, plan program with the speed feedforward / SONIC guard): crossings of the line and stop error;
* clips (C2 under f, C4 under d04, S0 under g): clip_measure's success per controller (ScaleBFM: best mode);
* the estimated-pose arm (c noise on): ScaleBFM's brake and expert walk-ins with the leg odometry against the truth.

    python3 scripts/experiments/controller_compare_20261013/p25_summary.py   # -> outputs/controller_compare_20261013/p25/p25.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

O = Path(__file__).resolve().parents[3] / "outputs" / "controller_compare_20261013"
P = O / "p25"
CELLS = ("nominal", "b20", "b40", "c", "d04", "f", "e050", "e150")
CARRY = ("nominal", "c", "g", "e050", "e150")


def rounds(d: Path):
    for j in sorted(d.glob("*.json")):
        npz = j.with_suffix(".npz")
        if npz.exists():
            yield json.loads(j.read_text()), dict(np.load(npz))


def end_check(kind, f, z):
    obj = z["bottle" if kind == "bottle" else "box"]
    if kind == "bottle":
        return float(obj[-1, 2] - obj[0, 2]) >= 0.03  # still lifted (l2_measure's rule)
    x, y = obj[-1, 4], obj[-1, 5]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
    return float(obj[-1, 2] - obj[0, 2]) >= 0.05 and tilt <= 15.0


def cell(kind, d: Path):
    rs = list(rounds(d))
    if not rs:
        return None
    ok = sum(bool(f.get("ok")) for f, _ in rs)
    fell = sum(float(np.min(z["pelvis"][:, 2])) < 0.40 for _, z in rs)
    end = sum(end_check(kind, f, z) for f, z in rs)
    return f"{ok}/{len(rs)}（末态 {end}，摔 {fell}）"


def manifest_cell(run: Path):
    """A SONIC batch with no state traces (every round ended at the stand): its manifest's verdicts."""
    rows = [json.loads(line) for m in sorted(run.glob("manifest_*.jsonl")) for line in m.open() if line.strip()]
    if not rows:
        return None
    whys = sorted({str(r.get("why"))[:30] for r in rows if not r.get("ok")})
    return f"{sum(bool(r.get('ok')) for r in rows)}/{len(rows)}（{'; '.join(whys)}）"


def carry(kind, d: Path, program: str | None = None):
    rs = [(f, z) for f, z in rounds(d) if program is None or str(f.get("program", "")).startswith(program)]
    if not rs:
        return None
    obj = "bottle" if kind == "bottle" else "box"
    lifted = [(f, z) for f, z in rs if float(np.max(z[obj][:, 2] - z[obj][0, 2])) >= 0.05]
    held = sum(float(z[obj][-1, 2] - z[obj][0, 2]) >= 0.05 for _, z in lifted)
    fell = sum(float(np.min(z["pelvis"][:, 2])) < 0.40 for _, z in rs)
    return f"{held}/{len(lifted)} 抬起局拿住（{len(rs)} 局，摔 {fell}）"


def brake(d: Path):
    f = d / "results.json"
    if not f.exists():
        return None
    rows = [r for r in json.loads(f.read_text())["rows"] if r.get("fell") is None and "frontmost_max_past_marker" in r]
    fell = sum(1 for r in json.loads(f.read_text())["rows"] if r.get("fell"))
    if not rows:
        return f"摔 {fell}"
    crossed = sum(r["frontmost_max_past_marker"] > 0 for r in rows)
    worst = max(r["frontmost_max_past_marker"] for r in rows)
    out = f"越线 {crossed}/{len(rows)}（最远 {100 * worst:+.1f} cm），摔 {fell}"
    if all("target_err" in r and r["target_err"] is not None for r in rows):
        e = np.array([100 * r["target_err"] for r in rows])
        out += f"，停位误差 {e.mean():+.1f} ± {e.std():.1f} cm"
    return out


def clips(d: Path):
    f = d / "complex.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text())["choice"]


def main() -> int:
    lines = [
        "# 25.5 扰动与估计位姿（TODO 25.5）",
        "",
        "生成：`p25_summary.py`。格式：成功 / 局（末态检查通过数，摔倒数）。",
        "",
    ]
    for kind in ("bottle", "warehouse"):
        lines += [f"## {kind} 生产程序", "", "| 扰动 | SONIC | ScaleBFM（25.7 移植专家） |", "|---|---|---|"]
        for c in CELLS:
            sd = O / kind / "sonic" / "states" if c == "nominal" else P / kind / f"sonic_{c}" / "states"
            s = cell(kind, sd) if sd.exists() else None
            if s is None:
                s = manifest_cell(O.parent / "runs" / f"controller_compare_p25_{kind}_{c}_20261013")
            b = cell(kind, P / kind / f"bfm_{c}")
            lines.append(f"| {c} | {s or '–'} | {b or '–'} |")
        lines += [
            "",
            f"## {kind} 带载走路（{'后退 2 m' if kind == 'bottle' else '转 180° + 1 m'}）",
            "",
            "| 扰动 | SONIC | ScaleBFM |",
            "|---|---|---|",
        ]
        for c in CARRY:
            sd, prog = P / kind / f"sonic_carry_{c}" / "states", None
            if c == "nominal":  # 25.2 / 25.3's plan rounds: the same plan's program only
                sd = O / kind / ("sonic_carry/states" if kind == "bottle" else "sonic_plans/states")
                prog = "carry_back2" if kind == "bottle" else "carry_turn180"
            lines.append(
                f"| {c} | {carry(kind, sd, prog) if sd.exists() else '–'} | {carry(kind, P / kind / f'bfm_carry_{c}') or '–'} |"
            )
        lines.append("")
    lines += [
        "## 刹车（0.55 / 0.35 / 0.25 m/s 各 10 局）",
        "",
        "| 扰动 | SONIC guard | ScaleBFM plan（按速度前馈） |",
        "|---|---|---|",
    ]
    lines.append(
        f"| 无 | {brake(O.parent / 'wbc_compare_20261012' / 'brake' / 'sonic_guard') or '–'} | {brake(O.parent / 'wbc_compare_20261012' / 'brake_ff' / 'bfm_plan_ffspeed') or '–'} |"
    )
    for c in ("b20", "b40", "d04"):
        lines.append(f"| {c} | {brake(P / 'brake' / f'sonic_{c}') or '–'} | {brake(P / 'brake' / f'bfm_{c}') or '–'} |")
    lines += ["", "## 片段", "", "| 扰动 | 片段 | SONIC | ScaleBFM 最好 mode |", "|---|---|---|---|"]
    for c in ("f", "d04", "g"):
        for clip, ch in clips(P / "clips" / c).items():
            lines.append(
                f"| {c} | {clip} | {ch['sonic']}（G {ch['sonic_gmpkpe_cm']} cm，摔 {ch['sonic_falls']}） | mode {ch['best_mode']} {ch['best']}（G {ch['best_gmpkpe_cm']}，摔 {ch['best_falls']}） |"
            )
    lines += ["", "## 估计位姿臂（ScaleBFM，c 噪声开）", ""]
    lines.append(
        f"- 刹车，里程计：{brake(P / 'odometry' / 'brake_odo_c') or '–'}；真值：{brake(P / 'odometry' / 'brake_truth_c') or '–'}"
    )
    for kind in ("bottle", "warehouse"):
        lines.append(
            f"- {kind} 移植专家，里程计：{cell(kind, P / 'odometry' / f'{kind}_odo_c') or '–'}；真值（c）：{cell(kind, P / kind / 'bfm_c') or '–'}"
        )
    (P / "p25.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
