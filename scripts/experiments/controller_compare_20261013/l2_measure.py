"""L2 judgement (TODO 25.8 / 25.9, plan §2 / §6): SONIC's production expert against the same program on ScaleBFM, paired by
seed, one rule for both.

Per round (bottle_states.py / warehouse_states.py files of SONIC's run; bfm_expert_*.py files of ScaleBFM's, paired
by label):
* success -- the expert's own verdict (`ok`: SONIC's result, ScaleBFM's port running the same checks);
* measured -- the same end state checked from the trace for both: bottle >= 3 cm above where it stood at the end
  (round_phases_approach's own `z > table top + 3 cm` passes a standing bottle, whose origin is 8 cm up), box >= 5 cm
  up and <= 15 deg tilted at the end (the expert's max_tilt);
* fell -- pelvis below 0.40 m; dropped -- the object was lifted (>= 3 cm) and is not at the end;
* ego -- p95 of the torso link's angular speed (the head camera rides it), rad/s, over the round.
Main criterion (§2): ScaleBFM's point estimate >= SONIC's in both scenes and an exact McNemar p < 0.05 in at least
one; guardrails per scene: falls <= SONIC + 1, drops <= SONIC + 1, ego p95 <= SONIC x 1.2; veto (combo tier): ScaleBFM
more than 10 points below SONIC in a scene. SONIC rounds with RTF < 0.85 are void (§6) and listed.

    python3 scripts/experiments/controller_compare_20261013/l2_measure.py --out outputs/controller_compare_20261013/l2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402

L2 = REPO / "outputs" / "controller_compare_20261013" / "l2"
SCENES = {
    "bottle": ("sim_lowcom", "bottle"),
    "warehouse": ("warehouse_box_45x32x33_mars", "box"),
}
RTF_VOID = 0.85


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p on the discordant pairs (b: only A, c: only B)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


class Torso:
    def __init__(self, scene):
        self.m = mujoco.MjModel.from_xml_path(str(REPO / "outputs" / "scene" / scene / "scene_locomani.xml"))
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in MUJOCO_JOINT_NAMES])
        self.b = self.m.body("torso_link").id

    def speed_p95(self, z) -> float:
        R = []
        for pel, q in zip(z["pelvis"], z["joints"]):
            self.d.qpos[:7], self.d.qpos[self.qadr] = pel, q
            mujoco.mj_kinematics(self.m, self.d)
            R.append(self.d.xmat[self.b].reshape(3, 3).copy())
        t = np.asarray(z["t"], float)
        w = []
        for i in range(1, len(R)):
            dt = t[i] - t[i - 1]
            if dt <= 0:
                continue
            c = np.clip((np.trace(R[i] @ R[i - 1].T) - 1) / 2, -1, 1)
            w.append(np.arccos(c) / dt)
        return float(np.percentile(w, 95)) if w else float("nan")


def round_facts(npz: Path, kind: str, torso: Torso) -> dict:
    z = dict(np.load(npz))
    f = json.loads(npz.with_suffix(".json").read_text())
    obj = z[kind]
    rise = obj[:, 2] - obj[0, 2]
    out = dict(label=npz.stem, ok=bool(f.get("ok")), why=f.get("why") or f.get("expert_reason") or "", rtf=f.get("rtf"))
    # a SONIC bottle round that ended before planning its walk (stand-up / spawn-view failure) has no edge
    out["pre_walk"] = bool(kind == "bottle" and (f.get("edge") or {}).get("line") is None)
    out["fell"] = bool(np.min(z["pelvis"][:, 2]) < 0.40)
    lifted_any = bool(np.max(rise) >= 0.03)
    if kind == "bottle":
        # still lifted at the end (the production check `z > table top + 3 cm` passes a bottle standing on the table:
        # its origin is 8 cm above its base)
        end_ok = float(rise[-1]) >= 0.03
    else:
        w, x, y, zq = obj[-1, 3:7]
        tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))
        end_ok = float(rise[-1]) >= 0.05 and tilt <= 15.0
        out["tilt_end_deg"] = round(tilt, 1)
    out["measured"] = bool(end_ok and not out["fell"])
    out["dropped"] = bool(lifted_any and not end_ok)
    out["rise_end_cm"] = round(100 * float(rise[-1]), 1)
    out["ego_p95"] = round(torso.speed_p95(z), 3)
    return out


def compare(sonic_dir: Path, bfm_dir: Path, kind: str, torso: Torso) -> dict:
    sonic = {p.stem.replace("sonic", "", 1): round_facts(p, kind, torso) for p in sorted(sonic_dir.glob("sonic*.npz"))}
    bfm = {p.stem.replace("bfm", "", 1): round_facts(p, kind, torso) for p in sorted(bfm_dir.glob("bfm*.npz"))}
    void = [k for k, r in sonic.items() if r.get("rtf") is not None and r["rtf"] < RTF_VOID]
    keys = sorted(k for k in sonic if k in bfm and k not in void)
    res = dict(n=len(keys), void=void)
    for key in ("ok", "measured"):
        a = [sonic[k][key] for k in keys]
        b = [bfm[k][key] for k in keys]
        only_s = sum(1 for x, y in zip(a, b) if x and not y)
        only_b = sum(1 for x, y in zip(a, b) if y and not x)
        res[key] = dict(
            sonic=sum(a), scalebfm=sum(b), both=sum(1 for x, y in zip(a, b) if x and y), only_sonic=only_s,
            only_scalebfm=only_b, p=round(mcnemar_exact(only_s, only_b), 4),
        )  # fmt: skip
    for key in ("fell", "dropped"):
        res[key] = dict(sonic=sum(sonic[k][key] for k in keys), scalebfm=sum(bfm[k][key] for k in keys))
    res["ego_p95"] = dict(
        sonic=round(float(np.nanmedian([sonic[k]["ego_p95"] for k in keys])), 3) if keys else None,
        scalebfm=round(float(np.nanmedian([bfm[k]["ego_p95"] for k in keys])), 3) if keys else None,
    )
    res["why"] = dict(
        sonic=sorted({sonic[k]["why"][:60] for k in keys if not sonic[k]["ok"]}),
        scalebfm=sorted({bfm[k]["why"][:60] for k in keys if not bfm[k]["ok"]}),
    )
    res["rounds"] = {k: dict(sonic=sonic[k], scalebfm=bfm[k]) for k in keys}
    rtf = [sonic[k]["rtf"] for k in sonic if sonic[k].get("rtf") is not None]
    res["rtf"] = [round(min(rtf), 3), round(float(np.median(rtf)), 3), round(max(rtf), 3)] if rtf else None
    res["missing_scalebfm"] = sorted(k for k in sonic if k not in bfm and k not in void)
    pre = [k for k in keys if sonic[k]["pre_walk"]]
    res["sonic_pre_walk"] = pre
    if pre:  # the same comparison without the rounds SONIC lost before its walk plan
        a = [sonic[k]["ok"] for k in keys if k not in pre]
        b = [bfm[k]["ok"] for k in keys if k not in pre]
        o_s = sum(1 for x, y in zip(a, b) if x and not y)
        o_b = sum(1 for x, y in zip(a, b) if y and not x)
        res["ok_without_pre_walk"] = dict(n=len(a), sonic=sum(a), scalebfm=sum(b), p=round(mcnemar_exact(o_s, o_b), 4))
    res["why_counts"] = dict(
        sonic=Counter(sonic[k]["why"][:48] for k in keys if not sonic[k]["ok"]).most_common(),
        scalebfm=Counter(bfm[k]["why"][:48] for k in keys if not bfm[k]["ok"]).most_common(),
    )
    return res


def judge(report: dict) -> list[str]:
    """Plan §2 against the numbers: main criterion, guardrails, the combo-tier veto."""
    out = []
    nominal = {sc: report.get(f"{sc}_nominal") for sc in SCENES}
    if all(nominal.values()):
        ge = all(r["ok"]["scalebfm"] >= r["ok"]["sonic"] for r in nominal.values())
        sig = any(r["ok"]["p"] < 0.05 and r["ok"]["scalebfm"] > r["ok"]["sonic"] for r in nominal.values())
        out.append(
            f"- 主判据（标称档，两个场景 ScaleBFM ≥ SONIC，且至少一个 p < 0.05 地更好）：{'满足' if ge and sig else '不满足'}"
            f"（点估计都 ≥：{'是' if ge else '否'}；有显著更好的场景：{'是' if sig else '否'}）"
        )
    for key, r in report.items():
        n = max(r["n"], 1)
        guards = [
            f"摔 {r['fell']['scalebfm']} ≤ {r['fell']['sonic']} + 1：{'过' if r['fell']['scalebfm'] <= r['fell']['sonic'] + 1 else '不过'}",
            f"掉物 {r['dropped']['scalebfm']} ≤ {r['dropped']['sonic']} + 1：{'过' if r['dropped']['scalebfm'] <= r['dropped']['sonic'] + 1 else '不过'}",
            f"ego p95 {r['ego_p95']['scalebfm']} ≤ {r['ego_p95']['sonic']} × 1.2：{'过' if r['ego_p95']['scalebfm'] <= 1.2 * r['ego_p95']['sonic'] else '不过'}",
        ]
        line = f"- {key} 护栏：" + "；".join(guards)
        if key.endswith("_combo"):
            gap = 100 * (r["ok"]["sonic"] - r["ok"]["scalebfm"]) / n
            line += f"；组合档否决（ScaleBFM 低 > 10 个百分点）：{'否决' if gap > 10 else '不否决'}（差 {gap:+.1f} 点）"
        out.append(line)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=L2)
    ap.add_argument("--bfm", default="bfm", help="ScaleBFM run dir name under l2/<scene>/ (tier suffix appended)")
    a = ap.parse_args(argv)
    report = {}
    lines = [
        "# L2 任务层（TODO 25.8）",
        "",
        "生成：`l2_measure.py`。成功 = 专家自己的判定；实测 = 两边同一条末态规则。p = exact McNemar（双侧）。",
        "",
    ]
    for scene, (scene_dir, kind) in SCENES.items():
        torso = Torso(scene_dir)
        for tier in ("nominal", "combo"):
            sd = a.out / scene / f"sonic_{tier}" / "states"
            bd = a.out / scene / f"{a.bfm}_{tier}"
            if not sd.exists() or not bd.exists():
                continue
            r = compare(sd, bd, kind, torso)
            report[f"{scene}_{tier}"] = r
            ok, me = r["ok"], r["measured"]
            extra = ""
            if r.get("ok_without_pre_walk"):
                w = r["ok_without_pre_walk"]
                extra = f"；去掉 SONIC 走近前就失败的 {len(r['sonic_pre_walk'])} 局：{w['sonic']} 对 {w['scalebfm']} / {w['n']}（p = {w['p']}）"
            lines.append(
                f"- **{scene} {tier}**（{r['n']} 局配对，作废 {len(r['void'])}，SONIC 实时率 {r['rtf']}）：成功 SONIC {ok['sonic']} 对 ScaleBFM {ok['scalebfm']}"
                f"（只 SONIC {ok['only_sonic']} / 只 ScaleBFM {ok['only_scalebfm']}，p = {ok['p']}）；实测 {me['sonic']} 对 {me['scalebfm']}"
                f"（p = {me['p']}）；摔 {r['fell']['sonic']} 对 {r['fell']['scalebfm']}，掉物 {r['dropped']['sonic']} 对 {r['dropped']['scalebfm']}，"
                f"ego 角速度 p95 中位 {r['ego_p95']['sonic']} 对 {r['ego_p95']['scalebfm']} rad/s{extra}"
            )
            for side in ("sonic", "scalebfm"):
                lines.append(f"  - {side} 失败：" + "；".join(f"{w} ×{c}" for w, c in r["why_counts"][side]))
    lines += ["", "## 对照计划 §2", ""] + judge(report)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "l2.json").write_text(json.dumps(report, indent=1))
    (a.out / "l2.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
