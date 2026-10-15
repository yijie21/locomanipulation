"""TODO 26.1 / 26.3.4 (user 2026-10-15): comparison videos -- SONIC's production expert against ScaleBFM with the expert
the RSI loop re-tuned, on a task's sealed starts (`--task warehouse`: phase-4 arm A's final incumbent, which arm A2
found too; `--task bottle`: the bottle loop's final incumbent).

Clips are chosen by a rule fixed before rendering (seed 20261015), not by looking at them:
* 3 starts drawn at random from those only ScaleBFM finished;
* 2 starts both finished, the ones whose torso-shake difference (SONIC - ScaleBFM) is closest to the median difference;
* 1 start drawn at random from those only SONIC finished (the counter-example).
(fewer when a category has fewer starts). Each clip is the whole round from the same start state, SONIC left, ScaleBFM
right, the commanded jaw centres as ghost spheres, preceded by a title card. Writes metrics.json, summary.png, the
source clips and <= 500 KB previews under outputs/rsi_demo_20261015/<task>/.

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/rsi_demo_20261015/compare.py --task warehouse|bottle [--metrics-only]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import demo_video as dv  # noqa: E402
from l2_measure import Torso, mcnemar_exact, round_facts  # noqa: E402

REPO = dv.REPO
SEED = 20261015


@dataclass
class Task:
    name: str  # outputs/rsi_demo_20261015/<name>
    title: str
    sonic: Path
    bfm: Path
    scene: str
    kind: str  # l2_measure's object key
    obj: str  # the object, in captions
    why: dict
    starts: tuple = ("reach", "handoff")  # events opening the hand-jerk window (the first one a round has)
    align: str | None = None  # render both rounds aligned on this event (None: the round clock)
    note: str = ""
    why_prefix: dict = field(default_factory=dict)


WAREHOUSE = Task(
    name="warehouse",
    title="warehouse 搬箱",
    sonic=REPO / "outputs" / "scalebfm_warehouse_rsi_20261014_hidden" / "pool_sealed" / "states",
    bfm=REPO / "outputs" / "scalebfm_warehouse_rsi_20261014" / "sealed" / "phase4_A",
    scene="warehouse_box_45x32x33_mars",
    kind="box",
    obj="箱子",
    why={
        "": "完成",
        "completed": "完成",
        "lost_attachment": "抬起时失去夹持",
        "reach_stalled": "伸手停在箱沿上方",
        "basket_tilt": "箱子倾斜",
        "stand_not_delivered": "站位没到",
        "empty_gripper": "合爪没夹到",
        "probe_not_attached": "试提没夹住",
        "unstable_hold": "保持不稳",
        "timeout": "超时",
    },  # fmt: skip
    note="专家由 RSI 闭环冷启动自动重标：3 次验收、约 3 小时，人没调参",
)
BOTTLE = Task(
    name="bottle",
    title="bottle 抓瓶",
    sonic=REPO / "outputs" / "scalebfm_bottle_rsi_20261015_hidden" / "pool_sealed" / "states",
    bfm=REPO / "outputs" / "scalebfm_bottle_rsi_20261015" / "sealed" / "final",
    scene="sim_lowcom",
    kind="bottle",
    obj="瓶子",
    why={"": "完成", "completed": "完成"},
    why_prefix={
        "simple lift did not raise the bottle": "抬起时没把瓶子抬上去",
        "[hold] object slipped": "保持时瓶子在手里滑脱",
        "[hold] object is back on the table": "保持后瓶子回到桌上",
        "simple no bilateral pad contact": "合爪后没有两侧接触",
        "simple bottle knocked before the close": "合爪前碰倒瓶子",
        "[grasp] simple bottle knocked": "合爪前碰倒瓶子",
        "simple entry could not align": "夹爪对不准瓶子",
        "simple entry timed out": "入位超时",
        "simple lost pad contact during the lift": "抬起时失去指垫接触",
        "simple support reference outside": "支撑超限",
        "fell": "摔倒",
        "[stand]": "站位够不着",
        "[grasp]": "抓取失败",
    },  # fmt: skip
    starts=("ready",),
    align="walk",
    note="专家由 RSI 闭环冷启动自动重标，人没调参",
)
TASKS = {t.name: t for t in (WAREHOUSE, BOTTLE)}
METRIC_KEYS = ("torso_w_p95", "hand_jerk", "object_w_p95")


def metrics(task: Task) -> tuple:
    return (
        ("torso_w_p95", "躯干晃动（角速度 p95, rad/s）"),
        ("hand_jerk", "手部 jerk 中位数（m/s³）"),
        ("object_w_p95", f"{task.obj}晃动（抬起后角速度 p95, rad/s）"),
    )


def why_zh(task: Task, why: str) -> str:
    if why in task.why:
        return task.why[why]
    return next((v for k, v in task.why_prefix.items() if why.startswith(k)), why[:28])


def measure(task: Task) -> dict:
    torso = Torso(task.scene)
    kin = dv.Kin(dv.scene_model(REPO / "outputs" / "scene" / task.scene))
    rows = {}
    for s in sorted(task.sonic.glob("*.npz")):
        start = s.stem.replace("sonic", "", 1).lstrip("_")
        b = task.bfm / f"{s.stem.replace('sonic', 'bfm', 1)}.npz"
        if not b.exists():
            continue
        row = {}
        for side, p in (("sonic", s), ("scalebfm", b)):
            z, f = dv.load(p)
            facts = round_facts(p, task.kind, torso)
            row[side] = dict(ok=bool(facts["ok"]), why=f.get("why", "") or "", rise_cm=facts["rise_end_cm"],
                             fell=facts["fell"], dropped=facts["dropped"], file=str(p.relative_to(REPO)),
                             **dv.smoothness(z, f, kin, task.starts))  # fmt: skip
        rows[start] = row
        print(start, row["sonic"]["ok"], row["scalebfm"]["ok"], flush=True)
    return rows


def summarise(rows: dict, task: Task) -> dict:
    keys = sorted(rows)
    only_b = [k for k in keys if rows[k]["scalebfm"]["ok"] and not rows[k]["sonic"]["ok"]]
    only_s = [k for k in keys if rows[k]["sonic"]["ok"] and not rows[k]["scalebfm"]["ok"]]
    both = [k for k in keys if rows[k]["sonic"]["ok"] and rows[k]["scalebfm"]["ok"]]
    out = dict(n=len(keys), sonic_ok=sum(rows[k]["sonic"]["ok"] for k in keys),
               scalebfm_ok=sum(rows[k]["scalebfm"]["ok"] for k in keys), only_scalebfm=len(only_b),
               only_sonic=len(only_s), both=len(both), p_two_sided=mcnemar_exact(len(only_b), len(only_s)),
               sonic_drops=sum(rows[k]["sonic"]["dropped"] for k in keys),
               scalebfm_drops=sum(rows[k]["scalebfm"]["dropped"] for k in keys), metrics={})  # fmt: skip
    for m, _ in metrics(task):
        for subset, ks in (("all", keys), ("both_ok", both)):
            pairs = [(rows[k]["sonic"].get(m), rows[k]["scalebfm"].get(m)) for k in ks]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if not pairs:
                continue
            x, y = np.array(pairs).T
            out["metrics"][f"{m}/{subset}"] = dict(
                n=len(pairs), sonic_median=round(float(np.median(x)), 3), scalebfm_median=round(float(np.median(y)), 3),
                scalebfm_smoother=int(np.sum(y < x)), wilcoxon_p=wilcoxon(x, y),
            )  # fmt: skip
    return out


def wilcoxon(x, y) -> float | None:
    from scipy.stats import wilcoxon as w

    d = np.asarray(x) - np.asarray(y)
    return None if np.all(d == 0) or len(d) < 5 else round(float(w(x, y).pvalue), 6)


def select(rows: dict) -> list[tuple[str, str]]:
    keys = sorted(rows)
    rng = random.Random(SEED)
    only_b = [k for k in keys if rows[k]["scalebfm"]["ok"] and not rows[k]["sonic"]["ok"]]
    only_s = [k for k in keys if rows[k]["sonic"]["ok"] and not rows[k]["scalebfm"]["ok"]]
    both = [k for k in keys if rows[k]["sonic"]["ok"] and rows[k]["scalebfm"]["ok"]]
    gap = {k: rows[k]["sonic"]["torso_w_p95"] - rows[k]["scalebfm"]["torso_w_p95"] for k in both}
    med = float(np.median(list(gap.values()))) if gap else 0.0
    picks = [("only_scalebfm", k) for k in rng.sample(only_b, min(3, len(only_b)))]
    picks += [("both", k) for k in sorted(both, key=lambda k: abs(gap[k] - med))[:2]]
    picks += [("only_sonic", k) for k in rng.sample(only_s, min(1, len(only_s)))]
    return picks


def figure(rows: dict, summary: dict, task: Task, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(dv.FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=dv.FONT).get_name()
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    n = summary["n"]
    ax[0].bar(["SONIC", "ScaleBFM\n+ RSI"], [summary["sonic_ok"], summary["scalebfm_ok"]], color=["#8a8f98", "#3b82f6"])
    ax[0].set_ylim(0, n)
    ax[0].set_title(
        f"封存集 {n} 个起点的成功数\n只 ScaleBFM 成 {summary['only_scalebfm']}，只 SONIC 成 {summary['only_sonic']}"
    )
    for i, v in enumerate([summary["sonic_ok"], summary["scalebfm_ok"]]):
        ax[0].text(i, v + 1, f"{v}/{n}", ha="center")
    rng = np.random.default_rng(0)
    for a, (m, name) in zip(ax[1:], metrics(task)):
        ks = [k for k in rows if rows[k]["sonic"].get(m) is not None and rows[k]["scalebfm"].get(m) is not None]
        data = [[rows[k]["sonic"][m] for k in ks], [rows[k]["scalebfm"][m] for k in ks]]
        if not ks:
            a.set_axis_off()
            continue
        a.boxplot(data, tick_labels=["SONIC", "ScaleBFM\n+ RSI"], showfliers=False)
        for i, d in enumerate(data):
            a.scatter(np.full(len(d), i + 1) + rng.uniform(-0.12, 0.12, len(d)), d, s=6, alpha=0.5)
        s = summary["metrics"].get(f"{m}/all", {})
        a.set_title(f"{name}\n越低越平稳；n={s.get('n')}，ScaleBFM 更低 {s.get('scalebfm_smoother')} 局")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fmt_side(task: Task, name: str, r: dict) -> list[tuple[str, tuple]]:
    ok = r["ok"]
    head = f"{name}：{'成功' if ok else '失败'} — {why_zh(task, r['why'])}，{task.obj}最终抬起 {r['rise_cm']:.1f} cm"
    m = f"躯干晃动 p95 {r['torso_w_p95']:.2f} rad/s · 手部 jerk {r.get('hand_jerk', float('nan')):.1f} m/s³"
    return [(head, dv.GREEN if ok else dv.RED), (m, dv.GREY)]


TITLES = {
    "only_scalebfm": "只有 ScaleBFM + RSI 完成的起点（随机抽取）",
    "both": "两边都完成的起点（躯干晃动差距取中位数附近）",
    "only_sonic": "反例：只有 SONIC 完成的起点（随机抽取）",
}


def render(task: Task, out: Path, rows: dict, summary: dict, picks: list) -> list[dict]:
    m = dv.scene_model(REPO / "outputs" / "scene" / task.scene)
    made = []
    clock = "同一起点、两边按开始走的时刻对齐" if task.align == "walk" else "同一起点、同一时钟"
    for i, (cat, start) in enumerate(picks, 1):
        r = rows[start]
        title = dv.card([
            (f"{task.title} · 起点 {start}", 34, (240, 240, 240)),
            (TITLES[cat], 26, (200, 210, 230)),
            (f"左：SONIC 生产专家    右：ScaleBFM + RSI 重标的专家（{clock}）", 22, dv.GREY),
            (f"封存集 {summary['n']} 局：SONIC {summary['sonic_ok']} 成功，ScaleBFM + RSI {summary['scalebfm_ok']} 成功", 22, dv.GREY),
            ("半透明小球 = 专家指令的夹爪中心（品红左手、青色右手）", 18, dv.GREY),
        ])  # fmt: skip
        src = out / "clips" / f"{i:02d}_{cat}_{start}.mp4"
        dv.render_pair(m, REPO / r["sonic"]["file"], REPO / r["scalebfm"]["file"], src,
                       [fmt_side(task, "SONIC", r["sonic"]), fmt_side(task, "ScaleBFM + RSI", r["scalebfm"])],
                       title=title, align=task.align)  # fmt: skip
        prev = dv.preview(src, out / "previews" / src.stem)
        made.append(dict(category=cat, start=start, source=str(src.relative_to(REPO)),
                         previews=[(str(p.relative_to(REPO)), p.stat().st_size) for p in prev]))  # fmt: skip
        print(made[-1], flush=True)
    return made


def overview(task: Task, out: Path, summary: dict) -> list[dict]:
    mt = summary["metrics"]
    lines = [
        (f"ScaleBFM + RSI 对 SONIC：{task.title}", 36, (240, 240, 240)),
        (f"封存集 {summary['n']} 个没见过的起点：SONIC 成功 {summary['sonic_ok']}，ScaleBFM + RSI 成功 {summary['scalebfm_ok']}", 26, (200, 230, 210)),
        (f"只 ScaleBFM 成 {summary['only_scalebfm']} 局，只 SONIC 成 {summary['only_sonic']} 局，McNemar 双侧 p = {summary['p_two_sided']:.2g}；"
         f"掉物 SONIC {summary['sonic_drops']}、ScaleBFM {summary['scalebfm_drops']}", 22, dv.GREY),
    ]  # fmt: skip
    for m, name in metrics(task):
        s = mt.get(f"{m}/both_ok")
        if s:
            lines.append((f"{name}（两边都成的 {s['n']} 局）：SONIC {s['sonic_median']} → ScaleBFM {s['scalebfm_median']}，"
                          f"更平稳 {s['scalebfm_smoother']}/{s['n']}", 20, dv.GREY))  # fmt: skip
    if task.name == "warehouse":
        lines.append(
            (
                "手部 jerk 的差距主要来自 SONIC 手上约 1 mm 的高频抖动（ScaleBFM 约 0.4 mm）；按 1 秒尺度看约 2 倍",
                18,
                dv.GREY,
            )
        )
    lines.append((task.note, 20, (230, 210, 160)))
    src = dv.still(out / "clips" / "00_overview.mp4", dv.card(lines), 8.0)
    prev = dv.preview(src, out / "previews" / "00_overview")
    return [dict(category="overview", source=str(src.relative_to(REPO)),
                 previews=[(str(p.relative_to(REPO)), p.stat().st_size) for p in prev])]  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--metrics-only", action="store_true")
    ap.add_argument(
        "--note", default="", help="replace the overview's last line (e.g. the loop's gate calls and hours)"
    )
    a = ap.parse_args(argv)
    task = TASKS[a.task]
    if a.note:
        task.note = a.note
    out = REPO / "outputs" / "rsi_demo_20261015" / task.name
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "metrics.json"
    rows = json.loads(mpath.read_text())["rounds"] if mpath.exists() else measure(task)
    summary = summarise(rows, task)
    picks = select(rows)
    mpath.write_text(json.dumps(dict(summary=summary, picks=picks, rounds=rows), indent=1, ensure_ascii=False))
    figure(rows, summary, task, out / "summary.png")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if a.metrics_only:
        return 0
    made = overview(task, out, summary) + render(task, out, rows, summary, picks)
    (out / "videos.json").write_text(json.dumps(made, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
