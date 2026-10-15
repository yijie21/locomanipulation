"""TODO 26.2 / 26.3 (user 2026-10-15): the evolution of one RSI loop run, from its ledger, as videos.

Reads a loop sandbox (phase4_sandbox.py layout): the ledger (reviews, probes, Manager decisions with their Chinese
`summary_zh`, gate verdicts), the gate's ledger (every candidate spec) and the diagnostic rounds of each incumbent.
Generations are the starting incumbent and every accepted candidate. Writes under --out:
* timeline.png -- the incumbent's acceptance-pool successes per gate call, every probe's diagnostic-pool result;
* 00_overview -- a card per iteration (hypotheses, predictions against probe results, the Manager's summary, the gate);
* NN_step_k -- for each accepted candidate, the previous generation against the new one on a diagnostic start the new
  one finishes and the previous does not (drawn with seed 20261015 from all such starts);
* NN_generations -- every generation side by side on one start where the first fails and the last finishes, preferring
  starts that flip at each step;
each clip as a source mp4 and a <= 500 KB preview. Missing diagnostic rounds of a generation (the last one: the loop
stops before diagnosing it) are run with the sandbox's own round code first.

    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/rsi_demo_20261015/evolution.py /workspace/code/rsi-phase4/arm-a2 \\
        --scene outputs/scene/warehouse_box_45x32x33_mars --task "warehouse 搬箱" --out outputs/rsi_demo_20261015/warehouse/evolution
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import demo_video as dv  # noqa: E402

REPO = dv.REPO
SEED = 20261015
REL = Path("scripts/experiments/scalebfm_warehouse_rsi_20261014")


def sandbox_loop(sandbox: Path, loop_rel: Path):
    spec = importlib.util.spec_from_file_location("sandbox_loop", sandbox / loop_rel / "loop.py")
    argv, sys.argv = sys.argv, ["loop.py"]
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def diff(a: dict, b: dict) -> str:
    fa, fb = flat(a), flat(b)
    ch = [f"{k}={fb[k]}" for k in sorted(fb) if fa.get(k) != fb[k] and k != "name" and not k.startswith("note")]
    return "，".join(ch) or "（无改动）"


def outcomes(d: Path) -> dict:
    return {p.stem.split("_", 1)[1]: bool(json.loads(p.read_text())["ok"]) for p in sorted(d.glob("bfm_*.json"))}


class Run:
    def __init__(self, sandbox: Path, loop_rel: Path):
        self.sb = sandbox
        self.loop = sandbox_loop(sandbox, loop_rel)
        self.L = self.loop.LOOP
        self.ledger = [json.loads(x) for x in (self.loop.STATE / "ledger.jsonl").read_text().splitlines()]
        gl = self.loop.GATE_DIR / "ledger.jsonl"
        self.gates = [g for g in map(json.loads, gl.read_text().splitlines()) if not g.get("dry_run")]
        start = self.loop.load_spec(self.loop.HERE / "incumbent.yaml")
        self.gens = [dict(spec=start, hash=self.loop.spec_hash(start), path=self.loop.HERE / "incumbent.yaml",
                          label="第 0 代（起点）", acc=None, call=0)]  # fmt: skip
        inc0 = next(
            (e["result"]["incumbent_ok"] for e in self.ledger if e.get("step") == "gate" and e.get("result")), None
        )
        self.gens[0]["acc"] = inc0
        for g in self.gates:
            if g["accepted"]:
                k = len(self.gens)
                self.gens.append(dict(spec=g["spec"], hash=g["hash"], path=Path(g["candidate"]), label=f"第 {k} 代",
                                      acc=g["result"]["candidate_ok"], call=g["call"]))  # fmt: skip

    def diag_dir(self, gen: dict) -> Path:
        d = self.L / "diag" / f"incumbent_{gen['hash']}"
        n = len(list((self.L / "pool_diag" / "states").glob("*.json")))
        if len(list(d.glob("bfm_*.json"))) < n:
            print("running the diagnostic pool for", gen["label"], gen["hash"], flush=True)
            self.loop.PROCS = 12
            self.loop.run_batch([(gen["path"], d, None, None)], d.parent / f"{d.name}.log", "0,1")
        return d


def timeline(run: Run, out: Path, task: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(dv.FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=dv.FONT).get_name()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    inc = run.gens[0]["acc"]
    steps_x, steps_y = [0], [inc]
    for g in run.gates:
        r = g.get("result") or {}
        if g["accepted"]:
            inc = r.get("candidate_ok", inc)
        steps_x.append(g["call"])
        steps_y.append(inc)
        y = r.get("candidate_ok")
        if y is None:  # rejected in stage 1: only the first 30 pairs ran
            ax[0].scatter([g["call"]], [steps_y[-2]], marker="x", color="#dc2626", zorder=3)
            ax[0].annotate(
                "第一段就拒", (g["call"], steps_y[-2]), textcoords="offset points", xytext=(6, -12), fontsize=9
            )
            continue
        ax[0].scatter([g["call"]], [y], color="#16a34a" if g["accepted"] else "#dc2626", zorder=3)
        ax[0].annotate(f"{'接受' if g['accepted'] else '拒绝'} {y}", (g["call"], y), textcoords="offset points",
                       xytext=(6, 4), fontsize=9)  # fmt: skip
    ax[0].step(steps_x, steps_y, where="post", color="#2563eb", label="incumbent（当前最好）")
    sonic = run.loop.sonic_accept_ok()
    ax[0].axhline(sonic, color="#6b7280", ls="--", label=f"SONIC {sonic}")
    ax[0].set_xlabel("验收门调用次数")
    ax[0].set_xticks(range(0, len(run.gates) + 1))
    ax[0].set_ylabel("验收池成功数（共 60）")
    ax[0].set_title(f"{task}：验收池上的进化")
    ax[0].legend(loc="lower right")
    probes = [e for e in run.ledger if e.get("step") == "probe"]
    for e in probes:
        for r in e["results"]:
            if "error" in r:
                continue
            ax[1].scatter([e["iter"] + 1], [r["ok_after"]], color="#9ca3af", zorder=2)
            ax[1].annotate(
                r["id"], (e["iter"] + 1, r["ok_after"]), textcoords="offset points", xytext=(5, -3), fontsize=8
            )
        if e["results"] and "ok_before" in e["results"][0]:
            ax[1].hlines(e["results"][0]["ok_before"], e["iter"] + 0.7, e["iter"] + 1.3, color="#2563eb")
    ax[1].set_xlabel("第几轮")
    ax[1].set_xticks(range(1, max([e["iter"] + 2 for e in probes] or [2])))
    ax[1].set_ylabel("诊断池成功数（共 50）")
    ax[1].set_title("每轮探针：各假设在诊断池上的结果（蓝线 = 当时的 incumbent）")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def iteration_cards(run: Run, task: str) -> list[np.ndarray]:
    cards = [dv.card([
        (f"RSI 闭环的进化过程：{task}", 36, (240, 240, 240)),
        ("给定控制器（ScaleBFM）和任务，闭环从原样移植的专家冷启动，人不调参", 22, dv.GREY),
        ("每轮：Reviewer 读轨迹提假设 → Prober 在 50 个诊断起点上闭环试 → Manager 选 → Engineer 做候选 → 验收门在 60 个新起点上判", 18, dv.GREY),
        (f"起点：验收池 {run.gens[0]['acc']}/60；最后：验收池 {run.gens[-1]['acc']}/60（SONIC {run.loop.sonic_accept_ok()}/60）", 22, (200, 230, 210)),
    ])]  # fmt: skip
    iters = sorted(
        {e["iter"] for e in run.ledger if "iter" in e and e.get("step") in ("review", "probe", "manager", "gate")}
    )
    for n in iters:
        es = [e for e in run.ledger if e.get("iter") == n]
        hyp_path = run.loop.STATE / f"iter_{n:02d}" / "hypotheses.json"
        if not hyp_path.exists():
            continue
        hyps = json.loads(hyp_path.read_text())["hypotheses"]
        probe = {r["id"]: r for e in es if e["step"] == "probe" for r in e["results"]}
        man = next((e for e in es if e["step"] == "manager"), None)
        gate = next((e for e in es if e["step"] == "gate"), None)
        lines = [
            (f"第 {n + 1} 轮（iter_{n:02d}）", 32, (240, 240, 240)),
            ("Reviewer 提的假设（预测修好几局 → 探针实测诊断池成功数）", 20, (200, 210, 230)),
        ]
        for h in hyps:
            delta = json.dumps(h["delta"], ensure_ascii=False)
            delta = delta if len(delta) < 70 else delta[:67] + "…"
            r = probe.get(h["id"], {})
            res = f"{r['ok_before']} → {r['ok_after']}" if "ok_after" in r else "（没有探针 / 没跑）"
            lines.append((f"{h['id']}：{delta}  预测修好 {len(h['predicted_fixed'])}，实测 {res}", 17, dv.GREY))
        if man:
            # the Manager counts iterations from 0; the cards from 1
            s = re.sub(r"第\s*(\d+)\s*轮", lambda m_: f"第 {int(m_.group(1)) + 1} 轮", man.get("summary_zh", ""))
            for line in wrap(s, 1180, 17)[:7]:
                lines.append((line, 17, (230, 210, 160)))
        if gate:
            r = gate.get("result") or {}
            verdict = "接受" if gate["accepted"] else "拒绝"
            nums = (
                f"候选 {r.get('candidate_ok')} 对 incumbent {r.get('incumbent_ok')}（60 局）"
                if r
                else gate.get("reason", "")[:60]
            )
            lines.append((f"验收门：{verdict} — {nums}", 22, dv.GREEN if gate["accepted"] else dv.RED))
        elif man and man.get("decision") == "review":
            lines.append(("Manager：证据不够，打回 Reviewer 重审（不调用验收门）", 20, (230, 210, 160)))
        cards.append(dv.card(lines))
    return cards


def wrap(text: str, width_px: int, size: int) -> list[str]:
    """Greedy wrap on the rendered width (CJK has no spaces; Latin words stay whole)."""
    from PIL import Image, ImageDraw

    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    font = dv._font(size)
    tokens = re.findall(r"[A-Za-z0-9_.+\-/%]+|\s+|.", text)
    out, cur = [], ""
    for tok in tokens:
        if d.textlength(cur + tok, font=font) > width_px and cur.strip():
            out.append(cur.strip())
            cur = tok.lstrip()
        else:
            cur += tok
    if cur.strip():
        out.append(cur.strip())
    return out


class _Why(dict):
    """The task's failure strings in Chinese, prefix rules included (compare.why_zh)."""

    def __init__(self, task, base):
        super().__init__(base)
        self.task = task

    def get(self, key, default=None):
        from compare import why_zh

        return why_zh(self.task, key)


def caption(gen: dict, f: dict, z: dict, why_zh: dict, extra: str = "") -> list[tuple[str, tuple]]:
    ok = bool(f["ok"])
    obj = z[dv.object_key(z)]
    rise = 100 * float(obj[-1, 2] - obj[0, 2])
    head = f"{gen['label']}：{'成功' if ok else '失败'} — {why_zh.get(f.get('why', ''), f.get('why', ''))}，物体最终抬起 {rise:.1f} cm"
    return [(head, dv.GREEN if ok else dv.RED), (extra or f"验收池 {gen['acc']}/60", dv.GREY)]


def render_multi(m, paths: list[Path], out: Path, captions, title=None, title_s=4.0, tail_s=1.0) -> Path:
    sides = []
    for p in paths:
        z, f = dv.load(p)
        sides.append((z, dv.view(m, f, z)))
    t_end = max(float(z["t"][-1]) for z, _ in sides) + tail_s
    strips = np.hstack([dv.caption_strip(c, dv.wv.W) for c in captions])
    width = len(paths) * dv.wv.W
    if title is not None and title.shape[1] != width:
        from PIL import Image

        title = np.asarray(Image.fromarray(title).resize((width, title.shape[0])))
    w = dv.Writer(out, width, dv.wv.H + strips.shape[0])
    if title is not None:
        w.write(title, int(title_s * dv.FPS))
    for s in np.arange(0.0, t_end, 1.0 / dv.FPS):
        tiles = []
        for z, v in sides:
            i = int(np.clip(np.searchsorted(z["t"], s), 0, len(z["t"]) - 1))
            tiles.append(v.frame(z["qpos"][i], dv.wv.ghosts_at(z, i)))
        w.write(np.vstack([strips, np.hstack(tiles)]))
    w.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sandbox", type=Path)
    ap.add_argument("--loop-rel", type=Path, default=REL)
    ap.add_argument("--scene", type=Path, default=None, help="default: the task's scene")
    ap.add_argument("--task", choices=("warehouse", "bottle"), required=True, help="compare.TASKS: captions, scene")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from compare import TASKS, why_zh

    task = TASKS[a.task]
    why = {w: why_zh(task, w) for w in set(task.why) | {""}}
    WHY = _Why(task, why)
    run = Run(a.sandbox, a.loop_rel)
    out = a.out if a.out.is_absolute() else REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    scene = a.scene or Path("outputs/scene") / task.scene
    m = dv.scene_model(scene if scene.is_absolute() else REPO / scene)
    dirs = [run.diag_dir(g) for g in run.gens]
    ok = [outcomes(d) for d in dirs]
    for g, o in zip(run.gens, ok):
        g["diag"] = sum(o.values())
    timeline(run, out / "timeline.png", task.title)
    made = []
    clips = out / "clips"
    # the iteration cards and the timeline as one overview clip
    cards = iteration_cards(run, task.title)
    from PIL import Image

    tl = np.asarray(Image.open(out / "timeline.png").convert("RGB").resize(cards[0].shape[1::-1]))
    w = dv.Writer(clips / "00_overview.mp4", cards[0].shape[1], cards[0].shape[0])
    for c in cards:
        w.write(c, 7 * dv.FPS)
    w.write(tl, 8 * dv.FPS)
    w.close()
    made.append(("00_overview", clips / "00_overview.mp4"))
    rng = random.Random(SEED)
    for k in range(1, len(run.gens)):
        prev, new = run.gens[k - 1], run.gens[k]
        cands = sorted(s for s in ok[k] if ok[k][s] and not ok[k - 1].get(s, False))
        if not cands:
            continue
        s = rng.choice(cands)
        title = dv.card([
            (f"{task.title} · 第 {k} 次接受（验收门第 {new['call']} 次调用）", 32, (240, 240, 240)),
            (f"改动：{diff(prev['spec'], new['spec'])}", 20, (230, 210, 160)),
            (f"诊断池 {prev['diag']} → {new['diag']}/50，验收池 {prev['acc']} → {new['acc']}/60", 22, (200, 230, 210)),
            (f"同一个诊断起点 {s}：左 {prev['label']}，右 {new['label']}（从这次改动修好的 {len(cands)} 个起点里随机抽）", 18, dv.GREY),
        ])  # fmt: skip
        paths = [dirs[k - 1] / f"bfm_{s}.npz", dirs[k] / f"bfm_{s}.npz"]
        caps = [caption(g, *reversed(dv.load(p)), WHY) for g, p in zip((prev, new), paths)]
        src = clips / f"{k:02d}_step_{k}_{s}.mp4"
        dv.render_pair(m, paths[0], paths[1], src, caps, title=title)
        made.append((src.stem, src))
    # every generation on one start: the first fails, the last finishes, prefer starts that flip at every step
    last = len(run.gens) - 1
    cands = [s for s in ok[last] if ok[last][s] and not ok[0].get(s, False)]
    if cands and last >= 1:
        score = {s: sum(ok[i].get(s, False) for i in range(last + 1)) for s in cands}
        best = min(score.values())
        s = random.Random(SEED + 1).choice(sorted(c for c in cands if score[c] == best))
        title = dv.card([
            (f"{task.title} · 每一代专家在同一个起点 {s} 上", 30, (240, 240, 240)),
            ("从左到右：" + " → ".join(f"{g['label']}（验收池 {g['acc']}/60）" for g in run.gens), 18, dv.GREY),
        ], size=(len(run.gens) * dv.wv.W, dv.wv.H + 64))  # fmt: skip
        paths = [d / f"bfm_{s}.npz" for d in dirs]
        caps = [
            caption(g, *reversed(dv.load(p)), WHY, extra=f"诊断池 {g['diag']}/50 · 验收池 {g['acc']}/60")
            for g, p in zip(run.gens, paths)
        ]
        src = clips / f"{last + 1:02d}_generations_{s}.mp4"
        render_multi(m, paths, src, caps, title=title)
        made.append((src.stem, src))
    index = []
    for stem, src in made:
        prev = dv.preview(src, out / "previews" / stem)
        index.append(dict(clip=stem, source=str(src.relative_to(REPO)),
                          previews=[(str(p.relative_to(REPO)), p.stat().st_size) for p in prev]))  # fmt: skip
        print(index[-1], flush=True)
    gens = [
        dict(label=g["label"], hash=g["hash"], call=g["call"], acceptance=g["acc"], diagnostic=g["diag"])
        for g in run.gens
    ]
    (out / "videos.json").write_text(json.dumps(dict(generations=gens, videos=index), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
