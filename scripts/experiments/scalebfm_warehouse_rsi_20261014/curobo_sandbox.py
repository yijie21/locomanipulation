"""TODO 26.5.6 (docs/handoff/scalebfm-warehouse-rsi-20261014.md §21): build the all-CuRobo redo's cold-start sandbox --
phase 4's arm-A loop, with both controllers reaching along the CuRobo route (user 2026-10-15) and none of phases 1-4's
answers.

As phase4_sandbox.py, but from HEAD (the route, the route cache, the analytic IK and the shared batches are in it), with
phase 4's own answers removed too (its scripts, final configs, demo scripts, the bottle instance that copied them, the
plan past §11). The pools are §4's (their start states do not depend on how SONIC reached; SONIC's own rounds in them are
its production, segmented reach -- SONIC on the CuRobo route went 1/30 against 3/10 on the same smoke seeds, plan §21),
with phase 0's SONIC envelope. The frozen task block is the redo's (`scalebfm_warehouse_curobo_task_20261015.yaml`:
reach_route planned with no fallback, analytic IK) -- bfm_round's default --config is pointed at it. The sealed set is
not copied: it does not exist for the loop.

    python3 scripts/experiments/scalebfm_warehouse_rsi_20261014/curobo_sandbox.py /workspace/code/rsi-curobo/arm-c
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase4_sandbox as p4  # noqa: E402

REPO, REL, PLAN, LOOP, HIDDEN, SCENE = p4.REPO, p4.REL, p4.PLAN, p4.LOOP, p4.HIDDEN, p4.SCENE
TASK_CONFIG = "configs/experiments/scalebfm_warehouse_curobo_task_20261015.yaml"
OLD_CONFIG = "configs/experiments/controller_compare_l2_warehouse_sonic_20261013.yaml"
# phase 1-3's (phase4_sandbox) and phase 4's answers, and the sandbox's own makers
DROP_REL = (
    "phase1.py", "sealed.py", "prompts/reviewer.md", "final.yaml", "final_route.yaml", "phase4_sandbox.py",
    "phase4_score.py", "phase4_sealed.py", "route_eval.py", "approach_motion.py", "curobo_sandbox.py",
    "curobo_sealed.py",
)  # fmt: skip
DROP_TREE = (
    "scripts/experiments/rsi_demo_20261015",
    "scripts/experiments/scalebfm_bottle_rsi_20261015",
    "tests/test_scalebfm_bottle_rsi.py",
    "tests/test_scalebfm_rsi_phase4.py",
    "docs/handoff/scalebfm-bottle-rsi-20261015.md",
)
MARKERS = re.compile(
    p4.MARKERS.pattern + r"|loaded_palm_hold|71743845d6b5|8eb038e21ad0|rsi-phase4|arm-a2|base_anchor: reference|"
    r"68 / 68|57 / 40|封存集 80 局：臂 A"
)
SEC0 = [
    "## 0. 本次运行（全 CuRobo 冷启动，TODO 26.5.6）",
    "",
    "- ScaleBFM 的伸手只走 CuRobo 路线：从走停时下垂的手一条路线到预抓点，没有原地抬手，规划不出来该局就失败"
    "（冻结配置：`visual.reach_route: planned`、`visual.reach_route_fallback: none`）。全身 IK 用解析雅可比"
    "（`carry.ik_jacobian: analytic`）。这三个字段和 §3 的任务字段一样冻结。",
    f"- 冻结的任务配置是 `{TASK_CONFIG}`（§3 原来那份 L2 配置加上面三项）。三个池子的起点同 §4；池子里 SONIC 自己的局"
    "是它的生产版（原地抬手再直线伸手），停止条件和对照都对它。",
    "- 起点是 incumbent（`incumbent.yaml`，25.7 的移植）。闭环从这里冷启动，自己找改进；人不调参。",
    f"- {p4.ARM_TEXT['A']}",
    "- 预算：验收门 12 次、墙钟 6 h、模型花费 150 USD，先到先停；验收池上比 SONIC 多成 ≥ 5 局即停。",
    "- 只读 `rules.md` 和本副本里的文件。",
    "",
]


def cold_plan() -> str:
    """Plan §1-§11 with this run's §0 (phase4_sandbox.cold_plan's arm text replaced)."""
    lines = p4.cold_plan("A").splitlines()
    start = next(i for i, x in enumerate(lines) if x.startswith("## 0."))
    end = next(i for i, x in enumerate(lines) if x.startswith("## 1."))
    status = next(i for i, x in enumerate(lines) if x.startswith("状态："))
    lines[status] = "状态：全 CuRobo 冷启动的副本，见 §0。本副本只有运行前定下的 §1–§11。"
    return "\n".join(lines[:start] + SEC0 + lines[end:]) + "\n"


def build(dest: Path, diag: Path, accept: Path, envelope: Path) -> None:
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", "HEAD"], cwd=REPO, check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    for rel in DROP_REL:
        (dest / REL / rel).unlink(missing_ok=True)
    for rel in DROP_TREE:
        target = dest / rel
        shutil.rmtree(target) if target.is_dir() else target.unlink(missing_ok=True)
    ext = dest / REL / "extensions.py"
    ext.write_text(p4.strip_hooks(ext.read_text()))
    gate = dest / REL / "accept_gate.py"
    old, new = p4.GATE_COMMENT
    if old not in gate.read_text():
        raise SystemExit("accept_gate.py: the comment to neutralise moved")
    gate.write_text(gate.read_text().replace(old, new))
    runner = dest / REL / "bfm_round.py"
    if runner.read_text().count(OLD_CONFIG) != 1:
        raise SystemExit("bfm_round.py: the default --config moved")
    runner.write_text(runner.read_text().replace(OLD_CONFIG, TASK_CONFIG))
    for name in ("STATE.md", "TODO.md", "DECISIONS.md"):
        (dest / name).write_text(p4.STUB.format(name=name.removesuffix(".md")).replace("one phase-4 arm", "the redo"))
    (dest / PLAN).write_text(cold_plan())
    hits = []
    for p in sorted(dest.rglob("*")):
        rel = p.relative_to(dest).as_posix()
        if p.is_symlink() or not p.is_file() or rel.startswith(p4.MARKER_SKIP):
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        hits += [f"{rel}: {m.group(0)}" for m in MARKERS.finditer(text)]
    if hits:
        raise SystemExit("answer markers left:\n  " + "\n  ".join(hits))
    (dest / "third_party").symlink_to(REPO / "third_party")
    shutil.copytree(REPO / SCENE, dest / SCENE)
    shutil.copytree(diag, dest / LOOP / "pool_diag" / "states")
    shutil.copytree(accept, dest / HIDDEN / "pool_accept" / "states")
    env = (
        dest / LOOP / "envelope"
    )  # SONIC's (phase 0, the same pool); ScaleBFM's start comes from this sandbox's rounds
    env.mkdir(parents=True)
    for f in ("sonic_v1_1.json", "sonic_v1_1_ok.json"):
        shutil.copy2(envelope / f, env / f)
    p4.git(dest, "init", "-q")
    files = p4.git(dest, "ls-files", "-o", "--exclude-standard", "-z").stdout
    if any(f.startswith(("third_party", "outputs")) for f in files.split("\0")):
        raise SystemExit("third_party or outputs would be tracked; check .gitignore")
    p4.git(dest, "add", "--pathspec-from-file=-", "--pathspec-file-nul", input=files)
    p4.git(dest, "-c", "user.name=rsi-curobo", "-c", "user.email=rsi-curobo@localhost", "commit", "-q", "-m",
           "all-CuRobo redo sandbox")  # fmt: skip
    print(f"{dest} ({len(files.split(chr(0))) - 1} files tracked)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--diag", type=Path, default=REPO / LOOP / "pool_diag" / "states")
    ap.add_argument("--accept", type=Path, default=REPO / HIDDEN / "pool_accept" / "states")
    ap.add_argument("--envelope", type=Path, default=REPO / LOOP / "envelope")
    a = ap.parse_args(argv)
    build(a.dest.resolve(), a.diag.resolve(), a.accept.resolve(), a.envelope.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
