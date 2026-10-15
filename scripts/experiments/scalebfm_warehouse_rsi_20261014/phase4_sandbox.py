"""TODO 25.9.1.6 (docs/handoff/scalebfm-warehouse-rsi-20261014.md §17): build one phase-4 arm's sandbox -- a checkout of
the code the phase-2 loop ran on with everything that carries phase 1-3's answers taken out, so a cold-start arm finds
only what it finds.

The sandbox is `git archive BASE` plus this checkout's loop driver and prompts, minus:
* STATE / TODO / DECISIONS (stubs), and the plan past §11 (results); §0 describes the arm instead;
* phase 1's extension hook, phase1.py, sealed.py and the phase-2 Reviewer prompt; in arm B also handoff.py and
  envelope.py (the flat arm has no handoff records and no envelope);
* the gate comment that names the retry field phase 1 found.
Data are copied, not linked, because a link's `..` climbs into this checkout's outputs: the diagnostic pool, the
acceptance pool, the warehouse scene, and in arm A phase 0's envelope. Only third_party is a link (the loop's audit
voids a role call that names a path through it). A marker grep refuses to finish while a phase 1-3 marker is left.
The sandbox is its own git repository with one commit: the loop tells the Engineer's edits apart, and no history
leads back here.

    python3 scripts/experiments/scalebfm_warehouse_rsi_20261014/phase4_sandbox.py A /workspace/code/rsi-phase4/arm-a
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REL = HERE.relative_to(REPO)  # scripts/experiments/scalebfm_warehouse_rsi_20261014
PLAN = Path("docs/handoff/scalebfm-warehouse-rsi-20261014.md")
LOOP = Path("outputs/scalebfm_warehouse_rsi_20261014")
HIDDEN = Path("outputs/scalebfm_warehouse_rsi_20261014_hidden")
SCENE = Path("outputs/scene/warehouse_box_45x32x33_mars")
# The phase-2 loop's code: src unchanged since phase 1's commit; before the CuRobo reach route (5a35010)
BASE = "e9f0530"
OVERLAY = (
    "loop.py",
    "batch.py",
    "prompts/engineer.md",
    "prompts/manager.md",
    "prompts/phase4",
)  # batch.py: loop.py imports it
DROP = {
    "A": ("phase1.py", "sealed.py", "prompts/reviewer.md"),
    "B": ("phase1.py", "sealed.py", "prompts/reviewer.md", "handoff.py", "envelope.py"),
}
GATE_COMMENT = (
    "(descend_retries\n    # only after a stall)",
    "(a retry count is\n    # read only after a failure)",
)
# phase 1-3 results by name. The retry fields themselves predate the loop (2026-09-27 configs and docs name them), and
# 25.7's search of a pelvis floor is in the frozen plan §2: neither is a marker
MARKERS = re.compile(
    r"pelvis_floor|phase1_best|h1_lift_reseat|272307df4fcb|descend_retries\W+2|reseat_retries\W+2|48 对 30|9 → 46|"
    r"骨盆关键点|重夹"
)
MARKER_SKIP = ("src/", "tests/", ".git/", "third_party", "outputs/")
STUB = (
    "# {name}\n\nThis checkout is the sandbox of one phase-4 arm (plan §0). The project's {name} is not part of it.\n"
)
ARM_TEXT = {
    "A": "本臂用 §6 的全套工具：交接记录（§6.1）、控制器包络（§6.2）、Prober 的干预归因（§6.3），加上验收门（§6.4）。",
    "B": (
        "本臂是扁平闭环（借 RoboRSI 的 flat revision）：Reviewer 读轨迹后直接提改动，Manager 选一条，Engineer 做成候选，"
        "验收门判。没有 Prober、包络和交接记录，§6.1–6.3 描述的是本臂没有的工具；§6.4 的验收门照用。"
    ),
}


def cold_plan(arm: str) -> str:
    """Plan §1-§11 (written and frozen before any result), its status line replaced, with §0 for the arm."""
    text = (REPO / PLAN).read_text()
    head = text[: text.index("\n## 12.")]
    lines = head.splitlines()
    status = next(i for i, x in enumerate(lines) if x.startswith("状态："))
    lines[status] = "状态：第 4 阶段一臂的冷启动副本，见 §0。本副本只有运行前定下的 §1–§11。"
    first = next(i for i, x in enumerate(lines) if x.startswith("## 1."))
    sec0 = [
        f"## 0. 本次运行（第 4 阶段，臂 {arm}）",
        "",
        "- 起点是 incumbent（`incumbent.yaml`，25.7 的移植）。闭环从这里冷启动，自己找改进；人不调参。",
        f"- {ARM_TEXT[arm]}",
        "- 预算和停止条件同 §9：验收门 12 次、墙钟 36 h、模型花费 400 USD；验收池上比 SONIC 多成 ≥ 5 局即停。",
        "- 只读 `rules.md` 和本副本里的文件。",
        "",
    ]
    return "\n".join(lines[:first] + sec0 + lines[first:]) + "\n"


def strip_hooks(src: str) -> str:
    """extensions.py with its registry and no registered hook (phase 1's pelvis floor is one)."""
    return src[: src.index("\n\n@hook(")] + "\n"


def leaks(dest: Path) -> list[str]:
    hits = []
    for p in sorted(dest.rglob("*")):
        rel = p.relative_to(dest).as_posix()
        if p.is_symlink() or not p.is_file() or rel.startswith(MARKER_SKIP):
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        hits += [f"{rel}: {m.group(0)}" for m in MARKERS.finditer(text)]
    return hits


def git(dest: Path, *args: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=dest, check=True, capture_output=True, text=True, **kw)


def build(arm: str, dest: Path) -> None:
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    if re.search(r"[^A-Za-z0-9/-]", str(dest)):
        raise SystemExit("use letters, digits and '-' only: the transcript audit maps the path to ~/.claude/projects")
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", BASE], cwd=REPO, check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    for rel in OVERLAY:
        src, dst = HERE / rel, dest / REL / rel
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    for rel in DROP[arm]:
        (dest / REL / rel).unlink()
    ext = dest / REL / "extensions.py"
    ext.write_text(strip_hooks(ext.read_text()))
    gate = dest / REL / "accept_gate.py"
    old, new = GATE_COMMENT
    if old not in gate.read_text():
        raise SystemExit("accept_gate.py: the comment to neutralise moved")
    gate.write_text(gate.read_text().replace(old, new))
    for name in ("STATE.md", "TODO.md", "DECISIONS.md"):
        (dest / name).write_text(STUB.format(name=name.removesuffix(".md")))
    (dest / PLAN).write_text(cold_plan(arm))
    hits = leaks(dest)
    if hits:
        raise SystemExit("phase 1-3 markers left:\n  " + "\n  ".join(hits))
    # data: copies, so no `..` climbs out
    (dest / "third_party").symlink_to(REPO / "third_party")
    shutil.copytree(REPO / SCENE, dest / SCENE)
    shutil.copytree(REPO / LOOP / "pool_diag" / "states", dest / LOOP / "pool_diag" / "states")
    shutil.copytree(REPO / HIDDEN / "pool_accept" / "states", dest / HIDDEN / "pool_accept" / "states")
    if arm == "A":
        env = dest / LOOP / "envelope"
        env.mkdir(parents=True)
        for f in ("sonic_v1_1.json", "sonic_v1_1_ok.json", "scalebfm_m22200_mode2_incumbent.json"):
            shutil.copy2(REPO / LOOP / "envelope" / f, env / f)
    git(dest, "init", "-q")
    files = git(dest, "ls-files", "-o", "--exclude-standard", "-z").stdout
    if any(f.startswith(("third_party", "outputs")) for f in files.split("\0")):
        raise SystemExit("third_party or outputs would be tracked; check .gitignore")
    git(dest, "add", "--pathspec-from-file=-", "--pathspec-file-nul", input=files)
    git(dest, "-c", "user.name=rsi-phase4", "-c", "user.email=rsi-phase4@localhost", "commit", "-q", "-m",
        f"phase 4 arm {arm} sandbox")  # fmt: skip
    print(f"arm {arm}: {dest} ({len(files.split(chr(0))) - 1} files tracked)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", choices=("A", "B"))
    ap.add_argument("dest", type=Path)
    a = ap.parse_args(argv)
    build(a.arm, a.dest.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
