# RSS 2027 论文：SelfAim

**SelfAim: Learned Self-Models for Aiming Frozen Humanoid Whole-Body Controllers**

- 投稿类型：RSS 2027 Stage 1，6 页 extended abstract，内容是假设、初步结果和实验计划。
- 截止时间：2026-12-04 AoE。
- 这个文件夹是论文写作会话（rss-35，原目录 `~/Research/rss/paper`）的全部内容，2026-11-27 并进 loco 仓库。
- 从这天起，**这里就是论文的正本**；原目录只是迁移前的副本，不再修改。

## 编译

```bash
cd rss2027_paper
latexmk -pdf -interaction=nonstopmode main.tex   # TeX Live 2026，pdfTeX
```

当前版本 5 页，0 个错误，0 个 overfull。

## 文件

| 路径 | 内容 |
|---|---|
| `main.tex` | 正文。每个数字旁有 `% src:` 注释写明出处；`\dnote{}` 是红色草稿批注，投稿前改 `\draftfalse` |
| `refs.bib`, `IEEEtran.cls`, `figures_preamble.tex` | 参考文献、模板、TikZ 图的公共设置 |
| `fig/teaser1_A.jpg`, `fig/teaser1_R.jpg` | 图 1 (a) 正在用的两张渲染 |
| `fig/teaser_v3/` | 图 1 v3 的原始渲染、README、`teaser_points.json`（图 1 (b) 的坐标从这里来） |
| `fig/teaser/`, `fig/teaser_v2/`, `fig/teaser_a.jpg`, `fig/teaser_b.jpg` | 被用户否掉的 v1、v2，留作记录 |
| `main_with_lml.tex` | 删掉 LocoManiLab (LML) 之前的完整版本 |
| `main_simple_with_heft.tex` | 以 SIMPLE 为主、还带 HEFT 和迁移部分的版本 |
| `main_pre_rewrite.tex` | 更早的版本 |
| `pending_updates.md` | 暂停期间收到、**还没写进 main.tex** 的结果和用户决定，按时间顺序 |
| `report.md` | bestpaper-writing skill 的作者报告（claim、claim map、图表计划等） |
| `template/` | RSS 官方模板原件 |
| `handoff/memory/` | 写作会话的长期记忆（工作规则、用户的框架决定、写作偏好） |
| `handoff/rss_status_snapshot_20261127.md` | 数字来源文件 `rss_status.md` 在 11-27 的**只读快照** |

## 工作规则（用户定的，接手前先读 `handoff/memory/`）

1. **数字只从 `rss_status.md` 和它链接的实验条目里取**，不凭记忆。
   - 正本在实验机器的 `~/research-notebook/paper/rss_status.md`，由"5090 机器 SSH 连接"那个会话维护。
   - 实验条目的镜像在本仓库 `docs/handoff/notebook-20261120/`。
   - 这里的快照不会更新，只用来对照。
   - 标"待复核"的可以先写，但要加批注；"已收回的说法"不能用；不要改实验条目。
2. **需要新数字或新实验，发给"5090 机器 SSH 连接"会话**，不自己跑。
3. **逐条更新暂停中**：5090 会话的工作做完后，问它要一次汇总，再一次性改 main.tex。期间收到的东西只记进 `pending_updates.md`。
4. **不写防御性内容**：不解释、不辩护没做的实验，也不把它们写成 limitation。
5. **不能出现的内容**：
   - 2×2 接口 × 求解器实验；
   - 常数偏移对照组；
   - SAM 3 那几轮；
   - 路线和自身模型"combine / stack"的说法；
   - LocoManiLab 的任何结果，包括只用了 LML 训练模型的结果，也不能改写成"another simulator"；
   - HEFT 步态不匹配只能留在内部批注里。
6. **SIMPLE 的复核措辞规则**：
   - 不写"each method fixes its own failure"，也不写"the route clears the edge by avoiding the obstacle"；
   - 不在 XMovePick、XMoveBendPick、BendPick 上比较 R_D 和 J / kin_rr；
   - R_D 在官方时限内的数字一律注明"计算时间不计入"；
   - 不用事后挑的三任务子集，只用登记的 P1。
7. **文字风格**按 bestpaper-writing skill：标题、摘要和引言里不用破折号；图 1 的图注不超过 100 词。
   - skill 不在本仓库里，在原机器的 `~/.claude/skills/bestpaper-writing`，需要单独拷到新机器。
   - 检查命令：`check_draft.py`（只对摘出来的摘要和引言跑）、`check_figures.py main.tex`。

## 当前状态（2026-11-27 中午）

**已在 main.tex 里**
- 以 SIMPLE 为主线，三条假设：H1 瞄准、H2 成功率、H3 部署。
- 表 I 是 ScaleBFM 和 SONIC 的 SIMPLE 结果；SONIC 的 R_D 格还是红色问号。
- 图 1 v3 已经放好：(a) 自建场景，往高处拿花瓶；(b) 按比例画的俯视图；(c) SIMPLE 柱状图。
- 计划部分是 E1（SONIC 在 SIMPLE 上）和 E2（三个控制器在 HumanoidArena 上）。

**已到、还没写进去**（详见 `pending_updates.md` 末尾）
- SONIC 在 SIMPLE 上的 R_D 已定稿，三次运行合并后**没测出差别**：45 s 内 89 → 103/360，p 0.50。
  - 写进去时要注明：这组的直接执行数和表 I 里的 29/18 不是同一次运行。
  - BendPick 的差不能算成模型带来的提高。
- ScaleBFM 在 HumanoidArena 上已经跑完，抓起并举起的成功数是直接执行 19/120、SelfAim 13/120（p 0.345），**没有提高**。
- 11-27 09:45 用户停掉了 HumanoidArena 的全部任务；HEFT 和 SONIC 在 HA 上不会再有结果。

**等用户决定**（用户 11-27 说"先不动，等我想好"）
- HumanoidArena 还留不留在论文里。
- 计划部分写什么：E1 已经跑完，E2 暂停，现在没有一个还没做的实验。跑过的实验不能写成"将要做的实验"。
- SONIC 的结果怎么放。

**其他待办**
- Method 里的成本数字：各控制器的仿真时间和 GPU 时间。
- SAW 的引用（arXiv 2409.10308）。
- 投稿前：`\draftfalse`，填 Paper-ID，精简参考文献。

## 没有放进来的东西

- `~/Research/rss/report_20261127/`：汇报 PPT 和视频（约 77 MB），不是写作会话做的。
- 写作会话的对话记录：里面有私人信息，状态已经整理在本文件和 `pending_updates.md` 里。
- bestpaper-writing skill：见上面规则第 7 条。

## 在新机器上接着写

1. clone 本仓库，切到 `sonic-selfmodel` 分支，进 `rss2027_paper/`。
2. 想让 Claude Code 带上原来的规则，把 `handoff/memory/*.md` 拷到新机器对应项目的 memory 目录（`~/.claude/projects/<项目路径>/memory/`）。
   - 记忆里写的路径 `~/Research/rss/paper` 换成本文件夹。
3. 先读 `pending_updates.md` 的最后几节，再读 `handoff/memory/rss-paper-workflow.md`。
