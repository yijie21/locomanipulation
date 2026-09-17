# 重构计划：阶段脚本扁平化（2026-09-17）

`/request-refactor-plan` 访谈结果。六个决定：a+b（局部性 + 概念少，Python 扁平脚本）、
实验配置全展开一文件、`stack.py` 保留为库并收紧、并行/跳过/注册表减到最简 + 保留资源准入、
按任务线分脚本 + `pipeline.py` 调度、验收 = GPU 闸门 + dry-run 送达测试 + 产物契约。

## Problem Statement

2026-09-16 的基础设施重构把"采集 → 转换 → 训练 → 评测"收敛成一个 CLI、七个 Hydra 配置组、
五个阶段模块、一个 Stack 类和一个 launcher。结果是：跑一次单臂采集，从命令行到专家进程之间
要经过 15 个文件、约 2200 行基础设施；配置分散在 29 个 YAML（732 行）里靠 Hydra 组合。
打开任何一个阶段文件都看不到全貌，改一个行为要在 4 层之间来回跳。

更严重的是它没有解决这个仓库的固有失败形态——"配置声明了，但没送到下游进程"。
2026-09-17 在评测线上一天查出 4 个这类缺陷：`eval.backend` 整块被阶段丢弃并替换（评测录像因此消失）、
只有 4 个字段被手抄给评测器（`spawn_edge` 因此从配置的 1.1–1.3 m 变成默认 0.7 m，100 局评测作废）、
初始 token 在阶段里硬编码、录像起点在 `[stand]` 阶段之后（结束最早的失败家族恰好拍不到）。
这些缺陷全部藏在"阶段 → 子进程命令行"这道缝里，而现有测试只覆盖进程内的消费者。

## Solution

每个阶段一个扁平的 Python 脚本，顶层顺序执行，读一份**全展开**的实验 YAML，把整份配置原样复制进
运行目录并原样交给下游进程；进程边界上不允许手抄字段。起栈仍是一个库（`stack.py`），但子进程环境
只在一处生成并落盘。没有 Hydra、没有 manifest hash、没有 launcher、没有注册表。
并行 = 一卡一个终端各跑一个 YAML；跳过 = 阶段产物已存在；资源准入 = 每个脚本开头 30 行检查，不够就拒绝。

验收分三层：GPU 闸门（采集行为不变）、dry-run 送达测试（YAML 里每个字段都出现在将要执行的
argv/env 里）、产物契约（重构前记录每个阶段产出的文件清单，重构后逐项比对）。

## Commits

每个 commit 都让仓库处于可运行状态；新路径和旧路径**并存**直到新路径通过闸门和契约，然后一次删除旧路径。

0. **落地 2026-09-17 的四个修复**作为基线：评测配置整份以 `--config` 交给评测器、`eval.backend` 不再被替换、
   录像从传送复位开始、`world_lift` 支持阶段目录布局、`check_dataset` 锚定 episode 结尾的那次闭合、
   eval 的 `sim_vis` 改回 `none`。带上 `test_stage_config_delivery.py`。重构从修好的基线开始。

1. **记录产物契约**。新脚本 `scripts/artifact_contract.py record <run_dir> <stage>`，把该阶段产出的
   文件清单（相对路径，按批编号的目录用通配）写成 `tests/contracts/<line>_<stage>.json`。
   从现有运行记录：单臂 collect/convert/train/eval（`bottle_simple_200_20260916`，eval 用 commit 0 之后
   带录像的那份）、双臂 collect（`crate_check_record20`）、双臂 eval（`crate_eval100_20260915` 的目录结构）。
   配套 `tests/test_artifact_contract.py`：给一个运行目录和一份契约，缺一个文件就失败。
   此时契约描述的是**旧路径**的产出，这是重构的验收标尺。

2. **写全展开的实验文件**。`configs/experiments/bottle_simple_200.yaml`、`bottle_simple_gate8.yaml`、
   `pick_place.yaml`、`crate_table30_record.yaml`。首版由 `locomani.cli show` 的组合结果导出再手工整理
   （去掉 Hydra 的 `defaults`、`schema_version`），字段一个不少。此时无消费者，只加不改。
   一个只读测试：每个实验文件的 task 块能被对应任务 dataclass 的 `from_dict` 接受（未知字段报错）。

3. **收紧 `stack.py`**。给 `Stack` 加一个从整份配置构造的入口（配置 dict + 运行目录 + GPU），
   子进程环境在**一个函数**里一次生成，并写进运行目录的 `stack_env.json`。旧的 `StackSpec` 构造路径保留、
   行为不变。测试：给定配置，生成的环境是确定的，且包含录像目录、渲染开关、编码器开关、实例端口。

4. **脚本公共部分** `scripts/run_common.py`，控制在 100 行内：读 YAML、复制到运行目录、资源检查
   （CPU 配额 / pid / 该卡显存，不够就 `SystemExit` 并打印原因）、`--dry-run` 的统一打印格式
   （每条将执行的命令：argv + 它专属的环境变量）。测试资源检查的判定函数（纯函数，喂假数据）。

5. **`scripts/collect_bottle.py`**。读实验 YAML → 复制场景 → 把 task 块 dump 成专家读的 JSON →
   按批：起栈、跑 `grasp_gym.py`、停栈 → 写 `collect/summary.json`。带 `--dry-run`。
   测试：dry-run 输出里，YAML 的 spawn / seed / successes / 录像目录 / 编码器开关逐项可见。
   **验收**：`bottle_simple_gate8.yaml` 跑 8 局固定库，`scripts/gate.py compare simple` 对冻结基线 PASS；
   产物契约 `bottle_collect` PASS。

6. **`scripts/convert.py`**。读上游 `collect/summary.json` 的 session/manifest 配对 → 调转换器 →
   GR00T stats → `check_dataset --run-dir`。dry-run 测试。
   **验收**：在 `bottle_simple_200_20260916` 的 session 上重跑，数据集 `meta/` 与现有 `train_dataset` 一致；
   契约 `bottle_convert` PASS。

7. **`scripts/train.py`**。读上游数据集路径 → 调 `train_bottle.sh`（环境变量从 YAML 的 train 块和 embodiment
   块生成）。dry-run 测试断言 MODALITY / MAX_STEPS / SAVE_STEPS / CUDA_VISIBLE_DEVICES。
   **验收**：10 步冒烟，契约 `bottle_train` PASS，冒烟 checkpoint 随即删除。

8. **`scripts/eval_bottle.py`**。起评测栈（`eval.backend`）→ 起策略服务 → 把 YAML 的 eval 块 dump 成 JSON
   以 `--config` 交给 `eval_closed_loop.py` → 停栈。dry-run 测试断言 spawn_edge、initial_token、
   video_include_init、SIM_VIDEO_DIR 全部送达。
   **验收**：3 局，契约 `bottle_eval` PASS——**录像目录非空**是这条契约的核心断言。

9. **`scripts/collect_crate.py`**。用 `tasks/crate/prepare.py` 准备场景、按 `task.expert` 选专家、
   按批起栈跑专家。dry-run 测试。**验收**：20 局 ready-stand，`gate.py compare crate` PASS；契约 `crate_collect` PASS。

10. **`scripts/eval_crate.py`**。只是调 `eval_crate_closed_loop.py`（它自己起栈和策略服务），把 YAML 的
    eval 块和运行目录传过去。dry-run 测试。**验收**：3 局，契约 `crate_eval` PASS。

11. **`scripts/pipeline.py`**。读 YAML 的 `task.plugin` 选脚本，按 collect → convert → train → eval 顺序
    `subprocess.check_call`；每个阶段"产物存在则跳过"。`--from <stage>` / `--to <stage>` 两个开关，没有别的。
    **验收**：全链路冒烟（1 集、10 步、2 局）通过；`--dry-run` 打印四条命令。

12. **删除旧路径**：`cli.py`、`conf.py`、`stages/`、`launch.py`、`embodiment_config.py`、`scene_config.py`、
    `collect_config.py`、`train_config.py`、`configs/{task,scene,embodiment,backend,collect,train,eval,experiment,jobs}/`、
    `outputs/registry.jsonl` 的写入、`pipeline/stack.py` 的 `resolve_env`（并入 `stack.py`）、
    以及 `test_conf`、`test_stages`、`test_launch`、`test_stack_delivery` 里测这些对象的用例。
    `Makefile` 的 `run`/`cli` target 改指 `pipeline.py`。提交钩子的全量 CPU 套件通过。

13. **文档**：`docs/infra.md` 重写（一个 YAML、六个脚本、并行就是多开终端）、README 快速开始、
    三份采集教程和训练/评测教程的命令、`reference/runbook.md`、`reference/change-map-runtime.md`、
    `config_fields.md` 重新生成、`STATE.md`。`lint_repository.py` 通过。

14. **收尾**：本文件记录进度和实测数字；记忆更新。

## Decision Document

- **脚本形态**：每个阶段一个 Python 脚本，顶层顺序执行，读起来是一条直线；bash 只保留 `paths.sh`
  这种环境设置。不用 Hydra，配置就是 `yaml.safe_load` 出来的 dict。
- **配置**：一个实验一个全展开的 YAML，包含 task / scene / embodiment / backend / collect / train / eval
  全部字段。新实验 = 复制一份改几行。脚本启动时把它**原样**复制到运行目录，这份就是 provenance，
  也是下游阶段读的那份。"这个值来自哪个文件"的答案永远是"运行目录里那一份"。
- **进程边界纪律**：脚本把整块配置（task 块 → 专家的 JSON、eval 块 → 评测器的 `--config`）交给下游，
  不手抄单个字段。唯一允许的例外是 `dataset_fps`（评测栈钉在数据集的 25 fps）。
- **校验**：脚本读自己那块配置时用现有 dataclass 的 `from_dict`（`BottleTaskConfig`、`CrateTaskConfig`、
  `SonicBackendConfig`、`EvalConfig`），未知字段照旧报错。`config_base.py` 因此**保留**（它们都依赖它）。
  运行级 schema（`RunConfig`、`EmbodimentConfig`、`SceneConfig`、`CollectConfig`、`TrainConfig`）删除，
  它们的字段直接成为 YAML 里的普通键。
- **Stack**：保留为库，地位等同于 `grasp_gym.py`——阶段脚本调它三行，不需要打开它。接口收紧为
  "从整份配置构造"，子进程环境在一处生成并落盘 `stack_env.json`。`StackSpec`/`resolve_env`/`child_env`
  三处各自决定子进程看到什么的现状结束。
- **并行**：一张卡一个终端各跑一个 YAML。没有 jobs 文件、没有 launcher、没有 GPU 分配。
- **跳过**：阶段产物存在就跳过（数据集目录、checkpoint 目录、`closed_loop.json`）；要重跑就删该产物。不算 hash。
- **注册表**：删除。对比实验 = 看各运行目录的 `summary.json`。
- **资源准入**：保留为每个脚本开头的检查——CPU 配额（`cpu.max`，不是 `nproc`）、pid 数（容器上限 3840，
  一个录制栈约 1800）、该卡空闲显存。不够就拒绝启动并打印原因，**不等待**。
- **任务线**：`collect_bottle` / `eval_bottle`（pick_place 也走这两条，同一专家同一 gym）、
  `collect_crate` / `eval_crate`、`convert` / `train` 两条线共用。`pipeline.py` 按 `task.plugin` 分派。
- **运行目录**：沿用 `outputs/runs/<name>/`，阶段产物路径固定（`collect/`、`train_dataset/`、`model/`、`eval/`），
  阶段之间靠固定路径 + `collect/summary.json` 认识彼此，不再有 manifest。
- **专家接口不变**：专家仍从运行目录读 `task_config.json` / `reach_config.json`；评测器仍用 `--config` + `--set`。
- **dry-run**：每个脚本 `--dry-run` 不起任何进程，打印将执行的每条命令（argv + 专属环境变量）。
  这是送达测试的观测点，也是人看"这次到底会跑什么"的入口。

## Testing Decisions

- **好的测试只看外部行为**：给一份 YAML，断言脚本**将要执行**的命令里有什么（dry-run 输出），
  以及跑完后目录里**有什么文件**（产物契约）。不测内部函数怎么拼字符串。
- **三层验收**，各抓一类问题：
  1. GPU 闸门（`scripts/gate.py`，现有）：单臂 8 局固定库对冻结基线 `baseline_simple.json`，
     双臂 20 局对 `baseline_crate.json`——抓"采集行为变了"。这是上次重构唯一的验收，它的盲区是 eval。
  2. dry-run 送达测试（新，`tests/test_dry_run_delivery.py`）：对每个脚本、每份实验文件，断言 YAML 里
     每个非默认字段出现在 argv 或 env 里——抓"配置声明了没送到"，即 2026-09-17 那 4 个缺陷中的 3 个。
     先例：`tests/test_stage_config_delivery.py`（commit 0）、`tests/test_eval_config_reaches_the_client.py`。
  3. 产物契约（新，`tests/test_artifact_contract.py` + `tests/contracts/*.json`）：重构前从旧路径记录
     每阶段文件清单，重构后逐项比对——抓"以前有的东西没了"，即评测录像消失那一类。
     契约在 commit 1 记录，之后每个脚本 commit 都要过它。
- **被测模块**：六个阶段脚本 + `pipeline.py`（dry-run 与契约）、`stack.py` 的环境生成（确定性 + 关键键存在）、
  `run_common.py` 的资源判定（纯函数）。
- **不测的**：专家、转换器、训练脚本、评测器内部——范围之外，它们各有自己的测试。
- **闸门噪声**：crate 的 `stand_off_m` 同一份代码两次可差 1.8 倍容差；闸门 FAIL 先在同一代码上复跑一次再判。
  单臂 8 局批次间有 ±1 局波动，比较失败集合而不是总数。

## Out of Scope

- 专家进程及其配置 dataclass：`grasp_gym.py`、`round_phases.py`、`crate/expert.py`、`bimanual_*.py`、
  `BottleTaskConfig`、`ReachConfig`、`CrateTaskConfig`。单臂专家冻结在 `stage4_27_wrap`。
- `convert_to_hf_format.py`、`check_dataset.py`、`world_lift.py`、`train_bottle.sh`、`serve_policy.py`、
  `eval_closed_loop.py`、`eval_crate_closed_loop.py` 的内部——脚本只调它们。
- `gate.py` 和 `tests/gate/` 基线。
- 2026-09-17 评测暴露的控制层根因（无编码器的评测栈下 deploy 前奏不接管 / 复位后倒地 / 走偏 0.5 m）。
  那是控制问题，重构不碰；重构后用新脚本继续查。
- 评测预览（`previews/`）：仓库里没有代码生成它，是按 `docs/rendering.md` 的人工导出步骤，不进契约。
- ScaleBFM 线（独立的进程内 runner）。
- Isaac 离线渲染流程。

## Further Notes

- **顺序原则**：先记契约（commit 1），再写新脚本，最后删旧路径（commit 12）。新旧并存期间两条路径都能跑，
  闸门可以在同一份代码上对两条路径各跑一次做 A/B。
- **配置"复活"陷阱**：把"声明但未送达"的字段接通本身就是行为变更。2026-09-17 修好 `eval.backend` 送达后，
  旧 JSON 带来的 `sim_vis: full` 首次生效，仿真图像发布率 33 → 19 Hz，机器人复位即倒地。
  commit 2 写实验文件时，每个字段都要问一句"这个值现在真的生效吗，还是从没生效过的陈年默认"。
- **删除纪律**：commit 12 前 `git diff --name-only main..HEAD` 检查没有 `outputs/` 或 symlink 混入；
  按名暂存，不用 `git add -A`。
- **运行前提**：单臂 streamed-pose 专家要求打过夹爪交接补丁的 deploy，`SONIC_DEPLOY_ROOT` 指向
  `outputs/bottle_continuous_20260914/deploy`；这个前提写进 `run_common.py` 的启动检查，缺了就拒绝。
- GitHub issue：本机没有 `gh`，计划先落在这里；需要 issue 时把本文粘贴过去。

## 进度

- **commit 0** `40aaed0`：四个送达修复 + 审计两处（世界 z、闭合锚点）落地，`eval_stack()` 抽出使这道缝
  可在无 GPU 下测试；全量 CPU 套件通过。计划文档 `4c09802`。
- **commit 1**：`scripts/artifact_contract.py`（record / check，~90 行）与六份契约
  `tests/contracts/*.json`。基准来源：单臂四阶段取自 `outputs/runs/bottle_simple_200_20260916`
  （eval 用 commit 0 之后带录像的那次 3 局），双臂 collect 取自 `outputs/runs/crate_check_record20`，
  双臂 eval 由 `outputs/crate_eval100_20260915` 的目录结构映射到阶段布局后手写——该参考目录的 mp4
  已在 2026-09-17 清理中删除，`videos_*/*.mp4` 依据当时的清单（videos_00..09 共 1.7 GB）补回。
  五份可自动记录的契约都已对各自基准运行 `check` 通过。
  契约只查"有没有"，不查内容；每份契约声明一条 `CRITICAL` 产物（例如 `eval/videos/*.mp4`），
  删掉它必须是明确动作而不是顺手。`test_artifact_contract.py` 21 项通过。
- **commit 2**：`configs/experiments/` 四份全展开文件（bottle_simple_200 391 行、bottle_simple_gate8 391、
  pick_place 390、crate_table30_record 192），由 `locomani.cli show` 导出后逐块核对。
  **等价性已钉住**：四份文件的 task / backend / eval / collect / train 五个块与 Hydra 组合结果逐字段相等
  （`test_experiment_files.py` 的过渡用例，随 `conf.py` 一起在 commit 12 删除）——因此闸门在任一路径上的
  结果对另一条路径同样成立。
  字段审计的结果：删掉 `embodiment.{arms,state_dim,action_dim,cameras,dataset_kind}` 和
  `scene.{name,preset}`（全仓库无消费者，只有 `name`/`gr00t_tag`/`gr00t_modality_config`/`dir`/`cases`
  被读），以及根字段 `stage`/`out`/`seed`（`out` 由 `name` 推出，`gpu` 改为命令行参数）。
  严格 `from_dict` 当场抓到一处：`task.reach` 必须带 `schema_version`，`cli show` 的输出把它剥掉了——
  这正是"展开后逐字段校验"的价值。
- **commit 3**：`StackSpec.settings()` 成为**唯一**决定子进程环境的函数（`child_env` 只剩"继承 + 覆盖"
  两步，并清掉可能从环境里漏进来的 `SIM_VIDEO_DIR`/`SONIC_DDS_DOMAIN`）；`Stack(run_dir=...)` 在 `up()`
  时把它写成 `<run>/stack_env.json`，事后还能回答"子进程到底被告知了什么"。
  新入口 `spec_from_config(cfg, run_dir, gpu, backend=...)`：整份实验配置进，一个 `StackSpec` 出；
  `backend` 显式传块而不是读固定键——采集栈要编码器、评测栈必须关编码器，这个差别留在调用处可见。
  旧的 `StackSpec(...)` 构造和 `resolve_env` 行为不变（`test_stack_delivery`、`test_stack_start_failure` 照旧通过）。
- **commit 4**：`scripts/run_common.py`（123 行，超出 100 行目标 23 行——多出来的是注释里那几条实测教训，
  没有为了凑数删掉）：`open_run`（读实验文件 + **逐字节复制**到 `<run>/config.yaml`，同名再跑但文件变了就拒绝）、
  `measure`/`shortfall`/`require_headroom`（CPU 配额 1 秒 delta、pid 余量 200、该卡空闲显存；
  **拒绝而不等待**，读不到的数不作为拒绝理由）、`require_deploy_patches`、`show`（dry-run 统一格式）、
  `summarise`。
  **与计划的一处偏差**：阶段摘要写成 `<run>/summary.json` 的一个键，而不是 `collect/summary.json`——
  这样跨运行对比就是 `ls outputs/runs/*/summary.json`，同时下游阶段照样能读上游那条记录。
  `test_run_common.py` 11 项：pid 上限卡在边界上（3840−200−1800 恰好通过，多一个就拒绝）、
  显存不足只拒绝训练不拒绝评测、读不到的测量不拒绝、改了文件的同名重跑被拒。
- **commit 5**：`scripts/collect_bottle.py`（198 行，顶层一条直线）+ `tests/test_dry_run_delivery.py`（7 项）。
  **验收**：`bottle_simple_gate8.yaml` 在 GPU0 上跑 8 局固定库 **6/8**，
  `gate.py compare simple` 对冻结基线 **PASS 5/5**（yield 0.75→0.75，停位中位 0.1469→0.1506 m，失败家族无新成员）；
  `artifact_contract.py check bottle_collect` PASS。
  dry-run 当场抓到一个新引入的 bug：`spec_from_config` 用了 `StackSpec` 的默认 deploy 路径（树内未打补丁的二进制），
  而旧路径让 `SONIC_DEPLOY_ROOT` 优先——已改回同样的优先级。资源准入也生效了一次：GPU1 被别的项目占了 31.9 GB，
  `require_headroom` 直接拒绝（`689 MiB free < 6000 needed`），改用 GPU0。
  **契约按计划被"故意修改"了四条**（每一条都是明确动作）：`collect/manifest.json` 和
  `run_config.{json,yaml}` 由 `<run>/summary.json` + 逐字节的 `config.yaml` 取代，新增 `stack_env.json`；
  `scene/scene_locomani_zone.xml` 删除——那是仿真**只在开可视化时**生成的文件，参考运行是在昨天
  `sim_vis=full` 的评测诊断里顺带留下的，属于记录契约时没筛掉的偶然产物。
  已知行为（沿用旧路径）：固定库运行保留数少于 `successes` 时脚本以非零退出，闸门运行因此总是"失败退出"。
- **commit 6**：`scripts/convert.py`（125 行）+ `run_common.load_run_config`（下游阶段只读运行目录里的
  `config.yaml` 副本）+ 5 项送达测试。
  **验收**：在 200 集那 30 个 session 上用新脚本重转，`meta/` 下 **10 个文件逐字节相同**
  （含 `gr00t.data.stats` 产的 `stats.json`、`relative_stats.json`），190 个 parquet、570 个视频数量一致；
  两个 split 的审计各 **32 项全过**；`artifact_contract.py check bottle_convert` PASS。
  验收时给旧路径产生的运行目录补了两样桥接物并已记录：`config.yaml`（直接拷贝 `bottle_simple_200.yaml`，
  commit 2 的等价性测试已证明它与 Hydra 组合结果逐字段相等）、`summary.json` 的 collect 条目（由
  `collect/inputs.json` 转写）。
  **抓到一个真 bug**：`check_dataset.py` 由 data-collection venv 启动、在 `--run-dir` 分支里 `import locomani`，
  一直依赖**上游隐式继承的 `PYTHONPATH`**（旧路径由 launcher 设在阶段进程上）。扁平脚本没有那份隐式环境，
  于是在转换完成、跑了 12 分钟之后才炸。修法是脚本显式声明子进程环境，并在 dry-run 里打印、由测试钉住。
  记下未处理的一项：`Stage.check` 的日志是**追加**而非覆盖，`audit_train.log` 里叠着多次运行的输出，读日志容易误判。
- **commit 7**：`scripts/train.py`（97 行，只做一件事：把 `train`/`embodiment` 两个块变成
  `train_bottle.sh` 读的环境变量）+ `configs/experiments/bottle_simple_smoke.yaml`（与 200 集那份**只差 8 个值**，
  commit 11 的全链路冒烟复用它）+ 4 项送达测试。
  **验收**：10 步冒烟在 GPU1 上跑通（10 步 36 秒，加载与保存是大头），落下 `model/checkpoint-10`，
  `artifact_contract.py check bottle_train` 对冒烟目录和 200 集目录**都 PASS**，随后删除该 24 GB 冒烟产物。
  契约按同样的方式**故意改了一条**：`train/manifest.json` 由 summary 里的 train 条目取代（checkpoint 路径、
  步数、训练所用数据集）。
- **commit 8**：`scripts/eval_bottle.py`（142 行）+ 5 项送达测试。
  **验收**：200 集那个 checkpoint 上跑 3 局（含重试共 6 次尝试）**0/6**（控制层根因未修，属范围外），
  但 `eval/videos/` 里有 **6 段录像**——`artifact_contract.py check bottle_eval` PASS，
  而这条契约存在的理由就是这个目录曾经空了一整天。契约同样故意改一条：`eval/manifest.json` → summary 的 eval 条目。
  配置改动：瓶子线的 `eval.video_include_init` 改为 true（`[stand]` 是结束最早、也是唯一拍不到的失败家族），
  并**镜像回** `configs/eval/closed_loop_100.yaml` 保持新旧路径等价（crate 那份一并对齐）。
  过程中两处自查：批量替换误改了 `task` 块里的同名字段，被等价性测试当场挡住；`--dry-run` 原本会创建运行目录
  并复制场景，现在 `open_run(write=False)` 之后**一个字节都不写**——测试留下的 `outputs/runs/delivery_check`
  正是靠这条被消除的。
- **commit 9**：`scripts/collect_crate.py`（161 行）+ 3 项双臂送达测试；`crate_collect` 契约同样故意改三条后 PASS。
  **验收（重要，结论与计划预期不同）**：20 局 ready-stand 两次：**16/20** 与 **12/20**。
  `gate.py compare carry` 分别 3/4 与 2/4，两次都只挂在 `failure_reasons` 上，且**每次多出来的家族都不一样**
  （第一次 `basket_displaced_before_capture`，第二次 `reach_stalled`，且前者没有复现）。
  同一代码血统的四份 20 局样本产出率 12–17，说明**这条线的 20 局样本判不了回归**——与记忆里
  "10-round pilots cannot measure this line" 和 "crate gate FAIL 可能是纯噪声" 一致；
  另需注意 `baseline_crate.json` 来自 **wall** 诊断专家，visual 线要对 `baseline_carry.json`（上次重构也是这么做的）。
  因此双臂的等价性改用**确定性证据**：`test_both_paths_build_the_same_stack` 断言四份实验在新旧两条路径下
  构造出的 `StackSpec` **20 个子进程环境变量与全部关键字段逐项相同**（实测 crate 线 0 差异）。
  连续量也一致：`end_depth_m` 基线 1.3306 → 1.3215 / 1.3295，`exit_majority` 两次都 PASS。
- **commit 10**：`scripts/eval_crate.py`（116 行）+ 3 项送达测试。**按用户决定不跑真实评测**
  （双臂 checkpoint 已在 2026-09-17 清理中删除），只保证逻辑一致；`crate_eval` 契约因此明确标注
  **UNVERIFIED**，并由一条测试钉住这个标注——第一次真实双臂评测必须验证它。
  写脚本时发现旧路径在双臂线上也有"声明未送达"：评测器接受 `--port`/`--budget-s`/`--execution-horizon`/
  `--initial-token-file`，而旧 stage 一个都没传，全用评测器默认值。更要紧的是共享的 eval 组让 crate 线
  带上了**单臂的开局 token**；新增 `configs/eval/crate_closed_loop.yaml`（crate token、端口 5700、
  spawn 1.2–1.5、`pick up the crate`）并让实验覆盖它，新旧两条路径继续逐字段相等。
  顺带验证了 provenance 规则：旧运行目录 `crate_flat_20260917` 的 dry-run 仍按它**自己那份** `config.yaml`
  输出端口 5590 和单臂 token——运行读的永远是它开始时的副本。
- **commit 11**：`scripts/pipeline.py`（88 行：按 `task.plugin` 分派、产物存在则跳过、`--from/--to`）+ 2 项分派测试。
  **验收**：`bottle_simple_smoke.yaml` 全链路通过——采集 2 批留 1 集 → 转换 1 集 495 帧 → 训练 10 步 →
  评测 4 次尝试 0 成功（冒烟只测通路）；`bottle_collect` / `bottle_convert` / `bottle_train` / `bottle_eval`
  四份契约全 PASS，随后删除 24 GB 冒烟 checkpoint。
  **资源准入第三次拦住真问题**：起跑时另一个项目（`real2sim_pipeline` 的 `remove_movables`）占了约 14.5 核，
  准入拒绝（`15.38 + 8 > 15.36`）。没有去动别人的进程，改成"等有余量自动开跑"，CPU 一空冒烟就自己跑完了。
  契约拆分：`bottle_convert` 与新的 `bottle_convert_val` 分开——验证集只在 `train.holdout > 0` 时存在，
  而冒烟实验刻意设为 0（集数太少），两种形态因此都可检查。
- **commit 13**：文档全面改写。`docs/infra.md` 按现状重写（一份实验文件、六个脚本、并行=多开终端、
  "删产物即重跑"）；README 快速开始、三份采集指南、训练/评测/数据集指南、runbook、standards 的 run 规则、
  silent_failures 的入口表全部跟进。**当前文档里不再出现组合 CLI、launcher 或 jobs 文件**；
  两份重构计划作为历史保留并在索引中链接。
- **commit 12**：删除组合路径，65 个文件、−2318 行。删掉：`cli.py`、`conf.py`、`launch.py`、`stages/`、
  四个运行级 schema、`configs/{experiment,jobs,embodiment,scene,collect,train}/`、
  `test_conf/test_stages/test_launch/test_stage_config_delivery`、两条过渡等价性测试、
  两个只为省敲命令而存在的 shell 包装（`run_wall_pilot.sh`、`run_bimanual_pilot.sh`）；`make cli` 改为 `make run`。
  **与计划的两处偏差（均已核实）**：
  (1) `configs/{task,backend,eval}/` **保留**——它们不是组合路径的输入，而是专家、评测器和 17 个测试
  按名字引用的 profile 库。新增 `src/locomani/profiles.py`（48 行，纯 YAML + 各自 dataclass 的 `from_dict`，
  不依赖 Hydra）取代 `conf.load_group`，18 个调用点已迁移；顺带给 5 个瓶子任务档补上
  `reach.schema_version`（此前靠 Hydra schema 提供，裸读会被拒）。
  (2) `pipeline/stack.py` **保留**——两个评测器直接调 `start_stack`/`stop_instance`，合并它就要改范围外的评测器。
  `test_pose_ownership` 里"起资源前必须拒绝未打补丁的 deploy"那条改写到 `collect_crate.py` 上，行为照旧被钉住。
  验收：全量套件 rc=0、`make lint` PASS（含 change-map 链接与 `config_fields.md` 重新生成）。

## 收尾（commit 14）

**十四个 commit 全部落地**，每一步都有验收，工作区干净，全量 CPU 套件与 `make lint` 通过。

### 实测数字

| 阶段 | 验收 | 结果 |
|---|---|---|
| collect（单臂） | 8 局固定库 + `gate.py compare simple` | 6/8，**PASS 5/5**（yield 0.75→0.75，停位 0.1469→0.1506 m，失败家族无新成员） |
| convert | 200 集 30 个 session 重转，与旧路径比对 | `meta/` **10 个文件逐字节相同**，190 parquet / 570 视频一致，两个 split 审计各 32/32 |
| train | 10 步冒烟 | `model/checkpoint-10` 落盘，契约 PASS，24 GB 产物随即删除 |
| eval（单臂） | 3 局闭环 | 0/6（控制层根因在范围外），**6 段录像**——契约要护的就是这个目录 |
| collect（双臂） | 20 局 ready-stand ×2 | 16/20 与 12/20；闸门 3/4 与 2/4，两次多出的家族各不相同 |
| eval（双臂） | 按用户决定不跑 | 契约标注 **UNVERIFIED**，由测试钉住 |
| 全链路 | `pipeline.py` 冒烟 | 采集 1 集 → 转换 495 帧 → 训练 10 步 → 评测 4 次尝试，四份契约全 PASS |

### 这次重构真正买到的东西

不是"层数变少"（那是结果），而是**三类以前无人看守的问题现在有了看守**：

1. **配置声明了没送达** → 每个脚本 `--dry-run` 打印将执行的命令与环境变量，`test_dry_run_delivery.py`
   逐字面断言。这一类问题 2026-09-17 一天内出现四次（`eval.backend` 被丢、`spawn_edge` 变成 0.7、
   token 硬编码、录像拍不到 stand），全部由这条防线复现并锁住。
2. **产物悄悄不再产生** → `tests/contracts/*.json` 从旧路径录下"一个阶段必须留下什么"，
   重构过程中共**故意修改 11 条**，每一条都是明确决定而不是顺手。
3. **资源不足硬跑成"机器人故障"** → `require_headroom` 在三次真实场合拒绝启动
   （GPU1 被别的项目占满 31.9 GB、未打补丁的 deploy、CPU 配额被别的项目吃掉 14.5 核）。

### 已知未了事项

- 双臂评测契约未经真实运行验证（checkpoint 已删）；用户手动 eval 时应跑
  `python scripts/artifact_contract.py check outputs/runs/<name> crate_eval`。
- 评测栈下 deploy 前奏不接管 / 复位后倒地的**控制层根因**仍未解决，属重构范围外。
- `Stage.check` 的日志是追加而非覆盖，同一运行目录里多次运行的输出会叠加。
- 固定库运行保留数不足时脚本非零退出，闸门运行因此总是"失败退出"（沿用旧语义）。
