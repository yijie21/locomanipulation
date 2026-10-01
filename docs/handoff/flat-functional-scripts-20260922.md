# 重构计划：bash 直接调功能脚本（2026-09-22）

`/grilling` 访谈结果，16 条决策。阶段 1–8 已于 2026-09-22 全部落地（见「阶段」与「进度」）；遗留项见根目录 TODO 1.14–1.20。

## Problem Statement

协作者报了四条重复：`collect_data.sh` 与 `start_stack.sh` 是同一份 bash 只差 `--mode`；
`run_vla_server.py` 与 `serve_policy.py`、`run_eval_single_arm.py` 与 `eval_closed_loop.py`、
`run_eval_bimanual.py` 与 `eval_closed_loop.py` 为什么不合并。

2026-09-22 第一次尝试把四个 driver 合成 `run_collect.py` / `run_eval.py` 两个共享 driver，
被用户否决并全部回退。否决理由是这次修改**加了一层**：共享 driver 骑在两条线的功能代码上面，
而协作者报的四条正是"层太多"的症状，不是"文件太多"的症状。

真正的问题比报告的更老：

- **四个 driver 在自己已经在跑的那个解释器里再 spawn 一个子进程。**
  `run_collect_single_arm.py` 由 `$VENV_SIM_PY` 启动，它 spawn 的是 `$VENV_SIM_PY grasp_gym.py`。
  两个 eval driver 同理。这一跳不跨 venv，不买任何东西。只有 `run_convert`、`run_train`、
  `run_vla_server` 跨到 Isaac-GR00T 的 venv，那三跳是物理必需。
- **这一跳已经制造过真 bug**：脚本以 `__main__` 跑、而别的模块按包名 import 它，会产生第二份模块副本，
  `--set wall_walk=true` 因此静默跑了错的变体（当时用 `sys.modules` 别名修的）。
- **仓库里已经有一处是对的**：`pipeline/eval_closed_loop.py:194` 直接
  `from locomani.tasks.bottle import grasp_gym` 并进程内调 `grasp_gym.main(...)`。采集路径没跟上。

## 规则

> **任何一个文件，如果它的 body 就是「给下一个文件拼参数」，它就不该存在。**

库（import 进来调函数）不算层；进程内组合不算层；**拼 argv 再 spawn 的中间人**才算。
配套的可检查形式：**只有跨 venv 边界才准 spawn 子进程**（最后一个阶段写进 `lint_repository.py`）。

## 决策

1. **范围**：collect 和 eval 的整个 driver 层删除（`run_collect_single_arm.py`、`run_collect_bimanual.py`、
   `run_eval_single_arm.py`、`run_eval_bimanual.py`）。`run_stack.py` 保留——它的 body 在构造两种
   stack spec，是真内容不是拼参数。
2. **「直接调」= 进程内**，不是「合成一个文件但仍然 spawn」。
3. **功能脚本直接读运行目录的 `config.yaml`**，不再由上游拼 20 个 flag 递下来。
4. **receipt**：保留 per-stage 的 `.log` 与 `timings.jsonl`；`commands.jsonl` 的 payload 从 argv 换成
   已解析的配置。不为了留 receipt 而保留子进程。
5. **跨 venv 的三个阶段（convert / train / serve）保持原样**，docstring 写明这一跳存在的理由。
6. **批次循环与 exporter 管理抽成库 `pipeline/batches.py`**，两条线各自 import。
   硬约束：该模块**不准 import `tasks/` 下任何东西，也不准拼 argv**，只做 session 开关与计数——
   这样它没法退化成 driver。
7. **两个 bash 脚本，各自 exec 自己那条线的功能模块**。不做「一个 bash 读 YAML 再分派」：
   那需要起一个 python 进程来解析，等于为省一个文件又加一次调用。
8. **`eval_closed_loop.py` 算功能脚本**，bash 直接调它。它的 body 是组合（把 VLA policy 装进 gym 的
   round loop、算 Wilson 区间、做失败分期），不是拼参数的中间人。
9. **`pipeline/lines.py` 保留但缩小**：只有 `run_stack.py` 用它（起栈必须支持两条线）。
   功能脚本不准再查这张表——它们已经知道自己是谁。
10. **`--dry-run` 改为打印解析后的配置**，加上仍会启动的子进程。`test_dry_run_delivery.py` 中
    collect/eval 的 argv 断言删除，换成配置 dataclass 的 round-trip；
    **stack / convert / train / serve 的断言原样保留满强度**——那里的缝还在。
    要点：测试变弱不是退步，是因为它防的那道缝被物理删除了。
11. **`grasp_gym` 的轮次循环改成接配置对象**；`main(argv)` 保留为薄解析边界，诊断脚本照常可用。
    `eval_closed_loop` 里那段为进程内调用拼 argv 的代码随之消失。
12. **验收**：新旧路径并存，新路径先过 `scripts/gate.py compare simple` / `compare crate` 的 GPU 闸门
    与产物契约，通过后一次性删旧路径。不接受「`make check` 绿就算过」——`--until-ok` 这类 flag
    悄悄掉了，CPU 测试发现不了，要等一整批采集完才看得出来。
13. **先写方案再实施**，分阶段、每阶段单独可运行、单独提交。
14. **collect 入口放 `tasks/<线>/collect.py`**（`tasks/bottle/prepare.py` 已是同形状的先例）。
    不放 `pipeline/`：放那儿它就会重新变成「对两条线通用的东西」，然后长回 driver。
    放在线目录下它**只能**认识自己这条线，是结构性防护而不是纪律要求。
    也不复用 `grasp_gym.py` 的 `__main__`：那会让一个有 4 个进程内调用方的现有入口含义静默改变。
15. **协作者的第 0 条一起做**：`collect_data.sh` 与 `start_stack.sh` 合成
    `bash scripts/start_stack.sh collect|eval <target>`。它与分层问题正交，两者都 exec 保留下来的
    `run_stack.py`。
16. **阶段顺序从库开始**，因为第一阶段是纯搬运、零行为变化、不占 GPU，不必等 GPU 窗口就能落地。

### 被否决的方案

- **共享 driver `run_collect.py` / `run_eval.py` + `lines.py` 分派表**（2026-09-22 第一次尝试，已回退）：
  减少了文件数，却在功能代码上又加一层。六条命令的 `--dry-run` 字节相同、`make check` 全绿，
  但方向错了——**验证通过不等于设计正确**。
- **一个 bash 读 `task.plugin` 分派**：见决策 7。
- **为了让 `--dry-run` 和那 30 条断言活着而保留子进程**：本末倒置（决策 4、10）。
- **`make check` 绿即验收**：见决策 12。

## 阶段

| # | 内容 | 需要 GPU | 状态 |
|---|---|---|---|
| 1 | 抽 `pipeline/batches.py`（纯搬运，旧 driver 改为调它） | 否 | 已完成 |
| 2 | `collect_data.sh` + `start_stack.sh` 合并（决策 15） | 否 | 已完成 |
| 3 | `grasp_gym` 轮次循环接配置对象，`main(argv)` 变薄（决策 11） | 否 | 已完成 |
| 4 | `tasks/bottle/collect.py` 新入口，与旧 driver 并存 | 闸门 | 已完成（闸门 PASS 5/5） |
| 5 | `tasks/crate/collect.py` 同上（仅 `visual`） | 闸门 | 已完成（闸门 PASS 4/4） |
| 6 | eval 两条线：bash 直接调 `eval_closed_loop.py` / `eval_crate_closed_loop.py` | 跑通 | 已完成 |
| 7 | 删旧 driver、`--dry-run` 改形、测试改形、文档 | 否 | 已完成 |
| 8 | `lint_repository.py` 加「只有跨 venv 才准 spawn」规则 | 否 | 已完成 |

第 8 阶段最后做的理由（已验证）：太早加会把还没迁完的旧路径全判红。

## 不在范围

`eval_closed_loop.py`（438 行）与 `eval_crate_closed_loop.py`（365 行）仍是并行演化的两套评测器
（协作者第 2、3 条指向的深层问题）。它们的 round loop、失败分期、resume 机制都不同，合并会动到
产出 `closed_loop.json` 的路径——台账里每个数字的来源。需要 checkpoint 和 GPU 闸门才能动，
不能只靠 CPU 测试。

## 进度

- **2026-09-22 阶段 1**：`pipeline/batches.py` 抽出 `new_session` / `open_exporter_session` /
  `alive_exporter` / `kept_in`，两个 collect driver 改为 import 它。
  顺带解除了 `run_collect_bimanual` 直接 import `run_collect_single_arm` 的跨线耦合。
  零行为变化，`make check` 全绿。
- **2026-09-22 阶段 2**：`collect_data.sh` 删除，`start_stack.sh` 接 `collect|eval` 位置参数。
- **2026-09-22 阶段 3**：`grasp_gym.run(options, runtime=)` 新增（解析之后的全部工作），
  `main(argv)` 缩成「解析 → run」。`eval_closed_loop` 不再为一个永远不会启动的进程拼 argv：
  它直接建选项集，经 `cli_args.resolve_options`（`main` 用的同一个合并函数）解析，
  resume 的 seed/rounds 偏移从改 argv 列表变成传参。
  选项集提到模块级 `gym_options()`，新测试 `tests/test_eval_gym_options.py`（7 条）钉住
  与 gym 默认值不同的那几项——掉一个键不会报错，只会让 100 局悄悄跑在 `spawn_lateral` 0.35 上。
  中性证据：7 组配置下「旧 argv → `parse_args`」与「新选项集 → `resolve_options`」解析出的
  `GymOptions` 完全相同（含 `sources`）；六条 driver 的 `--dry-run` 仍逐字节相同。

- **2026-09-22 阶段 4**：`tasks/bottle/collect.py` 落地，进程内跑 gym，批次/exporter 走
  `pipeline/batches.py`，与旧 driver 并存。**闸门 PASS 5/5**
  （`gate8_new_20260922`，GPU0、instance 0、8 局固定样本库）：
  yield 0.75→0.875、end_depth 0.1469→0.1721 m、无新阶段、1/8 新失败原因在 rule-of-three 界内。

  **进程内执行踩到的两个同源缺陷**（都是"per-run 的值在 import 时就被冻住"）：

  1. `privileged_pub.py` 的 `PRIVILEGED_PORT` / `EPISODE_PORT` 是模块常量兼默认参数，import 时求值。
     改为 `privileged_port()` / `episode_port()` 按需解析。闸门**看不见**这条（它跑 instance 0，
     默认端口恰好是对的），所以钉在 `tests/test_instance_ports_are_resolved_at_use.py`（5 条）。
  2. `locomani/paths.py:68` 的 `SIM_DIR`/`SCENE_META` 同样在 import 时从 `LOCOMANI_SIM_DIR` 解析，
     而 `locomani.paths` 是任何进程里第一个被 import 的 locomani 模块——`os.environ.update(stack.env)`
     永远来不及。**这条是闸门抓到的**：gym 读了解释器启动时的场景而不是本次运行的副本。
     改为把场景作为 `GymOptions.scene_meta` 显式传下去（AGENTS.md「parse at the entry, pass explicit
     values to consumers」），钉在 `tests/test_bottle_collect_entry.py`。
     它表现为固定样本库的场景指纹不匹配——**这是运气**：样本库自带指纹。没有固定样本库的运行会
     静默地对着错误的场景采集。

  结论（对后续阶段有约束力）：**"直接调 = 进程内"与"per-run 的路径在 import 时解析"是直接冲突的。**
  每个转成进程内的阶段都要先确认它依赖的 per-run 值不是 import 期冻结的。标准里那句
  「without import-order setup」正是这条，`os.environ` 先设后 import 的写法不许用。

  另：`grasp_gym.run` 现在在 `finally` 里关掉 setup 开的 `Primitives` 和 `EpisodeControlClient`
  （两者本就有 `close()`，只是没人调；旧路径靠进程退出兜底）。

  **闸门看不见的第二项验证——两批次同进程不泄漏**（`leakcheck2_20260922`，2 批 × 2 局）。
  闸门配置 `max_batches: 1`，所以它**结构上测不到**"一个解释器跑 40 批"这件事，
  而那正是改成进程内之后唯一的新风险。实测采样进程的 fd / 线程：

  | | fds | threads |
  |---|---|---|
  | 批 0 峰值 | 54 | 78 |
  | 批间（teardown 后）| 18 | 69 |
  | 批 1 峰值 | 56 | 79 |

  每批增长 fds +2 / threads +1，即无泄漏（若不关闭，每批会多留两个 ZMQ context）。
  两批产出**两个不同的 exporter session**，`convert` 依赖的 session k ↔ manifest k 按位置配对成立。

- **2026-09-22 阶段 5**：`tasks/crate/collect.py` 落地。`expert.py` 按 gym 的同一形状拆成
  `run(config, *, rounds, seed, manifest, ...)` + 薄 `main(argv)`，场景经 `scene_dir` 显式传入
  （`run_attempt` 的三处 `ArmIK(paths.SCENE_XML)` 同改）。**闸门 PASS 4/4**
  （`crate_new_20260922`，`gate.py compare carry` 对 `baseline_carry.json`）：
  yield 0.85→0.95、end_depth 1.3306→1.3143、无新失败原因（0/20）。

  **这一跑顺带验证了阶段 4 的端口修复**：`crate_table30_record` 用 `instance: 2`，栈起在
  5575/5576/5577/5600/5601/5602。日志显示进程内的 agent 绑到 `:5576`、`:5600`，连 `:5577`——
  全是实例 2 的端口。改之前那些是 import 期冻结的实例 0 默认值，这一跑会绑 5556 然后卡在等 5581。
  **instance 0 的闸门永远测不到这条，instance 2 的这一跑测到了。**

  **未转换**：`wall` / `carry` 两个诊断专家的 `main()` 仍直接读 `sys.argv` 且带 `global DUMP_DIR`
  模块级状态。`tasks/crate/collect.py` 对它们**显式拒绝并指向旧 driver**，不是静默跑成别的东西。
  这是阶段 7（删旧 driver）的前置：旧 driver 不能删，直到这两个也转完。

- **2026-09-22 阶段 6**（验收改为「跑通不报错」，用户 2026-09-22 决定：闭环成功率此处无意义）：
  两个评测器各加一个 `run` 位置参数——给它一个运行目录，它自己读 `config.yaml` / `summary.json`、
  附栈、跑局、写回 `summary.json`；显式 flag 原样保留给 `scripts/experiments/` 里那些
  「评测的不是一次 run」的诊断。`run_eval_single_arm.py` / `run_eval_bimanual.py` 删除。

  **实跑一次**（`bottle_simple_200_20260916`，checkpoint-10000，`--set rounds=1`，GPU0）：
  exit 0，`closed_loop.json` / traces / videos 齐全，`summary.json` 的 eval 条目写入。
  该局本身失败（`[vla] walk: never closer than 1.28 m`）——那是策略质量，不是管线缺陷。
  历史 eval 目录与 `summary.json` 的 eval 条目已还原，冒烟结果留在 `eval_smoke_20260922/`。

  **这一跑抓到一个既有缺陷**：评测路径的准入检查从来没拿到 2026-09-21 给采集的那个修复——
  它在 attach 之前就要满额 8 核，于是栈和策略服务都起着的时候必然被拒
  （`cpu: 7.68 cores in use + 8 needed > quota 15.36`）。改成「先 attach，再只要余下的」，
  与 `shortfall` docstring 里写的顺序一致。

- **2026-09-22 阶段 7**：四个旧 driver 全部删除，`scripts/<line>/*.sh` 直接 exec 功能模块。
  `bimanual_wall` / `bimanual_carry` 拿到 `main(argv)` 和 `--scene-dir`，`tasks/crate/collect.py`
  在**同一进程内**调它们（`visual` 走 `expert.run(config, ...)`，两个诊断仍经各自的 `main(argv)`——
  它们没有闸门也没有 checkpoint 可验证更深的重构，CLI 暂时留作接缝；进程边界已经没有了）。
  `--dry-run` 改为打印解析后的配置，`test_dry_run_delivery.py` 的 collect/eval argv 断言随之改为
  读解析结果（决策 10）；stack / convert / train 的断言**原样保留满强度**。
  `checkpoint_of` 三份合一，落在 `run_context`（旧 driver 删除后原来的两份随之消失）。

- **2026-09-22 阶段 8**：`lint_repository.spawn_errors` + `tests/fixtures/interpreter_spawns.json`
  （沿用 `architecture_debt.json` 的机制：条目要有理由，解决了的条目会被报成陈旧，清单只能变短）。
  检查的信号是**「一个列表的第一个元素是解释器」**——每一处都是这么写的，而且不管最后交给
  `subprocess` 还是 `Stage.run` 都能抓到；模块级常量会被解析，所以换个名字藏不住。
  外部工具（pgrep / nvidia-smi / ffmpeg）永远不算。

  **规则真的会拒**：`tests/test_interpreter_spawn_rule.py` 把删掉的那个形状重新写一遍并断言被拒
  （不会失败的检查比没有检查更糟）。

  **census 10 条**，其中跨 venv 4 条（`run_convert` / `run_train` / `run_vla_server` /
  `eval_crate_closed_loop` → Isaac-GR00T），其余 6 条各自记了理由。**规则本身又查出 3 处
  `sys.executable` spawn**：`bimanual_scene.py`、`reach_envelope/floor_scene.py`、
  `reach_envelope/boundary_search.py`——都是这次重构删掉的同一形状，活在诊断路径上、没有闸门可验证，
  先入清单不改。另外 `eval_crate_closed_loop → vla_rounds` 是同 venv spawn，理由是它要跑在
  每批重启的那个栈的环境里；要改需要一个双臂 checkpoint，磁盘上没有。
- **2026-10-01 遗留 1.15**：那 3 处 `sys.executable` spawn 改成进程内调用——`bimanual_scene.main(argv)` 调
  `build_sim_scene.main(argv)`，`floor_scene` 调 `bimanual_scene.main`，`boundary_search` 直接调 `generate_joint_bank`。
  `build_sim_scene.main` 退出时把 `--seed` / `--table-top` 改过的 `scene_lib` 常量还原，同进程下一次构建不继承。
  验证：新旧两条路径各建 3 个场景（默认、`--seed 3 --table-top 0.45`、floor），另在同一进程里先 seeded 再默认，
  `scene_locomani.xml` / `scene_meta.json` 逐字节一致；`_write_group` 写出的 bank 与脚本对同一 `sampling.json` 的输出一致。
  清单里 `same-interpreter` 三条删除。
- **2026-10-01 遗留 1.16–1.18**：
  - `eval_crate_closed_loop` 不再 spawn `vla_rounds`：每批先 `run_context.adopt_env(栈环境)`，再 import 并调
    `vla_rounds.run(config, ...)`。顺序不能反：import 时 IK 就按 `locomani.paths` 当时的场景建模型，`adopt_sim_dir`
    会拒绝移走已被持有的场景（第一次实跑就是这么失败的，有回归测试）。闸门是 cap120 checkpoint-5000 在实例 2 上
    同 seed 各跑 10 局：旧路径 4/10、新路径 5/10，逐 seed 不一致 1 : 2，产物文件集合相同。
  - `bimanual_wall` / `bimanual_carry`：`main` 只解析，`run(config, ...)` 干活。dump 目录和 ghost trace 改走回合上下文，
    `global DUMP_DIR` / `GHOST_TRACE` 删除；`prepare.expert_argv` 和 `collect.diagnostic_argv` 随之删除。
    这一步只有单元测试，没有实跑。
  - 「不在范围」那一节说的两套评测器：轮次循环仍是两份（单臂是 grasp_gym 自己的循环，双臂是 `vla_rounds` 分批 +
    批间重启 deploy），循环之外的运行目录准入、收栈、写 summary 和打分合进 `pipeline/closed_loop.py`。
    47 份历史 manifest 用新旧代码打分，逐字段相同。
  - `interpreter_spawns.json` 只剩 4 条：3 条是跨到 gr00t venv 的边界，1 条是栈自己的子进程。
