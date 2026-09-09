# 仓库重构与规范计划(2026-09-09 grill 定稿,待用户确认后执行)

13 个决定,一次 grill 逐条拍板。执行顺序在最后。任何一条要改,先改这里再改代码。

## 决定

| # | 问题 | 决定 |
|---|---|---|
| 1 | 重构结束时什么必须不变 | **(b)** 单臂正式线(07d / visible expert 的 run_learn_pilot 全流程)+ bimanual pilot 流程都可复现。历史 eval 变体不保证。 |
| 2 | 怎么锁行为 | **(c)** 先录 GPU 闸门基线 JSON(单臂 1 batch + bimanual 若干局,固定 seed;yield、走近退出原因、刹车深度、drop、阶段时长、失败原因集合,容差按 seed 方差:刹车深度 ±1 cm、yield ±15%),再抽纯函数 expert 做离线回放闸门(逐字节);之后离线为主,GPU 每周一次。在线部分只做容差比对。 |
| 3 | 目标结构 | **(a)** Pipeline 骨架 + Task 插件。骨架拥有 stack 生命周期、gym 主循环、录制、manifest、convert、stats/audit、train、eval、配对统计。Task 五件套:场景采样、expert 决策函数、验收判据、数据集 schema、失败分类。expert 签名 `decide(obs_snapshot, task_config) -> command`,不持 wall-clock,不持跨 tick 隐藏状态。 |
| 4 | 淘汰的实验代码 | **(a)** 删代码留裁定。docs 里已有裁定的分支删代码与 config,裁定表加"删除于 commit X"。ReachConfig 收缩到当前协议字段。例外:`--ensemble` 推理开关和 lockstep 评测保留;planner 退化模式删。**2026-09-09 补充裁定(D 组)**:RTC 整条(推理侧 + 训练 patch)和 `--client blocking` 保留,之后还要用;47 个无人 import 的 `diag_*.py`、无人 import 的 `tools/`(留 crate 教程的 4 个预检)、旧 `collect_episode` 采集线 + `run_batch` 及其解锁的 stream_reach 死尾巴删。证据表 `docs/handoff/refactor-step2-candidates.md`。 |
| 5 | legacy/ 规则 | 不适用(4 选了 a)。 |
| 6 | ScaleBFM | **(a)** 划出范围,`src/teleop_free/scalebfm/` 原地冻结到其 Goal 结束;骨架预留 `Backend` 维度(stack 启动、动作空间、执行器),SONIC 是第一个实现,ScaleBFM 日后作为第二个后端接入。 |
| 7 | 文档分类 | 只允许四类,四类之外不新建:**教程**(每个 pipeline 阶段一篇,collection.md 的章节结构)、**台账**(每条产品线一个 append-only 文件,一个结论只记一处)、**参考**(静默失败表 + 从 dataclass 注释生成的配置字段表)、**状态**(STATE.md、ROADMAP.md)。其余移 `docs/archive/` 加索引。带日期的诊断不再单独建文件。 |
| 8 | 旋钮放哪 | 一个 Task 一个 config dataclass,一个 Backend 一个;每个字段带注释;代码禁 `os.environ.get`,白名单只留 CUDA_VISIBLE_DEVICES / 端口 / 路径,测试强制;命令行只剩 config 路径 + 输出目录 + 字段覆盖;每次运行写完整 config 快照即 provenance。 |
| 9 | 执行手段 | 工具拦,不靠 review:pre-commit = ruff format + 基础 lint + import 隔离测试 + env 白名单测试 + 配置字段表一致性;纯逻辑测试与 GPU 测试用 marker 分开;**文件 ≤ 600 行,函数 ≤ 80 行**;GPU 闸门手动跑,结果 JSON 提交,PR 必须贴;规范写成 `docs/reference/standards.md` ≤ 2 页,CLAUDE.md 指向它。 |
| 10 | 节奏 | **(a)** 冻结两周:main 上只修 bug、跑实验,不加功能。每步一个 PR 过闸门合 main,不开长期分支。DAgger 等 expert 改动在新骨架的 `tasks/bottle` 上做,作为第 4 步验收用例。 |
| 11 | 多 session 协同 | Codex 的约 50 个未提交文件先由其自己原样提交进 scalebfm/ 冻结范围;冻结期其他 session 只跑实验不改主代码;STATE.md 加 "refactor in progress" 节列出正在移动的路径;清理类操作先在 STATE.md 登记 24 小时。 |
| 12 | 包名与布局 | `src/locomani/{pipeline, backends/sonic, tasks/bottle, tasks/crate, primitives, cli.py}`;`scripts/` 只留闸门脚本和 stack shell;一次性 runner 放 `outputs/experiments/<name>/launch.sh`;git mv 保历史;`teleop_free` 名字消失;`src/gr00t` 不动。 |
| 13 | 测试 | 被删代码的测试随删;新增 Task contract 测试(决策函数同 snapshot 两次同输出、不读 wall-clock、config 往返序列化、失败分类覆盖 acceptance 每种失败),bottle 与 crate 都过;GPU 测试打 marker;不追求覆盖率。 |

## 执行顺序(每步一个 PR)

0. 录基线 JSON、写 `scripts/gate.py`。零主代码改动。可在 visible200 结果出来前做。
1. 工具:ruff、pre-commit、marker、行数 lint。先报警一周,再转拦。
2. 删淘汰分支与 config,裁定表加删除 commit。
3. 旋钮进 config,`os.environ` 白名单测试。
4. 抽骨架:grasp_gym + stream_reach → `pipeline` + `tasks/bottle` + `backends/sonic`;离线回放闸门上线。
5. bimanual → `tasks/crate`,过 contract 测试。
6. 文档重组(从第 2 步起并行)。

## 不在范围
- ScaleBFM 线的任何代码。
- 训练配方、模型结构、相机、动作空间。
- 历史 eval 变体(planner / RTC 训练侧 / a18)的可运行性。
