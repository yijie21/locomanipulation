# 代码与文档规范

两页以内。规则能用工具拦的都用工具拦(`make check` = pre-commit),靠人 review 的只有"是否属于四类文档"和
"Task 接口是否被绕过"两条。来源:`docs/handoff/refactor-plan.md` 的 13 条决定(2026-09-09)。

## 1. 代码

**布局**(重构第 4 步起生效;之前的 `src/teleop_free` 按同样规则约束)
```
src/locomani/pipeline/      stack 生命周期、gym 主循环、录制、manifest、convert、stats/audit、train、eval、配对统计
src/locomani/backends/<x>/  Backend:stack 启动、动作空间、执行器。sonic 是第一个
src/locomani/tasks/<x>/     Task 五件套:scene、expert、acceptance、schema、failures。bottle、crate
src/locomani/primitives/    真正共享的底层(arm_ik、scene_lib、primitives)
src/locomani/cli.py         唯一入口:locomani collect|convert|train|eval|gate --config X --out Y
scripts/                    只放 GPU 闸门与 stack 的 shell;一次性 runner 放 outputs/experiments/<name>/launch.sh
```

**expert 的唯一签名**:`decide(obs_snapshot, task_config) -> command`。不读 wall-clock,不持跨 tick 的隐藏状态;
需要的状态显式放进 snapshot。动作必须是策略可观测量的函数:同一 snapshot 两次调用输出相同(contract 测试强制)。

**旋钮**:只在 config dataclass 里。每个字段一行注释写含义和裁定来源。代码禁 `os.environ.get`,白名单在
`scripts/lint_env.py`(设备、端口、路径)。命令行只剩 config 路径、输出目录、字段覆盖。每次运行把完整 config
快照写进输出目录,那就是 provenance。

**尺寸**:文件 ≤ 600 行,函数 ≤ 80 行(`scripts/lint_sizes.py`;2026-09-16 起 `--strict`)。

**格式与 lint**:`ruff format` 说了算,不讨论风格。lint 只开 pyflakes、语法、import 顺序和四条 bug 模式。

**注释**:写"为什么"和"哪次实验定的",不写"做了什么"。超过 10 行的考古叙事放台账,代码里留一行链接。

**删除优于保留**:有裁定的实验分支删代码,台账里记"删除于 commit X"。没有 `legacy/`。

## 2. 测试

三层,`tests/conftest.py` 分层:
- **unit**:纯逻辑,秒级,pre-commit 每次跑。
- **sim**(`--run-sim`):需要 stack。
- **torch**(`--run-torch`):需要 GR00T venv。

**Task contract 测试**:每个 Task 必须通过同一组测试:决策函数同 snapshot 两次同输出、不读 wall-clock、config
往返序列化、失败分类覆盖 acceptance 的每种失败。

**行为闸门**:`scripts/gate.py`。每个改动主代码的 PR 必须附 `gate.py compare` 的输出(bottle + crate),结果 JSON
提交到 `outputs/gate/`。容差:yield ±0.15,深度 ±1 cm,drop ±5 mm,阶段时长 ±30%,阶段与失败原因集合不得新增。

## 3. 文档

只有四类,四类之外不新建:

| 类 | 位置 | 规则 |
|---|---|---|
| 教程 | `docs/<stage>.md` | 每个 pipeline 阶段一篇,章节固定:场景 / 跑一局并读日志 / pilot 与验收 / 控制逻辑 + 流程图 + 判据表 / 数据形状 / 随机化 / 相机 / 带日期的坑 |
| 台账 | `docs/ledger/<line>.md` | 每条产品线一个,append-only,一条 = 日期 + 变量 + 数字 + 裁定。一个结论只记一处,别处链接 |
| 参考 | `docs/reference/` | 静默失败表、本规范、配置字段表(从 dataclass 注释生成,不手写) |
| 状态 | `docs/STATE.md`、`docs/ROADMAP.md` | 只这两个 |

带日期的诊断不再单独建文件,写进台账。其余历史文档在 `docs/archive/`,只加索引不改内容。

## 4. 协作

- 冻结期(2026-09-09 到 09-23):main 上只跑实验、只修 bug。ScaleBFM 线 2026-09-10 起是 `backends/scalebfm` + `tasks/bottle_visual`,同规则。
- 重构在 git worktree 里做,每步一个 PR,过闸门合 main,不开长期分支。
- 删 `outputs/` 目录或权重前,先在 STATE.md 登记 24 小时。
- 每个 session 从 STATE.md 开始,结束前更新它。
