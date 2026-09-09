# locomanipulation

GR00T VLA + SONIC 全身控制的 G1 loco-manipulation finetune 项目。项目文档只分四类,见
`docs/README.md`:教程(setup / collection / dataset / training / evaluation / rendering /
crate_collection)、台账(`docs/ledger/`)、参考(`docs/reference/`)、状态(STATE / ROADMAP);
仓库级坑与规范在 `docs/reference/silent_failures.md`(静默失败表必读)。

## Agent skills

### Issue tracker

GitHub Issues(`gh` CLI,仓库 yijie21/locomanipulation)。See `docs/agents/issue-tracker.md`.

### Triage labels

默认五角色标签(needs-triage / needs-info / ready-for-agent / ready-for-human / wontfix)。See `docs/agents/triage-labels.md`.

### Domain docs

单上下文布局:根目录 `CONTEXT.md` + `docs/adr/`(按需惰性创建)。See `docs/agents/domain.md`.

### Session workflow

每个 session 从 `docs/STATE.md` 开始,结束前更新它,然后 `/clear`。See `docs/agents/session-workflow.md`.

### Code and doc standards

`docs/reference/standards.md` 是唯一的规范。`make check` = ruff + 尺寸/环境变量 lint + unit 测试;改主代码的 PR 附
`scripts/gate.py compare` 输出。

### Refactor freeze (2026-09-09 to 2026-09-23)

仓库重构进行中,计划与 13 条决定见 `docs/handoff/refactor-plan.md`。冻结期内 main 上只跑实验、只修 bug,不加新功能;
`src/teleop_free/scalebfm/` 归 Codex 线程,其余主代码的移动/删除由重构 PR 完成。删除 `outputs/` 目录或权重前先在
`docs/STATE.md` 登记 24 小时。每个重构 PR 必须附 `scripts/gate.py compare` 的输出。
