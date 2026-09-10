# 重构审核台账

## 2026-09-10：重构后的独立审核（未修实现）

范围：`git diff 78536ae...1b32a48`；ScaleBFM 以迁移前已提交原始代码 `a3e2fee` 另作对照。依据 `docs/handoff/refactor-plan.md`、步骤执行记录、`docs/reference/standards.md` 和 ROADMAP 的延期裁定。审阅中其他会话在修改 `docs/crate_collection.md`、`docs/reference/silent_failures.md`、`scripts/run_wall_pilot.sh`；这些未提交修改不属于本次定稿结论。

裁定：目录、配置集中和文档分类方向成立，但尚不能以“行为保持、可复现性完整”验收。以下区分可复现缺陷、文档问题和已接受的延期；不以尺寸超限代替行为审核。

### Spec：行为与复现（4 项）

1. **P1，配置输入校验退化。** `src/locomani/config_base.py:66` 丢弃未知 JSON 字段和 schema_version，不验证标量类型；client/through 也没有枚举校验。CPU 实测 `EvalConfig.from_dict({"schema_version":99,"round":100,"rtc":"false"})` 得到 rounds=20、rtc 为真值字符串；`EvalConfig().with_overrides(["client=asnyc"])` 与 `PilotConfig().with_overrides(["through=collec"])` 均接受。`eval_closed_loop.py:206` 将未知 client 当 blocking；run_learn_pilot 的三个提前返回均不匹配错误 through，会继续后续训练/评估。旧 argparse choices 会拒绝这些值。应在副作用前验证类型、未知字段、版本和枚举，不能让运行结果与配置文字不符。
2. **P1，ScaleBFM 后端完全漏出源码指纹。** `tasks/bottle_visual/pipeline/provenance.py:16` 的 `parents[4]` 得到 `src/backends/scalebfm`，实际应在 `src/locomani/backends/scalebfm`。实测目标目录不存在，line_sources 返回 26 个文件、后端文件 0 个。修改动作编解码、执行器或底层网络不会改变续跑/视频复用的源码指纹。修路径并断言关键文件覆盖；修好后源快照还应保留相对目录，避免两个 config.py 同名覆盖。
3. **P1，自定义 ScaleBFM 场景配置不生效。** `tasks/bottle_visual/scene.py:28` 从模块默认 BASE_SCENE 生成场景；`:72` 又覆盖传入 backend.scene_dir。round.py 虽接受并保存 `--set-backend scene_dir=...`，实际仍走默认场景。以 mock 替代 R.Sim（不启动仿真）实测传 `/tmp/review-requested-scene`，构造器收到 `outputs/scene/sim`。应从有效 backend 配置产生所有高度变体，记录实际资产身份。
4. **P2，SONIC 新配置快照丢失输入资产身份。** `pipeline/run_learn_pilot.py:89` 仅传 task/pilot 给 write_config_snapshot；`pipeline/provenance.py:55` 不记录冻结场景和外部 initial token。旧版本明确对 token 与冻结场景文件哈希。源文件哈希不能覆盖任意外部输入；仅采集时也不能依赖后续 eval 补记录。恢复有效输入的解析路径和内容哈希。

### Standards：文档与维护（3 项）

1. **P1，当前参考文档给出失效的帧率设置。** 基线 `docs/reference/silent_failures.md:104,138` 指导 `TF_DATASET_FPS=25`，但 eval_closed_loop 采用 cfg.backend.dataset_fps（默认 30）。应改为当前入口的 `--set backend.dataset_fps=25`。该文档也推荐已删除的 diag 工具；应移为历史描述或替换成存活入口。此文件正被其他会话修改，需在对方完成后复核。
2. **P2，台账把机制假设写成结论。** `docs/ledger/scalebfm.md:16` 的“学生无法从下蹲恢复”强于已知证据。单步上升目标最多 1.836 cm 是历史测量事实，但不能排除多步恢复；ABCD 无最终结果。应追加更正，标明恢复能力与初始下沉原因尚未因果验证。
3. **P2，Task 能力说明超出实现。** `tasks/base.py:5` 声称 pipeline 只经 Task 调用，且确定性/no-wallclock 已由 contract 强制；实际 pipeline 仍直接调用任务脚本，测试只检查配置往返、可调用、schema 与人工标签。纯 decide(snapshot,cfg) 和回放闸门已明确延期，缺失本身不列为新增回归；应改正文档声明，并让失败分类测试覆盖真实 acceptance 的输出。

### 验证与边界

- `make check`：格式/基础 lint/env/config 文档检查通过；尺寸为 6 文件 >600 行、91 函数 >80 行（目前仅报告）。首次 unit 在 `test_offhand_guard_reuses_current_observation` 因 0.2 秒墙钟观测超时失败。STATE 已记录该历史问题，不能归罪本次 ScaleBFM 迁移。
- 限制线程后运行完整默认套件：224 passed、9 skipped（共收集 233）；命令如下。不同条件下的两次结果提示时序敏感，不证明缺陷已消失。

```bash
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 \
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -m pytest tests -q --continue-on-collection-errors

PYTHONPATH=src python3 - <<'PY'
from locomani.tasks.bottle_visual.pipeline.provenance import line_sources, _BACKEND
from locomani.pipeline.eval_config import EvalConfig
from locomani.pipeline.pilot_config import PilotConfig
print(_BACKEND, _BACKEND.exists())
print([str(p) for p in line_sources() if 'backends' in str(p)])
print(EvalConfig.from_dict({'schema_version':99,'round':100,'rtc':'false'}))
print(EvalConfig().with_overrides(['client=asnyc']).client)
print(PilotConfig().with_overrides(['through=collec']).through)
PY
```

- 未启动 GPU/stack/训练/闭环实验；历史 gate 数字仅来自已有记录，未重跑。模型丢失导致 eval 路径缺少配对验证，不能宣称端到端等价。
- 第三视角只调 mj_kinematics、未调 mj_camlight 的变暗问题仍在 follow_video.py:130；是已记录旧 bug，不是重构新引入。
- 本次仅写审核台账与 STATE，不修改实现、不提交、不恢复已取消 Goal。

### 下一步

先修配置 fail-fast、源码/资产指纹、场景传递，分别补有实际失败触发器的 CPU 回归测试；复核其他会话对参考文档的更改，并补台账因果限定。恢复可用模型后，按既定计划做固定 checkpoint/seed 的旧新 eval 配对闸门。之后再把配置和每 tick 状态显式传给阶段逻辑，减少全局变量和反向模块依赖；不继续单纯为行数机械拆文件。通用 Backend 集成、纯 expert seam 和 ScaleBFM 驱动 CLI 统一属于已接受的延期。
