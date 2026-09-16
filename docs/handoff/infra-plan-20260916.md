# 基础设施重构计划（2026-09-16 grill 定稿，待用户确认后执行）

目标：单机多卡上并行收集 / 训练 / 评估、并行跑多个配置、一行切换 task × embodiment × scene，sim / 控制 agent / 导出进程干净分离，训练与评估用版本化 config。8 个决定，一次 grill 逐条拍板。前提：2026-09-09 的 13 条 refactor 决定继续有效（Pipeline + Task + Backend、frozen dataclass config、`--set` 覆盖、config 快照即 provenance、600/80 行上限、四类文档）。

## 决定

| # | 问题 | 决定 |
|---|---|---|
| 1 | 并行规模 | **单机多卡**。一张卡一个 job；job = 自包含 spec（config 引用 + GPU + 输出目录 + 阶段），Python launcher 本机按卡排队。多机时只换 launcher，不换 job。 |
| 2 | config 组合 | **Hydra + OmegaConf 结构化配置**。现有 dataclass 注册进 ConfigStore 当 schema（`__post_init__` 校验保留），yaml 只能写 dataclass 里存在的字段。`configs/experiment/<name>.yaml` 一个文件一个版本，只增不改。`-m` 做扫描；`.hydra/` 快照替代自写 provenance。关闭 Hydra 改工作目录。 |
| 3 | 进程生命周期 | **Python `Stack` 进程管理器**：`locomani stack up <spec>` 拥有 sim / deploy / exporter 三个子进程（各自 log、健康检查：端口就绪、LowState 心跳、pid 数；进程组一把杀；deploy 掉线按策略自愈）。**agent 是独立进程**（`locomani agent <spec>`），只通过端口和 spec 连接。每个 stack 自带 exporter（废除"只有 instance 0 能录"）。shell 启动脚本和 env 翻译层（`ENV_FIELDS`）删除，子进程直接读 config JSON。 |
| 4 | job 粒度 | **阶段即 job**：collect / convert / train / eval / stats 各一个 CLI，输入输出只有目录 + manifest（config 快照 + 上游 hash）。流水线是列阶段和依赖的 yaml；launcher 做最小 DAG（上游 manifest 存在且 hash 匹配则跳过）。需要 stack 的阶段声明 `needs: stack`。不用 Snakemake / DVC / Prefect。 |
| 5 | 切场景的轴 | **四个正交组**：`task/`（expert 决策、验收判据、失败分类）、`embodiment/`（动作空间、状态维度、相机集、GR00T modality/data config；单臂、双臂各一份）、`scene/`（桌高、物体集、背景、随机化范围）、`backend/`。Task 声明兼容的 embodiment，不兼容组合时报错。GR00T 的 Python data config 从 embodiment yaml 生成。训练/评估只依赖 embodiment，不依赖 task。 |
| 6 | 资源准入 | **静态声明 + 上机前实测闸门**：job 声明 `resources: {gpu, cpu_cores, pids}`；起跑前读 `cpu.stat` 5 s 差分、`pids.current`、该卡的外来进程；不够就等（有上限、有日志）；manifest 记起跑时的实测值。一张卡同一时间只属于一个 job（stack 或训练）。 |
| 7 | 迁移方式 | **原地改**。单臂专家冻结在 root `stage4_27_wrap`（worker `a694e12`），迭代暂停。安全网：动手前录基线（固定 bank `regression8_lowcom.json` seed 30700 的 8 局逐局结果与失败家族、数据集 schema、convert 输出逐字节 hash；crate 用现有 20 局 ready-stand 闸门），改完同 seed 复跑对照。改动期间不跑实验。 |
| 8 | 结果登记 | **launcher 自动写 `outputs/registry.jsonl`**：每 job 一行（job id、experiment 名、config hash、阶段、指标摘要、目录、起跑资源实测、上游 job id）；`locomani ls` / `locomani compare` 从它读；markdown 台账只写结论。W&B 可选，只用于训练曲线，注册表不依赖它。 |

## 参考的现成结构

- Hydra 结构化配置 + config 组 + `experiment/` 覆盖文件 + multirun：本计划的 config 层照搬。
- LeRobot（`lerobot/scripts/{train,eval}.py` + dataclass config + `policies/ envs/ datasets/` 分包）：阶段 CLI 和 embodiment/数据 schema 的分法。
- Isaac Lab（`gym.register` + `EnvCfg` dataclass，`--task` 一行切换）：task × scene 的注册与组合方式。
- robomimic（每算法一份层级 JSON config，训练与数据解耦）：训练侧 config 版本化的样子。
- supervisor / systemd 的 program 模型（健康检查、重启策略、每进程日志）：`Stack` 管理器按此实现，但不用它们本身（动态 GPU/端口/目录）。

## 执行顺序（每步一个 PR，过闸门合 main）

0. 录基线（bottle simple 8 局 + crate 20 局 + schema/hash），写 `scripts/gate.py` 的对照模式。零主代码改动。
1. config 组：`configs/{task,embodiment,scene,backend,collect,train,eval,experiment}/`，dataclass 进 ConfigStore，5 个入口改 `@hydra.main`；GR00T data config 由 embodiment 生成。
2. `Stack` 进程管理器 + agent 独立进程；删 `scripts/stack/*.sh`、`env_deploy.sh`、`ENV_FIELDS`。
3. 阶段 CLI + manifest 约定（collect / convert / train / eval / stats）；`run_learn_pilot` 拆成流水线 yaml。
4. launcher：DAG、资源准入、注册表、`locomani ls/compare`。
5. 基线对照复跑；删旧入口。
6. 文档：教程按 collection.md 章节结构重写 collect / train / eval 三篇；`docs/reference/standards.md` 加 job/manifest/registry 规则。

## 进度（2026-09-16）

- 步骤 0 完成：root 冻结提交 `ba18972`（分支 `refactor/infra-20260916` 从它切出）；基线 `tests/gate/baseline_simple.json`（stage4_wrap8p，6/8）、`baseline_carry.json`（`outputs/gate_baseline_20260916/crate`，17/20）、`baseline_simple_dataset.json`（8p 的 session 经 convert：6 集、3358 帧、37 个特征、parquet/meta 的 sha256）；`scripts/gate.py` 新增 `simple`、`carry` 两臂，`simple` 的停位容差 2 cm、接触侧归一化，对 8n/8o/8p 三批干跑全部 PASS。
- 步骤 1 完成（`7105545`）：`locomani.conf` + `configs/{task,embodiment,scene,backend,collect,train,eval,experiment}/` + `python -m locomani.cli`；OmegaConf 对 frozen 嵌套 dataclass 需清只读标记（`conf.writable`），`tuple`/`Optional` 字段全部通过。
- 步骤 2：`locomani.stack.Stack`（GPU1 实机 up/status/restart-deploy 30 s/down 验证通过，无残留进程与端口）；shell 启动/停止/重启脚本已删；`pipeline/stack.py` 的 `start_stack`/`stop_instance` 改为桥接，`deploy.restart` 经 `Stack.attach`。子进程仍读环境变量（sim/exporter 的 `TF_*`、`LOCOMANI_SIM_DIR` 等），`ENV_FIELDS` 翻译层留给步骤 3 的 agent 进程改造一并删。首次端到端复跑 `outputs/runs/infra_check_simple8` 2/8：主机负载 21、CPU 用满配额、GPU1 有外来 3 GB 进程，失败家族是已知的负载家族（停位过深 ×5、站位看不到瓶子 ×2），不能判定；负载降到 7 核以下后重跑。

## 待定（执行时决定，不影响上面 8 条）

- OmegaConf 对 `tuple[float, float]`、`Optional[...]`、frozen dataclass 的支持要先用现有 5 个 config 类试一遍；不行的字段改 list。
- GR00T finetune 是第三方脚本、独立 venv：train 阶段 CLI 只是包一层，参数从 train config 生成命令行；三个 venv 各阶段声明自己的解释器。
- agent 进程需要场景 XML 做 IK：由 spec 里的 scene 路径提供，还是 stack 通过消息下发模型摘要？倾向前者（只读路径）。
- 端口分配：按 job 从端口池取，还是沿用 INSTANCE 端口块？倾向端口池 + 写进 manifest。
- 阶段 CLI 是否复用 `locomani.cli`（`cli.py` 已存在）作为唯一入口。
- 谁执行：本 session 还是 9d session（角色分工见 STATE.md）。

## 不在范围

- 单臂专家逻辑本身（冻结）。
- 多机调度、ScaleBFM 线、训练配方、模型结构。
