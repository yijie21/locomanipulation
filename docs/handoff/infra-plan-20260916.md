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

- 步骤 3：`src/locomani/stages/`（collect / convert / train / eval / stats 各一个模块，`<out>/<stage>/manifest.json` 约定：配置组 hash、上游 manifest hash、outputs、metrics、起跑资源），`locomani.cli <stage>` 直接调用，不再桥接 `run_learn_pilot`；crate 的 collect 仍走 `run_crate_pilot`。`train_bottle.sh` 读 `MODALITY`/`BASE_MODEL`；convert 的 GR00T stats 改用 embodiment 的 modality 配置（旧 pilot 用 2cam 算 stats、3cam 训练，不一致）。

- 步骤 4 完成（`fde9cc0`、`dab109a`）：`locomani.launch run <jobs.yaml> --gpus 0,1`（每阶段一个子进程、manifest 判定跳过、准入 = 一卡一 job + 实测 CPU/pid/显存、`outputs/registry.jsonl`、`ls`/`compare`；`configs/jobs/example.yaml`）。
- 步骤 5（进行中）：新路径端到端复跑 `outputs/runs/infra_check3_simple8`（`locomani.cli collect +experiment=bottle_simple_lowcom`，负载 8.6 核、GPU1 空闲时起跑）**7/8**，`scripts/gate.py compare simple` 对冻结基线 **PASS 5/5**（yield 0.75→0.875，停位中位 0.147→0.153 m，失败集合无新成员）。之前两次复跑不能判定：一次是主机负载 21（2/8，停位过深/看不到瓶子），一次是 `backend/sonic.yaml` 从旧 JSON 带来的 `no_encoder: true`（0/8，全部站起失败：无编码器时 deploy 丢弃规划器手臂/腰参考；已改为 false，eval 阶段自行强制 true，`a2d5d1e`）。已删 `run_learn_pilot.py`、`conf.pilot_from`、`run_crate_pilot.py`、`run_ready_pilot.py`、`PilotConfig`、`configs/pilot/`（`c04ada5`）：crate 的 collect 直接在 `stages.collect` 里跑专家（`tasks/crate/prepare.py` 保留场景拷贝/组合拒绝/专家命令行，`task.expert` 选 visual/wall/carry），eval 阶段按 plugin 分派到 `eval_crate_closed_loop`；`test_config_groups_match_json.py` 把 task/backend/eval 的 YAML 钉在 JSON 双胞胎上（JSON 仍被约 30 个测试和专家进程读，删 JSON 是下一步；退役的 image_servo 只保留 JSON）。仍保留：旧 shell 跑批脚本（`collect*.sh`、`run_closed_loop.sh`、`run_overnight.sh`、`run_eval_arm.sh`、`par_batch.sh`、`spawn_sweep.sh`、`seq_ab.sh`，已改为调 stack CLI，未删）；`ENV_FIELDS` 翻译层仍给 agent/sim/exporter 进程供参数（子进程改读 JSON 是独立一步）。
- 步骤 6（部分）：`collection.md` 加了统一入口/配置组/阶段 manifest 一节，五篇文档里的 pilot 命令块全部改写为 `python -m locomani.cli <stage>`，`standards.md` 加了运行/阶段/launcher 规则，change-map 四行；collect/train/eval 三篇教程按 collection.md 章节结构的整体重写尚未做。

- 步骤 5 收尾：crate 线经新路径复跑 `outputs/runs/crate_check_record20` **15/20**，`gate.py compare carry` 对 `baseline_carry.json` **PASS 4/4**（yield 0.85→0.75 在容差内、失败集合相同）。JSON 双胞胎已删（`configs/{tasks,backends,eval}/*.json`，只留退役的 `bottle_image_servo.json`），约 20 个测试和 4 个源文件改用 `conf.load_group(group, name)`；OmegaConf 把 tuple 字段还原成 list 的问题在 `conf._typed` 里用各 dataclass 自己的 `from_dict` 修正。旧 shell 跑批脚本 9 个已删，文档引用改为 CLI / launcher。`ENV_FIELDS` 翻译层已删：backend 只通过 `LOCOMANI_BACKEND_JSON` 一个变量到达 agent / sim / exporter，`lockstep_sim` 读同一份 JSON；sim 的 `SIM_VIS`/`SCENE_RANDOM` 由 `StackSpec` 显式设置。
- 步骤 6 完成：`docs/training.md`、`docs/evaluation.md` 按 `collection.md` 的八章结构重写（新 CLI、manifest、launcher），`collection.md` 的命令块已换。

- 全通路 smoke（`configs/jobs/smoke_pipeline.yaml`，`outputs/runs/smoke_pipeline`）：1 集采集 → convert → 10 步训练 → 闭环评估 → stats，`locomani.launch` 全部 ok，registry 落行。暴露并修掉 4 个 bug：convert 在 `holdout=0` 时仍对不存在的 val 集算 stats；eval 阶段起策略服务没传 embodiment 的 modality 配置；单臂 embodiment 误指 3cam（bottle 线的观测契约是 `gr00t_sonic_2cam.py`，已改回，crate 仍 3cam bimanual）；`clean_clock` 的 commander 包装不接受今天上午加的 `enter_streamed(prime=...)`（评估路径自那次改动后从没跑过）。launcher 的 pid 声明从 2700 降到 1800（本仓库 exporter 4 线程，1.8k 基线上稳起）；CPU 声明 8 核在主机 8–9 核占用下会拒绝，smoke 用 6。

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
