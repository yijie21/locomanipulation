# STATE — 2026-10-07（会话：mars 外观与 D435、训练图加相机延迟、频率时序对齐官方 GR00T + SONIC、T7、合入 henry-dev）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认：到 `e6d7e80`（合入 henry-dev）为止都已推到 origin/main；之后只有本次 handoff 提交，未推送。
已确认（10-07 收尾时）：本机没有留下仿真栈、策略服务、镜像或 Unreal 编辑器。GPU1 上常驻别的项目的 depth_server 约 3.7 GB；SimpleRoboJev 的进程时有时无。
别的会话：henry-zhao 直接推 origin/main 或 `henry-dev`，推送不跑钩子，可能自带检查失败（记忆 origin-main-can-carry-failing-checks）。

## 磁盘

- 已确认（10-07）：`/workspace` 剩 97 GB（91%），根分区 `/` 剩 1.6 GB。跑 Isaac、UE、pytest 前 `export TMPDIR=/workspace/tmp`。
- 待用户：11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12、13.6.12.1、13.6.17（本次新增约 3.2 GB）、14.7 的删除都没定；要删先在这里写明 24 小时。

## 当前进度

- 已确认（DECISIONS 2026-10-04/05）：机器人外观是缎面银 `g1_satin_20261004`，三个渲染器一致（MuJoCo 外壳 rgba 0.62）；mars 小臂银色、腰部电机和骨盆黑。
- 已确认（DECISIONS 2026-10-06）：mars 第一视角可选脸上的 D435（场景 `*_mars_d435`，实验 `warehouse_mars_smoke_d435_{unreal,isaac,mujoco}`）。D435 检查采集 10 局保留 4（不是曾报的 5）。
- 已确认（DECISIONS 2026-10-06，`a0f05af`）：离线补渲的训练图按真机相机变旧：`images.camera_delay: {fps: 30, latency_ms: [20, 30]}`，warehouse mars 线 9 个实验文件打开；审计按 `dataset.fps` 算预期滞后（25 fps 约 1.3 帧，50 fps 约 2.6 帧）。数值是估计（13.6.16.1）。
- 已确认（DECISIONS 2026-10-07 两节，`804f962`）：warehouse mars 线的频率时序对齐官方。
  - smoke 系列 7 个实验文件改 `dataset.fps: 50`、`execution_horizon: 20`、`latency_dist: configs/eval/latency_serve_policy_50fps_20261007.json`（本机 serve_policy 实测，3/4/5 拍 = 132/31/1 块）。`warehouse_mars_200` 是已做的 25 fps 运行，不改。
  - lockstep 评测按运行的 `dataset.fps` 定节拍：50 时一行一拍、第一块不混合、执行 20 拍再推理、块尾保持；25 不变。
  - 评测图用训练同一延迟模型变旧（`vla_lockstep.Camera`、sim `observe`、镜像按携带状态渲）；`camera_frames` 与 `camera_delay` 同设被拒。
  - 已确认：真实栈只验了机制（25 fps 旧 checkpoint，成功率无意义）：24 拍一周期、图晚 40–60 ms（`outputs/t7_lockstep50_smoke_20261007/`）；Isaac、Unreal 镜像按携带状态渲各 2 局通过（`outputs/runs/t7_mirror_check_{isaac,unreal}_20261007/`，`613681e`）。
  - 已确认：同 seed 两遍 MuJoCo 轨迹从第一块起就分叉，25 fps 旧路径对照也一样，原因是已记录的 MuJoCo 4× 多重采样（腕部图差一个灰度级），不是本次改动。
  - 未验证：策略层面（要 50 fps checkpoint，16.4 → 16.7.2）。
- 已确认（`e6d7e80`）：合入 henry-dev `dc3b909`：真机部署用官方式 `src/real_robot/run_gr00t_server.py`（`scripts/run_gr00t_server.sh`）和按官方逐函数重写的客户端（调 `get_action`）。单元测试 15 个过、服务器 `--help` 能起；未加载过真实 checkpoint，没连过仿真栈或真机（16.2.1、16.5、16.6）。
- 已确认（henry 设计 D2 被推翻）：50 fps 数据集按 30 fps 相机加延迟出图，T4 的 Unreal 补渲带 `camera_delay`，约 15 h 单卡。henry 还不知道（16.10）。

## 待验证 / 卡点

- 待验证（16.11）：合并提交时钩子测试失败一次，重跑与前后两次全量 `make check` 都过；是哪个测试未知。假设是不稳定测试，未证实。
- 待用户（13.6.18）：warehouse mars 11 个实验文件顶层 `backend.dataset_fps: 30.0` 与 `dataset.fps` 不一致（采集不读它）。
- 待协作者：真机三台相机的帧率、端到端延迟、相互时间差（13.6.16.1，可用官方相机服务器 `--test-latency`）；D455/D405/D435 内参（13.3、13.6.15.2）；腰固件力矩上限（13.6.11.5）。
- 待验证（同前未结）：13.6.5.6 只走 ym 的 mars 采集没实跑；13.6.5.7 mars 上 runbook 第 5–7 步没跑；13.6.15.1 D435 产出率无同种子 D455 对照；14.4、14.5、9.11.1、12.9.13、12.20.4、12.9.6。

## 下一步（编号见 `TODO.md`）

- **16.4**：新运行 `warehouse_mars_200_50hz`：复制运行目录、删补渲标记、Unreal 全量补渲（带 `camera_delay`，约 15 h，开渲前看 GPU 与 STATE）→ 转换 → 训练；留 25 fps 对照。
- **16.2.1**：用真实 checkpoint 起 `run_gr00t_server.sh`，客户端在本机连上核对（不上真机）。
- **16.5**：sim2sim：官方结构客户端连本仓库 MuJoCo 栈（`--hand-wire sim`），之后把 T7 的延迟分布换成这条链路实测的。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export TMPDIR=/workspace/tmp MUJOCO_GL=egl
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; ss -ltn | grep -E ':(55[5-9][0-9]|5700) '
make check                                                     # 全量检查（ruff、lint、pytest -x）；提交钩子也跑它
# warehouse 闭环评测（runbook docs/warehouse.md 第 6 步；$RUN 的 eval 块决定 lockstep 节拍与延迟）
bash scripts/start_stack.sh eval outputs/runs/$RUN --gpu 0
bash scripts/run_vla_server.sh outputs/runs/$RUN --gpu 0
bash scripts/run_isaac_mirror.sh outputs/runs/$RUN --gpu 1    # 或 start_unreal.sh --gpu 1 后 run_unreal_mirror.sh
bash scripts/bimanual/run_vla_inference.sh outputs/runs/$RUN --gpu 0
# 本次的检查脚本（四步串起来、收尾先杀子进程）：outputs/runs/t7_mirror_check_{isaac,unreal}_20261007/check_run.sh
# 延迟分布：从 lockstep 评测 traces 生成（每行 infer_s）
"$VENV_SIM_PY" -m locomani.tasks.crate.vla_lockstep latency-dist <run>/eval/traces --out x.json --inference-only --fps 50
# 真机/sim2sim（src/real_robot/README.md）
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gr00t_server.sh --model-path outputs/runs/<run>/model/checkpoint-<step>
bash scripts/run_real_robot_inference.sh --host localhost --init-token-file configs/deployment/warehouse_initial_token.json [--hand-wire sim]
# 提交：按文件名 add（不要 -A，别带上 assets/hero.png、.serena、node_modules）→ commit → 用户说了才 push
```
