# STATE — 2026-09-29（会话：warehouse 默认四边采集 + 采集前提醒；双臂线并行采集通道；已推送）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，与 origin/main 同步于 `4dd5649`（之后只有这次 handoff 提交）。
已确认：本会话起的栈、Isaac 补渲都已收掉；2026-09-29 01:49 UTC 实例端口 5555–5699 没有任何程序监听，GPU0 2 MiB、GPU1 3698 MiB。
别的会话：`locomanipulation-loco_agent` 检出目录会在实例 1、GPU 1 起栈；GPU 1 常驻无关的 `depth_server.py`（约 3.7 GB）。

## 磁盘（先处理）

- 已确认：`/workspace` 只剩 19 GB（99%）。
- 计划删除（TODO 9.9，上一会话写明）：`outputs/runs/{bottle_simple_smoke,crate_smoke,warehouse_smoke}/model/`（各约 24 GB）和
  `outputs/docs_smoke_20260928/try2/`（24 GB），**2026-09-29 11:45 之后**；用户同意可提前。运行目录其余文件保留。
- 本会话产物约 1 GB + exporter session 1.2 GB（TODO 11.9 列了路径）：最早 2026-09-30 02:00 UTC 可删，待用户同意；
  `outputs/runs/warehouse_smoke_lanes_20260928` 的 manifest 与 `collect/` 是并行通道的证据，保留。

## 当前进度（TODO 第 10、11 节；DECISIONS 2026-09-28 最后两节）

- 已确认（10）：warehouse 默认采集是四边随机——`warehouse_smoke` / `warehouse_200` 用 `warehouse_edges_compact_45x32x33_20260928`
  的 task 块，场景 `warehouse_box_45x32x33`；只从一个地方开始的是 `warehouse_{smoke,200}_fixed_start`（只差摆放，有测试）。
  采集第一局前打印摆放设置（`expert.layout_lines`）。实跑 `warehouse_smoke_edges_20260928`：4 局抽到 yp/xm 两边，保留 2 条。
- 已确认（11）：双臂线采集默认并行。`start_stack.sh collect` 用 `pipeline/lanes.py` 量空闲 CPU（3 秒）、线程数、每卡显存、
  空闲实例（`stack.instance_busy` 查栈、锁、端口），定通道数、打印、逐个起栈、写 `collect/lanes.json`；`run_agent_server.sh`
  通道 0 在本进程，其余 spawn 子进程；批次在文件锁下按编号领取（`batches.claim_batch/finish_batch`）。
- 已确认（11.1 实测，脚本在 `outputs/warehouse_parallel_20260928/`）：一条 warehouse 通道 CPU 平均 +2.6 / p90 +3.7 核、显存 +1.5 GB、线程 +180 起每录一局约 +130。
- 已确认（11.2.1）：`warehouse_smoke_lanes_20260928` 两条通道（实例 2/GPU0、实例 3/GPU1）8 局保留 5 条，RTF 0.974–0.996，
  容器线程峰值 3279/3840；Isaac 补渲 5 集 + convert（4 训练 + 1 验证）通过，两条通道的 session/manifest 配对正确。

## Isaac 新外观 + `images.source: unreal`（`loco_unreal`；TODO 第 12 节，DECISIONS 2026-09-29 各节）

- 已确认：G1 用 Unitree 官方材质；warehouse（含四边线）stage `*_env`、按 Blender 标；`box_platform_srgb`；2× 超采样。旧 stage 别删；新外观不重训（用户定），新旧评测数字不可直接比。已确认（12.9，分支 `loco_unreal`，未合回 main）：`images.source: unreal` 已接通。warehouse_textured 和四个 warehouse_box_* 各有关卡，补渲 + convert 和 3 局 lockstep 闭环（3/3，Isaac 训练的 checkpoint，分布偏移）都实跑过。还没用 Unreal 图训练（12.9.6）。命令见 [handoff](docs/handoff/unreal-urlab-20260928.md)；URLab 的端口已挪到 5800–5812（插件补丁的 PortBase），编辑器常驻时实例 0 的栈照常可用。

## 待验证 / 卡点

- 待验证（10.7）：终端里“打印后等 10 秒再开第一局”这条分支没实跑过（两次实跑都是 nohup）。
- 待验证（11.7）：多通道的产出率是否与单通道相同，只有 8 局冒烟；完整规模要按产出率和 RTF 再核。
- 假设（11.6）：crate 这种实时渲染通道的成本沿用旧单栈预算（10 核、6000 MiB、一卡一条），没实测；这台机器上 crate 目前只开得出一条。
- 待用户（10.4）：本机 `outputs/runs/warehouse_smoke` 是旧配置跑的，按 runbook 同名重跑会报 `config.yaml differs`——删旧目录还是文档写换 `--name`。
- 待定（11.10）：旧单栈闸门（1800 线程、8 核）仍管单臂线和没有 `lanes.json` 的栈，是否按实测改小。
- 上一会话留下：9.3 bottle lockstep 对照；9.10 crate 完整规模 lockstep；9.8 新文档写法待认可；8.6.4 四边 compact 采集待定；7.2；8.11 可删 trial 目录。

## 下一步（编号见 `TODO.md`）

- **9.9**：2026-09-29 11:45 之后删冒烟的 `model/` 和 `try2/`（磁盘只剩 19 GB）。
- **10.4**：问用户旧 `outputs/runs/warehouse_smoke` 怎么处理。
- **10.5 / 11.7**：用户同意后按默认（四边 + 自动通道）跑 `warehouse_200` 采集，顺带核对多通道产出率。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh     # 路径、PYTHONPATH，并读 /workspace/locomani.local.sh
df -h /workspace; cat /sys/fs/cgroup/pids.current; timeout 30 bash scripts/cpu_headroom.sh 8; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# 通道计划：只打印、不起栈（每项资源允许几条、跳过了哪些实例）
bash scripts/start_stack.sh collect configs/experiments/warehouse_smoke.yaml --dry-run --name plan_check
# 默认采集（所有卡、自动通道）；--lanes 1 = 旧行为；--gpu 0,1 限卡；*_fixed_start = 只从一个地方开始
bash scripts/start_stack.sh collect configs/experiments/warehouse_smoke.yaml --name <run>
bash scripts/bimanual/run_agent_server.sh configs/experiments/warehouse_smoke.yaml --name <run>
# 驱动被强杀后关掉所有通道的栈
for i in $(python3 -c "import json,sys;print(' '.join(str(l['instance']) for l in json.load(open(sys.argv[1]))['lanes']))" outputs/runs/<run>/collect/lanes.json); do bash scripts/stop_stack.sh --instance $i; done
du -sh outputs/runs/*_smoke/model outputs/docs_smoke_20260928/try2             # 9.9 要删的东西
# 提交：make check → 按文件名 add（不要 -A）→ git commit → fetch；远端有新提交就 merge（不 rebase，TODO 里记着提交哈希）→ push
```
