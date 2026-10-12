# STATE — 2026-10-12（会话：抓取来源接口与文档——默认离线缓存、graspnet 可选；之前：执行侧杠杆、GraspNet 对比、单臂 CuRobo 路线）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。
已确认（10-12）：实例 1–4 空，graspnet 和 CuRobo 服务都已停；GPU1 上是别的项目的 Isaac（`sim_server.py` + `run.py`，约 25 GB、约 3 核），它在跑时开 3 条以上仿真通道会把 CPU 配额压满。
已确认：henry-zhao 直接推 origin/main 或 `henry-dev`，可能自带检查失败。

## 提交（10-12，用户：commit and push）

- 已确认：主线 CuRobo 出路线开关（TODO 22.6）提交为 `0035add`；GraspNet 线、抓取缓存线、执行层改动、抓取来源接口和文档（21.3.4.5、23.2.8、23.7）为其后一个提交；合入 origin 的 4 个新提交（real_robot、check_dataset 等）后已推送，`git log origin/main` 为准。
- 没提交（不是这条线的、来路不明）：`papers/`、`node_modules/`、`package*.json`、`uv.lock`、`step1.md`–`step7.md`、`.serena/`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`、`assets/hero.png`。
- 提交钩子会跑整套 CPU 检查，并先把未暂存的改动藏起来；未跟踪的测试依赖未暂存的 src 改动时钩子会失败。STATE.md 上限 60 行。

## 磁盘

- 已确认（10-10）：`/workspace` 剩 62 GB（94%）；跑仿真 / pytest 前 `export TMPDIR=/workspace/tmp`。容器内存 91 GB。
- 已确认（10-10 15:00，用户）：10-10 上午的 749 个 `gnb_obj_*` 运行目录已删（158 → 150 GB），下午的 `_gateA/B`、`_highA/B` 260 个新增；`presort_b/ghost_rounds/`（392 MB）、`cache/videos_best_full/`（202 MB）还在（TODO 23.2.7）；缓存 `scenes/obj_*/grasp_cache.npz` 留。
- 上一会话新增（删不删待用户，TODO 21.3.4.6、22.7）：GraspNet 线约 4.4 GB（`ghost_rounds/`、`fixed/`、`videos_single_full/`、`gnb_obj_*_fixed*`），主线 A/B `outputs/runs/bottle_route_*` 5.2 GB。
- 待用户（更早）：11.9、12.19.4、12.20.3、12.20.5、13.5、13.6.12、13.6.12.1、13.6.17、14.7 的删除都没定；要删先在这里写明 24 小时。

## 当前进度

- 已确认（本会话，handoff `grasp-cache-20261010.md`）：缓存 64/65 物体有验证通过的抓取（中位 243/300）；验收第 0 局（同朝向）缓存 25/65 对 graspnet 16/65（配对 18 对 9，p 0.061 单侧）；三局至少一局 45/65；193 局里执行失败 103（碰倒 49、无双指接触 30、抬掉 18）、规划够不到 17、没候选 8 → 位姿问题基本排除，剩下在执行层（23.4）。
- 已确认（23.4.4 做完，handoff 第 13 节）：进入段闸门（偏差 > 1.5 cm 停推进，最多 2 s）`_gateA/B` 32、33/65，碰倒 7、7；合爪时偏差不变，是手偏着时不顶。高度优先 `_highA/B` 24、24。此前 23.4.6.1–3 三根控制侧杠杆都否定（第 12 节），默认仍 `--base-freeze off`、`--arm-pin arm`、kp 120。运行目录：10-10 的 `_gate*`/`_high*` 260 个还在（约 4 GB）。
- 已确认（23.4.5 做完，handoff 第 10 节）：右臂 arm pin 两批 48 对不 pin 40 / 130（p 0.19），碰倒 35 → 22；IK 右臂参考自碰撞是新发现（肩横滚上限 0.05 + 闭环吃规划器限位后才跟得准）；剩余横向 2.9 cm 在骨盆晃动。默认仍 off（23.4.5.2 待用户）。src 改了 `supported_motion.py`（joint_limits 形参）、`run_sim_locomani.py`（pin 重力前馈、腰用控制器增益）、`privileged_pub.py`（pin 日志只打转换）。
- 已确认：单臂 curobo 开关（关节空间目标，抬到原地抬手的抬起位姿）D / D' 36、35 对 A / A' 35、34；抬手 76 局 0 接触；可排除成功率降超约 6 个百分点。停下到预抓慢 2 s。

## 下一步（编号见 `TODO.md`）

- 用户 10-12：抓取来源接口和文档做完（TODO 23.7，读者页 `docs/object_grasp.md`）；默认缓存 + IK + 闸门两批 32、33 / 65。待用户：**23.7.4**（graspnet 在现在默认下两批）、**23.3**、**23.2.7**（剩余产物）。
- **22.3.6**（待用户）：默认改成 curobo 与否；改前先做 **22.3.7**（实例 1 / 2 上跑一组 curobo 查 neutral waist）。

## 未解决 / 卡点

- 待用户：23.2.7、23.3–23.5、23.7.4、22.3.6、22.3.8、22.5、22.7、21.3.4.2–21.3.4.4、21.3.4.6、21.3.4.8.1、21.3.1 留出目标、19.8.1、19.8.3、19.18.3.1、16.14.1、16.15.4、13.6.18。
- 已确认（工具）：CuRobo 0.7.4 批量为 1 有两个崩溃 bug；有的请求会让 MotionGen 之后全 IK_FAIL，只有重建能救（服务已自愈）；同一请求换调用历史路线不复现（中位 0.03 rad），只有成败稳定。
- 已确认（环境）：主机内存碎片时 numpy 大页缺页卡在内核整理——离线 numpy 重活设 `NUMPY_MADVISE_HUGEPAGE=0`（记忆 numpy-hugepage-compaction-stall）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && export TMPDIR=/workspace/tmp
make check; make test TESTS="tests/test_bottle_simple.py tests/test_arm_route.py"
# CuRobo 路线服务（GPU0）——curobo 开关的局需要它：
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src nohup setsid third_party/SIMPLE/.venv/bin/python -m locomani.primitives.curobo_service \
    outputs/graspnet_bottle_20261008/curobo/g1_right_curobo.json --port 5795 > outputs/curobo_route_20261009/service.log 2>&1 &
# 一组 40 局（实例在 yaml 里）：先起栈再跑专家，专家跑完自己收栈
bash scripts/start_stack.sh collect configs/experiments/bottle_route_curobo_js_20261010.yaml --gpu 0
bash scripts/single_arm/run_agent_server.sh configs/experiments/bottle_route_curobo_js_20261010.yaml --gpu 0
# 分析：table_clearance.py <run>（离桌间隙）→ compare_runs.py A=<run> D=<run> --pairs A:D（scripts/experiments/curobo_route_20261009/）
# 抓取缓存线（TODO 23.2）：缓存 5 分钟；验收批 65 物体 × 3 朝向约 1 小时（4 通道，实例 1–4）；汇总 → outputs/graspnet_bottle_20261008/cache/summary.md
bash scripts/experiments/graspnet_bottle_20261008/cache_all.sh 12
ROUNDS=3 EGL_GPU=1 bash scripts/experiments/graspnet_bottle_20261008/run_batch.sh 4 1 0 _cache --grasp-source cache --planners ik --rounds 3 --yaws 3
python3 scripts/experiments/graspnet_bottle_20261008/cache_summary.py _cache
# 停服务：按端口查 PID 再 kill（不要 pkill -f）；停栈：bash scripts/stop_stack.sh --instance N
```
