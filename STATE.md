# STATE — 2026-09-28（分支 `loco_unreal`：Unreal / URLab 替代 Isaac 渲染）

工作目录 `/workspace/code/locomanipulation-loco_unreal`（worktree），分支 `loco_unreal`，基于 `9955487`，落后 main（main 已到 `7135a83`，STATE 归别的会话管）。
主仓库 `/workspace/code/locomanipulation` 的 STATE / TODO 不在这里维护；本文件只管这条分支。

**未提交**（等用户说，TODO 10.10）：`scripts/experiments/unreal_{export_mjcf,smoke,warehouse_bake,warehouse_level,robot_materials,clear_import,g1_lookdev,mirror,trace_video}_20260928.py`、`patches/urlab_fresh_capture.{patch,md}`。
`assets/hero.png` 是用户放的 URLab 主页图，不要提交。
没有运行中的 UE、镜像或策略服务。GPU0 空闲；GPU1 上约 5 GB 是别的项目的。实例 1 上有另一个会话的采集栈（`/workspace/tmp/p03_i1/stack.json`），不要动。

## 当前进度

- 已确认：UE 5.7.4 在本容器可以无显示器运行（Vulkan，`-RenderOffscreen`）。首次编译着色器 389 s，之后 10 s 启动。
- 已确认：插件补丁 `"fresh"` 取图后，交替两个姿态 0/50 张拿到旧姿态。纯机器人关卡单相机 54–67 ms；warehouse 3 相机截 8 张 679 ms/次；第三视角 1280×960 单相机 276 ms/帧。
- 已确认：warehouse 从 Blender 烘焙导入，UE 里的包围盒与 (x, −y, z)·100 的坐标完全一致；锯齿和噪点经非 2 的幂贴图缩放 + TSR + camera cut 后明显减轻。
- 已确认：URLab 自带的 G1 与我们的 G1 是同一套网格（37/37 逐字节相同）。hero.png 的外观来自材质和灯光。用户认可 `mujoco` 预设。
- 已确认：一局闭环 VLA 看 Unreal 图，seed 6000、lockstep、latency 2，1/1 ok。运行目录 `/workspace/code/locomanipulation/outputs/runs/unreal_eval_20260928`，实例 2。
- 已确认：同一局第三视角的 Unreal / Isaac / MuJoCo 三路视频已出，逐帧对齐（739 帧，59.1 s）。
- 假设：一局成功不说明 Unreal 图对这个 checkpoint 没影响（它是用 Isaac 图训练的）。要 100 局才能判断（TODO 10.8）。

## 下一步（编号见 `TODO.md` 第 10 节，顺序等用户定）

- **10.7**：取图速度。扫 `urlab.FreshCaptureRepeats`（2–8），看画质和耗时。
- **10.3.2 / 10.3.3**：天花板灯板不发光；对照 Cycles 标定曝光。
- **10.8**：Unreal 图 100 局评测，对比 Isaac 88/100。

## 卡点 / 待验证

- 待验证：天花板灯板为什么是暗的（glTF 的 emissive_strength 乘 683 已设，但 Lumen 截图里不亮）。
- 待验证：纯机器人关卡里的白盘，在近裁剪改成 2 cm 后是否也消失（只在 warehouse 里验证过）。
- 已确认的坑：URLab 导入的蓝图不写盘。每次开编辑器都要先 clear 再 setup，然后跑材质脚本（TODO 10.11）。
- 已确认的坑：`dynamic.xml` 里多余的世界相机会每帧渲染，直到被截过一次。镜像脚本启动时会先把所有相机截一次。

## 产物位置

- UE 安装和工程：`/workspace/ue_57/`
  - `UE_5.7.4/` 引擎
  - `projects/URLabSmoke/` 宿主工程；插件在 `Plugins/UnrealRoboticsLab`，分支 `loco-fresh-capture`
  - `urlab_bridge/.venv` 是 Python 客户端，mujoco 3.10.0
- 导出的 MJCF：`outputs/unreal_warehouse_20260928/mjcf/`
  - `dynamic.xml` 是机器人 + 纸箱 + 三个数据集相机 + `reverse_room`（1280×960）
  - `scene.xml` 是 MuJoCo 参考用的完整场景
- Blender 房间：`outputs/unreal_warehouse_20260928/room/`（`room.glb`、`room_scene.json`、`bake_report.json`）
- 结果：`outputs/unreal_warehouse_20260928/`
  - `v13/summary.png`、`g1_lookdev/g1_dex1_unreal_vs_mujoco.png`
  - `vla_round/{keyframes.jpg, unreal_vs_mujoco_preview_part{1,2}of2.mp4}`
  - `vla_round/third_person/previews/{unreal,isaac,mujoco}_third_person.mp4` 和 `three_way.mp4`

## 关键命令

```bash
. /workspace/ue_57/ue_env.sh      # HOME/TMPDIR 放到 /workspace；UE_ROOT、UE_PROJECT、BRIDGE_PY
cd /workspace/code/locomanipulation-loco_unreal
# 1. 编辑器：限 4 核（taskset 28-31）+ 看守；常驻的 TCP 命令服务 :18801；GPU 号作为参数
UE_EXEC_CMDS="py /workspace/ue_57/ue_cmd_server.py" bash /workspace/ue_57/guarded_editor.sh 0 > /workspace/ue_57/logs/guard.log 2>&1 &
until grep -q 'uecmd\] listening' /workspace/ue_57/logs/editor_run.log; do sleep 2; done
# 2. 每次开编辑器：清掉旧导入 → 导入机器人 + 纸箱到 warehouse → 材质
M=$PWD/outputs/unreal_warehouse_20260928/mjcf
python3 /workspace/ue_57/uepy.py -f scripts/experiments/unreal_clear_import_20260928.py
"$BRIDGE_PY" scripts/experiments/unreal_smoke_20260928.py setup --xml $M/dynamic.xml --existing-level /Game/Levels/Warehouse
python3 /workspace/ue_57/uepy.py -f scripts/experiments/unreal_robot_materials_20260928.py
# 3a. 闭环：栈 + 策略服务起来后，用 Unreal 镜像替代 Isaac 镜像（端口取自栈的 TF_PRIVILEGED_PORT / TF_ISAAC_MIRROR_PORT）
MUJOCO_GL=egl "$BRIDGE_PY" scripts/experiments/unreal_mirror_20260928.py --sim-scene <run>/warehouse_textured/scene_locomani.xml \
    --privileged-port 5601 --mirror-port 5578 --out <run>/eval/unreal
# 3b. 离线：某局 trace 的第三视角（Unreal | MuJoCo）
MUJOCO_GL=egl "$BRIDGE_PY" scripts/experiments/unreal_trace_video_20260928.py --xml $M/scene.xml \
    --trace <run>/eval/state_trace/<label>_state_trace.npz --trace-joints $M/scene_joints.json --camera reverse_room --out <dir>
# 重建 warehouse 关卡（改了 Blender 烘焙之后）
WAREHOUSE_DIR=$PWD/outputs/unreal_warehouse_20260928/room UE_EXEC_CMDS="py /workspace/ue_57/ue_cmd_server.py, py $PWD/scripts/experiments/unreal_warehouse_level_20260928.py" \
    bash /workspace/ue_57/guarded_editor.sh 0 > /workspace/ue_57/logs/guard.log 2>&1 &
# 改插件后重编（先杀 UE）
bash "$UE_ROOT/Engine/Build/BatchFiles/Linux/Build.sh" URLabSmokeEditor Linux Development "-Project=$UE_PROJECT" -WaitMutex
# 停 UE：用 kill -KILL（SIGTERM 会 segfault，不影响）；不要 pkill -f 带自身命令行的模式
for p in $(pgrep -f '^/workspace/ue_57/UE_5.7.4/'); do kill -KILL $p; done
```

一局 Unreal 闭环评测的完整流程（建运行目录、起栈、策略服务、镜像、跑局）照这次的做法：

1. `make_runs.py unreal_eval_20260928 collect.instance=2 ... policy_port=5720 images=isaac`
2. `start_stack.sh eval`
3. `run_vla_server.sh`
4. 起 Unreal 镜像
5. `scripts/bimanual/run_vla_inference.sh`
