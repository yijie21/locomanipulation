# Unreal（URLab）渲染线：产物、命令、坑（分支 `loco_unreal`，2026-09-28）

`STATE.md` 只写进度和下一步，本文件放可以直接照抄执行的命令和路径。
决策与被否决的方案见 `DECISIONS.md` 的 2026-09-28、2026-09-29 各两节，待办见 `TODO.md` 第 12 节（2026-09-29 合入 main 时由第 10 节改号，台账 `blender_scenes.md` 的分支各节同时顺延为第 4–8 节）。

## 产物位置

- UE：`/workspace/ue_57/`
  - `UE_5.7.4/` 是引擎
  - `projects/URLabSmoke/` 是宿主工程：
    - `Config/DefaultEngine.ini`：`r.Shadow.Nanite=0`、`urlab.FreshCaptureRepeats=8`、`urlab.CameraNearClipCm=2.0`；
      `EditorStartupMap=/Game/Levels/Warehouse`，编辑器一启动就打开 warehouse 关卡（2026-09-29）
    - `Config/DefaultEditorSettings.ini`：`bSendUnattendedBugReports=False`、`bSendUsageData=False`，崩溃报告不再上传 Epic（2026-09-29）
    - 编辑器视口关闭（`ShowFlag.Rendering 0`）由 `/workspace/ue_57/ue_cmd_server.py` 在启动时设。`ShowFlag.*` 是作弊类 cvar，工程 ini 设不了
    - 插件在 `Plugins/UnrealRoboticsLab`，分支 `loco-fresh-capture`；改动未提交，补丁在 `patches/urlab_fresh_capture.patch`
  - `urlab_bridge/.venv` 是 Python 客户端，mujoco 3.10.0
  - `logs/`：`editor_run.log`、`guard.log`
  - `backups/`：房间原 Nanite 网格、原 `DefaultEngine.ini`
- MJCF：`outputs/unreal_warehouse_20260928/mjcf/`
  - `dynamic.xml`：机器人 + 纸箱 + 三个数据集相机 + `reverse_room`（1280×960）
  - `scene.xml`：MuJoCo 参考用的完整场景
- Blender 房间：`outputs/unreal_warehouse_20260928/room/`（`room.glb`、`room_scene.json`）
- 视频：`outputs/unreal_warehouse_20260928/vla_round/`
  - `unreal_vs_mujoco_preview_part{1,2}of2.mp4`：镜像实时取的图，约 1.4 张/秒，一顿一顿
  - `dataset_cams/`：trace 补渲 8 遍，Nanite
  - `dataset_cams_stream/`：stream，Nanite
  - `dataset_cams_default/`：现默认（无 Nanite + stream，42 ms/帧）
  - `third_person/previews/`：Unreal / Isaac / MuJoCo 第三视角
- 测速与 A/B：`outputs/unreal_render_speed_20260928/`
  - 各配置的 `bench.json`
  - `i_nanite_shadows/`：无 Nanite 的耗时 + 画面 A/B；另有 `ue_up.sh`，一键开编辑器 + 导入
  - `j_viewport_quality/summary.txt`：视口开/关各对 64 遍参考
  - `k_default_check/`：默认生效的验证

## 关键命令

```bash
. /workspace/ue_57/ue_env.sh
cd /workspace/code/locomanipulation-loco_unreal
M=$PWD/outputs/unreal_warehouse_20260928/mjcf
T=/workspace/code/locomanipulation/outputs/runs/unreal_eval_20260928/eval/state_trace/crate_vla_000_state_trace.npz
# 1. 编辑器（4 核 + 看守，TCP 命令服务 :18801）：杀旧的（连看守一起）→ 起 → 等就绪
for p in $(pgrep -f '^bash /workspace/ue_57/guarded_editor.sh') $(pgrep -f '^/workspace/ue_57/UE_5.7.4/'); do kill -KILL $p; done
UE_EXEC_CMDS="py /workspace/ue_57/ue_cmd_server.py" bash /workspace/ue_57/guarded_editor.sh 0 > /workspace/ue_57/logs/guard.log 2>&1 &
until grep -q 'uecmd\] listening' /workspace/ue_57/logs/editor_run.log; do sleep 2; done
# 2. 只在 MJCF（dynamic.xml）变了时重新导入：清旧导入 → 导入机器人 + 纸箱 → 材质 → 去 Nanite（不跑最后这步，镜像和视频拒绝出图）。
#    setup 会把蓝图存盘（2026-09-29 起），所以平时开编辑器就能直接出图，不用再跑这一步
python3 /workspace/ue_57/uepy.py -f scripts/experiments/unreal_clear_import_20260928.py
"$BRIDGE_PY" scripts/experiments/unreal_smoke_20260928.py setup --xml $M/dynamic.xml --existing-level /Game/Levels/Warehouse
python3 /workspace/ue_57/uepy.py -f scripts/experiments/unreal_robot_materials_20260928.py
python3 /workspace/ue_57/uepy.py -f scripts/experiments/unreal_no_nanite_20260928.py
# 3a. 闭环：栈 + 策略服务起来后，用 Unreal 镜像替代 Isaac 镜像（端口取栈的 TF_PRIVILEGED_PORT / TF_ISAAC_MIRROR_PORT）
MUJOCO_GL=egl "$BRIDGE_PY" scripts/experiments/unreal_mirror_20260928.py --sim-scene <run>/warehouse_textured/scene_locomani.xml \
    --privileged-port 5601 --mirror-port 5578 --out <run>/eval/unreal
# 3b. 某局 trace 按仿真时间补渲（上 Unreal、下 MuJoCo），默认 --mode stream；第三视角用 --camera reverse_room
MUJOCO_GL=egl "$BRIDGE_PY" scripts/experiments/unreal_trace_video_20260928.py --xml $M/scene.xml --trace $T \
    --trace-joints $M/scene_joints.json --camera head_camera,left_wrist,right_wrist --stride 2 --out <dir>
# 4. 测速：模式:相机:遍数
"$BRIDGE_PY" scripts/experiments/unreal_capture_bench_20260928.py --trace $T --trace-joints $M/scene_joints.json \
    --configs fresh:all:1 fresh:all:8 stream:all:1 --out <dir>
# 5. 画面 A/B：同一会话每种设置渲两次（a、b）当噪声底；--repeats 64 作收敛参考
"$BRIDGE_PY" scripts/experiments/unreal_pose_set_20260928.py render --trace $T --trace-joints $M/scene_joints.json --out <dir> --tag x_a [--cvar ...] [--repeats N]
"$BRIDGE_PY" scripts/experiments/unreal_pose_set_20260928.py compare --out <dir> --tags x_a x_b y_a [--ref ref64]
# GPU profile：发 ProfileGPU，从 editor_run.log 读表
python3 /workspace/ue_57/uepy.py "unreal.SystemLibrary.execute_console_command(None, 'ProfileGPU')"
# 改插件后重编（先杀 UE），约 25 s；改完重新生成补丁
bash "$UE_ROOT/Engine/Build/BatchFiles/Linux/Build.sh" URLabSmokeEditor Linux Development "-Project=$UE_PROJECT" -WaitMutex
git -C "$URLAB_ROOT" diff efbc5e2 > patches/urlab_fresh_capture.patch
# 只改现有关卡的自发光材质或后处理体积（先停 PIE；变量名见脚本开头，WAREHOUSE_EMISSIVE_ONLY / WAREHOUSE_EXPOSURE_ONLY）
python3 /workspace/ue_57/uepy.py "import os; os.environ.update(WAREHOUSE_DIR='$PWD/outputs/unreal_warehouse_20260928/room', WAREHOUSE_EXPOSURE_ONLY='1')
try: exec(open('$PWD/scripts/experiments/unreal_warehouse_level_20260928.py').read(), {'__name__': '__main__'})
finally: os.environ.pop('WAREHOUSE_EXPOSURE_ONLY')"
# 对 Blender 参考图标定（reverse_room；每个候选重启一次 PIE）
"$BRIDGE_PY" outputs/unreal_blender_match_20260929/match.py --tag <tag> --candidates base 'b4:lumen_diffuse_color_boost=4'
# 重建 warehouse 关卡（改了 Blender 烘焙之后；导入时已关 Nanite）
WAREHOUSE_DIR=$PWD/outputs/unreal_warehouse_20260928/room UE_EXEC_CMDS="py /workspace/ue_57/ue_cmd_server.py, py $PWD/scripts/experiments/unreal_warehouse_level_20260928.py" \
    bash /workspace/ue_57/guarded_editor.sh 0 > /workspace/ue_57/logs/guard.log 2>&1 &
```

一局 Unreal 闭环评测的完整流程：
1. `make_runs.py <run> collect.instance=2 ... policy_port=5720 images=isaac`
2. `start_stack.sh eval`
3. `run_vla_server.sh`
4. 上面第 1、2、3a 步起 Unreal 镜像
5. `scripts/bimanual/run_vla_inference.sh`

## 已确认的坑

- **URLab 导入的蓝图原先不写盘（2026-09-29 已修，TODO 12.11）。**
  - 原因：插件的 `ImportXmlSync` 只在内存里建蓝图；网格和材质实例由后面的步骤存盘，关卡由 `save_level` 存。下次开编辑器，关卡里的机器人 actor 就没有类了。
  - 现在 `unreal_smoke_20260928.py setup` 导入后会把 `/Game/MuJoCoImports/<xml 名>` 存盘。
  - 验证：重启编辑器后不重新导入，和刚导入时的图差 0.5–0.7，跨会话噪声底是 0.65–1.1（`outputs/unreal_persist_20260929/`）。插件崩溃后重启也一样能直接用。
  - URLab 经 Interchange 导入，默认会开 Nanite，所以重新导入后仍要跑去 Nanite。
- **glTF 母材质不渲自发光（2026-09-29 已修，TODO 12.3.2）。** 天花板灯板原先只是被照亮的漫反射面。关卡脚本现在把自发光材质换成 `M_LocoEmissive`，见台账 `docs/ledger/blender_scenes.md` 第 7 节。
- **客户端 `close()` 会把插件切回 live，插件偶发段错误。** 崩溃点在 `UURLabZmqPublishTransport::PostStep` 的空指针，2026-09-29 出现过。
  - 另一次崩溃是在前一个脚本不调 `close()` 就退出之后，下一个客户端发 `hello` 时。
  - 两次之前都在 PIE 里直接改过相机的后处理（这种改法对截图本来就无效，已不用）。因果未查明。
  - `match.py` 先停 PIE 再 `close()`，没有再崩。
- **PIE 运行时，`get_editor_world()` 和 `list_assets()` 都返回空**：改编辑器世界和资产前先停 PIE。
- **截图遍数（`urlab.FreshCaptureRepeats`）。** 一遍不够，是因为 Lumen 的全局光照和 TSR 抗锯齿都靠多帧累积；fresh 模式第一遍做 camera cut，历史清零。
  下表是三路相机对 64 遍收敛参考的平均差（0–255），在无 Nanite、视口关的配置下测；两次 64 遍之间差 0.3（`outputs/unreal_repeats_20260929/`）：

  | 遍数 | 与参考差 | 整体偏暗 | 每次取图 |
  |---|---|---|---|
  | 1 | 3.4–3.9 | 2.1–2.9 | 56–64 ms |
  | 2 | 2.4–2.9 | 1.4–2.1 | 68–74 ms |
  | 4 | 1.7–2.1 | 1.0–1.5 | 92–94 ms |
  | 8（默认） | 1.2–1.4 | 0.6–1.0 | 137 ms |

  同一遍数渲两次的差：1 遍 0.9–1.0，8 遍 0.5–0.7。
  偏暗是系统性的，训练图和评测图用同一遍数就互相一致；遍数少的代价是离收敛远、两次之间的抖动大一些。
- **`dynamic.xml` 里多余的世界相机会每帧渲染**，直到被截过一次。脚本启动时会先把所有相机截一次。
- **阴影设置只能在启动时改**（`UE_EXTRA_ARGS="-ini:Engine:[/Script/Engine.RendererSettings]:..."`）：
  - 开 VSM（`r.Shadow.Virtual.Enable=1`），第一次按需取图 GPU 就卡死；
  - 运行时改 `r.ShadowQuality`，编辑器断言崩溃。
- **UE 崩溃时，CrashReportClient 原先会自动把报告上传到 Epic（2026-09-29 已关，TODO 12.12）。**
  - 引擎默认 `bSendUnattendedBugReports=True`；编辑器以 `-unattended` 运行时，这个值为 False 就根本不启动 CrashReportClient（`UnixPlatformCrashContext.cpp` 的 `bSkipCRC`）。
  - `GetIni EditorSettings:/Script/UnrealEd.CrashReportsPrivacySettings bSendUnattendedBugReports` 读回的合并值是 False。之后两次插件崩溃都没有生成新的崩溃报告目录。
- **比较 Unreal 图时要控制噪声底。** 必须在同一编辑器会话里同一设置渲两次，拿这两次的差当噪声底；跨会话的差本身就有 0.65–1.1。
- **编辑器 SIGTERM 会 segfault**（无害），停 UE 一律用 `kill -KILL`。
  - 不要用 `pkill -f` 匹配带自身命令行的模式，会把自己的 shell 也杀掉。
- **提交钩子会对整棵树跑 `ruff format --check`**，没提交的实验脚本也会让提交失败。
