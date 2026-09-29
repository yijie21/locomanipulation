# Unreal（URLab）渲染线：产物、命令、坑（2026-09-28 起；2026-09-29 接成 `images.source: unreal`）

`STATE.md` 只写进度和下一步，本文件放可以直接照抄执行的命令和路径。
决策与被否决的方案见 `DECISIONS.md` 2026-09-28、2026-09-29 各节，待办见 `TODO.md` 第 12 节。
怎么选 Unreal 出图、和 Isaac 的对照见 [rendering.md](../rendering.md#图像来源imagessource)；安装见 [setup.md](../setup.md) 第 8 步。

## 代码与产物位置

- 正式代码 `src/locomani/sim/unreal/`（在 `$BRIDGE_PY` 里跑）：
  - `editor.py`：向运行中的编辑器发 Python（TCP :18801）、开关卡、查出图默认值（无 Nanite 网格、`r.Shadow.Nanite 0`、视口关）
  - `view.py`：`UnrealView`，镜像和数据集补渲共用的出图对象（每次 fresh，截 `repeats` 遍）
  - `mirror.py` / `render_dataset_cameras.py`：评测镜像、数据集补渲；协议和补渲流程与 Isaac 共用（`sim/mirror.py`、`sim/view_render.py`）
  - `export_mjcf.py` / `build_level.py`：场景 → 机器人 + 纸箱 + 三路数据集相机的 MJCF → 该场景的关卡。2026-09-29 起导入里还有第三视角录像用的
    自由刚体相机 `loco_third_person_rig`（fovy 取场景机位，720×540）和两只 ghost 夹爪（材质 `/Game/Loco/M_LocoGhost`，半透明），平时停在地板下 5 m；
    `source.json` 的 `film` 记下相机的 fovy 和尺寸。之前建的关卡补渲第三视角时报 `rebuild`，重建即可（镜像和数据集补渲不受影响）
  - `in_editor/`：在编辑器里跑的脚本（命令服务、清导入、G1 材质、去 Nanite、房间关卡）
- 脚本：`scripts/start_unreal.sh` / `stop_unreal.sh`（`scripts/unreal/editor_guard.sh` 看守：限 4 核、CPU 打满或 GPU 卡死就杀）、
  `run_unreal_mirror.sh`、`render_dataset_cameras_unreal.sh`、`setup_unreal.sh`（宿主工程模板 `scripts/unreal/host/`）、
  `scripts/unreal/warehouse_bake.py`（在 Blender 里烘焙房间）
- 本机安装 `$UNREAL_ROOT=/workspace/ue_57`（`/workspace/locomani.local.sh` 指过去；按 setup 装的机器默认 `third_party/unreal/`）：
  - `UE_5.7.4/` 引擎；`projects/URLabSmoke/` 宿主工程（插件 `Plugins/UnrealRoboticsLab` 在本地分支 `loco-fresh-capture`，提交 `58c8a98`（取图模式）、`923b689`（端口），合起来 = `efbc5e2` + 补丁）
  - `urlab_bridge/.venv` 客户端（`$BRIDGE_PY`，mujoco 3.10.0）；`logs/`（`editor_run.log`、`guard.log`）；`backups/`（房间原 Nanite 网格、原 ini）
  - 关卡：`/Game/Levels/Warehouse` 是房间关卡（无机器人）；每个场景一个 `/Game/Levels/<场景名>`（2026-09-29 建了 warehouse_textured 和四个 warehouse_box_*）
- 每个场景导出的 MJCF 和 `source.json`（场景 XML 的哈希）：`$UNREAL_ASSETS/<场景名>/`，默认 `$UNREAL_ROOT/levels/`，本机是 `/workspace/ue_57/levels/`（2026-09-29 从 worktree 的 `assets/unreal/` 挪来，TODO 12.9.11）
- 实验脚本（诊断用）：`scripts/experiments/unreal_{smoke,trace_video,capture_bench,pose_set,g1_lookdev,export_mjcf}_20260928.py`
- 09-28/29 的测速、A/B、对 Blender 标定：`outputs/unreal_render_speed_20260928/`、`outputs/unreal_blender_match_20260929/`；
  接成开关时的验证：`outputs/unreal_images_source_20260929/`

## 关键命令

```bash
. scripts/paths.sh
bash scripts/start_unreal.sh --gpu 1                     # 常驻编辑器；已在跑就直接返回
bash scripts/build_scene.sh warehouse_textured --unreal  # 场景 MJCF 变了才重建；建好的关卡存盘，重开编辑器不用再导入
bash scripts/render_dataset_cameras_unreal.sh outputs/runs/<run>        # 数据集补渲（convert 之前）
bash scripts/run_unreal_mirror.sh outputs/runs/<run>                    # 评测镜像（栈起来之后、跑局之前）
bash scripts/run_overnight.sh <实验.yaml> --gpu 0 --render-gpu 1 --line bimanual   # 整条线，按 images.source 选渲染器
bash scripts/stop_unreal.sh
# 在编辑器里跑 Python / 一个编辑器侧脚本
"$BRIDGE_PY" -m locomani.sim.unreal.editor 'print(unreal.SystemLibrary.get_engine_version())'
"$BRIDGE_PY" -m locomani.sim.unreal.editor -f src/locomani/sim/unreal/in_editor/no_nanite.py
# 房间关卡（只在改了 Blender 烘焙之后）；只改它的自发光或后处理体积：把 WAREHOUSE_EMISSIVE_ONLY / WAREHOUSE_EXPOSURE_ONLY 设为 1
"$BRIDGE_PY" -m locomani.sim.unreal.build_level --room outputs/unreal_warehouse_20260928/room
# 对 Blender 参考图标定（reverse_room；每个候选重启一次 PIE）
"$BRIDGE_PY" outputs/unreal_blender_match_20260929/match.py --tag <tag> --candidates base 'b4:lumen_diffuse_color_boost=4'
# 诊断（实验脚本）：trace 视频（默认 stream）、测速、同一会话画面 A/B（--repeats 64 作收敛参考）、GPU profile
"$BRIDGE_PY" scripts/experiments/unreal_trace_video_20260928.py --xml <scene.xml> --trace <npz> --trace-joints <json> --camera reverse_room --out <dir>
"$BRIDGE_PY" scripts/experiments/unreal_capture_bench_20260928.py --trace <npz> --trace-joints <json> --configs fresh:all:8 --out <dir>
"$BRIDGE_PY" scripts/experiments/unreal_pose_set_20260928.py render --trace <npz> --trace-joints <json> --out <dir> --tag x_a
"$BRIDGE_PY" -m locomani.sim.unreal.editor "unreal.SystemLibrary.execute_console_command(None, 'ProfileGPU')"
# 改插件后重编（先 stop_unreal），约 25 s；改完重新生成补丁
bash "$UE_ENGINE/Engine/Build/BatchFiles/Linux/Build.sh" URLabSmokeEditor Linux Development "-Project=$UE_PROJECT" -WaitMutex
git -C "$(dirname "$UE_PROJECT")/Plugins/UnrealRoboticsLab" diff efbc5e2 > patches/urlab_fresh_capture.patch
```

## 已确认的坑

- **URLab 导入的蓝图原先不写盘（2026-09-29 已修，TODO 12.11）。**
  - 原因：插件的 `ImportXmlSync` 只在内存里建蓝图；网格和材质实例由后面的步骤存盘，关卡由 `save_level` 存。下次开编辑器，关卡里的机器人 actor 就没有类了。
  - 现在 `build_level` 导入后会把 `/Game/MuJoCoImports/<关卡名>` 存盘。
  - 验证：重启编辑器后不重新导入，和刚导入时的图差 0.5–0.7，跨会话噪声底是 0.65–1.1（`outputs/unreal_persist_20260929/`）。插件崩溃后重启也一样能直接用。
  - URLab 经 Interchange 导入，默认会开 Nanite，所以重新导入后仍要跑去 Nanite。
- **不要复制关卡资产再打开（`EditorAssetLibrary.duplicate_asset`）：编辑器直接退出。** 副本 World 留在内存，Map_Load 的泄漏检查是致命错误（EditorServer.cpp:2524，2026-09-29）。`build_level` 改为打开房间关卡、`save_map` 另存、再打开副本。
- **URLab 原版的端口和实例 0 撞车（2026-09-29 已修）。** 原版写死 5555–5559（相机从 5558 往上找），正是实例 0 的端口段；编辑器常驻时实例 0 的栈报 `Address already in use`。补丁让所有端口从插件 ini 的 `[BridgeServer] PortBase` 派生（默认 5800：状态 5800、控制 5801、info 5802、相机从 5803、step RPC 5804），客户端 `editor.connect` 读同一个 ini。实测：编辑器在 PIE 时占 5800–5806 和 18801，实例 0 的栈和 lockstep 评测同时正常。
- **URLab 的导入要在一个 Python 里跑网格预处理（trimesh、scipy、networkx、Pillow）。** 插件从自己的 `Config/LocalUnrealRoboticsLab.ini`（`PythonPath=`）找这个 Python；没有就弹窗问要不要装进 UE 自带的 Python，无人值守的编辑器会自动取消，导入失败（`ImportAssetsAutomated returned no objects`）。`setup_unreal.sh` 把这几个包装进客户端 venv，并写好这个 ini。
- **删掉编辑器正开着的关卡会让编辑器崩溃**（`ULevelInstanceSubsystem::OnAssetsPreDelete`）。宿主工程一启动就打开房间关卡，所以 `in_editor/warehouse_level.py` 重建房间前会先切到引擎的空地图。
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
