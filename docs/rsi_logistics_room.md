# 场景 RSI 在物流间：双手抱起纸箱

同一套场景 RSI 流程，场景换成扫描的物流间：机器人走到托盘平台前，用两只手把纸箱抱起来、举高、稳住。流程机制和 ADT 公寓那条线一样，见[场景 RSI 详解](rsi_design.md)；ADT 公寓的命令见 [runbook](rsi.md)；这条线的决策、理由和每次运行的记录见[方案与记录](handoff/box-bimanual-rsi-plan-20261023.md)。代码在分支 `henry-logistic-room`。

任务族 `box_bimanual`，两个任务：纸箱长边朝外（`box_long_side`）和短边朝外（`box_short_side`），纸箱近侧面离最近的台沿 0–0.15 m，机器人出生在房间里任何能走到的地方，走路 0.25 m/s。成功判据（2026-10-24 起）：
- 两只手的手指都碰着纸箱；
- 纸箱最低的那个角高出平台顶面 20 cm 以上；
- 纸箱底面和水平面的夹角小于 60°（2026-10-24 用户从 30° 放宽）；
- 以上三条在一局最后 5 秒里每 0.1 秒判一次，每次都成立。这 5 秒包括技能的最后 3 秒（机器人可以还在站起来或抬手）和技能结束后原地保持的 2 秒。

耗时是 2026-10 在 SRA0889（一张 RTX 5090）上的实测；模型费按 Claude Opus 计。

| 步骤 | 耗时 | 模型费 |
|---|---|---|
| 建场景（每台机器一次） | 不到 1 分钟 | 无 |
| 可行性探针 | 约 1 分钟 | 无 |
| 内环一轮 | 约 30–40 分钟 | 约 17 USD |
| 封存集 50 局 | 约 2 分钟 | 无 |
| 视频：MuJoCo / Unreal | 几秒 / 约 1 分钟一段 | 无 |

现状（2026-10-24，沙盒 `~/workspace/code/rsi-scene/box-b`，`--max-iters 3`）：内环第 0 轮的候选被接受后现任在验收池上已过 70% 的停止线，循环自己停了，只跑了 1 轮，约 1 小时、20.47 USD。

| 池 | 起步技能 | 第 0 轮接受的版本 `6834ec73d077` |
|---|---|---|
| 诊断池 30 局 | 0 / 30 | 草稿 22 / 25 / 22 / 30，采用第 2 份 |
| 验收池 40 局 | 0 / 40 | 34 / 40（85%） |
| 封存集 50 局（最终成绩） | 0 / 50 | **41 / 50（82%）**：长边朝外 25 / 25，短边朝外 16 / 25 |

封存集的 9 局失败：6 局举起后在 5 秒窗口里掉了（`dropped_hold`），2 局滑落（`slip`），1 局手没送到（`side_reach`）；没有摔倒。停止原因 `the incumbent wins 34/40 >= 70%`。逐步记录见[方案与记录页](handoff/box-bimanual-rsi-plan-20261023.md)第 5.2 节。

## 0. 每个新终端先执行

```bash
. scripts/paths.sh
export TMPDIR=$LOCOMANI_TMP NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2
export RSI_GPUS=0 RSI_PROCS=6 RSI_SIM_SLOTS=6
PY=$VENV_SIM_PY
```

做什么：设置路径变量和仿真用的 Python，固定临时目录和线程数（`OMP_NUM_THREADS` 改成 1 会改变仿真结局）。第三行是一张卡的机器用的：内环、验收门和封存集的仿真进程都放到 0 号卡、6 个并行；两张卡的机器去掉这一行，下面命令里的 `CUDA_VISIBLE_DEVICES=0`、`MUJOCO_EGL_DEVICE_ID=0` 可以改成 1。
前提：[安装](setup.md)里“场景 RSI”一列的步骤做完（仿真环境、ScaleBFM 权重、Claude CLI）；`git lfs pull` 取到场景包 `assets/scene/logistics_room_box_v1/`；`bash scripts/setup_sonic.sh` 做过（场景包引用里面的机器人网格）。这条线不需要 ADT 数据。

## 1. 建场景（每台机器一次）

```bash
$PY scripts/prepare_logistics_scene.py outputs/scene_discovery/logistics_room_mars --robot mars --rsi
SC=outputs/scene_discovery/logistics_room_mars
$PY scripts/scene_settle.py --scene $SC --write
$PY scripts/scene_lint.py --scene $SC
```

做什么：第一条把仓库里的场景包装成 MuJoCo 场景，机器人换成 mars 版 G1（和 ADT 线同一个机器人，Dex1 两指夹爪）。`--rsi` 再补上场景 RSI 要读的字段：从房间网格切出导航地图（墙和货架在仿真里不碰撞，只靠这张地图让导航和出生点避开）、出生区、平台四条能站人的边、纸箱的属性（要两只手、1.2 kg、开口箱），并写出不带机器人的 `env_only.xml`。第二条让纸箱落稳，第三条检查场景。
得到什么：`$SC/` 下 `scene_locomani.xml`、`env_only.xml`、`scene_meta.json`、`settle.json`、`lint.json`；最后一条打印 `[lint] 0 errors, 0 warnings`。第一次跑一局时会在旁边生成 `logistics_room_mars_physics/`（去掉纯视觉网格的物理副本），不用管。场景放在 `outputs/scene_discovery/` 下是有意的：沙盒只把这个目录链接进去。

## 2. 检查代码

```bash
$PY -m pytest tests/test_strict_box_goal.py tests/test_box_bimanual_family.py tests/test_scan_room.py tests/test_scalebfm_motor.py tests/test_rsi_stage.py tests/test_rsi_failure_class.py; echo "exit $?"
```

做什么：这条线新加和改过的代码的单元测试：严格判据（`lifted` 谓词、5 秒判定窗口）、任务族文件、地图切片、双手动作（`grip` / `release` 的 `"both"`、`squeeze`、`reach(keep_squeeze=True)`）、双手检查和失败分类。
得到什么：退出码 0，约 10 秒。仓库的 pytest 配置已经带了 `-q`，命令行再加 `-q` 会连“N passed”那行也不打印，看退出码。

## 3. 可行性探针（开循环之前必做）

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src \
  $PY scripts/rsi/probe_box.py --out outputs/scene_rsi_20261016/box_probe_$(date +%Y%m%d)
```

做什么：先用一段人写的序列证明这个任务做得到，再花钱跑循环。默认用的是 crate / 纸箱线人工定义的双臂抓法（`MODE` 为 `"wall"`）：机器人从场景的默认出生点走到离纸箱最近的台沿外 0.38 m，骨盆降到 0.76 m；两手各对准机器人左右两侧的箱壁，抓点在箱沿下 1 cm、离箱壁外表面 8 mm，张开的夹爪跨在 1 cm 厚的壁上（一根手指在箱内、一根在箱外），手掌向下俯 60°；先到抓点上方 9 cm、往身体方向退 4 cm 的预抓位置，再直线下到抓点，两手同时合爪；然后两手一起向上 32 cm、骨盆站直到 0.79 m，保持。纸箱按两个任务的摆放各随机放 5 次。序列在 `scripts/rsi/probe_box_skills/`，不进沙盒；参数都是大写常量，用 `--set` 改（如 `'{"PALM_PITCH_DEG": 45.0}'`；`MODE` 为 `"hug"` 是两手张开从两侧顶住箱壁的抱法）。每局按任务族的判据判，记录和循环的同格式。
得到什么：终端打印每局成败、失败类型和步骤；`--out` 下 `rounds/`（每局记录和 50 Hz 轨迹）、`videos/`（每局一段回放，`previews/` 里是 ≤ 500 KB 的预览）、`summary.json`。约 1 分钟，出视频再加约 1 分钟；加 `--no-video` 不出视频。
怎么用结果：5 / 10 以上开循环；1–4 / 10 调一次参数再试；0 / 10 停下，先查是控制器做不到还是抱法不对。2026-10-24 严格判据下的 6 种设置见第 9 节。

## 4. 跑内环

```bash
S=$LOCOMANI_WORKSPACE/code/rsi-scene/box-b
python3 scripts/rsi/sandbox.py $S box_bimanual
cd $S && CUDA_VISIBLE_DEVICES=0 nohup setsid $PY scripts/rsi/loop.py run box_bimanual --max-iters 1 > loop.log 2>&1 &
cd -
```

做什么：第二条把仓库**已提交**的代码导出成沙盒（没提交的改动沙盒看不到），技能目录换成任务族文件 `sandbox.seed` 指的起步版本（现在是 `scripts/rsi/seeds/box_strict_start`：循环之前写出的双手抱箱技能，箱子只抬 3–16 cm），并抽诊断池 30 局、验收池 40 局。第三条在后台跑内环：Reviewer 读诊断池的记录和轨迹写假设，Prober 试参数，Manager 选，三个 Engineer 并行改技能，Composer 合成一个候选，验收门在角色看不到的验收池上和现任逐局比。`--max-iters 1` 只跑一轮，用来确认代码没有问题；去掉它就跑到停止条件（验收池 70% 以上、12 次验收门、6 小时或 Manager 停）。
得到什么：账本在 `$S/outputs/scene_rsi_20261016/box_bimanual/loop/ledger.jsonl`，每行一步；每轮的提示词、角色回答、草稿、候选和验收结论在 `loop/iter_NN/`。一轮约 30–40 分钟、约 17 USD。

```bash
cd $S && $PY scripts/rsi/loop.py status box_bimanual; cd -
```

做什么：打印进度：第几轮、每次验收的结论、现任在验收池上的成绩、已花的时间和模型费。
怎么判断一轮没有出错：账本里每一步都有一行，最后一行是 `gate` 或 `stop`；没有 `"void": true`（越界作废）；`loop.log` 末尾没有 Traceback；`iter_00/` 下各角色的 `*_raw.json` 里 `permission_denials` 是空的（见第 10 节）。候选被不被接受不是判断代码有没有问题的依据。

## 5. 封存集

```bash
cd $S && $PY scripts/rsi/sealed.py box_bimanual; cd -
cat $S/outputs/scene_rsi_20261016_hidden/box_bimanual/sealed/summary.json
```

做什么：在 50 局封存集上跑起步版本和每一版被接受的现任。封存集连验收门都没见过，是报告用的成绩。
得到什么：`summary.json` 的 `versions` 里每个版本的成功数、按任务的成功数、摔倒和掉落数；约 2 分钟。被接受的版本在 `$S/outputs/scene_rsi_20261016/box_bimanual/loop/iter_NN/` 下：Composer 合成的在 `candidate/`，原样采用第 k 份草稿的在 `candidate_k/`（账本 `compose` 行的 `adopted` 是 k），里面是 `skills/` 和参数覆盖 `candidate.json`；要留下来，复制到 `scripts/rsi/seeds/` 下以版本哈希命名的目录再提交。这条线不写回 `src/locomani/tasks/scene/skills/`。

## 6. 看过程：MuJoCo 视频

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 $PY scripts/rsi/report/evolution_media.py \
  --sandbox $S --family box_bimanual --out outputs/scene_rsi_20261016/box_media --only task --versions final
```

做什么：读沙盒的账本，每个任务取最终版本做成的最短一局（没有成功就取最短的失败局），回放到完整的物流间场景上渲染，单画面，3 倍速，带字幕卡。去掉 `--versions final` 是起步版本和最终版本左右对比；去掉 `--only task` 还出每个被接受改动的改前改后对比和总览图（runbook 第 8 节）。脚本从沙盒的任务族文件读场景，并把场景名写进输出目录的 `media.json`，第 7 节靠它知道去哪个 Unreal 关卡拍。
得到什么：`final_NN_<任务>.mp4`、同名海报 `.jpg`、`previews/`（≤ 500 KB，给人看、放网页、打包只用这些）、`tasks_final.json`。几秒一段。

## 7. Unreal 渲染

前提是[安装](setup.md)第 8 步（Unreal 引擎、URLab 插件、`$BRIDGE_PY`）做完，并且物流间的房间关卡 `/Game/Levels/Warehouse` 已经建过（warehouse 场景用的同一个，`$UNREAL_ASSETS/` 下有 `warehouse_box_*` 就说明建过）。编辑器一台机器只能开一个；一张卡的机器不要和内环同时跑。

### 7.1 建场景关卡：每台机器一次

```bash
bash scripts/start_unreal.sh --gpu 0
bash scripts/build_scene.sh logistics_room_mars --unreal
```

做什么：第一条起编辑器并等它接受命令（首次启动编译着色器要几分钟，之后约 15 秒）。第二条按 `configs/scenes.yaml` 里 `logistics_room_mars` 的 `unreal.build` 建场景关卡：复制房间关卡，导入机器人、纸箱、第三人称机位，把场景文件的哈希记在 `$UNREAL_ASSETS/logistics_room_mars/source.json`。
得到什么：宿主工程里的 `/Game/Levels/logistics_room_mars` 和 `source.json`（`film` 的 `fovy` 应是 45）；约 45 秒。只有第 1 节的场景改过（`scene_locomani.xml` 变了）才重做，否则渲染会因哈希不符拒绝。

### 7.2 渲染视频：每次

```bash
bash scripts/start_unreal.sh --gpu 0
PYTHONPATH=src MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 "$BRIDGE_PY" scripts/rsi/report/report_unreal.py all \
  --media outputs/scene_rsi_20261016/box_media --out outputs/scene_rsi_20261016/box_media_unreal
bash scripts/stop_unreal.sh
```

做什么：第一条在编辑器没开着时才需要。第二条读第 6 节输出目录的索引和 `media.json`（场景是 `logistics_room_mars`，不用加 `--scene`），把每段视频的那几局回放进 Unreal 关卡，沿 MuJoCo 版同一条机位轨迹以 1280 × 960 拍，再用同样的起点和字幕卡合成；`all` 或点名（如 `final_01_box_long_side`）。第三条关编辑器。
得到什么：`--out` 下每段 `<名字>.mp4`、`previews/`、海报、`<名字>/panel0/NNNNN.jpg` 逐帧，以及 `compare_*.jpg`（同一时刻 Unreal 和 MuJoCo 上下对照，先看它确认对齐）；约 1 分钟一段。渲染期间不要跑别的重任务，编辑器的守护脚本在 CPU 占满时会把它杀掉，日志在 `$UNREAL_ROOT/logs/`。加 `--scene` 和 `configs/scenes.yaml` 里的场景名可以指定别的场景；不带 `--media` 也不带 `--scene` 时它渲染的是 ADT 进展报告自己的清单。

## 8. 改判据

判据写在两个文件里，改了要提交、重建沙盒，旧沙盒的验收和封存数字就不能和新的比了：
- `scripts/rsi/families/box_bimanual_tasks.jsonl` 每个任务的 `goal`：现在是 `held(nor_cal_box)` 加 `lifted(nor_cal_box, 0.20, 60)`；`lifted` 的两个数是最低角高出平台多少米、底面倾斜上限多少度。
- `scripts/rsi/families/box_bimanual.json` 的 `judge_s`（最后几秒要一直成立，5）和 `hold_s`（技能结束后原地保持几秒，2）。

角色读的规则在 `scripts/rsi/rules.md` 第 35、36 条，读的方案在 `scripts/rsi/plans/box_bimanual.md`；判据改了这两处也要改，改一次算一次人工干预。

## 9. 2026-10-24 严格判据下的探针

现有技能（`seeds/box_strict_start`）在新判据下：探针的 10 个起点 0 / 10，封存集 50 局 0 / 50，几乎都是 `low_lift`（箱子只抬了 3–16 cm）。人写探针的各种设置：

| 设置 | 结果 |
|---|---|
| 抬 25 cm，骨盆 0.79 m | 举高 15–25 cm，9 局倾斜 30–59° |
| 抬 30 cm，慢一些 | 6 局倾斜，3 局滑落，1 局不够高 |
| 抓点在箱高 85% | 不再倾斜，8 局从两手间滑落 |
| 抬 25 cm、骨盆 0.79 m，走路改 0.5 m/s | 停位仍在 1.5 cm 内，9 局倾斜 30–59°，1 局一只手没贴上 |
| 改用人工定义的夹壁抓法：俯 60°、抬 25 cm | 0 / 10，但 2 局最低角 19.8 和 20.0 cm，只差几毫米；没有大角度倾斜 |
| 夹壁抓法、抬 32 cm（现在的默认） | **3 / 10**：长边朝外 3 / 5，短边朝外 0 / 5；失败是一侧没夹住后滑落或碰歪（入位误差 3–8 cm、手腕姿态差 22–37°） |
| 夹壁抓法、抬 32 cm，倾斜上限放宽到 60° | 仍是 3 / 10：失败的 7 局都是手离开了箱子，没有一局败在倾斜 |
| 夹壁抓法、俯 45° | 0 / 10：姿态误差降到 12–28°，但 4 局举起后在 5 秒窗口里掉了 |
| 夹壁抓法、俯 45°、站得近 8 cm | 0 / 10：7 局手碰歪了箱子 |
| 抓点在箱高 75% | 5 局滑落，3 局不够高，2 局倾斜 |
| 抓点 75%，两手各向内顶 8 cm | 6 局倾斜，3 局不够高，1 局举起后又掉 |

结论（探针）：抱箱时手指只用侧面压壁，箱子不是绕两手连线翻就是往下滑；夹壁抓法的转轴在箱沿、高于重心，举起后不再翻，长边朝外能做成；短边朝外时要夹的侧壁在深处，手送不准。探针只到 3 / 10，用户决定照样开循环；循环从抱箱技能起步，一轮就把它改到封存集 41 / 50（上面的现状）。

## 10. 常见错误

- 角色的命令被拒、账本没问题但 `iter_NN/<角色>_raw.json` 里 `permission_denials` 不为空：这台机器的 Claude CLI 不让 `--permission-mode bypassPermissions` 放开 Bash。`scripts/rsi/loop.py` 里调 `claude -p` 的命令带着 `--allowedTools "Bash"`（2026-10-24 加的）；别的机器上如果仍被拒，查组织策略。
- `scene_lint.py` 报 error：不要开循环，先按第 1 节重建场景。
- 内环第一行就崩、`loop.log` 里 `No module named ...`：任务族文件的 `sandbox.keep` 缺了 `loop.py` 新 import 的脚本（2026-10-23 起要 `scripts/rsi/diag_pool.py` 和 `try.py`）；`sandbox.closure` 里不能写 `try`。
- `report_unreal.py` 报 `films ... at fovy 73.74, the scene's shot is ... at 45`：场景文件里还留着场景包的固定机位 `video_camera`。用现在的 `prepare_logistics_scene.py --rsi` 重建场景（它会去掉这个字段），再做 7.1。
- 探针不到 5 / 10：不要开循环，按第 3 节的规则处理。
