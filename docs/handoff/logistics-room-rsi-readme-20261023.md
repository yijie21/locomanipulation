# 物流间双臂抱箱的 RSI 线：一步一步跑通（2026-10-23）

这一页给要在另一台机器上复现这条线、或接着往下做的开发者。它把 TODO 38 做过的每一步按顺序写成：命令、做什么、得到什么、怎么检查、踩过的坑。决策和理由在[方案页](box-bimanual-rsi-plan-20261023.md)，流程机制在[详解](../rsi_design.md)，ADT 公寓那条线的通用命令在 [runbook](../rsi.md)。代码在分支 `henry-logistic-room`。

这条线和 ADT 线的差别只有三处：场景换成扫描的物流间（一个平台、一个纸箱），动作层多了双手动作（`grip` / `release` 的 `"both"`、`squeeze`、`reach(keep_squeeze=True)`），任务族 `box_bimanual` 的成功判据是两手把箱子抬离平台。内环、沙盒、验收门、封存集、视频脚本一行没改，靠任务族文件里的 `scene` 字段换场景。

## 0. 前提

- [安装](../setup.md)里“场景 RSI”一列的步骤做完：仿真 venv、ScaleBFM 权重、Claude CLI（角色用 `claude -p` 调 `claude-opus-5-5`）。ADT 数据这条线不需要。
- 每个新终端先执行 runbook 第 0 节的四行：

```bash
. scripts/paths.sh
export TMPDIR=$LOCOMANI_TMP NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=2
PY=$VENV_SIM_PY
```

- 只有一张卡的机器（SRA0889 就是）再加 `export RSI_GPUS=0 RSI_PROCS=6 RSI_SIM_SLOTS=6`，并且所有带 `CUDA_VISIBLE_DEVICES=1` 或 `MUJOCO_EGL_DEVICE_ID=1` 的命令改成 0。下面的命令都按一张卡写。
- 场景包 `assets/scene/logistics_room_box_v1/` 走 Git LFS，先 `git lfs pull`；它引用 `third_party/GR00T-WholeBodyControl` 里的机器人网格，所以 `bash scripts/setup_sonic.sh` 要做过。

## 1. 建场景

```bash
$PY scripts/prepare_logistics_scene.py outputs/scene_discovery/logistics_room_mars --robot mars --rsi
SC=outputs/scene_discovery/logistics_room_mars
$PY scripts/scene_settle.py --scene $SC --write
$PY scripts/scene_lint.py --scene $SC
```

做什么：第一条把场景包装成运行时场景（机器人换成 mars 版 G1，和 ADT 线同一个机器人），`--rsi` 再补上场景 RSI 要读的 v2 字段：
- `room`：从网格切出来的。`src/locomani/scene_discovery/scan_room.py` 沿每个三角形的边每 2.5 cm 采样，取 0.10–1.50 m 高度带内的点落进 5 cm 格子，占用格子合并成矩形写进 `obstacles`（墙和货架都在里面，`walls` 留空），离障碍 ≥ 机器人半径 0.35 m + 0.10 m 余量、且和默认出生点连通的格子合并成 `spawn.logistics_room`。墙和货架在 MuJoCo 里不碰撞，只靠这张地图挡住导航和出生点。
- 台面 `A` 的 `yaw: 0`、四条 `approach_edges`、`room`；箱子的 `semantic`（`graspable: bimanual`、质量 1.2 kg、容器）；`randomization`。
- `env_only.xml`：没有机器人的场景，给沉降和检查用。
- 箱子刚体补一个 `quat="1 0 0 0"` 属性：`scene_settle.py --write` 用正则改 `pos="…" quat="…"`，场景包里的箱子只有 `pos`，不补就断言失败。
第二条让箱子落稳并把位姿写回三个文件；第三条检查场景（竖直轴、静置 30 s 不掉、不漂）。

得到什么：`$SC/` 下 `scene_locomani.xml`、`env_only.xml`、`scene_meta.json`、`settle.json`、`lint.json`；第一条打印的 JSON 里 `rsi.obstacles` 约 81、`rsi.spawn_rectangles` 约 55；第三条打印 `[lint] 0 errors, 0 warnings`。三条加起来不到 1 分钟。第一次运行场景任务时 `tasks/scene/world.physics_scene` 会在旁边生成 `logistics_room_mars_physics/`（去掉纯视觉网格的物理副本），不用管。

怎么检查：想看地图，跑下面这段，把 `room` 块画成字符图（`#` 障碍、`s` 出生区、`R` 默认出生点）：

```bash
PYTHONPATH=src $PY - <<'EOF'
import json, numpy as np
m = json.load(open("outputs/scene_discovery/logistics_room_mars/scene_meta.json")); r = m["room"]; b = r["bounds"]; res = 0.1
nx, ny = int((b["x"][1]-b["x"][0])/res)+1, int((b["y"][1]-b["y"][0])/res)+1
img = [["."]*ny for _ in range(nx)]
def put(polys, ch):
    for (xa, ya), _, (xb, yb), _ in polys:
        for i in range(int(round((xa-b["x"][0])/res)), max(int(round((xa-b["x"][0])/res))+1, int(round((xb-b["x"][0])/res)))):
            for j in range(int(round((ya-b["y"][0])/res)), max(int(round((ya-b["y"][0])/res))+1, int(round((yb-b["y"][0])/res)))):
                if 0 <= i < nx and 0 <= j < ny: img[i][j] = ch
put(r["spawn"]["logistics_room"], "s"); put(r["obstacles"], "#")
rx, ry = m["robot_start"]["xy"]; img[int(round((rx-b["x"][0])/res))][int(round((ry-b["y"][0])/res))] = "R"
print("\n".join("".join(img[i][j] for i in range(nx)) for j in range(ny-1, -1, -1)))
EOF
```

场景登记在 `configs/scenes.yaml` 的 `logistics_room_mars` 条目（`bash scripts/build_scene.sh logistics_room_mars` 等于第一条命令；Unreal 先标 `unsupported`）。放在 `outputs/scene_discovery/` 下是有意的：沙盒把这个目录整个链接进去，越界审计也只认这里的链接。

## 2. 动作层的双手动作

代码已经在仓库里，这一步只是知道它们是什么、怎么验证。`src/locomani/backends/scalebfm/motor.py`：

| 动作 | 做什么 |
|---|---|
| `reach("both", (左, 右), (R左, R右), via=[(左, 右), …])` | 两只夹爪中心一起走直线、一起 IK、一起用夹爪中心 + 姿态伺服到位（原来就有） |
| `grip("both")` / `release("both")` | 两只夹爪同时合、同时开；`held` / `opening_cm` 按左右两项给 |
| `squeeze(inward_m, seconds)` | 两只夹爪中心沿它们的连线各向内偏置 `inward_m` 并保持：偏置记在骨盆偏航系里，加在每只手的伺服参考上，经过 `posture`、`hold`、`walk_to` 都还在，`release` 或该手的 `reach` 才撤掉。被箱子挡住时伺服积分沿推力方向最多涨到偏置本身、横向也最多偏置大小，所以指令最多进壁 2 × `inward_m`。返回两手手指有没有碰到同一个刚体 |
| `reach(..., keep_squeeze=True)` | 伸手时保留夹持：双手抬箱就是“两手一起向上 reach、保持夹持”，可以同时给 `pelvis_z` 让腿一起抬 |

配套：`src/locomani/tasks/scene/stage.py` 的 `bimanual_grip_check(m, body)`（两手手指都碰到）和 `bimanual_lift_check(world, m, key, z0)`（抬高且两手仍碰到）；`scripts/rsi/failure_class.py` 对 `bimanual` 物体用双手失败类型：`walk_off`、`side_reach`（两手都没到箱面 4 cm 内）、`one_hand_short`、`knocked`、`no_lift`（两手到位、箱子没抬 3 cm）、`slip`、`fell`、`other`。规则文件 `scripts/rsi/rules.md` 第 20 版第 35 条写给角色看。

验证：

```bash
$PY -m pytest tests/test_scalebfm_motor.py tests/test_rsi_stage.py tests/test_rsi_failure_class.py tests/test_scan_room.py tests/test_box_bimanual_family.py; echo "exit $?"
```

得到什么：退出码 0。注意 `pyproject.toml` 已经给 pytest 加了 `-q`，命令行再加 `-q` 会把“N passed”那行也吞掉，看退出码最稳。

## 3. 可行性探针（开循环前必做）

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src $PY scripts/rsi/probe_box.py --out outputs/scene_rsi_20261016/box_probe_$(date +%Y%m%d)
```

做什么：机器人从场景的默认出生点（平台 -y 边前 1.1 m）出发，箱子按任务族的两种摆放各随机放 5 次，用一段人写的序列试举：走到离箱子最近的台沿外 0.38 m → 骨盆降到 0.72 m → 张开夹爪、手腕朝前，两手伸到箱子两侧壁外 6.35 + 0.5 cm（张开时外侧手指的侧面离夹爪轴 6.35 cm，碰壁的是它）→ `squeeze(0.06, 1 s)` → `reach("both", 向上 10 cm, keep_squeeze=True, pelvis_z=0.78)` → 保持 2 s。序列在 `scripts/rsi/probe_box_skills/`，参数都是大写常量，可以 `--set '{"INWARD_M": 0.08}'` 改；它不是种子，不进沙盒。每一局走 `scene_round.py`，记录和循环的同格式（`steps`、`failure_class`、`events`、50 Hz 轨迹）。

得到什么：`--out` 下 `pool/`（10 个起点）、`rounds/`（记录和 `.npz`）、`videos/`（每局一段回放，`previews/` ≤ 500 KB）、`summary.json`；终端打印每局成败和步骤链。约 1 分钟（出视频再加 1 分钟）。

判定（方案 D5）：≥ 5 / 10 开循环；1–4 / 10 调 `INWARD_M` 和抓高再试一次；0 / 10 回头查控制器。2026-10-23 本机：**10 / 10**，抬高 2.8–6.9 cm。

探针量到、写进方案页第 7 节给角色看的事实：
- 顶边捏取（手腕朝下跨住箱壁顶边）姿态差 50–75°，控制器给不出，还把箱子推开，0 / 4。
- 闭合夹爪去抱：夹爪底座（半宽 2.9 cm）比闭合手指（1.9 cm）宽，先碰壁的是底座，`held` 只算手指接触。要张开夹爪用外侧手指的侧面。
- 夹持偏置 0.04 时 1 / 4（手指顺壁上滑），0.06 和 0.08 都 4 / 4。
- 站直（`posture()`）不能当抬箱用：它按关节空间保持手臂，前倾下站直把手甩到头顶。

## 4. 任务族文件

`scripts/rsi/families/box_bimanual.json` 和 `box_bimanual_tasks.jsonl`，字段逐个说：

| 字段 | 值 | 为什么 |
|---|---|---|
| `scene` | `outputs/scene_discovery/logistics_room_mars` | 换场景就改这里，`family.py`、`sandbox.py`、越界审计都从它读 |
| `scene_note` | “in a simulated logistics room (…)” | 五个角色提示词开头的那句话；没有这个字段的任务族沿用“in a simulated apartment” |
| `tasks` | `box_long_side`、`box_short_side` | 都是 `held(nor_cal_box)`，只差箱子怎么放 |
| 任务行的 `placement.yaw_deg` | `[85, 90]` / `[0, 5]` | 箱子 x 轴（0.45 m 那条边）和最近台沿外法线的夹角：90 是长边贴着台沿（长边朝外），0 是短边朝外 |
| 任务行的 `placement.depth_m` | `[0.18, 0.31]` / `[0.25, 0.375]` | 量的是箱子**中心**到最近台沿的距离，不是近侧面；近侧面要在台沿 0–0.15 m 内，就是半宽（0.16 / 0.225）加 0–0.15，下限再留 2 cm 让脚印在台面里 |
| `spawn` | `{"rooms": "all", "distance_m": [0.5, null]}` | 全房间出生、离箱子 ≥ 0.5 m，和 grasp_atomic 一样 |
| `pools` | 诊断 30（种子 430000）、验收 40（530000）、封存 50（630000） | 每个任务 15 / 20 / 25 局；种子段和别的任务族不重叠 |
| `plan` | `scripts/rsi/plans/box_bimanual.md` | 角色读的方案：目标、结构、动作层、决策、探针量到的事实、做不到的起点怎么处理 |
| `sandbox.seed` | `scripts/rsi/seeds/atomic` | 空种子：`plan_for` 不覆盖任何目标 |
| `sandbox.keep` | 比 grasp_atomic 多 `scripts/rsi/diag_pool.py`、`scripts/rsi/try.py` | `loop.py` 2026-10-23 起 import 它们，不带进沙盒第一行就崩（grasp_atomic、grasp_prep 的 keep 还没加，建它们的沙盒前要补） |
| `sandbox.closure` | 不写 `try` | closure 靠 `import 模块名` 算依赖，`try` 是关键字 |

`tests/test_box_bimanual_family.py` 核对这些数字和箱子尺寸自洽，场景建好时还抽一个起点。

## 5. 假角色冒烟

```bash
S=$LOCOMANI_WORKSPACE/code/rsi-scene/box-smoke
python3 scripts/rsi/sandbox.py $S box_bimanual
cd $S && RSI_GPUS=0 RSI_PROCS=6 RSI_SIM_SLOTS=6 CUDA_VISIBLE_DEVICES=0 $PY scripts/rsi/loop.py run box_bimanual --stub --max-iters 1; cd -
```

做什么：第二条把**已提交**的代码导出成沙盒（没提交的改动沙盒看不到），抽诊断池 30 局和验收池 40 局。第三条用写死回答的假角色走一轮内环。

得到什么：`$S/outputs/scene_rsi_20261016/box_bimanual/loop/ledger.jsonl` 八行：`review` → `probe`（空种子 30 局全是 `no_plan`，几秒）→ `manager` → 三个 `engineer` → `compose` → `gate`（`"reason": "dry run: static checks passed"`）。建沙盒 15 秒，一轮 6 秒。看完删掉 `$S`。

## 6. 真角色试跑两轮

```bash
S=$LOCOMANI_WORKSPACE/code/rsi-scene/box-a
python3 scripts/rsi/sandbox.py $S box_bimanual
cd $S && RSI_GPUS=0 RSI_PROCS=6 RSI_SIM_SLOTS=6 CUDA_VISIBLE_DEVICES=0 nohup setsid $PY scripts/rsi/loop.py run box_bimanual --max-iters 2 > loop.log 2>&1 &
cd -
```

做什么：和 runbook 第 5.2 节一样的内环，只跑两轮就停，目的是看角色读不读得懂新场景和双手动作、代码有没有 bug。每轮：Reviewer 读诊断池记录写假设 → Prober 真跑参数试验 → Manager 选 → 三个 Engineer 并行写技能 → Composer 合成一个候选 → 验收门在隐藏的验收池上和现任配对比较。

怎么看进度：

```bash
cd $S && $PY scripts/rsi/loop.py status box_bimanual; cd -
tail -f $S/outputs/scene_rsi_20261016/box_bimanual/loop/ledger.jsonl
```

账本每行一个步骤：`review` 的 `n_hypotheses`、`void`（越界作废）、`cost_usd`；`probe` 每条假设救回 / 弄坏几局；`manager` 的 `decision`；`engineer` 的 `changed`（改了哪些文件）、`stray`（技能目录外的文件，有就作废）；`compose` 的 `decision`；`gate` 的 `accepted`、`result`（多赢 b、少赢 c、按失败类型的分布）。每轮的提示词、角色回答、草稿、候选、验收结论都在 `loop/iter_NN/`。

怎么判断“没有 bug”：两轮都走到 `gate` 行，没有退出码 3（`loop.log` 末尾没有 Traceback，账本里没有 `step: "stop"` 带异常原因），角色没有因为读不到文件而越界作废（`void: false`），Prober 和验收门跑的局都有记录（`gate` 的 `void` 不是 true）。候选接不接受不是 bug 的判据。

两轮的实测见第 10 节。

## 7. 续跑到停止条件（这次没做）

```bash
cd $S && RSI_GPUS=0 RSI_PROCS=6 RSI_SIM_SLOTS=6 CUDA_VISIBLE_DEVICES=0 nohup setsid $PY scripts/rsi/loop.py run box_bimanual > loop.log 2>&1 &
```

做什么：从账本接着跑，直到验收池 70% 以上、12 次验收门、墙钟 6 小时或 Manager 停。一张卡的机器仿真部分比两张卡慢，一轮估计 60–90 分钟，大头是模型生成。

## 8. 封存集与快照（这次没做）

```bash
cd $S && $PY scripts/rsi/sealed.py box_bimanual; cd -
cat $S/outputs/scene_rsi_20261016_hidden/box_bimanual/sealed/summary.json
```

做什么：在 50 局封存集上跑种子和每一版被接受的现任。空种子的任务族不写回 `src/locomani/tasks/scene/skills/`（新目录和同名文件会冲突）；最终版技能目录在 `$S/outputs/scene_rsi_20261016/box_bimanual/incumbents/<哈希>/skills/`，要留快照就复制到 `scripts/rsi/seeds/box_<哈希>`，像 grasp_c11 那样。

## 9. 视频

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 $PY scripts/rsi/report/evolution_media.py --sandbox $S --family box_bimanual --out outputs/scene_rsi_20261016/box_media
```

做什么：runbook 第 8 节的脚本，读账本里被接受的版本，每个改动一段改前改后对比、每个任务一段，预览 ≤ 500 KB。Unreal 版（第 9 节）这条线还没接：`warehouse_textured` 的房间关卡已经有，要给 `logistics_room_mars` 建场景关卡并把 `scenes.yaml` 的 `unreal.unsupported` 换成 `level`，等 MuJoCo 视频确认后再做（TODO 38.17）。

## 10. 2026-10-23 的两轮试跑

（跑完后补：每轮的耗时、费用、假设数、Manager 的决定、候选改了什么、验收门结论，以及发现并修掉的问题。）

## 11. 这条线改了哪些文件

| 文件 | 作用 |
|---|---|
| `scripts/prepare_logistics_scene.py` | `--rsi`：v2 字段、`env_only.xml`、箱子的 `quat` 属性 |
| `src/locomani/scene_discovery/scan_room.py` | 从网格切 `room` 块 |
| `configs/scenes.yaml`、`tests/test_run_context.py` | 场景登记 |
| `src/locomani/backends/scalebfm/motor.py` | `grip` / `release` 的 `"both"`、`squeeze`、`reach(keep_squeeze)`、夹持时的伺服上限 |
| `src/locomani/tasks/scene/stage.py` | 双手检查 |
| `scripts/rsi/failure_class.py`、`scripts/rsi/loop.py`（`CLASS_SUGGEST`、`SCENE_NOTE`）、`scripts/rsi/prompts/*.md` | 双手失败类型、提示词的场景句 |
| `scripts/rsi/rules.md` | 第 20 版第 35 条 |
| `scripts/rsi/families/box_bimanual*.json*`、`scripts/rsi/plans/box_bimanual.md` | 任务族和角色读的方案 |
| `scripts/rsi/probe_box.py`、`scripts/rsi/probe_box_skills/` | 探针 |
| `tests/test_scan_room.py`、`test_box_bimanual_family.py`，`test_scalebfm_motor.py`、`test_rsi_stage.py`、`test_rsi_failure_class.py` 的新用例 | 测试 |
| `TODO.md` 第 38 节、`DECISIONS.md` 2026-10-23、`docs/handoff/box-bimanual-rsi-plan-20261023.md`、本页 | 记录 |

## 12. 已知限制

- 墙和货架不碰撞：机器人手臂穿过货架不会被挡，只有导航和出生点避开它们。出生点离障碍 ≥ 0.45 m，目前没见到穿墙。
- 单手的可抓性、挑选器、身体表、准备动作层对双手任务不适用，规则第 35 条告诉角色别引用它们。
- 失败分类里“手到没到箱面”按夹爪中心和两个手指外侧面到箱面的最近距离算（阈值 4 cm），抱箱的手指侧面接触算到位；其他抱法要是用别的部位碰箱，会被分到 `side_reach`。
- `grasp_atomic.json`、`grasp_prep.json` 的 `sandbox.keep` 缺 `diag_pool.py` 和 `try.py`，再建那两族的沙盒前要补。
