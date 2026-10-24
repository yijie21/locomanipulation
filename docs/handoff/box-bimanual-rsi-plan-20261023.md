# 场景 RSI 迁到物流间：双臂移动抓箱的方案（2026-10-23）

这一页记 2026-10-23 和用户逐条问答定下的目标、决策和理由，任务清单在 [TODO 38](../../TODO.md)。流程机制见[详解](../rsi_design.md)，命令见 [runbook](../rsi.md)，前一条线见[从原子技能开始的抓取进化](grasp-atomic-line-20261023.md)。开发在分支 `henry-logistic-room` 上。

## 1. 目标

把场景 RSI 的内环从 ADT 公寓搬到扫描的物流间（`warehouse_textured_mars` 那间），给动作层加第 9 个原子动作（双手同时合爪、向内夹持），然后让循环从空种子长出“走到平台、双手抬起纸箱”的技能。只跑 L1（一个任务族的内环和封存集），不跑外环。

## 2. 起点：已有的东西

- 机器人和控制器不变：mars 版 G1、Dex1 两指夹爪、ScaleBFM。
- 动作层已有双手同时伸手：`Motor.reach(side="both", pos=(左, 右), rot=(…))`，两手一起做 IK、一起用夹爪中心加姿态伺服到位。缺的是 `grip` / `release` 的 `"both"`（现在两手只能先后合爪）和保持夹持力的动作。
- 谓词 `held(X)` 已有双手分支：X 离开所有台面 2 cm 以上，且两手的手指都接触（只算手指，手掌和手腕不算）。一局结束前固定保持 2 s 再判。
- 场景：`prepare_logistics_scene.py --robot mars` 出的 scene_meta 是 v1 格式，只有台面 A 和物体 `nor_cal_box`，没有 `room.spawn` / `bounds` / `walls`，起点抽样会直接报错；房间网格是纯视觉，墙不碰撞。平台 1.2 × 0.9 m，四边净空 1.05–1.63 m，四边都站得下。
- 内环代码对场景的依赖：`family.py`、`sandbox.py`、`loop.py`、`batch.py`、`scene_round.py`、`gate.py`、`episode.py`、`world.py` 只读 scene_meta；写死 ADT 的是越界审计里对 `outputs/scene_discovery` 的假设、角色提示词里的“公寓”、外环和 `rsi_kit` 工具（这次不用）。
- 失败分类、可抓性、站位挑选器、身体表都只认一只手。物体名禁令（规则第 30 条）在这个场景只禁 `nor_cal_box` 和它的刚体名，`box` 这个词不禁。
- 同一场景、同一个纸箱上有一条 ScaleBFM 调参 RSI（手写专家、自己的 runner、只从 ym 边，封存 48 / 80，CuRobo 变体 68 / 80，TODO 25.9.1）。
- 物流间已有 Unreal 关卡：`warehouse_textured` 房间关卡和四个 `warehouse_box_*_mars` 场景关卡。

## 3. 决策（用户 2026-10-23）

| 编号 | 决定 | 理由 |
|---|---|---|
| D1 | 只用房间自带的纸箱一种；箱子在平台上的位置随机、机器人出生点随机 | 第一次运行分清是流程问题还是尺寸问题；多尺寸留给后续任务族 |
| D2 | 成功判据 `held(nor_cal_box)`，不加 `upright`，不另写谓词 | 现成谓词、不改冻结层；抬高下限等封存集数字出来再议 |
| D3 | 出生点是房间任何能走到的地方、离平台 0.5 m 以上；平台四边都允许接近 | 出生区已扣掉平台占地加机器人半径，不会出生在物体里；站哪边由循环学 |
| D4 | 第 9 个原子动作保持原子：`grip` / `release` 的 `"both"`，加 `squeeze(inward_m, seconds)` 两手向内偏置并保持；不含站位、抓点换算、抬起 | 要进化的正是站位、抓高、夹紧时机和抬法；手指只有约 20 N、纸板软、台账记过物体从夹爪滑出，只靠合爪可能夹不住 1.2 kg |
| D5 | 先手写探针：默认出生点、10 个随机箱位，`reach("both")` → `squeeze` → 站直抬起 → 保持 2 s；5 / 10 以上才开循环 | 10-14 的数据是回放 SONIC 轨迹时抬起的，没人用动作层从头抓过这个高度（抓高约 0.5 m、双手间距 0.45 m，在 crate 线标定之外） |
| D6 | 只跑 L1 | 外环的发现、分桶、能力表都绑着 ADT 的房间和物体目录 |
| D7 | 场景参数化：一套 `scripts/rsi/` 按任务族文件的 `scene` 跑两个场景 | 复制一份会让规则版本、验收门和工具很快分叉 |
| D8 | 配置沿用 grasp_atomic（30 / 40 / 50，三 Engineer + Composer，6 h，12 次验收门），先 `--max-iters 2` 看一眼 | 已验证的配置；先确认角色读得懂新场景和新动作 |
| D9 | 经验库带 ADT 单手经验，不带代码 | 站位选择、分段走近、抬起后检查对双手也适用 |
| D10 | 视频先 MuJoCo 回放，确认无误再接 Unreal | 房间关卡已有，Unreal 只是让 `report_unreal.py` 认新场景 |
| D11 | 场景文件由 `prepare_logistics_scene.py --rsi` 补 v2 字段，输出到 `outputs/scene_discovery/logistics_room_mars/`；墙只进导航器，不加碰撞盒；不加第二个台面 | 放在 `outputs/scene_discovery/` 下沙盒链接和越界审计都不用改；出生点离墙 0.5 m 以上后手臂穿墙几乎不出现 |
| D12 | 箱子朝向两档：长边朝外（`yaw_deg [0, 5]`）、短边朝外（`[85, 90]`），分成 `box_long_side`、`box_short_side` 两个任务 | 每个任务在诊断池至少 2 个起点，验收门按任务分层；斜放留给后续 |
| D13 | 箱子近侧面离最近台沿 0–0.15 m（`placement.depth_m`） | 平台 1.2 × 0.9，放中间时任何一边都够不到（前伸上限 0.6–0.7 m）；不限制会让三个池混进注定失败的局 |
| D14 | 单手工具：只把失败分类扩成双手，其余声明只对单手有效 | 失败类型进验收门结论和 Manager 的修理提示，用得最多；挑选器和准备动作层在 ADT 上也还没接进技能 |
| D15 | 同场景的调参 RSI 只作参照数字，不做同池对照 | 用户 10-22：研究初期不做严格对照；接那个专家进场景运行器工作量不小 |

## 4. 被否决的方案

- 把 warehouse 双臂专家整套移植进种子（approach、grasp_bimanual、put_down）：任务退化成调参数，和 grasp_atomic 从空种子长出技能的意义不一致。
- 一个 `grasp_both(...)` 伸、夹、抬一体的动作：把抬起策略从进化里拿走。
- 成功判据加 `upright`：用户不要；第一次运行只看 `held`。
- 箱子朝向 ±180° 完全随机：第一次运行先两档。
- 不限制箱子深度、让够不到的起点由 Reviewer 声明做不到：稀释验收门。
- 现在顺手加第二个台面给将来的搬运任务族：用户不要。
- 把场景放在 `outputs/scene/` 下并改越界审计：改审计是独立的小任务（38.10），不挡这条线。

## 5. 结果

### 5.1 可行性探针（2026-10-23，TODO 38.7–38.8）

`scripts/rsi/probe_box.py`：默认出生点，10 个随机箱位（两个任务各 5），手写序列（`scripts/rsi/probe_box_skills/`，不进沙盒）。结果 **10 / 10** 举起（`held(nor_cal_box)` 成立），抬高 2.8–6.9 cm；记录和视频在 `outputs/scene_rsi_20261016/box_probe_20261023/`（预览 ≤ 500 KB）。D5 的线是 5 / 10，开循环。

到这个结果之前试过、没用的：
- 顶边捏取（手腕朝下、夹爪跨住箱壁顶边，crate 线的抓法）：这个前伸距离下手腕朝下的姿态控制器给不出（差 50–75°），手把箱子推开，0 / 4。
- 抱箱时闭合夹爪：夹爪底座半宽 2.9 cm 比闭合手指的 1.9 cm 宽，先碰到箱壁的是底座，`held` 只算手指接触。改成张开夹爪，用外侧手指的侧面压壁（离夹爪轴 6.35 cm）。
- 夹持伺服不设上限：积分沿推力方向涨到 8 cm 上限，箱子被两手挤飞（抬起 32–81 cm 后掉）。`motor._servo` 改成：夹持时沿推力方向的积分不超过偏置本身，横向也不超过偏置大小。
- 用 `posture()` 站直来抬：它按关节空间保持手臂，前倾姿态下站直把手甩到头顶。抬起改成 `reach("both", 向上 10 cm, keep_squeeze=True, pelvis_z=0.78)`：腿抬、手臂保持夹持。
- 偏置大小：0.04 时 1 / 4（手指顺着箱壁往上滑），0.06 和 0.08 都是 4 / 4，默认 0.06。

这些是人为探针写的，循环的种子里没有；它们进了方案页第 7 节，角色会读到。

### 5.2 第一次内环（2026-10-23，TODO 38.14）

沙盒 `box-a`，`--max-iters 2`：第 0 轮 32 分钟、17.23 USD，Reviewer 3 条假设、三个 Engineer 的草稿诊断池 10 / 25 / 15 / 30、Composer 合并、验收门 39 / 40 对 0 / 40 接受（哈希 `42ff17cb110f`），现任超过 70% 停止线，循环自己停了。逐步数字、代码无 bug 的判据和这台机器上 `bypassPermissions` 不生效的问题在 [readme 页第 10 节](logistics-room-rsi-readme-20261023.md)。封存集 50 局：接受的版本 49 / 50（长边 25 / 25、短边 24 / 25），种子 0 / 50；快照 `scripts/rsi/seeds/box_42ff17cb110f/`。视频（38.16）和 Unreal（38.17）没做。

## 6. 命令

前提同 [runbook](../rsi.md) 第 0 节（环境变量、`$PY`）。同一套 `scripts/rsi/`，任务族文件的 `scene` 指向这个场景；角色读的方案是 `scripts/rsi/plans/box_bimanual.md`。

```bash
$PY scripts/prepare_logistics_scene.py outputs/scene_discovery/logistics_room_mars --robot mars --rsi
SC=outputs/scene_discovery/logistics_room_mars
$PY scripts/scene_settle.py --scene $SC --write
$PY scripts/scene_lint.py --scene $SC
```

做什么：第一条从仓库里的场景包建 MuJoCo 场景，`--rsi` 补上场景 RSI 要的字段：从网格切出房间的范围、障碍和出生区（墙和货架不碰撞，只进导航地图）、台面的四条可站边、箱子的 `semantic`，并写 `env_only.xml`。后两条和 runbook 第 2 节一样：让箱子落稳，检查场景。
得到什么：`$SC/` 下的 `scene_locomani.xml`、`env_only.xml`、`scene_meta.json`（`room` 块里约 80 个障碍矩形、55 个出生区矩形）、`settle.json`、`lint.json`，最后一条打印 `[lint] 0 errors`。一共不到 1 分钟。

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src $PY scripts/rsi/probe_box.py --out outputs/scene_rsi_20261016/box_probe
```

做什么：开循环之前的可行性探针：机器人从默认出生点出发，箱子随机放 10 次，用一段人写的序列（走到最近的台沿、放低骨盆、两手张开夹爪从两侧抱住、`squeeze`、保持夹持抬手）试举。这段序列不进沙盒，循环从空种子开始。
得到什么：打印每局成败和步骤，`summary.json`，`videos/` 下每局一段回放和 `previews/`。2026-10-23 是 10 / 10，约 1 分钟；少于 5 / 10 不要开循环，先查控制器。

```bash
S=$LOCOMANI_WORKSPACE/code/rsi-scene/box-a
python3 scripts/rsi/sandbox.py $S box_bimanual
cd $S && RSI_GPUS=0 RSI_PROCS=6 RSI_SIM_SLOTS=6 nohup setsid $PY scripts/rsi/loop.py run box_bimanual > loop.log 2>&1 &
```

做什么：建沙盒、跑内环，和 runbook 第 5.2 节一样；两张卡的机器去掉三个 `RSI_` 变量。先加 `--max-iters 2` 看一轮角色读不读得懂新场景和双手动作，再去掉续跑。封存集、看过程的视频同 runbook 第 5.3、8 节，任务族名换成 `box_bimanual`；空种子的任务族不写回技能目录。
