# 场景文件与任务文件格式（2026-10-03 定）

TODO 9.5。方法的输入是一份场景文件（[决策记录](scene-discovery-plan-20261002.md) #4、#5），输出是任务文件（#7、#10）。
本页定字段（用户 2026-10-03 过目，§9 三处按推荐定）；读取、校验与谓词判定的代码在 `src/locomani/scene_discovery/`。能力说明在 `configs/scene_discovery/capability.yaml`（#8，G1 + Dex1 + SONIC）；mars + ScaleBFM 的实测表是 `capability_mars_scalebfm.yaml`（场景 RSI 方案 §9，键相同，`scripts/scene_feasibility.py --capability` 选一张）。

## 0. 原则

- **是 `scene_meta.json` 的超集。** 现有读取方（导航、走近、抓取、判据、`SceneModel`，约 40 个文件）读的字段原样保留、含义不变；
  新信息全部放进新增的键，旧读取方看不到也不受影响。
- **每个推断出来的值都标出处**：`adt`（数据集原值）、`rule`（几何规则计算）、`claude`（Claude 估计）、`measured`（仿真里实测，如掉落测试）。
  失败时据此分清是属性估错还是技能问题。
- **全部用世界系**（z 朝上，米）。ADT 是 glTF 的 y 朝上，转换时统一旋转一次，旋转记在 `source.transform`。
- 文件里只有引用，没有网格；ADT 衍生的场景文件只放 `outputs/`（#17）。

## 1. 顶层

| 键 | 新/旧 | 内容 |
|---|---|---|
| `surfaces`、`objects`、`robot_start`、`target_object`、`target_pool`、`goal_object`、`object_phrase`、`scene_xml`、`nbody` / `ngeom` / `nq` | 旧 | 不变。`target_*` / `goal_object` 对发现出来的场景只是"当前这局的任务"，由采集入口按任务文件改写 |
| `schema_version` | 新 | `2` |
| `source` | 新 | `{kind: adt, dataset_root, sequence, frame_ns, transform, built_by, built_at}`；`kind` 以后也可以是 `procedural`、`scan` |
| `room` | 新 | 墙、房间、门洞、出生区（§4） |
| `randomization` | 新 | 哪些物体能挪、能挪到哪些台面、干扰物池（§5） |

## 2. 物体 `objects.<key>`

旧字段不变：`body`、`surface`、`init_pos`、`init_quat`、`height`、`scale`、`mesh`、`bimanual`、`container`、`box`、`free`、`qpos_adr`、`body_id`。
新增两块：

**`semantic`**

| 字段 | 出处 | 含义 |
|---|---|---|
| `asset` | adt | glb 路径（相对 `source.dataset_root`） |
| `adt` | adt | `{instance_name, prototype_name, category, motion_type, rigidity}` 原值 |
| `phrase` | claude | 英文短名，给提示词和 VLA 指令用（"white trash bin"） |
| `room` | rule | 所在房间（§4） |
| `movable` | rule + claude | 仿真里是否做成自由体。ADT 的 `static` 只表示录制时没动过（书架上的书也是 `static`），所以按类别判：家具、柜体部件、墙饰 → 固定；其余 → 可移动 |
| `part_of` | rule | 柜门 / 抽屉这类部件属于哪个家具（按名字前缀和位姿归属）；第一轮只记录，不做关节（#6） |
| `mass_kg`、`friction` | claude → measured | Claude 按类别和尺寸估，设上下界；掉落测试后实测稳定才算通过 |
| `upright_axis` | adt / rule | 物体系里朝上的轴，判 `upright` 用。原来取 ADT 的 `canonical_pose.up_vector`，但 64 个自由物体里有 28 个照它算是躺着的：书架里竖着的书、盆栽、水果、钥匙、遥控器、钱包，都比它们静置的姿态偏了 80–126°。所以 2026-10-18（用户定，TODO 19.15）起，凡是偏离静置朝上方向超过 30° 的，改取离静置朝上方向最近的物体轴（±x、±y、±z）。见 `adt.resting_upright`；沉降之后由 `scene_settle.py` 做，已沉降的场景用 `--upright-only`。ADT 原来的值留在 `upright_axis_adt`（出处 `adt`；改过的 `upright_axis` 出处 `rule`）。物体斜着放、离哪根轴都超过 30° 的（ADT 里就斜着 36° 的两个收纳盒），轴不变，由 `scripts/scene_lint.py` 报警告 |
| `graspable` | rule | `single` / `bimanual` / `none`，按 `capability.yaml` 的夹爪开口、箱宽、重量算 |
| `container` | claude + rule | `{is: bool, opening_half_extent, rim_height}`；Claude 判"是不是容器"，规则核对包围盒和开口尺寸 |
| `support` | rule | 这个物体本身能不能当台面（顶面够平够大），能的话在 `surfaces` 里有对应项 |

**`provenance`**：`{字段名: adt|rule|claude|measured}`，覆盖 `semantic` 里每个字段和旧字段里被推断的那几个（`bimanual`、`container`）。

## 3. 台面 `surfaces.<key>`

旧字段不变：`name`、`body`、`xy`、`top`、`half_extent`、`flat_half_extent`、`circumradius`、`body_id`。新增：

| 字段 | 出处 | 含义 |
|---|---|---|
| `yaw` | adt / rule | 台面朝向。旧场景都是 0；ADT 家具有任意朝向 |
| `of_object` | rule | 由哪个物体的顶面得来（一个书架可以有好几层台面） |
| `approach_edges` | rule | 哪几条边前面有 ≥ 机器人身宽 + 余量的空地，可以从那条边走近（10-03 按 SONIC 的规则算；mars + ScaleBFM 用场景运行器的站位搜索重算成覆盖文件，场景文件不改，检查器和提任务用 `--approach-edges` 读，见 scene-rsi 方案 §9.2） |
| `room` | rule | 所在房间 |
| `provenance` | — | 同上 |

台面由规则从家具网格提取：水平面积 ≥ 阈值、高度在 `capability.yaml` 关心的范围里（地面以上 0.1–1.3 m）。如果一个台面 ≥ 90% 被另一个高出不到 10 cm 的台面盖住，就不算台面（`surfaces.shadowed`，2026-10-18 用户定）。ADT 厨房的短台面和中台面，其实是长台面板下 3.6 cm 处的橱柜顶：东西放上去落在长台面上，`on(X, 中台面)` 在一开局就成立。已建好的场景用 `scripts/scene_drop_shadowed.py` 去掉这类台面：它的行、它在各物体摆放列表里的名字、它的板，以及可站边覆盖文件里的对应行。

台面高度 `top` 取这一层朝上三角形按**面积**加权的中位数（`surfaces.horizontal_regions`，2026-10-20，TODO 27.14）。原来按三角形**个数**取中位数：板面是两个大三角形，边条是几百个小三角形，高度就落在边条上。ADT 的 17 / 69 个台面因此比板面低 0.5–3.7 cm（电视柜顶 0.512 对 0.545），物体开局就埋在板里。已建好的场景用 `scripts/scene_fix_surface_heights.py --write`（`/venv/main`）原地改，不从 ADT 重建（重建会丢掉之后定下的门、去掉的台面和沉降后的位姿）：
- 改每个台面的 `top` 和它的板（两个 XML）；
- 改最低台面动过的家具的机器人碰撞体：只改当初按最低台面切过的，浮板的碰撞体是整件的外形，不动；
- 升高台面上的物体跟着抬高；
- 拿掉面板和碰撞盒，由后面几步按新高度重建。

改动前的文件备份在 `<场景目录>.bak_20261020_before_surface_heights/`，新旧高度记在 `<场景目录>/surface_heights.json`。`scripts/experiments/scene_rsi_20261016/surface_height_audit.py` 逐个台面比较 `top`、按面积的中位数和最高的大块朝上面。改完后只有书架中层（DisplayShelves_1__1）差 0.8 cm，它本身是两级的面。

MuJoCo 里每个台面是一块 2 cm 厚的板（`adt_slab_<key>`，`mjcf.SLAB`），物体和机器人都碰。板带 1 mm 接触余量（`adt_to_scene.py` 的 `SLAB_MARGIN`，2026-10-18）。没有余量时，物体凸包落在板上只有 1–4 个接触点，在运行器的 5 ms 步长下每步都变，静置的物体会自己转：钱包 2°/s，盆栽 4°/s；加扭转摩擦、椭圆摩擦锥、noslip 都止不住。有了余量，接触点集稳定，物体停在板上方约 0.8 mm，原来是陷进去 0.75 mm。余量只加在板上，不加在物体上：一次接触取两个 geom 中较大的余量，所以爪和物体的接触不变。`scripts/experiments/scene_rsi_20261016/spin_check.py` 测每个物体静置 30 s 转过的角度和滑动的距离，结果在 `outputs/scene_rsi_20261016/physics/`。

建完 MuJoCo 场景后还有四步（2026-10-18，方案 §10.2；碰撞盒和检查 2026-10-19）：
- **家具面板**（`scripts/scene_add_panels.py`，`/venv/main`）：家具碰撞体只和机器人碰，而且只到最低台面以下，所以书架上一排书在端头没有东西挡，往外倒；往后也一直滑，最后从背面掉下去。脚本在每个台面周围的家具网格里找面板：两个短端只要网格有面板就加；两条长边只在机器人不从那边靠近时才加（场景文件和可站边覆盖文件的 `approach_edges`，机器人干活的那一面保持敞开，和“门按开着处理”一致）。每块面板是 1 cm 厚的盒子，只和自由物体碰（contype 2；自由物体的碰撞体 conaffinity 3）。视觉被换成凸包的家具（`FURNITURE_FACES` 以上）不加面板。现在这个场景有 52 块，分在 15 件家具上，见 `<场景目录>/panels.json`。
- **墙和家具上部的碰撞盒**（`scripts/scene_add_colliders.py`，`/venv/main`，2026-10-19，方案 §14.1）：外壳（墙）原来什么都不碰，有台面的家具只在最低台面以下和机器人碰，学出来的技能有手臂穿墙、头进书架、手臂穿过椅背。脚本从 ADT 原始网格的表面生成体素盒子：墙在 0.10–2.0 m 之间，2 cm 一格，机器人和自由物体都碰（contype 6）；有台面的家具从原有碰撞体的顶往上，1.5 cm 一格，只和机器人碰（contype 4）。只取表面、不填实心，柜格里面是空的。台面那层让给台面板；上方被家具盖住的台面（书架、柜子的内层），可站边前面到上一层之间清空（门按开着处理，第 4 节）；上方敞开的台面边上的东西保留（椅背、推车的篮筐沿和把手）。盒子收紧到网格表面外 1/4 格，最薄 1.5 cm。现在这个场景有 6642 个墙盒子、5117 个家具盒子（35 件），见 `<场景目录>/colliders.json`（每件家具的盒子数、每个台面判出的顶）。墙也挡物体，所以接着要跑沉降。
- **沉降**（`scripts/scene_settle.py`，sim venv）：ADT 位姿配上比物体略大的凸包，成排的书互相嵌进最多 1.3 cm，物体嵌进台面板最多 8 mm，砧板悬在岛台上方。先对平移加重阻尼、对转动加轻阻尼跑 4 s，嵌入被慢慢推开，物体不会获得速度，圆的东西（胡萝卜）还能滚到自己稳的那一面；再撤掉阻尼跑 20 s 检查，最后把落定的位姿写回 `scene_meta.json`（`init_pos`、`init_quat`）和两个 XML。报告在 `<场景目录>/settle.json`。
- **检查**（`scripts/scene_lint.py`，sim venv，2026-10-19，方案 §12）：任何内环开跑前跑一次。竖直轴偏离静置姿态、被盖住的台面、静置 30 s 掉下去、目标开局就成立是错误，退出码 1；静置漂移、斜着放、扁物体是警告。结果写在 `<场景目录>/lint.json`（或 `--out`）。

起点池里的摆放是按抽样那天的台面高度抽的。场景运行器每局摆好物体后先调 `SceneWorld.seat`（2026-10-20）：如果物体底面在它下方某个台面的板面以下 1.2–5 cm，就把它抬到板面上方 0.5 mm。抬了多少记在每局记录的 `seated` 里。之所以从 1.2 cm 起算，是因为池里的摆放本来就让凸包嵌进板里最多 0.9 cm，开局静置会把它推开；在旧场景里，1496 个摆放没有一个够得上 1.2 cm。

## 4. 房间 `room`

| 字段 | 出处 | 含义 |
|---|---|---|
| `bounds` | rule | `{x: [min, max], y: [min, max]}`，整个场景的范围；替换导航里写死的 `ROOM_XLIM` / `ROOM_YLIM` |
| `walls` | rule | 墙的线段 `[[x0,y0],[x1,y1]]`，从 `ApartmentEnv` 在 0.1–1.5 m 高度切片得到 |
| `obstacles` | rule | 挡路家具的轮廓多边形（占 0.05–1.8 m 高度带的固定物体，ADT 包围盒的凸包）；导航把它们画进占用图 |
| `doors` | adt + rule | 门洞线段和对应的门实例（`Apartment_BedroomDoor` 等）；第一轮门按"开着"处理：占用图里不算障碍，MuJoCo 里也没有碰撞体（D3 曾给门加了凸包，浴室门半开堵住半个门洞，2026-10-16 去掉） |
| `rooms` | rule + claude | `{<名字>: {polygon: [[x, y], ...], name_source: claude}}`：多边形是墙 + 门洞围成的连通区域（规则算），名字由 Claude 按里面的家具起（"kitchen"） |
| `spawn` | rule | `{<房间>: [多边形, ...]}`，能出生的区域（离墙和家具 ≥ `body_radius` + 余量） |

## 5. 随机化 `randomization`

| 字段 | 含义 |
|---|---|
| `movable_pool` | 可以挪位置的物体 |
| `placements` | 每个可挪物体能去的台面列表，及在台面上的可放区域（离边 ≥ 余量） |
| `distractor_pool` | 可以随机增减的干扰物 |
| `spawn_rooms` | 出生房间（第一轮 = 任务所在房间，#9） |

## 6. 任务文件（`outputs/` 下，每个场景一份）

```yaml
- id: put_banana_in_white_bowl
  room: kitchen
  goal: [in(Banana_A, WhiteBowl)]            # 谓词合取，由仿真真值判定
  prompts: ["Put the banana in the white bowl.", "..."]   # 3–5 句，英文
  bucket: 1                                   # 1 现有技能可做 / 2 缺技能 / 3 超出能力
  plan: [{primitive: approach, args: {...}}, ...]          # bucket 1 才有，Claude 写
  missing: []                                 # bucket 2：缺的技能或谓词，附规格
  reason: ""                                  # bucket 3：超出能力的原因，引用 capability.yaml 的键
  checks: {...}                               # 程序可行性检查逐项结果
  provenance: {proposed_by: claude, model: opus, prompt_sha: ...}
```

## 7. 谓词库 v1

| 谓词 | 判定（仿真真值） | 现有代码 |
|---|---|---|
| `on(X, S)` | X 底面在 S 顶面上方 `PUT_ON_HEIGHT_M` 内、xy 在 S 轮廓里 | `SceneModel` 的 put-on 判据；plan 以 `put_down_single` 结束时另要求手里没东西、手不碰 X（`plan_episode.criterion_for`） |
| `in(X, C)` | X 在 C 的开口内、低于沿 | `place_phase.in_container` |
| `held(X)` | X 离开原台面，且同一只手的两根手指都接触它（或两只手都接触）；2026-10-16 起（TODO 27.9），之前一根手指接触就算 | 现有"末了还持着目标"判据 |
| `upright(X)` | X 的 `upright_axis` 与世界 z 夹角 < 15° | 新 |
| `near(X, Y, d)` | xy 距离 < d | 新 |
| `in_room(robot, R)` | 骨盆在房间多边形内 | 新 |

谓词不够用时（"把桌面收拾干净""把书叠起来"），Claude 写进任务的 `missing`，算缺口报告的一部分（#7）。

## 8. 读取方要跟着改的地方（9.6 做，不在本格式里）

- 导航：用 `room.bounds` 和 `room.walls` 建占用图，台面按 `yaw` 画成带转角的方框。
- `SceneModel.front_line` 和走近的站位规划：读 `yaw`，从 `approach_edges` 里选边。
- 场景编译：只把本局任务涉及的物体和抽到的干扰物做成自由体，其余作为固定几何（公寓里 350 个自由体会拖慢仿真）。

## 9. 用户定的三处（2026-10-03，均按推荐）

1. **带转角的台面**：存 `yaw`，导航和站位规划改为支持它（9.6）。不做"吸附成轴对齐、其余丢掉"：那会悄悄丢家具，是半生效配置。
2. **自由体的范围**：每局只把任务物体和抽到的干扰物做成自由体，其余作为固定几何。
3. **房间命名**：规则按墙和门洞切区域，Claude 按区域里的家具命名，出处标 `claude`。
