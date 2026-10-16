# 场景文件与任务文件格式（2026-10-03 定）

TODO 9.5。方法的输入是一份场景文件（[决策记录](scene-discovery-plan-20261002.md) #4、#5），输出是任务文件（#7、#10）。
本页定字段（用户 2026-10-03 过目，§9 三处按推荐定）；读取、校验与谓词判定的代码在 `src/locomani/scene_discovery/`。能力说明在 `configs/scene_discovery/capability.yaml`（#8）。

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
| `upright_axis` | adt | 物体系里朝上的轴（ADT `canonical_pose.up_vector`），判 `upright` 用 |
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
| `approach_edges` | rule | 哪几条边前面有 ≥ 机器人身宽 + 余量的空地，可以从那条边走近 |
| `room` | rule | 所在房间 |
| `provenance` | — | 同上 |

台面由规则从家具网格提取：水平面积 ≥ 阈值、高度在 `capability.yaml` 关心的范围里（地面以上 0.1–1.3 m）。

## 4. 房间 `room`

| 字段 | 出处 | 含义 |
|---|---|---|
| `bounds` | rule | `{x: [min, max], y: [min, max]}`，整个场景的范围；替换导航里写死的 `ROOM_XLIM` / `ROOM_YLIM` |
| `walls` | rule | 墙的线段 `[[x0,y0],[x1,y1]]`，从 `ApartmentEnv` 在 0.1–1.5 m 高度切片得到 |
| `obstacles` | rule | 挡路家具的轮廓多边形（占 0.05–1.8 m 高度带的固定物体，ADT 包围盒的凸包）；导航把它们画进占用图 |
| `doors` | adt + rule | 门洞线段和对应的门实例（`Apartment_BedroomDoor` 等）；第一轮门按"开着"处理 |
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
