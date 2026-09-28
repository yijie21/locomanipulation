# Primitive 契约与 Session 对象设计（2026-09-28）

承接 [agentic 决策记录（2026-09-03）](agentic-plan.md) 的 Q1（agent 只在任务规划层）、Q3（相位边界闭环）、
Q4（特权状态文本，几何查表）、Q6（模板文法，判据由 sim 计算）、Q9（JSON 进出）。那份记录写在
`grasp_gym` 单一专家之上；此后仓库多了 `simple` 单臂专家、`visual` 双臂专家和 warehouse 配置，
本文回答"这三条已调顺的线怎么切成 agent 能拼的 primitive"。未写代码。

阅读前提：[单臂详解](../single_arm_design.md)、[双臂详解](../bimanual_design.md) 的阶段描述，
和 [标准](../reference/standards.md#interfaces-and-configuration) 的接口规则。

## 0. 结论与范围

- **primitive 的粒度是"技能"，不是"阶段"。** 两条线的抓取核心（单臂 `SimpleGraspPolicy._run`
  的 takeover→recover，双臂 `CoordinatedReach` 的 observe→hold）各自共用一个连续参考流、
  一个 IK 姿态先验和一套反馈滤波状态；按阶段切开就把被调顺的连续性切掉了。每个核心整体是一个 primitive。
- **能独立抽出的是核心两侧的动作**：走近与刹车、原地转向、抬手、交还规划器、放置、后退。
- **缺的能力**：持物前进（只有倒退）、双臂放下。它们进目录但标 `unvalidated` / `missing`。
- 本文只设计接口和状态对象；场景生成（Q10）、agent 的失败恢复验收（Q11）不在本文范围。

## 1. 词汇

| 词 | 含义 |
|---|---|
| primitive | 一个阻塞执行、自带时序的动作模块：读 session 与特权状态，驱动栈到一个声明好的后置状态，返回 `Outcome` |
| session | 一局内在 primitive 之间传递的全部状态：栈句柄、场景模型、手臂通道归属、朝向锚、夹爪、持物、日志 |
| profile | 一组绑定在一起的参数：走近方式 + 抬手姿态 + 抓取参数。见 §6 |
| plan | agent 给出的 primitive 调用序列（JSON） |
| executor | 校验前置、执行 primitive、记录、把结果回给 agent 的循环 |
| 发现层 | 从 `scene_meta` 枚举可做的任务模板实例（§8） |

## 2. 现在散落着、要收进 session 的状态

| 现状位置 | 内容 | 进 session 的字段 |
|---|---|---|
| `RoundContext.p`、crate `run_attempt` 的 `p` | `Primitives`：commander + 特权订阅 | `stack` |
| `RoundContext.meta`、crate `placement` / `meta` | scene_meta 与本局采样后的物体位姿 | `scene` |
| `RoundContext.fv`、crate `heading` + `arm._face` | 走近后一直保持的世界朝向 | `heading` |
| `ctx.coordinated.arm`、crate 局部变量 `arm`（`HangingWalk` / `PlannerArm` / `StreamedArm`，类型即阶段） | 谁在发上身参考 | `arm_owner` + `arm` |
| `StreamedArm.hands`、`p.cmd.left_hand/right_hand` | 两手闭合量 | `hands` |
| 单臂 `hold_offset` / `palm_frame_offset`，双臂 `closed_relative` / `attachment` | 持物及其在手中的位置 | `held` |
| `StreamedArm.q`、`PlannerArm` 的 q17 | 最后一次发出的上身参考 | 通过 `arm` 读，不复制 |
| `RoundContext.execution`、crate `record` | 阶段日志 `ExecutionRecord` | `record` |
| `RoundContext.rm`（reach map）、`rm.graspable` | 可达查表 | `scene.reach` |
| 走近结束时的停位、深度、横向、朝向误差（现在只在日志里） | 抓取前置要查的量 | `stance`（`approach` 的后置产物） |

不进 session 的：`task_grasp`、`grip`、`obj0`、`entry_cmd`、`descend_from` 等抓取核心的内部量。它们留在黑盒里。

## 3. Session 对象

```python
# src/locomani/skills/session.py
class ArmOwner(Enum):
    PLANNER_SWING = "planner_swing"  # 规划器自己摆臂（HangingWalk）；行走用
    PLANNER_FROZEN = "planner_frozen"  # 上身钉在 q17（PlannerArm / FROZEN_UPPER_BODY）；腿归规划器
    POSE_STREAM = "pose_stream"  # 29 DoF 全身 POSE 流（StreamedArm）；腿也归我们


@dataclass
class HeldObject:
    key: str
    sides: tuple[str, ...]  # ("right",) 或 ("left", "right")
    grip_in_palm: dict[str, np.ndarray]  # 各手掌系下的物体位置，place 的伺服基准
    hold_offset: np.ndarray | None  # 单臂：躯干系保持偏移；双臂：None


@dataclass
class Stance:
    target: str  # 停在哪个物体 / 表面前
    profile: str  # 用哪个 profile 走近的（§6）
    depth_m: float  # 前沿到最前点
    lateral_m: float
    yaw_err_rad: float
    quiet: bool  # 静止准入是否通过


@dataclass
class Session:
    stack: Primitives  # 注入，测试给假 obs
    scene: SceneModel  # §8
    record: ExecutionRecord
    clock: Callable[[], float]  # 仿真时钟；wall-clock 与 lockstep 由此区分
    arm_owner: ArmOwner = ArmOwner.PLANNER_SWING
    arm: Any = None  # HangingWalk | PlannerArm | StreamedArm
    heading: float | None = None  # 世界朝向锚；None = 还没 reanchor
    hands: dict[str, float] = field(default_factory=lambda: {"left": 0.0, "right": 0.0})  # 0 开 1 闭
    held: HeldObject | None = None
    stance: Stance | None = None
    trace: list[dict] = field(default_factory=list)  # 每次 primitive 调用一条（§7）
```

不变量（executor 在每个 primitive 前后断言）：

1. `arm_owner` 与 `arm` 的类型一致；`POSE_STREAM` 时 `arm` 必须是 `StreamedArm` 且未 `stop`。
2. `held is not None` ⇒ 对应手 `hands[side] > 0`；`hands` 里的值与 `arm.hands` 一致。
3. 任何原地循环（转向、抬手、抓取）之前 `heading is not None`。未锚定的朝向是随机游走，
   DECISIONS 里 50° 的漂移就是这个来源。
4. `stance` 只由 `approach` 写；任何走动的 primitive 结束时把它清为 `None`。
5. `held` 只由 `grasp_*` 置、`release` / `place` 清；`hand_back` 保留它。

## 4. Primitive 契约

```python
# src/locomani/skills/contract.py
@dataclass(frozen=True)
class Violation:
    check: str  # "stance_depth"
    measured: float
    bound: tuple[float | None, float | None]


@dataclass
class Outcome:
    ok: bool
    reason: str  # "[tag] message"，tag 与 tasks/base.bracket_label 同一套
    stages: list[dict]  # 本次进入过的 record 阶段
    metrics: dict
    duration_s: float
    produced: Any = None  # 后置产物：approach → Stance；grasp → HeldObject


class Primitive(Protocol):
    name: str
    requires: frozenset[ArmOwner]  # 允许的进入通道
    yields: ArmOwner  # 成功时的通道

    def preconditions(self, session: Session, **args) -> list[Violation]: ...  # 纯函数，只读 obs/scene
    def run(self, session: Session, **args) -> Outcome: ...  # 阻塞，驱动栈
```

规则：

- **前置在 primitive 外检查**。`preconditions` 不发命令，executor 先调它；有 `Violation` 就不进 `run`，
  违规列表原样回给 agent（Q3 的闭环需要机读的失败原因，不是字符串）。
- **后置是 session 的变化 + `produced`**。`run` 成功必须把 `arm_owner` 变成 `yields`，把该写的
  session 字段写掉；executor 用 §3 的不变量校验。
- **失败标签沿用 `[tag]` 约定**，`scripts/gate.py` 和 `failure_reasons` 已经按它归类。
- **超时属于 primitive**，在 `run` 里；executor 不加外层超时。
- **安全块归 executor**：跌倒、物理时钟停滞、钳子空闭，这三件事现在分别写在两条线的每拍循环里
  （`simple_policy.py:950-1006`、`coordinated_reach.py` 各阶段）。黑盒包装期它们留在原地；
  迁出后统一为 `session.watch()`，每个 primitive 的 tick 调它。
- **黑盒内部的重试保留**（单臂 entry 的 retreat 重进、双臂 reopen / reseat），executor 层不做隐式恢复。
  恢复决策是 agent 的事（Q11），层间加隐式重试会把它的贡献抹掉。

## 5. 目录

| primitive | 前置（`requires` / 检查） | 后置（`yields` / session 变化） | 来源代码 | 状态 |
|---|---|---|---|---|
| `approach(target, profile)` | SWING 或 FROZEN；目标在 `scene`；到前沿 ≤ `start_edge_max` | 同进入通道；写 `stance`（含 `profile`）；`heading` 锚定 | crate `VisualApproach` + `HangingWalk` + `GuardBrake`（`primitives/foot_watch.py`，两线共用）；bottle `EdgeGuard`；`face_target` | 抽取：把"桌边线"泛化成 `scene` 给的前沿线（crate 已有 `box_front_line`） |
| `turn_to(yaw)` | SWING 或 FROZEN；`heading` 锚定 | 同通道；`heading` 更新；`stance = None` | `Primitives.turn_to`、`face_target` | 可用；小于 9° 不准，进前置文档 |
| `walk_to(xy, heading)` | SWING 或 FROZEN | 同通道；`stance = None` | `Primitives.walk_to` / `navigate`（占用图） | 可用；持物时未验证，见 `carry_to` |
| `ready(pose)` | SWING 或 FROZEN；静止准入 | FROZEN；`arm = PlannerArm` | crate `prepare_stationary_ready`（`lifted` / `forward`）；bottle `_lift_ready` | 抽取；两线已共用 `BimanualIK.lifted_ready` |
| `grasp_single(obj, side, profile)` | FROZEN；`stance.target == obj` 且 `stance.profile == profile`；`rm.graspable(rel, side)`；物体在相机准入内 | POSE_STREAM；`held`；`hands[side] = 1` | `SimpleGraspPolicy._run` G0–G8（`simple_policy.py:627`） | 黑盒包装，右手；左手是新校准 |
| `grasp_bimanual(obj, profile)` | FROZEN；`stance` 同上；全身 IK 准入（`unsupported_assisted_reach`） | POSE_STREAM；`held` 双手 | `CoordinatedReach.step` observe→hold（`coordinated_reach.py:428-725`） | 黑盒包装 |
| `hand_back()` | POSE_STREAM | FROZEN；`held` 保留；`hands` 保留 | `StreamedArm.stop(preserve_reference=True)`（`streamed_arm.py:213`） | 可用，两线相同 |
| `place(container, side)` | POSE_STREAM；`held` 单手；容器在 `scene` 且在 ego 帧内 | POSE_STREAM；`held = None`；`hands[side] = 0` | `bottle/place_phase.py`（turn, carry, descend, release, raise） | 黑盒包装；只有单臂；内部的 `place_turn` / `place_carry` 是短距原地伺服，不是行走 |
| `release(sides)` | POSE_STREAM 或 FROZEN | 同通道；`held = None`；`hands = 0` | `Primitives.release`、`place_phase._release` | 可用 |
| `retreat(distance)` | FROZEN；`held` 可有 | FROZEN；`stance = None` | `crate/retreat.py` | 可用；倒退，上身冻结 |
| `carry_to(xy \| relative)` | FROZEN；`held` | FROZEN；`stance = None` | `catalogue/locomotion.py`（`walk_to` + `LoadWatch`，转向先 slow_walk） | 已实现并测（§9a、§9b）；持瓶 1–2 m 漂移 < 1 cm |
| `take_over()` | FROZEN；静止准入 | POSE_STREAM；`arm = StreamedArm`；`held`、`hands` 保留 | `StreamedArm.start(hold_planner_pose=True)`（`expert.py:728-739`） | 可用；`hand_back` 的逆，`place` 之前要它 |
| `put_down_bimanual(surface)` | POSE_STREAM；`held` 双手 | POSE_STREAM；`held = None` | `catalogue/crate_put_down.py` | 首个实现（§9b） |
| `navigate_to(xy \| target)` | SWING 或 FROZEN | 同通道；`stance = None` | `catalogue/locomotion.py`（`Navigator` 距离场分段） | 已实现（§9b） |

"把桌上的垃圾扔进垃圾桶"按这份目录展开是：

```
approach(trash, bottle_simple) → ready(lifted) → grasp_single(trash, right, bottle_simple)
→ hand_back() → carry_to(bin 前) → take_over() → place(bin, right) → hand_back() → retreat(0.35)
```

九步里的 `carry_to` 于 2026-09-28 实现并在持瓶上测过（台账 bottle §41）；其余七种 primitive 都有现成代码可包。

## 6. Profile：抓取参数与走近方式绑定

同一份双臂代码在两个场景上切了六个开关（`approach_stop`、`ready_pose`、`reach_path`、`close_trigger`、
`hold_frame`、`walk_steer`），`visual_config.py` 的注释逐条写着后一个是为了补前一个的后果：
guard 刹车决定停位散布，散布逼出 lifted 抬手、staged 路径和 contact 触发。
所以 approach 参数和 grasp 参数不能各自独立选。

```python
# src/locomani/skills/profiles.py
class GraspProfile:
    name: str
    hands: str  # single | bimanual
    config: Any  # 解析后的 BottleTaskConfig / CrateTaskConfig，包装专家的 primitive 整个交给它
    approach: Mapping  # approach_stop, stand_off, walk_speed, guard_*, face_after_brake, walk_steer ...
    ready: Mapping  # ready_pose, ready_tool, ready_pitch_deg, elbow_min_deg ...
    grasp: Mapping  # 单臂 SimpleGraspConfig 的其余字段；双臂 reach_path / close_trigger / hold_frame ... + carry


PROFILE_SOURCES = {
    "bottle_simple": "configs/experiments/bottle_simple_200.yaml",  # reach.simple + guard
    "crate_low_table": "configs/task/crate_visual_low_reach.yaml",  # depth / forward / direct / gate / world
    "warehouse_box": "configs/experiments/box_centre_base_20260927.yaml",  # guard / lifted / staged / contact / pelvis
}
```

- profile 的值不在代码里重写，从现有 yaml 加载并冻结；一次运行读到的每个值仍然在运行目录的 `config.yaml` 里。
- `approach` 把 `profile.name` 写进 `Stance`；`grasp_*` 的前置检查 `stance.profile == profile`。混配直接拒绝。
- 选哪个 profile 由物体类别决定（§8）：单手可抓 → `bottle_simple`；`bimanual` 标记 + 低桌 → `crate_low_table`；
  `bimanual` + 地面箱 → `warehouse_box`。agent 不选 profile，它选任务。

## 7. Executor 与 agent 的交互

```
agent  --JSON {"primitive": "approach", "args": {...}}-->  executor
executor: preconditions → [Violation] 非空则回传，不执行
          run → Outcome；断言 §3 不变量；record.enter(...)；trace 追加一条
executor --JSON {outcome, session 快照（owner/heading/hands/held/stance）, 场景快照}-->  agent
```

- 每次调用一条写进 `<run>/agent_trace.jsonl`：输入、违规、Outcome、耗时、session 快照。
  这是 Q9"全部输入输出落盘"的落点，也是数据集转换切"思考窗口"（Q5）的依据。
- 一局的成败**不由 agent 判**（Q6）。任务模板编译出判据（§8），executor 在 `plan` 结束或 agent 发 `abort` 时用特权状态算。
- 失败局照录，manifest 标失败的 primitive 与 `[tag]`。
- 与现有 `Task.run_round` 的关系：`run_round` 仍是固定专家的入口；plan 执行是新入口（`scripts/plan_episode.py`，
  按标准一个扁平脚本读 `summary.json`），两者共用 `RoundResult` 和 acceptance。

## 8. 场景发现接口

`scene_meta.json` 已经有发现层需要的字段：`objects[key]` 的 `box`、`bimanual`、`container`、`surface`、`init_pos`；
`surfaces[key]` 的 `top`、`xy`、`half_extent`；`robot_start`。

```python
# src/locomani/skills/scene_model.py
class SceneModel:
    def __init__(self, meta: dict, reach_map): ...
    objects: dict[str, ObjectInfo]     # ObjectInfo: key, box, on_surface, bimanual, container, pose()
    surfaces: dict[str, SurfaceInfo]
    reach: ReachMap                    # rm.graspable 的宿主
    def front_line(self, target) -> Line:            # approach 的刹车线：表面近沿或 box 前壁
    def affordances(self) -> list[Affordance]:       # pick(obj, hands) / place_into(obj, container) / put_on(obj, surface)
    def instances(self) -> list[TaskInstance]:       # Q6 模板文法的实例：put <obj> in <container> | on <surface>
```

- `affordances` 只做几何和标记判断，不猜：单手 pick 需要 `box` 最大边 ≤ Dex1 开口且非 `bimanual`；
  `place_into` 需要目标 `container` 且 `in_container`（`place_phase.py:50`）可算。
- `instances` 是 agent 的菜单。"发现任务"= agent 从菜单里选、排序、给出 plan；菜单之外的任务 executor 拒绝。
  这与 Q6"agent 说了不算"一致，也让判据可编译。
- 判据编译：`put X in C` → 终局 `in_container(X, C)` 且机器人未倒且未接触 X；`put X on S` → X 的 z 在 `S.top` 上、xy 在 `half_extent` 内。

## 9. 代码落位

- 新包 `src/locomani/skills/`：`session.py`、`contract.py`、`profiles.py`、`scene_model.py`、`executor.py`、
  `catalogue/`（每个 primitive 一个文件）。不复用 `src/locomani/primitives/`，那一层是运动学与刹车几何，
  名字撞了但层次不同；`backends/sonic/primitives.py` 的 `Primitives` 类改名不在本文范围，先原样作 `session.stack`。
- 依赖方向：第一步 `skills → tasks`（适配器包住 `SimpleGraspPolicy`、`CoordinatedReach`、`place_phase`，
  不搬代码）；`tasks` 不得 import `skills`。抓取核心搬进 `skills/catalogue/` 是第二步，等回归过了再做。
- 每个适配器只做三件事：把 `Session` 翻译成原代码要的对象（`RoundContext` 子集 / `cfg`）、调原入口、把结果翻译成 `Outcome` 并更新 session。
  适配器体量小，删掉它复杂度会重新出现在每个调用方，通得过 deletion test。
- 可测性：`Session.stack` 注入（现有测试已对 `Primitives.__init__` 打补丁）；`preconditions` 是 `(obs, scene, session) → list`
  的纯函数，可以直接在**已录的 trace** 上回放，用成功局 / 失败局校验前置带宽是否分开；`executor` 用桩 primitive 测不变量与 trace 落盘。

## 9a. 已搭起来的部分（2026-09-28，D1–D7 按推荐值定）

`src/locomani/skills/`：`session.py`、`contract.py`、`profiles.py`、`scene_model.py`、`executor.py`、
`catalogue/{locomotion,arms,bottle,crate,pending}.py`；测试 `tests/test_skills_*.py`（78 个含双臂离线回归，CPU）。
双臂专家的每拍逻辑抽到了 `tasks/crate/attempt.py`（`TickSource` / `walk_step` / `reach_step` / `record_tick`），
`run_attempt` 改为调用它们，对象构造和 `handoff()` 留在 `expert.py`，离线测试的 monkeypatch 原样有效。

| 目录条目 | 状态 |
|---|---|
| turn_to、walk_to、retreat、ready、take_over、hand_back、release | 适配器，依赖注入，假栈上测过 |
| approach | 按 profile 分派：单臂 = 朝向桌法线 + 中性腰 + `_phase_navigate` + `_phase_approach`；双臂 = `attempt.walk_step` 循环（guard 或 depth） |
| grasp_single | `execute_grasp` → `SimpleGraspPolicy.execute()`；**它自己在 finally 里交还 POSE 流**，所以产出 FROZEN 通道 + 持物 |
| grasp_bimanual | `expert.handoff` + `attempt.reach_step` 循环 + `evaluate`；成功停在 POSE 流持物，失败自己交还 |
| place | `place_phase._phase_place` 跑在 **PlannerArm（FROZEN 通道）** 上，自建 PlannerArm，结束时后退 + 验收 |
| carry_to | `catalogue/locomotion.py`：冻结上身 + `walk_to` + 持物守卫，转向先 `turn_to(mode=slow_walk)`；持瓶 1–2 m 漂移 < 1 cm，180° 转 4/4（台账 bottle §41），持箱未量 |
| put_down_bimanual | 占位，missing |

真栈证据（2026-09-28，`scripts/plan_episode.py`，GPU1 / instance 1）：bottle plan approach → grasp_single 冒烟 3/3、10 局 9/10（台账 bottle §40）。
pick_place plan（approach → grasp_single → place）冒烟 1/3（台账 pick_place §7）；warehouse plan（approach → ready → grasp_bimanual）冒烟 1/3（台账 crate 末节）。
三条线的通道交接都在真栈上走通；成功率对照要按 10 局 pilot 再跑。

与 §3–§8 的差别：
- `GraspProfile` 多了 `hands` 和 `config`（primitive 直接拿真实 dataclass）。
- `yields` 可以是 `frozenset`（approach：guard 模式把手臂交给行走策略 → SWING；depth 模式先抬手再走 → FROZEN）。
- `grasp_single` 要求 SWING（单臂 guard 走近后手臂在行走策略手里；抬手在黑盒内）、产出 FROZEN；`place` 要求 FROZEN、产出 FROZEN，不需要 `take_over`。
- `Session` 多了 `reset`、`camera`、`directory`、`scratch`（approach → grasp 的私有交接）。
- 单臂 approach 的站位取自 `plan_for_target` + `sample_stand`（无抖动）；双臂的刹车线和走入轴取场景 `robot_start` 朝向，与 `run_attempt` 一致。
- `Violation.check` 目前有：heading、held、profile、target、stance、prepared_policy / prepared_attempt、arm、camera、container、view、side、arm_owner、unknown_primitive、unknown_arg、unimplemented。

## 10. 验证计划

1. **单臂回归**：plan `approach → ready → grasp_single → place → release → retreat` 对 `configs/experiments/pick_place.yaml`
   现有入口，同 seed 10 局（pilot 规模），`scripts/gate.py compare bottle` 的 16 项阶段时长不退步。
2. **双臂回归**：plan `approach → ready → grasp_bimanual → hand_back → retreat` 对 `box_centre_base_20260927.yaml`
   同 seed 10 局；对照的是那份 yaml 自己的接触基线，不是 crate 低桌。
3. **前置带宽回放**：拿两条线现有运行目录的 trace，跑 `preconditions`，报告成功局中被误拒、失败局中被放行的比例。
4. **`carry_to` 测量**：先于实现。持瓶 / 持箱各 10 局，走 1 m 与 2 m，记掌系漂移与脱手；这决定它是 `walk_to` 加守卫还是独立实现。
5. 以上都在 wall-clock 栈上做；lockstep 评测线不受影响。

## 11. 待定决策（推荐值在前）

| # | 问题 | 推荐 | 备选 |
|---|---|---|---|
| D1 | 包名 | `skills`，避免与运动学层 `primitives` 混 | 复用 `primitives` 并把现有的改名 |
| D2 | profile 混配 | executor 硬拒绝（`stance.profile != profile` 是 `Violation`） | 只告警 |
| D3 | 前置里用特权状态 | 允许（Q4 已定；采集不是部署） | 只用相机准入 |
| D4 | 黑盒内重试 | 保留；executor 不加隐式恢复 | 抽到 executor |
| D5 | `hand_back` / `take_over` 是否独立 primitive | 独立。agent 要在持物流态与行走态之间显式切换，`walk_to` 的前置依赖它 | 并进 grasp / place 的后置 |
| D6 | `carry_to` 先测还是先写 | 先测（§10 第 4 条），接口先占位标 `unvalidated` | 直接把 `walk_to` 当 carry 用 |
| D7 | 左手单臂抓取 | 不做，前置里 `side == "right"`；需要时是新校准 | 镜像右手参数 |

## 9b. 第二批（2026-09-28 晚，TODO 7.4–7.10）

| 条目 | 状态 |
|---|---|
| `walk_to` / `approach` 持物放行（7.7） | 持物时 `walk_to` 直接走 `carry_to` 的守卫；单臂 `approach(target)` 持物时 = `carry_to` 到目标的规划站位（`stand_plan`）+ 原地 slow_walk 转向桌法线 + 写 `Stance`，从 `planner_frozen` 进出。默认 plan 变成 `approach → grasp_single → approach(goal) → place` |
| `put_down_bimanual`（7.8） | `catalogue/crate_put_down.py`：在 POSE 流上用 profile 的 `BimanualIK` 把两只工具点同步下放到"物体静止高度"（表面顶 + 箱半高），松手、抬开；判据只看物体（高度、倾角、无手接触）。真栈样本见台账 crate 末节 |
| `navigate_to`（7.9） | `catalogue/locomotion.py`：沿 `Primitives.planner`（占用图距离场）逐路点走；空手每段 `walk_to(face_target=True)`，持物每段先 `turn_to(slow_walk)` 再守线走，`LoadWatch` 每拍查 |
| `LoadWatch` | 从 `carry_to` 抽出的持物守卫（单臂掌系漂移 + 接触；双臂高度底线 + 双手接触；接触闪断 0.3 s 容忍），`carry_to` / `navigate_to` / 持物 `walk_to` / 持物 `approach` 共用 |
| 站位准入带（7.4） | `profiles.STANCE_BANDS` + `session.stance_violations`（纯函数），`grasp_single` / `grasp_bimanual` 前置调用；`scripts/replay_preconditions.py` 在 932 条已录站位上回放：bottle 成功 / 失败深度中位 0.324 / 0.325，warehouse 0.436 / 0.454，任何上界都同时误拒成功局；**带保持全开**，失败在抓取内部 |
| `claude -p` 规划器（7.6） | `skills/planner.py`：每局一个会话（`--session-id` / `--resume`），`--json-schema` 结构化输出，`--tools ""`；请求 = 任务 + 菜单 + 目录手册 + 特权状态文本 + session 快照 + 上一条 trace；`plan_episode.py --agent` 接线，`agent_io_NN.jsonl` 落盘 |
| 新物体标定（7.10） | 没有按几何自动落位的机制；流程写在 `docs/skills.md` §5（新 profile 登记 → `--profile` 10 局 pilot → 按失败族改一个旋钮 → gate 对照 → 回放定带） |

§5 表里的 `put_down_bimanual` 从 **missing** 改为"首个实现"，`carry_to` 从 unvalidated 改为持瓶已测（§9a）、持箱见台账。
