# 箱体边沿抓点规划

本页描述可迁移双臂抓箱的第一步：从箱体几何、6D 位姿和机器人实际停车状态，
选择一对边沿抓点并生成静态全身姿态。下一步见[仿真执行](box_rim_execution.md)；
正式采集仍使用[现有专家](bimanual_design.md)。

## 支持范围

支持具有四条直线顶边、已知壁厚的刚性开口矩形箱体。箱体尺寸、位姿和机器人
停车状态是输入，不依赖蓝色篮子的对象名或固定抓点。不从任意三角网格自动推断
箱口、软硬程度或可夹边缘；封闭箱、翻盖与复杂边沿需要另外的几何描述和策略。

箱体坐标系的原点、尺寸、位姿约定由输入接口明确规定。边沿上的禁抓区和箱角
余量决定可用线段，左右抓点是这些线段上的变量。仿真目标碰撞体必须与输入几何
一致；仅修改几何描述的尺寸，不能作为跨尺寸验证。

## 规划过程

1. 在两组相对边沿生成抓点，组合夹爪倾角与插入深度，再按箱体完整位姿变换。
2. 根据机器人实际基座、关节和双手状态分配左右目标，执行双臂 IK 初筛。
3. 对候选使用全身 IK 检查双手误差、脚部锚点、机器人重心偏移和关节限制。
4. 检查机器人自身、箱体及环境的碰撞，区分允许的夹指接触与不允许的穿插。
5. 在通过检查的候选中，按姿态改变量、关节余量和抓取中点偏离重心等指标排序。

输入停车状态建立支撑脚锚点。全身解可以在支撑约束内调整身体，但规划器不能
把机器人整体平移到另一个站位来制造可达性。没有通过检查的候选就返回失败与
拒绝原因；重新走近、重新停车属于后续在线策略的职责。

搜索是有限采样和有限次数的 IK 求解。失败表示当前候选与预算内未找到通过检查
的解，不是对所有可能姿态的不可达证明；未进入精修的候选必须与检查失败区分。

抓取中点与重心的距离是排序启发式，不是力闭合证明。脚部锚点误差与机器人重心
偏移是现有全身 IK 的静态检查，不等于含箱体负载的动力学平衡认证。
碰撞检查只覆盖模型中启用的碰撞体；物流间货架等视觉网格不自动成为碰撞障碍。

## 输出与使用边界

结果保留所选边沿、左右抓点、夹爪方向、全身姿态、候选诊断及拒绝原因。
ghost 图表示该静态参考姿态。成功返回只表示通过了实现中的静态检查，不代表：

- 从停车姿态到抓取姿态的整条路径无碰撞；
- 夹爪闭合、摩擦承重或纸板强度足够；
- SONIC 能稳定跟踪参考并完成抬升。

下一步在线接入需要预抓取路径检查、受限重规划、实际接触和试提验收。
正式采集还需要记录输入几何、实际停车状态、所选抓点以及失败原因。

## 代码入口

- [箱体几何与位姿](../src/locomani/tasks/crate/box_rim_geometry.py)
- [静态抓点规划](../src/locomani/tasks/crate/box_rim_planner.py)
- [离线 JSON 与渲染入口](../src/locomani/tasks/crate/box_rim_cli.py)
- [目标碰撞几何与夹指检查](../src/locomani/tasks/crate/box_rim_collision.py)

## 输入约定

JSON 入口使用 `schema_version: 1`；未知字段会被拒绝。

| 字段 | 含义 |
|---|---|
| `scene_xml` | 相对请求 JSON 所在目录解析，也可使用绝对路径 |
| `target.body` | 场景中目标物体的自由关节 body 名称 |
| `target.collider_mode` | `matched_target` 检查现有碰撞体；`replace_with_open_box_proxy` 在临时场景中替换为匹配的开口箱体 |
| `box.size_m` / `wall_thickness_m` | 外部三维尺寸与壁厚，单位米 |
| `box.pose` | 箱体外底面中心的世界位置与 `quaternion_wxyz` |
| `box.local_com_m` | 箱体局部重心；省略时使用几何中心作为排序代理 |
| `box.forbidden_intervals` | 指定边沿及沿边坐标区间；检查包含指垫宽度的接触区域 |
| `robot_state` | `source: explicit` 时提供实测 29 关节与 `pelvis` 位姿；`scene_default` 仅适用于静态示例 |
| `planner` | 搜索预算、候选角度和检查阈值，默认值见自动生成的[字段表](reference/config_fields.md#planneroptions) |

位姿四元数采用 wxyz 顺序且必须为单位长度。`x-` / `x+` 边沿的沿边坐标是箱体局部 y，
`y-` / `y+` 的沿边坐标是局部 x。两手的输出顺序始终为 left、right，结果同时给出
每只手对应的边沿名称与局部坐标，不能仅凭箱体 x 的正负推断左右手。

`matched_target` 当前要求尺寸、壁厚与五个未旋转的局部 box 碰撞体一致，并拒绝
额外启用的目标碰撞体。箱体整体仍可通过其 body 位姿旋转。代理模式保留原始场景，
但预览与检查针对的是替换后的几何，不能当作原资产的精确碰撞认证。

## 运行

仓库提供[完整请求示例](../configs/planning/box_rim_example.json)，使用随仓库分发的
小型几何测试场景，将其中的自由物体替换为开口箱体。它不依赖物流间运行产物。
从仓库根目录运行静态规划，不启动控制栈：

```bash
. scripts/paths.sh
PYTHONPATH=src "$VENV_SIM_PY" -m locomani.tasks.crate.box_rim_cli \
  configs/planning/box_rim_example.json \
  --output outputs/box_rim_example/plan.json
```

成功返回退出码 0；本次搜索未找到静态有效姿态返回 2，仍写出候选诊断。
增加 `--render-png` 可保存 ghost 图；渲染前按共享资源规则选择空闲 GPU，并通过
受管渲染入口运行，例如：

```bash
PYTHONPATH=src MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  "$VENV_SIM_PY" -m locomani.pipeline.render_job run --out outputs/box_rim_example \
  -- "$VENV_SIM_PY" -m locomani.tasks.crate.box_rim_cli \
  configs/planning/box_rim_example.json \
  --output outputs/box_rim_example/plan.json \
  --render-png outputs/box_rim_example/ghost.png
```

要用于真实停车状态，复制请求文件并填入该时刻的 `robot_state` 和箱体位姿。
原场景、旧专家和正式采集配置都不会因此改变。

## 验证记录

两种箱体、三种站位及物流间场景的输入、结果与限制见
[2026-09-20 静态规划验证](handoff/box-rim-planner-20260920.md)。测试入口为
`tests/test_box_rim_geometry.py` 与 `tests/test_box_rim_planner.py`；它们使用真实
MuJoCo 几何，但不运行机器人动态抓取。
