# Blender 场景 → MuJoCo + Isaac

把一个 `.blend` 房间变成这条流水线能用的两套场景:**MuJoCo 管物理和策略看到的图像,Isaac 管离线重渲**。
两边不是同一份东西的两种格式,分工见 [§5](#5-为什么两边装的不一样)。

首次跑通的完整记录和实测数字在[台账](ledger/blender_scenes.md)。

## 1. 准备

```bash
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src"
which blender                      # 需要 Blender 5.x,它自带 USD API
```

把 `.blend` 放进 `assets/blender/`。房间必须是**公制、1 单位 = 1 米**(`blender_to_usd.py` 会打印实际值,不对就在 Blender 里改)。

## 2. 出 Isaac 场景

```bash
blender --background assets/blender/<room>.blend \
  --python scripts/blender_to_usd.py -- outputs/scene/<name>
```

产出 `outputs/scene/<name>/<room>.usdc` + `textures/`,**几何、材质、相机、灯光都带过来**。

它同时修掉 Blender USD 导出器的四处单位错误。每一处都是渲出来才发现的:

| 项 | 导出值 | 症状 |
|---|---|---|
| 相机 `focalLength`/`aperture` | 米(0.185 / 0.36) | 视场角算出来对,但 RTX 不认亚毫米焦距,**取景完全跑偏** |
| 灯光 `intensity` | 瓦特(20.69) | 画面几乎全黑,需要 ~1200 倍 |
| 曲线 `widths` | `2.0`,而舞台单位是米 | **一根地面走线渲染成 2 米宽的灰飘带** |
| `DomeLight` 环境贴图 | 一张纯色 EXR | RTX 报 `MirroredBall environment format not supported yet` 后**整个环境光失效**,背光的角落全黑 |

灯光强度**不写进 USD**,见 [§4](#4-已知未解决)。

## 3. 出 MuJoCo 场景

```bash
blender --background assets/blender/<room>.blend \
  --python scripts/blender_to_mujoco.py -- outputs/scene/<name>_mj
python scripts/build_mujoco_scene.py outputs/scene/<name>_mj
```

第一步按**材质**(不是按物体)导出 OBJ —— MuJoCo 一个 geom 绑一个材质,而房间是几千个物体共用几十个材质。
第二步在现成的 `scene_locomani.xml` 上做手术,生成新的场景:

| 加了什么 | 为什么 |
|---|---|
| 每材质一个视觉 geom,`contype=0` | 房间只是布景,不参与碰撞 |
| 托盘/桌面一个 box 碰撞体 | 它就是工作面 |
| 目标物 5 块板(底 + 四壁) | 中空;凸包会变成实心块,手伸不进去 |
| 地面 plane 保留碰撞、**关闭渲染** | plane 是平面碰撞的精确图元;它的棋盘贴图会和房间地板网格打架 |

**机器人树、关节默认值、执行器全部从 base 场景继承**,不重写 —— 重建机器人等于引入一次新标定。

贴图会自动转成 PNG:**MuJoCo 不读 JPG**。

## 4. 渲一帧并对比

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 "$VENV_SIM_PY" \
  scripts/render_mujoco_frame.py outputs/scene/<name>_mj/scene_locomani.xml \
  /tmp/mj.png --camera <相机名>

python scripts/compare_frames.py <isaac.png> /tmp/mj.png --out /tmp/pair.png
```

`compare_frames.py` 报两个数,回答的是不同问题:

- **raw** —— 直接逐像素差,主要被曝光支配,而曝光只是个旋钮
- **structural** —— 对齐均值和方差之后的差,**旋钮修不掉的那部分**:阴影、全局光、噪点

判断标准是 structural。布置对不对看它,不看 raw。

### 已知未解决

**`blender_to_usd.py` 产出的 USD 单独渲染会偏暗且结构不符**(实测 structural 19.4%,而同一份内容在组合场景里是 4.2%)。
排除过:贴图文件、贴图路径、色彩空间、材质网络、相机、累积采样、资产流式加载、灯光强度写入时机、Hydra 结构性重同步 —— **九条都不是原因**。
所以灯光增益**不烘进 USD**,留给渲染时的调用方;仓库的数据集重渲走 `locomani.sim.isaac.isaac_render`,不受影响。
不要再从上面九条里找。

## 5. 为什么两边装的不一样

**MuJoCo 跑物理,并且生成策略训练用的图像;Isaac 只做采集完成后的离线重渲。**
所以房间在 Isaac 里是纯视觉背景,在 MuJoCo 里也是纯视觉 —— 但**碰撞只有地面、工作面和目标物**。

489k 三角面当碰撞体撑不住 50 Hz 的全身控制器,而且任务本来也碰不到墙和货架。
这不是偷懒,是仓库既有场景的一贯做法:现有的每个 `scene_locomani.xml` 都只有**一个 plane,一堵墙都没有**。

## 6. 接进采集之前还差什么

场景能渲不等于能采。专家读的是 `scene_meta.json`:

- `surfaces.A` —— 工作面的 `top`(高度)、`half_extent`、`xy`
- `objects.<名字>` —— `init_pos`、`height`、`mesh`、`surface`

没有这两块,场景搭好了专家也不知道在哪儿干活。字段含义见 [双臂详解](bimanual_design.md)。

## 7. 进一步阅读

- [Blender 场景台账](ledger/blender_scenes.md):逐次实测数字、四个单位错误的发现过程、未解决项。
- [MuJoCo 仿真与 Isaac 渲染](rendering.md):既有的重渲链路与视频导出。
- [双臂](bimanual.md):这条线要对接的采集流程。
