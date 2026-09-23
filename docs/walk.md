# walk 场景：空地上只有一台 G1

给 SONIC 下的 G1 测运动性能用：MuJoCo 原生蓝地板，除了机器人什么都没有。场景名 `walk`（名字表 `configs/scenes.yaml`）。

## 构建

```bash
. scripts/paths.sh
export PYTHONPATH="$LOCOMANI_ROOT/src"
"$VENV_SIM_PY" -m locomani.tasks.walk.scene outputs/scene/walk
```

构建过程是 [触达包络的空地场景](reach_envelope.md#1-准备一次) 再加一台相机，所以机器人（双手 dex1、手指垫、夹爪齿比）
与 crate / warehouse 采集线的物理完全一致。`scene_meta.json` 里 `objects` 为空、`robot_start.xy` 为原点。

## 相机

| 相机 | 用途 |
|---|---|
| `walk_follow` | 视频默认相机（`video_camera`）。`trackcom` 模式，始终在机器人质心右侧 3 m、高 1 m 处横看，走多远都在画面中间；不随机器人转身 |
| `reach_overview` / `reach_side` | 固定在原点附近，机器人走出一米多就出画 |
| `head_camera` / 两个腕相机 | 机器人自带 |

跟拍检查图：`outputs/scene_previews/walk/follow_check.png`（原点、x=4 m、(4, 3) 转 90° 三个位置）。

## 已知

- XML 里 pelvis 在 x=-1.5（沿用 crate 场景的起点），reset 才把机器人放到 `robot_start`。跟拍相机的位置按 XML 里的 pelvis 算，
  改起点位置不用动相机。
- 还没有 walk 的实验文件或探测脚本；运动性能探索从这里开始。
