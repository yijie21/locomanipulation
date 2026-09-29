# STATE — 2026-09-29（分支 `loco_unreal`：Unreal 渲染 + Isaac / Unreal 外观按 Blender 和 Unitree 官方资产标定）

工作目录 `/workspace/code/locomanipulation-loco_unreal`（worktree），基于 `9955487`，落后 main。本文件只管这条分支。
Unreal 的命令、产物和坑见 [docs/handoff/unreal-urlab-20260928.md](docs/handoff/unreal-urlab-20260928.md)；Isaac / Unreal 对 Blender 的测量见 [台账](docs/ledger/blender_scenes.md) 第 2–6 节。

**已提交**（2026-09-29，用户说）：本分支 09-28 / 09-29 的全部代码、实验脚本、补丁和文档（TODO 10.10、10.14）；插件改动提交在插件仓库本地分支 `loco-fresh-capture` 的 `58c8a98`（未推送）。未推送，未合并到 main。
仓库外也改了（不进 git）：`/workspace/ue_57/ue_cmd_server.py`、宿主工程的 `DefaultEngine.ini` / `DefaultEditorSettings.ini`。
`assets/hero.png` 是用户放的，不要提交。
没有运行中的 UE 或 Isaac；GPU0 空闲。GPU1 上有别的会话的 `depth_server`，还可能有 `loco_agent` worktree 的 instance 1 栈，渲染前先看 `nvidia-smi`。
AnyImageNav 的 `run_hm3d_when_free.sh` 在等 GPU0 空闲（< 1000 MiB 且没有 sonic 进程），占 GPU0 前先看一眼。

## 当前进度

**已确认（09-29）：G1 外观 = Unitree 官方 Isaac 资产的材质（用户定）。** Isaac `stage.G1_LOOKS`，Unreal `unitree` 预设；Dex1 手指按 body 名查色，不再落到近白。

**已确认（09-29）：Isaac 外观。**
- warehouse 按 Blender 标：stage `warehouse_textured_env`，ISO 80、间接光 ×2、Hejl–Hable，灰度平均差 11.1 → 6.9。旧 stage `warehouse_textured` main 还在用，不要删。
- 2× 超采样（用户定）：`isaac.supersample: 2`，镜像、数据集补渲、视频三处统一读。实测三路相机 95 → 209 ms（2.2 倍，不是估的 1.4 倍）。
- box_platform 盒子颜色转线性：新 stage `box_platform_srgb`，纸箱从近白变成纸板色。
- 新外观不重训（用户定）。
- 验证：`outputs/isaac_ssaa_20260929/`（测速、1×/2× 对照、两个场景的生产视频）；单元测试 1631 过、7 跳过。

**已确认（09-29）：Unreal。**
- 编辑器一启动就打开 warehouse 关卡，机器人已在（导入的蓝图已存盘）。只在 MJCF 变了时才重新导入。
- 视口默认关（`ue_cmd_server.py` 设 `ShowFlag.Rendering 0`）；崩溃不再上报 Epic。
- 灯板改用自建自发光材质（167 → 230，Blender 238）；曝光不改；Lumen 漫反射增益 4（天花板 90 → 114，Blender 174）。
- 截图默认 8 遍（用户定），137 ms；1 遍 56–64 ms 但偏暗 2–3。天花板比 Blender 暗约 60，用户接受。
- 09-28 的：全部网格非 Nanite + `r.Shadow.Nanite=0`；一局闭环 VLA 看 Unreal 图 1/1 ok（`/workspace/code/locomanipulation/outputs/runs/unreal_eval_20260928`，那局是 Nanite 版）。

**假设 / 待验证。**
- 现有 Isaac checkpoint 看的是旧外观（发白、FXAA、ISO 30），在新外观下没有评测数字；用户定不重训。
- 实时模式的间接光依赖前几帧：比较 Isaac 设置时必须交错渲染、重复、先把相机挪开再回来。
- URLab 插件会段错误（TODO 10.17），两次都在 `close()` / 重连附近；标定脚本改为先停 PIE 再 `close()` 后没再崩。

## 关键命令

```bash
# Isaac 生产视频（这个 worktree 没有 assets/isaac，ISAAC_ASSETS 指向 main）
ISAAC_ASSETS=/workspace/code/locomanipulation/assets/isaac bash scripts/render_isaac_videos.sh <run> --traces <run>/eval/state_trace --out <dir> --insets --gpu 0
# 重建 box_platform stage
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/build_isaac_primitive_scene.py /workspace/code/locomanipulation/outputs/scene/box_platform --label box_platform_srgb --assets /workspace/code/locomanipulation/assets/isaac
# Unreal：起编辑器（4 核 + 看守），就绪即可出图；其余命令见 handoff 文档
bash outputs/unreal_blender_match_20260929/ue_up.sh
```

坑：等后台进程时不要用 `pgrep -f <脚本名>`，它会匹配到等待循环自己的命令行，循环永远不退出。

## 下一步（编号见 `TODO.md` 第 10 节，顺序等用户定）

- **10.8 / 10.9**：Unreal 100 局评测、正式接入 `images.source: unreal`。
- 合并到 main / 推送：等用户说。`scripts/build_isaac_warehouse_scene.py` 取自 main 的版本再改，合并时以本分支为准。
