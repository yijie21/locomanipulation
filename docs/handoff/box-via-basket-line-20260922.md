# 纸箱走篮子线（2026-09-22）

用户决定：两个任务只是场景不同，抓取代码应完全相同。箱沿线（`box_rim_*`，5,000 行自有逻辑，见
[深度扫描记录](box-rim-depth-sweep-20260922.md)）不再往上修；篮子线的视觉 expert 参数化后直接跑纸箱。

## 改了什么（`0abb131`）

- `grasp_region.target_key(meta)` / `grasp_points_for(obj)`：目标名读 `scene_meta.target_object`，抓点由 `objects[key].box`
  生成（每侧端壁一点，壁内 8 mm、沿顶下 25 mm）；蓝篮子复现 `GRASP_POINTS` 到 0.06 mm。
- `expert.py`、`visual_approach.initial_robot_xy`、`prepare.py`、`evaluation.evaluate(key=)`、`AssistedGuidance(key, box)`
  全部按 key 工作；`CoordinatedReach` 收到 `CoordinatedTarget(object_key, grasp_points_local)`。
- `scene_lib`：目标 `nor_cal_box`（0.45×0.32×0.33，1 cm 壁，1.2 kg，长轴横放、近壁齐桌沿）+ 预设 `box_bimanual`；
  `bimanual_scene.py --preset box_bimanual --table-top 0.371`（托盘高度）。资产 `assets/crates/nor_cal_box/model.xml`，
  30 MB 视觉 mesh 不入库，从 `outputs/scene/logistics_mujoco/mesh/` 复制。
- 场景 `outputs/scene/box_table37`：箱心 x −0.141，沿顶 0.703 m，机器人起点 −1.5 m。

## 两次 20 局（`outputs/runs/box_pilot_20260922`、`box_pilot_env_20260922`）

| 配置 | 结果 | 失败 | 最远阶段 |
|---|---|---|---|
| 篮子线原配置，仅 `stand_off` 0.50 | 0/20 | reach_stalled 11, 推动箱子 2, 试提未附着 2, 摔倒 2, 视觉丢失 2, 初始距离 1 | reach 13, lift 2 |
| + 包络旋钮（反馈 1.0/0.15、全身上限、实测锚定，同 `crate_env2`） | 0/20 | 推动箱子 8, reach_stalled 6, 摔倒 2, 抬起脱手 1, 视觉 2, 初始 1 | reach 15, lift 1 |

实测站位中位 0.32–0.34 m（篮子线成功局是 0.24）。第一版手停在抓点 5–8 cm 外、修正量顶在 4 cm 上限；放大反馈后手到了箱子，
但变成把箱子推走（8 局）——手从上落到端壁时先碰壁顶。3 局合了爪抬了起来又脱手（1 cm 壁 vs 指垫）。视频在各 run 的 `videos/`。

## 意义

篮子线在纸箱上端到端跑通了：走入、停稳、落手、合爪、抬起都发生过。剩下的是**同一条代码路径上的调参**，和篮子线自己
2026-09-13 从 0/10 走到 17/20 的过程同类：站位（箱子更深，端壁中点在桌沿内 14 cm）、`pregrasp_above`、落手方向、
薄壁的夹持。不再有"计划不完美就不许试"的闸门。

## 未做

- 调参：站位（目标 0.24–0.28）、抓点沿端壁往近沿挪、`pregrasp_above`、掌面俯仰；每次 20 局对照。
- 薄壁夹持：1 cm 壁在 dex1 指垫间的保持力未量。
- 视觉丢失 2–3/20：纸箱比篮子高，ego 相机的可见性门限按篮子标定。
