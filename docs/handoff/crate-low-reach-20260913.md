# G1 低位双手抓取与全身前倾 — 2026-09-13

本次从 `f8983c1` 建立 `exp/crate-low-reach-20260913`，调查 Pico 能大幅俯身、
原 carry 却主要下蹲的差别，并实现可选的全身前倾控制。原始证据保留在
`outputs/crate_low_reach_20260913/`；各轮源码、失败和修正分开记录。
最终批次验证中，以下阶段性结果不代表最终可靠性。

## 控制路径与角度含义

G1 腰部 pitch 的模型范围约为 ±0.52 rad（±29.8°），不能靠扩大软件限位
获得更大的真实腰关节行程。上身相对世界的倾角还包含骨盆倾斜，以及腰部
三个旋转的组合；因此上身超过 30° 与腰关节没有超过限位可以同时成立。
[Unitree 官方模型](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_rev_1_0.xml)
给出关节范围，本次 IK 还直接读取实际加载模型的范围。

Pico 支持多种控制通道。旧 carry 使用 planner 的 17 维上身参考，不能直接
指定骨盆前倾姿态；Pico 的 POSE 路径可以传全身关节和根朝向。VR 三点路径
又是另一种契约，不能把它们统称为同一个“弯腰指令”。
[NVIDIA 官方 Pico 文档](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/tutorials/vr_wholebody_teleop.md)
说明模式选择；本地实际检查了 `pico_manager_thread_server.py` 的模式分发、
`zmq_manager.hpp` 和部署端参考注入逻辑。

独立探针源码 `7be8366`，SONIC v1.1 / encoder 开启，GPU 0 / instance 1。
每种模式都从远离桌子的站姿开始，6 s 连续进入参考、6 s 保持，不走路、不接触。
上身参考相同，腰 pitch 0.5 rad；VR / POSE 额外请求约 20° 骨盆前倾，
POSE 同时调整髋关节以保持浅膝。末 2 s 实测中位数如下。

| 通道 | 上身离竖直 | 骨盆 pitch | 骨盆高度 | 左 / 右膝关节 |
|---|---:|---:|---:|---:|
| frozen 上身参考 | 22.7° | −1.6° | 78.7 cm | 14.0° / 12.7° |
| VR 三点 | 30.4° | 4.7° | 78.6 cm | 16.0° / 14.0° |
| POSE 全身参考 | 44.3° | 19.5° | 78.1 cm | 14.2° / 12.4° |

三次都未跌倒。这支持“当前仿真也能浅膝、大幅俯身”，不证明抓取成功，
也不是完全相同输入的因果 A/B：三个模式能表达的输入本来就不同，初始状态
也分别采样。`lean_probe/` 保留 manifest、逐帧记录、视频和部署回执，
`lean_report.json` 记录计算结果。用户的 Pico 具体模式尚未确认。

## 实现与边界

- [低位配置](../../configs/tasks/crate_low_reach.json) 通过现有
  `carry.reach_control=whole_body` 选择全身流；默认仍使用 planner。
  carry 的 `waist_dofs=all` 开放 yaw / roll / pitch 的真实模型范围。
  `pitch` 用于旧方式对照；未传 carry 配置的 wall 接口保持原契约。
- [全身 IK](../../src/locomani/tasks/crate/whole_body_ik.py) 从实际站位、脚位和
  COM 规划前倾。固定脚的六维目标，保持 COM 水平投影，限制膝关节参考
  不超过 0.85 rad，并惩罚骨盆下降。骨盆参考最低默认 65 cm，前倾默认
  不超过 0.70 rad；所有关节保持物理限位。这是几何规划，不是动态平衡证明，
  也未提供全路径碰撞保证。
- 全身通道实际发送根朝向、29 维关节位置及速度。规划的根位置不是直接
  写入仿真，而是由腿部姿态和 SONIC 动态交付。缓冲的每帧朝向、位置、速度
  一一对应；已有固定根、零速度调用保持兼容。
- 从桌沿 1.2–1.5 m 外走近，再规划俯身；五次多项式限制位置、速度、加速度，
  双手在俯身和下探时使用一致的 60° 抓取朝向。参考到位后必须连续观察到身体
  稳定，随后重算实际双手可达性；不因计划“可达”就跳过实测。
- 仅 whole_body 模式用上述几何检查替代旧的横向 4 cm 站位启发式。
  桌沿下限、朝向、双手捕获区、闭合双侧接触、提起至少 6 cm、倾角不超过
  15° 的验收仍保留。未移动箱子帮助抓取，未使用吸附、焊接或外力托举。
- 抓取保持命令连续，不再套用旧 planner 的腕 pitch 额外补偿。提起从已经
  发出的双手参考出发，固定求解坐标系，共同上移 10 cm，保留双手朝向，
  连续到达后保持 2 s。实际接触、提起量和倾角仍独立验收。

这仍是 carry 机械诊断策略，依赖仿真物体状态。它没有替代
[RGB 专家的信息契约](crate-visual-expert-design.md)。全身参考完整写入
`*_motion.jsonl` 供审计，但当前正式数据动作格式未验证可重放这条 POSE 路径，
因此此模式的 `--record` 在启动资源前拒绝。不能把漏掉身体动作的数据直接
当作 VLA 训练集；后续接入视觉策略须同时明确可观察目标和完整动作表示。

## 试验记录

沿用 30 cm 场景，落稳把手约 44.5 cm 高、箱子名义内移 4 cm。
XML SHA-256：`a2af77f13691b642fa19dc3a0cb7d674d575f7ad64d375c401036afcb01bc760`。
机器人、箱体、摩擦、夹爪和碰撞几何与前次 30 cm 验证相同。

| 子目录 | 源码 | 结果及修正依据 |
|---|---|---|
| `whole_body_smoke` | `3983cb8` | 计划 3 次，实际启动 1 次；构造器 keyword-only 参数错误，未进入俯身；余下未运行。无 manifest，保留退出码和原始视频 |
| `whole_body_smoke2` | `d462118` | 0/3，均被旧横向站位启发式拒绝；改为全身可达性检查 |
| `whole_body_pilot` | `83525c2` | 0/5；4 次俯身后实际双手不可达，1 次朝向拒绝。参考发完时身体仍在动，且预抓取水平手掌要求过大反向腕角 |
| `whole_body_settled` | `f4643ce` | 1/5；5 次均捕获且双侧接触。4 次提起后回落至 4.4–5.9 cm。加入实测稳定窗口和一致抓取朝向后已能到达低把手 |
| `whole_body_continuous` | `37f5713` | 3/3；命令连续的抓取与提起。固定控制逻辑，以新 seed 41 做后续 10 次验证 |

`whole_body_settled` 捕获末 0.3 s 的实测上身倾角为 57.3–60.5°、
骨盆高 73.4–74.8 cm、两膝约 13.4–33.6°。这明显区别于
[旧下蹲批次](crate-table30-20260913.md) 的骨盆约 53.9 cm、膝约 98°；
两个批次的控制与准入不同，不能解释成单参数成功率 A/B。

记录还显示旧提起求解会把腰参考从约 30° 拉回接近 0°，并跟随实时骨盆
重复改变上身目标。`37f5713` 改为从捕获命令延伸一条固定的提起轨迹；
这是针对观察到的不连续所作的设计修正，最终平稳性须看实测回归。

姿态指标由 [报告脚本](../../scripts/experiments/crate_low_reach_report_20260913.py)
将 motion 的观测 seq 精确匹配到 trace qpos，再对 torso 做 FK。各阶段末
0.3 s 去重取中位数。脚的 XY 最大位移不是“打滑”判定，身体角速度 P95
包含该阶段全部运动和可能接触。不同版本不合并分母；失败视频全部保留。

## 复现

按 [crate collection](../crate_collection.md) 安装环境并准备桌高场景后运行：

```bash
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m locomani.pipeline.run_crate_pilot --expert carry \
  --config configs/tasks/crate_low_reach.json --scene <30cm场景目录> \
  --rounds 10 --seed 11 --gpu 0 --instance 1 --out <新输出目录>
```

独立通道探针为 `scripts/experiments/crate_lean_probe_20260913.py --scene ... --out ...`，
固定使用 GPU 0 / instance 1，须先遵循共享资源检查。每批 `commands.jsonl`、
`config_snapshot.json`、`inputs.json` 和 `deploy_receipts.jsonl` 保留实际启动身份。
视频交付按 [压缩规则](../rendering.md#video-export) 保留完整时长，原片另存。
