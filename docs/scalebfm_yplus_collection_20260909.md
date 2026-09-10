# ScaleBFM y+ 边中部采集分布

2026-09-09，用户要求修改采集位置。此次只调整初始化逻辑并做有界验证，没有恢复旧 Goal、训练模型或启动批量采集。

## 当前默认

`run_visual_bottle.py` 和 `collect_visual_cohort.py` 默认 `--scene-profile yplus_center_v1`。

| 项目 | 新规则 |
|---|---|
| 接近侧 | 只允许 y+；机器人向桌内的 -y 方向接近 |
| 瓶子沿边位置 | x = 桌面中心x + uniform(-0.08,0.08)m；替换旧名义x=-0.16m |
| 高桌/低桌近瓶深度 | 距y+边向内 uniform(0.12,0.18)m；旧值为4–8cm |
| 低桌深瓶深度 | 保留向内18–24cm |
| 机器人起步 | 距y+边外1.1–1.3m；相对瓶子x横向±20cm |
| 初始朝向 | 指向瓶子的方向±20° |
| 桌高 | 高76.1139cm、低62.1139cm；不改变桌子资产 |
| 动作 | 保留a16接近、伸手、站位及执行器规则 |

当前桌面中心为(0,0)，y+边约0.5044m，因此近瓶x∈[-0.08,0.08]、y∈[0.3244,0.3844]m；
深瓶y∈[0.2644,0.3244]m；普通机器人起步x的边际范围[-0.28,0.28]、y∈[1.6044,1.8044]m。
恢复示范 `stop_short` 仍按其定义直接从计算站位后方10cm开始，但也只在y+侧。

12–18cm与±8cm是本次实现对“向内一点、沿边靠中间”的具体取值，已向用户说明；不是通过VLA选优得到的参数。
瓶子移深后，a16高近的目标骨盆到边距离为15cm，低近/低深为22cm。

## 隔离旧实验与重放

旧分布名称为 `legacy_edges_v1`，保留旧采样函数和随机数顺序；共享 `run_bottle.py` 未修改。
旧比较驱动显式选择旧分布，旧正式评估默认仍为旧分布。新分布评估需要传 `--scene-profile yplus_center_v1`。
重放从原case的 `scene_profile` 恢复分布，旧case无该字段时按旧分布解释。
新配置及每集结果保存分布版本；不能向旧输出目录追加新分布。C复用B时检查分布一致且B已完成。

旧文件未改写，旧 Goal 仍blocked/用户已取消。旧a16图示、资格30/30、B17 17/60等均属于旧三侧分布。
本次修改后，当前部分源码哈希与旧冻结记录不同是预期行为；旧实验原始快照继续保留在其 `source/` 目录。

## 命令

以下CPU三场命令已执行，结果保留，不要为了恢复上下文重复运行：

```bash
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 \
third_party/Isaac-GR00T/.venv/bin/python \
src/teleop_free/scalebfm/run_visual_bottle.py \
  --out outputs/diagnostics/scalebfm_yplus_center_20260909 \
  --seeds 18001,18002,18003 --scene-profile yplus_center_v1 \
  --strata high_near,low_near,low_deep --policy-device cpu \
  --native-codec --action-hz 25 --grip-kp 1 --grip-kd 0.025 \
  --lift-height 0.12 --carry-clearance 0.12 --integral-gain 1 \
  --edge-clearance 0.22 --high-table-front 0.20 \
  --reference-style direct --facing edge
```

以后需要保存学生RGB/本体/动作数据时，在**新输出目录**运行相同a16参数并增加 `--record-observations`。
批量B组入口是 `collect_visual_cohort.py --arm B --scene-profile yplus_center_v1 --out <新目录>`；
该入口使用GPU，先按 `docs/STATE.md` 核对资源；本次没有运行批量入口。
不要用旧 `run_visual_comparison.py` 启动新分布研究，它固定旧三侧实验协议。

## 验证与范围

- 42项 visual 回归测试通过，包括新采样范围、拒绝其它边、旧分布随机数不变、A/B采样适配一致、恢复示范禁止混用分布等。
- 新采样范围测试覆盖三个场景各300个种子；测试使用平移后的桌面以检查居中依赖桌面坐标而不是世界原点。
- CPU完整专家冒烟：18001高近、18002低近、18003低深全部成功，3/3。
- 姿态、视觉、腕连续等离线检查3/3通过，见诊断目录 `audit_summary.json`。
  三场合爪窗口绝对前倾角中位数分别0.47°、28.35°、30.96°，只代表这三个样例。
- 没有重训VLA，也没有新分布闭环学习结论；三场不是正式资格测试。

代码入口：`visual_sampling.py`、`run_visual_bottle.py`、`collect_visual_cohort.py`、`visual_baseline.py`、
`eval_visual_vla.py`、`run_visual_comparison.py`；新增测试 `tests/test_bfm_visual_sampling.py`。
