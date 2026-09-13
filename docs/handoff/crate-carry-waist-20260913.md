# Carry / wall 前俯腰部修复 — 2026-09-13

## 请求与实现

用户指出 `bimanual_carry.BimanualArms.__init__` 仍将三个腰关节限位写为零，
要求允许机器人弯腰，抓取桌内较深位置的篮子。基线为 `3118005`；工作分支
`fix/crate-carry-waist-20260913`，工作树为
`/workspace/code/locomanipulation-crate-waist-20260913`。

- 前俯上限改为 0.4 rad（22.9°），同时受真实模型限位约束；扭转和侧弯保持中立。
- carry / wall 共用的 `BimanualArms` 改用现有 `BimanualIK`，同时求解双臂及共同的腰部。
  左右臂不再依次改变腰部；从测量值求解时也读取腰部。wall 的额外肩部限位、
  姿态权重、ghost 记录接口保留。
- IK 在关节达到限位后将剩余求解方向分配给可动关节，避免持续向限位外求解后截断、
  留下本可消除的双手误差。RGB 专家也使用这一共享算法，信息来源和配置入口不变。
- carry 站位目标不进入既有的桌沿距离下限内，停稳后再次检查 13 cm 下限。
  目标还预留现有 `STAND_TOL`（6 cm）作为停步缓冲；该缓冲不保证消除所有惯性越界。
  远于旧距离带的目标还需通过双手 IK 残差检查；原有横向、朝向、接触和抬升判据保留。
  检查使用同一次停稳后的观测，失败仍写入 manifest。

没有新增 CLI 配置字段；现有入口与参数位置见
[使用说明](../crate_collection.md#waist-and-reach)。carry / wall 仍是使用仿真物体状态的
历史专家，视觉指导示范入口是 [visual](../crate_visual_collection.md)。本次没有证明 VLA 学习效果。

## 可复查证据

本机产物根目录：`outputs/crate_carry_waist_20260913/`。保留全部成功、失败、
原始录像和运行日志。`diagnose.py` 是隔离的控制诊断，不是采集专家。
各 pilot 的 `config_snapshot.json` 记录源文件、场景和预启动配置哈希；
`deploy_receipts.jsonl` 记录实际 deploy / policy 身份，`commands.jsonl` 记录命令。

初始红测试：`make test TESTS=tests/test_crate_carry_waist.py`，
`ik.hi[14] == 0.0` 导致 `carry still locks forward waist pitch` 断言失败。
原日志：`/workspace/tmp/crate_carry_waist_red_20260913.log`。

### CPU 几何对照

`cpu_comparison.json` 使用仓库自带几何 fixture、零初值、双手横向坐标 ±0.1933 m、
高度 −0.026 m、手掌俯仰 60°、220 次迭代。距离和高度均相对骨盆。
原代码另存为 `baseline_bimanual_carry.py`；不是依赖历史文字结论的推断。

| 目标距离 | 原版锁腰：左 / 右误差 | 仅解锁原依次求解器 | 本次同时求解：左 / 右误差 |
|---|---|---|---|
| 0.44 m | 5.95 / 5.95 cm | 2.80 / 0.03 cm | 0.075 / 0.075 cm |
| 0.50 m | 13.29 / 13.29 cm | 0.35 / 0.36 cm | 0.46 / 0.46 cm |
| 0.60 m | 25.30 / 25.30 cm | 14.63 / 14.64 cm | 8.82 / 8.82 cm |

这说明单独取消零限位不能保证两个已求解目标同时成立；也说明放开腰部后仍有不可达目标。

### SONIC v1.1 实际指令交付

GPU 0 / instance 1，经共享 launcher 启动，`no_encoder=False`。
命令：`LOCOMANI_ROOT=<worktree> PYTHONPATH=src <venv_sim>/bin/python -u
outputs/crate_carry_waist_20260913/diagnose.py`。

同一个新求解器，仅将诊断中的前俯上限在 0 与 0.4 rad 间切换。每个条件重新站稳，
通过实际 `BimanualArms` / `PlannerArm` 发送 17 维指令。各条件一次，非随机顺序；
末尾 50 个控制 tick 的均值见 `tracking_summary.json`，全部样本见 `tracking.json`。

| 双手目标距离 | 上限 | 指令前俯 | 测得前俯 | 左 / 右工具点误差 |
|---|---|---|---|---|
| 0.44 m | 0 | 0° | −1.67° | 5.22 / 6.12 cm |
| 0.44 m | 0.4 rad | 17.11° | 15.96° | 2.90 / 1.54 cm |
| 0.50 m | 0 | 0° | −1.82° | 10.46 / 10.45 cm |
| 0.50 m | 0.4 rad | 22.92° | 21.52° | 4.02 / 3.69 cm |

四个条件均未跌倒。误差按当前骨盆坐标系计算，不能把骨盆前移计为手臂伸长；
0.50 m 解锁条件的骨盆平面漂移约 6.0 cm，原始记录中保留该量。
这是自由空间跟踪诊断，证明该通道实际交付前俯及该组目标误差降低；
并不证明所有手臂姿态的前俯都会增加前伸，也不等于带接触抓取成功率。

### 任务回归

场景从 `outputs/scene/crate_bimanual` 复制，仅将篮子初始 x 向桌内平移
0.12 m、0.16 m 或 0.18 m，同时更新 XML 与元数据，不改变质量、接触或桌子几何。
各场景、seed、命令和失败保存在对应子目录。所有运行均为非 exporter pilot。

初版 `carry_deep`（12 cm，seed 11，2 集）为 2/2，抓取时测得前俯
21.30° / 19.95°，抬升 12.24 / 11.78 cm。第 0 集停步后距离桌沿仅 9.37 cm；
这是新增停稳后桌沿检查的原因，该结果不作为最终门槛下的成功计数。
原始记录保留，禁止覆盖或删去该局限。

| 产物目录 / 版本 | 成功 | 观察 |
|---|---|---|
| `carry_deep18` / 初版，18 cm | 0/2 | 双手 IK 残差超限一次、横向站位偏差一次 |
| `carry_near` / 普通篮子位置 | 0/1 | 横向站位偏差 |
| `wall_near` / 共享 IK，普通位置 | 1/1 | 完成抓取抬升；采用默认不走路的 wall 协议 |
| `visual_smoke` / 共享 IK，RGB 路径 | 1/1 | 从桌沿外 1.211 m 开始，完成走近、抓取、抬升；`intervention=false` |
| `carry_final16` / 已加停稳检查，尚无目标缓冲 | 0/3 | 两次距离下限拒绝；一次 IK 检查通过后，实际双手捕获超时 |
| `carry_final12_buffer` / 最终控制逻辑，12 cm | 1/3 | 两次分别因距离下限和双手 IK 超限拒绝；一次完成抓取抬升 |
| `carry_final16_buffer` / 最终控制逻辑，16 cm | 0/2 | 一次横向偏差；一次抓取后 hold 倾斜 15.4°，未通过成功判据 |

最后两项是最终 carry 逻辑的验证；不与中间版本混合计算成功率。
12 cm 成功集为 `bimanual_s11_02`：停稳时距桌沿 17.06 cm，抓取时前俯指令
22.77°、测得 19.74°，抬升 12.00 cm，最终倾斜 4.43°。这建立了一个完整任务中的
可行样例；总体深处抓取仍不稳定。18 cm 拒绝及 16 cm 抓后倾斜失败没有通过调阈值改为成功。

实际命令见各目录的 `commands.jsonl`；最终两项也可由保留的
`regress_final.py` 查看调用协议。源代码无功能变化后仅清理了 `solve` 文档字符串；
整合了远端 `4d940b0` 的 manifest 汇总修复，该修复不参与控制回路。

成功集预览：`previews/carry_deep12_success.mp4`，**346,647 bytes**，
保留原片完整过程，11.92 s → 11.90 s（10 fps 重采样）；完整解码、尺寸、
起止画面已检查，记录在 `previews/verification.json`。原片 5,860,687 bytes 保留。

历史 wall gate 执行结果为 **FAIL，6/7**：`exit_majority` 从历史 `at the stand`
变为当前 `None`。当前 wall smoke 不走路，不能据此宣称行为等价或成功率改善。
命令及结果：`wall_historical_gate.log` / `wall_historical_gate.json`。

## 验证范围

回归测试覆盖实际 carry 构造入口、两手共同目标、测得腰部作为初值、
真实 PlannerArm 的 17 维上身字段、wall 调整后的关节限位、超远目标残差，
以及桌沿、横向和朝向门槛。CPU 成功不替代 GPU 测量；单次小 pilot 不证明可复现成功率。
既有 crate 低成功率仍不是用户要求的阻塞项，失败必须如实保留。
`make check`（含 CPU 套件）通过；相关验证原始日志为 `final_reach_tests.log`、
`integrated_make_check.log`。检查覆盖当前修改及 `4d940b0` 的整合，未重新运行 bottle GPU。
所有本次 GPU 栈已通过各自 launcher 的 `finally` 停止，instance 1 锁已释放。
