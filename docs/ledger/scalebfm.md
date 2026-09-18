> **台账(append-only)**:本文件只在末尾追加新的日期化章节,不改写、不删除已有结论。
> 要更正一条结论,在新章节里写清楚并链接回旧的那一条,不要就地覆盖。
> 分类与格式规则见 `docs/reference/standards.md` §3。每条末尾的"(来源: ...)"是可核对的出处;
> 2026-09-10 outputs/ 事故后,所有 `outputs/...` 证据目录都已不存在,只剩这些文档里的文字记录。

# ScaleBFM 线实验总表(2026-09-07 → 2026-09-10)

ScaleBFM(上海 AI Lab,arXiv 2607.15163,MIT)是 SONIC 的候选替代 controller:mode 2 只接收 **pelvis + 双腕** 的 6D 未来 6 帧目标,腿、腰、肘由模型 inpaint;单进程、无墙钟、逐帧可复现。这条线上的成功率有三种完全不可互比的口径:**expert 成功率**(脚本专家自己抓到)、**VLA 闭环**(GR00T 学生控制)、**开发池 x/12**(Codex Goal 的 CPU 候选筛查)。三天里 controller 层面被验证可用,但没有任何一个 VLA 模型超过 SONIC 线的 50/100 基线;最高的 VLA 数字是 yplus200 的 25/100 与 B17 的 17/60。

## 0. 一页结论

1. **controller 替换成立,动作表示是主战场**:mode 2 走路停位 1–2 cm、yaw 0.3°,双臂搬箱 9/10 与 10/10,单手抓瓶 expert 8/10(第 1、2 节)。但 per-row 骨盆系 keypoint 动作里没有速度,VLA 学生 0/29 根本不动;加速度列后能走却停短 6–10 cm,三个 pilot 全部 0/60(第 3 节)。
2. **v3/v4 的所有精度数字都是在错误几何上量的**:expert 一直从桌角斜着够 0.45–0.49 m;换成 SONIC 的 edge 几何 + BIAS 随站立 yaw 旋转后 expert 在三条边 2/2(第 4 节)。
3. **~20° 腰部扭转是 mode 2 的够物策略,不是站距的函数**:STAND_D 0.25–0.55 每档都 ~20°;单变量证明走路→伸手的腕旋转突变(~60°/12 cm)贡献了一半的胸部 roll(18.4→8.9°),但抓取率不变(第 4、5 节)。
4. **Codex 视觉 expert Goal(A/B/C 三臂,固定预算)在单 seed 上 13/17/11 of 60,McNemar p=0.585,用户 09-09 07:07 UTC 取消**;B 高桌 0/20,seed 29 未跑完,无胜者(第 7 节)。
5. **yplus200 25/100** 是这条线 VLA 的最高绝对数;失败中 38/100 骨盆沉到 0.60 m 以下,而相对骨盆高度动作被 q99 截到 +1.836 cm,学生无法从下蹲恢复(第 9、10 节)。
6. **2026-09-10 outputs/ 被误删**:yplus200 checkpoint、200 条数据、100 场评估、ABCD 实验目录、场景构建全部消失;只剩源码、tests、configs 和这些文档(第 13 节)。

## 1. 2026-09-08 spike:接口、停位散布、双臂搬箱(分支会话)

**为什么跑**:SONIC 的 planner 通道站位散布 7–11 cm、yaw ±7°,腰 pitch 从不交付;验证 ScaleBFM 能否直接替换。

| 变量 | 数字 | 裁定 |
|---|---|---|
| 策略调用 | 9 个 tensor(root_quat/base_ang_vel/dof_pos/dof_vel/last_action 各 3 帧历史,target_pos[1,6,14,3]/target_quat,mode_index=2,time_offsets 0..5),50 Hz,PD 200 Hz decimation 4;eager torch 5090 上 6.5 ms/步,不需要 TensorRT | 接口固定;FK MJCF 必须用 ScaleTrack 的 `g1_29dof_fk.xml` |
| pilot-23 双臂脚本 | **10/10**;停位 0.248–0.267(目标 0.24),横向 <1 cm,yaw 0.1–0.4°(SONIC 横向 7–11 cm,yaw ±7) | 停位散布比 SONIC 小一个量级 |
| 开环手误差 | hover 处 3–6 cm,右手固定内偏 4–8 cm;腕参考加积分 servo(Ki 2/s,上限 8 cm)后消除;双臂抬起时骨盆自己后退 ~5 cm | servo 留在 expert 里并限幅 |
| 涌现弯腰 pilot(run_bend) | 站距 U(0.28,0.44),**9/10**(0.62 桌),矮桌 0.45 **10/10**;最远成功站距 0.426,失败一局 0.466(手短 8 cm = servo 上限);躯干前倾 14–30° 随站距增大;ego 相机箱子可见 ≥0.91 | 前倾是涌现的,SONIC 从未交付 |
| envelope 对照 | 84 格里 pad 误差 ≤10 cm:ScaleBFM 49、SONIC 站立 6、SONIC 下蹲 8;开环 pad 误差 x 0.30 处 ~2 cm,x 0.35 以后每 5 cm 增 2.5 cm | controller 层面可替换 |
| 论文数字 | global mode 成功率 97.8 vs SONIC 59.4;local mode MPKPE 持平 0.043 | 真机路线未验证 |

(来源: docs/handoff/scalebfm_handoff.md「一句话」「策略接口」「已知未决」§6–7;memory scalebfm-interface-facts.md)

## 2. 2026-09-08 单手 expert:v3(时间参数化)8/10,v4(无时钟)0/3

**为什么跑**:要一个能录 200 集的单手抓瓶 expert。v3 原则:每帧命令 = 瓶子位姿 + 可见状态的函数,无积分器、无 hover、无骨盆 servo。

| 变量 | 数字 | 裁定 |
|---|---|---|
| 探针(bottle_probe) | 开环手短 6–11 cm、低 3–4 cm、掌心低头 13°;头相机在站距 ≥0.60 时丢瓶子(前倾 35° 看地),腕相机全程 0.85–0.96 | 需要前馈偏置 |
| v3 参数 | 站点 = 瓶子 − (d−0.06)·接近方向 + 0.15 m 向左,d~U(0.45,0.55);走路 0.5 m/s、加速度 0.6;剩余 0.5 m 时伸手;BIAS (10.0,−3.0,2.8) cm + 13° 抬头 + KP 1.0 上限 5 cm;合爪线索 pad 在抓点 2/1.5 cm 内;抬 10、carry 后退 15 | 瓶子在右肩前时抓点误差 0.6–1.7 cm,正前方 2–3 cm |
| v3 结果(seed 1) | **8/10**;停位比目标远 4–7 cm;合爪 1.0–1.3 s 触发;抬起 9 cm;静止帧 0.08;失败:一集瓶子在爪里转 37°,一集 carry 摔倒 | 采集推荐版本 |
| pad 硬化 | 无硬化时瓶子在爪里转 8–18°;运行时补 priority 1/condim 4/solref 0.002 后 6–14° | 瓶子场景缺箱子场景的接触硬化 |
| v4 无时钟(纯状态 carrot) | 六轮 **0/3**;策略对静态偏差死区 ~10 cm(12 cm 骨盆 carrot 只给 0.11 m/s,20 cm 手 carrot 不动);加 lead 0.25/v 0.5 能走但入刀 2/3 撞倒瓶子 | 参考必须带速度且领先身体;若重做,用实测进度 u 参数化 v3 曲线 |

(来源: docs/handoff/scalebfm_handoff.md「v3 单手 expert 的动作逻辑」「已知未决」§1–2;memory scalebfm-interface-facts.md)

## 3. 2026-09-08 三个 VLA pilot 全部 0/60:动作表示缺速度

**为什么跑**:用户定的第一个实验——v3 固定条件采 ~200 集,训一个与 SONIC 47.5% 基线同条件的模型,看闭环能否超过 ~50%。

| pilot | action | 开环 | 闭环 | 失败形态 |
|---|---|---|---|---|
| bfm_pilot_v3 18-d | pelvis + 右腕位姿,同行骨盆系 | nmse 0.04 | **0/29** | 从未离开出生点:expert 0.4 m/s 时 plan 恒在骨盆前 ~10 cm,每行都写 "+0.10",速度不在数据里;controller 以 ~7 cm/s 接近静态目标 |
| v3 26-d(离线中心差分速度 7 列,`--anchor live`) | + pelvis v xyz/yaw rate/腕 v xyz | nmse 0.12(离线速度尖峰:腕 p95 0.57 m/s,峰 1.8 m/s) | **0/60 + 0/60** | 能走能站,合爪短 6 cm、高 7–9 cm(expert std 3 mm,policy std 4 cm),站位远 ~8 cm |
| bfm_pilot_v4 23-d(原生平滑速度 4 列) | pelvis v + yaw rate | body_mae 0.44 cm | **0/60** | 停短 10 cm(0.55–0.61 m,前倾时漂到 0.62–0.70 m);stand offset 停在 +0.02——停下的步态读作 "stand",训练帧里没有"停了还得走"的样本 |

执行器同日修的两个缺陷:每次 replan 后 4 个延迟 tick 回退到 settle hold(20% 的 tick,trace `phase` 出现 `vvvvvvvvhh`);`--level-pelvis` 必需(相对骨盆旋转累积成 0.4 rad 后仰,把腕目标抬高 20 cm);0.4 s replan 在腕 x 0.34 处卡住,0.8 s 能完成。recorder 从 tick 0 写入的 46 行 idle 前缀(15% 帧)是吸收态隐患(`--trim-idle 5`)。**裁定**:执行器旋钮用尽,修法是数据(停短状态 DAgger 或 STAND_D curriculum),与 SONIC 线 policy-stop-point-is-set-by-the-crawl 同一课。

两条并发事故:(a) check_dataset 的 grasp-causality 用原始 pelvis 系 z 判抬升,19/200 有效集误判 FAIL(读 3.8 cm 中位/2.0 最小,真值 9.3/8.8),pilot 在训练前中止——任何物体位姿列先用 root_position/orientation 重建世界系;(b) GPU1 上的 ScaleBFM sim 探针触发 SONIC deploy 自愈重启,07d-200 采集的 15–17 三个 session 被切成 1–6 s 短集——**录制 stack 旁绝不跑 sim 探针**。

(来源: memory scalebfm-action-needs-velocity.md;docs/STATE.md「Recent results」2026-09-08 bfm_pilot_v3/v4 条与「ScaleBFM stop-short」条;memory object-pose-is-pelvis-frame-in-audits.md;memory no-sim-probes-beside-recording-stack.md)

## 4. 2026-09-08 场景几何:桌角→桌边、BIAS 修正、STAND_D 扫描、扭转裁定

**为什么跑**:用户从 pilot 视频里看出 run_bottle.py 一直沿 +x 接近,而瓶子名义位置离 −x 边 34 cm、离 +y 边 12 cm——expert 站在桌角斜够 0.45–0.49 m,与 SONIC gym(目标离最近边 4–8 cm,正面接近,骨盆离边 0.17–0.32 m)不是同一个任务。v3/v4 的所有精度数(3 mm expert 容差、cm 级 policy 失败)都是这个几何上的。

| 变量 | 数字 | 裁定 |
|---|---|---|
| BFM_SCENE=edge(默认)/corner(复现 v3/v4) | 换边后第一版 "bottle pushed" 3/3、stand err 13 cm | 腕 servo BIAS (0.10,−0.03,0.028) 是 +x 上量的 body 系偏移,当世界常量用时在 ±y 边把手送偏 10 cm;改为 BIAS_w 随站立 yaw 旋转 |
| BIAS 修正后 | x−、y−、y+ 三条边各 **2/2** | 任何在单一朝向上调的常量都可疑 |
| STAND_D 扫描(每档 6 局) | 0.25–0.30 **3/6**;0.30–0.35 **4/6**;0.35–0.40 **3/6**;0.40–0.45 **2/6**;0.45–0.55 **5/6** | 0.45–0.55 + 重叠伸手仍是最好组合;近站距败在 carry(CARRY_BACK 0.15 m 在桌边把瓶子拉掉) |
| 腰 yaw/roll 峰值 | 每档都 ~20°,伸手开始后 1 s;到位后再伸手(BFM_REACH_START=0)yaw 23→14–18°,roll 不变,成功降到 3/6 | 扭转是 mode 2 对"远腕目标 + 水平定高骨盆 + 无躯干约束"的全身解,不是站距;mode 表里没有 pelvis+torso+wrist 模式 |

(来源: memory scalebfm-scene-geometry-and-twist.md;docs/STATE.md「ScaleBFM twist」条与「Recent results」2026-09-08 edge geometry 条)

## 5. 2026-09-08 侧倾单变量诊断(Codex,CPU,seed 1/2/3)

**为什么跑**:第 4 节的扭转/侧倾在采集数据里普遍存在;要分清诱因。方法:`probe.py` 在内存副本里替换单个因素,edge 场景、STAND_D 0.45–0.55、GRIP_KP 40/KD 1。

| 设置 | 伸手胸部 roll 峰值 s1/s2/s3 | 腰 yaw 峰值 | 抓取 |
|---|---|---|---|
| 当前 expert | 18.40 / 17.20 / 19.93° | 25.95 / 21.25 / 24.87° | 1/3 |
| 仅 Slerp 平滑腕旋转 | **8.90 / 8.80 / 9.70°** | 15.52 / 17.00 / 15.53° | 1/3 |
| yaw 改为桌沿法线(瓶子在右 15 cm) | 21.11 / 19.97 / 19.25°(整轮) | — | 3/3 |
| 去掉 BIAS | 17.94 / 17.61 / 18.90° | — | 0/3 |
| 走完再伸手 | 19.77 / 16.69 / 20.08° | ~11–12° | 有 QACC 警告,不采信 |

切换处腕目标单 tick 旋转跳 59.40/60.58/59.74°、位置跳 11.81/11.54/12.13 cm(旋转经 15.15 cm 的 pad 偏移变成位置跳)。v4 数据集审计:210 集里 **199 集(94.8%)胸部 roll >15°**,39 集 >20°,中位/P90/最大 18.81/20.48/22.46°;腰 roll 22.22/24.01/25.10°。旧 v4 第 0 集 3.82 s:胸 roll 20.0°、骨盆 2.2°、腰 roll 18.5°——主要是腰部侧弯;recorder 还原的骨盆命令 roll/pitch 只有 ~2.4e−7/6.6e−7°,不是记录器写歪。**裁定**:腕旋转突变是已证实诱因,不是全部;采集验收缺姿态门槛;mode 6 也不锁 torso(只有 mode 7 含 torso 但启用全部 body)。15° 是诊断阈值不是验收标准。

(来源: docs/scalebfm_sidelean_diagnosis_20260908.md §1–5)

## 6. 2026-09-08 磁盘清理:ScaleBFM checkpoint 全部删除

/workspace 到 97%(37 GB 空余),GR00T checkpoint 每个 24 GB;用户批准后删到 77%。**删除**:bfm_trainsmoke、bfm_pilot_v3 model + model_vel、bfm_pilot_v4 model,以及 v3 的部分 eval/probe/replay_check;数据集与 pilot.log 保留。第 3 节的数字自此只存在于 pilot.log/STATE/memory/本台账;重算要在保留的数据集上重跑 `bfm_pilot.sh`(每个 ~80 min)。2026-09-09 branch 65 再清 outputs/experiments 195G→101G(bfm_trainsmoke 目录一并删除)。

(来源: memory kept-checkpoints-2026-09-08.md;docs/STATE.md「Disk cleanup」两条与「The v3/v4 ScaleBFM checkpoints are gone」条)

## 7. 2026-09-08 18:56 → 09-09 07:07 UTC Codex 视觉 expert Goal:A/B/C 固定预算比较(取消)

**为什么跑**:用户要"看着瓶子走近、接近时逐渐抬手、够得到就直立抓、必要时前倾、少侧倾扭腰"的 expert,而且**由固定预算的 VLA 闭环**而不是 expert 成功率来选采集策略。VLA 输入只有两路 RGB(ego_view、right_wrist,320×240,当前帧 + 0.4 s 前)+ 91 维本体(53 + 自身速度 9 + 上一动作 29);动作原生 29 维(pelvis 9 + 右腕 9 + 速度 7 + 夹爪 1,6D 旋转,当前骨盆 yaw 系),25 Hz 动作/50 Hz 控制,每 0.4 s replan + 80 ms 延迟;首次观测 1.60 s、首个动作 1.68 s。expert 的感知是可见性/延迟/5 mm 量化/1 s 记忆约束的仿真代理,**不是 RGB 定位网络**。

**登记的验收门**(v1,资格前冻结):躯干 |roll| 峰 ≤20° 且 >15° 持续 ≤0.2 s;腰 |yaw| ≤25°;pitch ≤40°;high_near 合爪窗 |pitch| 中位 ≤15°;50 Hz 腕增量 ≤2 cm/5°;资格 ≥24/30 且每层 ≥6/10;恢复 ≥8/12;≥90% 合爪前决策至少一路相机 ≥32 目标像素且可见比例 ≥20%;同状态重放 ≥9/10。三个场景层:high_near(桌面 0.7611386 m,瓶子向内 0.04–0.08)、low_near(0.6211386 m,0.04–0.08)、low_deep(0.18–0.24);三条边 x−/y−/y+。加高桌的理由:低桌无负载标定 0.25–0.40 m 站距 pitch 20–33°,高桌 7–21°,低桌上"深度"给不出直立层。

**开发池(CPU,x/12,seed 101..112)**:a00 原始 9/12(腕跳 ~60°/12 cm);a01 姿态滤波 5/12 拒绝;a02 连续切换 9/12;a03 grip kp8/kd0.15 7/12;a04 近站 + 短 carry 10/12;a05 观测驱动 0/3 卡在 reach(预抓误差 4–5 cm > 3.5 cm 门);a06 1/3;a07 稳定夹爪 PD 6/12;a08 可见抬升修正 11/12;a09 本体跟踪补偿 low_near 12/12;a10 8/12、a11 3/12(高桌碰撞盒未随视觉网格缩放,高桌结论无效);a12 修场景 9/12(高近 pitch 24–27°,不直立);a13 8/12(低 8/8,高 0/4);mode6 可行性探针腕误差 4.9–8.4 cm vs mode 2 3.1–3.4 cm,**不换模式**;a14 直接参考 低/深 8/8、高 0/4(瓶子在骨盆前 5 cm 左 8 cm,pad 横向误差 15.5 cm——朝向瓶子抵消了右肩偏置);a15 朝向桌沿法线 12/12,资格 29/30、恢复 11/12,**但审查发现内部参考到位而身体仍差 >10 cm 时输出零速度(停短 bug)**,已采的 cohort_B 145 次尝试退役并继续扣预算;a16 实测到达修正 12/12,**资格 30/30**(每层 10/10,合爪窗 |pitch| 中位 高 2.98°/低近 25.23°/低深 30.61°),恢复 **11/12**(1710 high_near object_shift 提前碰动),启动契约 30/30,同环境延迟 chunk 重放 **10/10** 误差 0,3 cm 腕扰动重放 3/3。

**数据(每臂 200 train/20 val,上限 880 次尝试)**:

| 臂 | 内容 | train/val 帧 | 计费尝试 | 层分布 train(高/低近/低深) |
|---|---|---|---|---|
| A 旧脚本经公共执行器 | 200 常规 | 56,522/5,693 | 225 | 66/68/66 |
| B a16 | 200 常规 | 50,572/5,102 | 367(含退役 145;新 222) | 67/67/66 |
| C B 的 150 常规 + 50 恢复 | 停早 12/腕偏移 15/遮挡 12/物体移动 11;val 只有停早 2/腕偏移 3 | 47,616/4,758 | 225(含复用 165;新 60) | 67/69/64 |

训练协议:GR00T N1.7 冻结 vision/LLM,调 projector + diffusion;batch 4×accum 16 = 64,LR 1e-4,AdamW8bit,decay 1e-5,warmup 0.05,state dropout 0.2,`allow_padding=True`(否则负历史索引绕到未来帧);seed 17/29 各 3000 步,上限 18,000;只评最终 3000。选择池 60 场(seed 21001..21060,48 常规 + 12 恢复),确认池 100 场(31001..31100);4 个去噪步、40 行 chunk、最长 24 s。两次基础设施故障(OmegaConf bool resume 字段;验证 loader 共享 dict 丢 action)各 0 次优化器更新。

**正式结果(只有 seed 17 完成)**:

| 指标 | A17 | B17(a16) | C17 |
|---|---:|---:|---:|
| 任务成功 | **13/60**(21.7%) | **17/60**(28.3%) | **11/60**(18.3%) |
| 登记姿态违规 | 49/60 | 0/60 | 0/60 |
| 高桌近 / 低桌近 / 低桌深(各 20) | 8 / 3 / 2 | 0 / 10 / 7 | 0 / 8 / 3 |
| 常规 /48、恢复 /12 | 11、2(遮挡 2/3) | 14、3(停早/腕偏/物移各 1/3) | 10、1(腕偏 1/3) |
| 提前碰动 / 高度终止 / 超时 | 24 / 17 / 6 | 15 / 4 / 24 | 11 / 14 / 24 |
| 从未合爪 / 停住未合爪 | 27 / 14 | 15 / 7 | 12 / 1 |
| 训练时长 / 末次 loss | 16832 s / 0.0157 | 8923 s / 0.0189 | 9647 s / 0.0191 |

配对(单 seed,双侧 exact McNemar):B vs A 17:13 共同 0,**p=0.5847**;C vs A 10:12 共同 1,p=0.8318;C vs B 6:12 共同 5,p=0.2379。"高度终止"全部是骨盆 z<0.45 m(A17 17 场终止高度 0.4483–0.4499 m,首触 7.10–21.78 s,中位 11.98),0 条 MuJoCo 警告,固定样例是屈膝下蹲不是倒地。"提前碰动" = 合爪命令减小 >2 mm 前瓶子水平移动 >5 mm,不等于碰倒。q01/q99 归一化截断有损:腕位置误差 >1 cm 的训练帧 A 2.39%/B 1.74%/C 2.00%,最大 10.25/6.00/6.21 cm。推理延迟中位 A 65.52 ms、B 66.36 ms;可见比例 98.8%/99.65%。学生"抬手起点"诊断:成功常规场景移动重叠 A 11/11、B 12/14、C 8/10,但检测点世界手高实际升 ≥1 cm 只有 3/60、3/60、0/60(骨盆下沉会触发相对阈值),**自然抬手验收未通过**。

**取消**:用户 2026-09-09 说"先暂停,取消之前的 goal";07:05:44 UTC SIGTERM 驱动/审计(exit 143),07:07 保存停止记录。A29 停在 40 次优化器更新,无 checkpoint;B29/C29 未训;总消耗 9040/18000 更新;18/36 检查点审计;无 `selection_locked.json`、无确认。平台 Goal 因三轮自动续跑缺人工授权标为 blocked。**裁定**:B 单 seed 点估计最高但无统计优势;B/C 高桌 0/20;C 是合成恢复不是学生态 DAgger;不能宣布胜者。

(来源: HANDOFF.md §0–7;docs/scalebfm_goal_plan.md「Measurable gates」「Fixed VLA budget」「C2 reach-domain」「Bounded development repair」「Geometry-based follow-up」「Review correction」;docs/scalebfm_goal_history_20260908.md「Run ledger」;docs/scalebfm_goal_history_20260908_c4.md「CURRENT: formal C4 v2」「20:35/20:43 UTC」;docs/scalebfm_goal_history_20260909_c5_044144.md「A17/B17 selection」;docs/scalebfm_goal_state.md「User cancellation」;docs/scalebfm_vla_comparison_results.md;docs/scalebfm_acceptance_evidence.md;docs/scalebfm_visual_single_arm_design.md「观测和动作协议」;docs/STATE.md「ScaleBFM visual-expert Goal」条)

## 8. 2026-09-09 yplus_center_v1 几何、固定视频检查、policy 审计(有界,CPU)

| 变量 | 数字 | 裁定 |
|---|---|---|
| 新采集分布 `yplus_center_v1`(用户要求) | 只从 y+ 接近;瓶子 x = 桌心 ± 0.08 m;近瓶向内 0.12–0.18(旧 0.04–0.08),深瓶 0.18–0.24;机器人离边 1.1–1.3 m、横向 ±20 cm、朝向 ±20°;桌高 76.1139/62.1139 cm;y+ 边 0.5044 m → 近瓶 y∈[0.3244,0.3844];站位到边 高近 15 cm、低 22 cm | 12–18 cm 与 ±8 cm 是实现取值,不是 VLA 选优结果;旧分布 `legacy_edges_v1` 保留随机数顺序,新旧结果不混 |
| CPU 冒烟 seed 18001/18002/18003 | 3/3;合爪窗 |pitch| 中位 0.47/28.35/30.96°;42 项 visual 测试 | 不是资格测试 |
| "检查采集视频"(`scripts/scalebfm_video_check.sh`) | 10 集覆盖(高近 4/低近 3/低深 3),抓取 **10/10**、逐集审计 10/10;800×696 25 fps 侧前方跟随;44 项测试;40 张成片解码帧已看;重复调用 20 个哈希不变 | 覆盖示范,不是成功率;目录 `scalebfm_check10_yplus_center_v1_26cc7ae54f7a`(已丢失) |
| policy 审计(修复 5 项) | 新增 reset t=0 / 1.60 s / 学生首帧三道头相机可见硬门(≥32 像素,robot-unoccluded ≥0.2);相机选择改为两门都过;entry 锁 yaw 后立刻重算腕朝向;抬升判定改用合拢时可见瓶高(去掉桌高 + 8.2 cm 固定瓶半高);恢复数据首帧独立取证;49 项测试;正常 3/3,不可见注入 2/2 拒绝 | 学生接口无真值输入;但 expert 仍读 `sim.crate()` 精确中心与桌面真值,**桌边没有独立可见性门**;固定路线暗含桌子与世界轴对齐 |
| 合拢/抬起规则(当前) | 站位 ≤0.5 m 开始抬手;距站位 ≤10 cm 且掌距预抓点 ≤3.5 cm 进入伸手;沿接近轴误差 ≤2 cm、横向 ≤1.8 cm、速度 ≤0.1 m/s 才合拢;抬 12 cm,看到 ≥8 cm 抬升才回收,后收 7 cm;骨盆目标高 0.76 m | 没有"够不到再弯腰"的显式规划器,前倾由手目标 + 全身控制产生 |

(来源: docs/scalebfm_yplus_collection_20260909.md;docs/scalebfm_video_check.md「实现与验证」;docs/scalebfm_policy_audit_20260909.md「结论」「完整运行链」「已复现并修复」「仍存在的先验」;HANDOFF.md 顶部三条批注)

## 9. 2026-09-09 yplus200:当前 expert 200 集 → seed 17/3000 步 → 100 场闭环 = 25/100

**为什么跑**:用户授权独立于已取消 Goal 的一次直线实验(先验缺口有意不修)。目录 `outputs/experiments/scalebfm_yplus200_s17_20260909/`,12:39:49 UTC 完成。

| 变量 | 数字 |
|---|---|
| 采集 | 200/200 次尝试全部接受(高 67/低近 67/低深 66) |
| 训练 | seed 17,3000 步;首次启动因强制 offline 阻断 tokenizer 元数据失败(0 步),去掉 driver 的 HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE 后恢复 |
| 闭环(100 普通 reset,y+,seed 31001:31101,无恢复扰动) | **25/100**:高桌 **10/34**、低桌近瓶 **10/33**、低桌深瓶 **5/33** |
| 失败 | 超时 29、闭合前碰瓶 17、physics warning/fall 29 |

**裁定**:这条线 VLA 的最高绝对数,仍是 SONIC 基线 50/100 的一半;与第 7 节 B17 17/60 不可配对(不同分布、不同池)。

(来源: HANDOFF.md「最新完成状态」批注;docs/STATE.md「yplus200 experiment COMPLETE」与「yplus200 update」条)

## 10. 2026-09-09 yplus200 失败分析(只读回放,未改策略)

| 发现 | 数字 | 裁定 |
|---|---|---|
| 第三视角导出变暗 | `visual_follow_video.py` 与 300 视频 export.py 对新 MjData 只调 mj_kinematics 未调 mj_camlight:light_xdir 保持 (0,0,0);同帧平均亮度 42.28 → 65.95(加 camlight 或 mj_forward 后像素一致) | 导出 bug,不是训练 RGB 变暗;ZIP 未重导 |
| 夹爪犹豫 | 200/200 expert 与 25 条成功 VLA 都只有一次 open→close;**7/100** ≥3 次显著开合(最多 9 次:6 超时 1 高度终止);**55/100** 从未下过 grip<−0.02 的近全闭命令(12 超时/17 碰瓶/26 高度终止);29 条超时里 16 条掌距瓶心 <7 cm 累计 ≥1 s;seed 31087 7 cm 内 7.94 s 切换 6 次 | 至少三种犹豫;归因(感知/阶段歧义/训练不足)未消融 |
| 下沉后无法恢复 | expert 200 集骨盆最低 ≥0.73556 m,解码目标 0.75292–0.76633;VLA **38/100** 骨盆 <0.60 m(高 1/34、低近 16/33、低深 21/33);29 条 fall 标签全部是 root z<0.45、0 条 MuJoCo 警告;动作第 2 维是相对实时骨盆高度偏移,checkpoint q01=−0.02916513、**q99=+0.01836114 m**,输出 clip 到 [−1,1] → 单次修正最高 +1.836 cm;身体到 0.50 m 后目标最高 0.51836 m | 限制单次恢复幅度,不是最初下沉的因果证据 |
| "后仰摔倒"不成立 | pelvis/torso pitch 从未 <−15°(expert 最低 −1.69/−14.14°,VLA −1.76/−14.92°) | 表述为"屈膝持续下沉、失去够取姿态、触发低高度终止" |
| 深瓶更难 | 低深成功 5/33:成功 inboard 均值 19.40 cm,失败 21.52;低近成功/失败 15.05/15.01 | 不能单独证明某深度必败 |

(来源: docs/scalebfm_failure_analysis_20260909.md §1–5;docs/STATE.md「Failure analysis 2026-09-09」条)

## 11. 2026-09-09 骨盆 ABCD 对照 protocol v2(GPU0,进行中时被事故打断)

**为什么跑**:第 10 节把失败指向下沉;用固定诊断场景做干预对照。30 个固定场景(每层 10),同 30 个 seed/干预:**A** 原始 VLA;**B** 平滑保持骨盆 76 cm;**C** 低于 72 cm 持续 0.1 s 后恢复;**D** 同触发下暂停右手 1 s 再 0.5 s 渐放。模块 `visual_pelvis_ablation.py` + `test_bfm_pelvis_ablation.py`(3 项),原 VLA/expert 未改,不涉及训练。

| 变量 | 数字 | 裁定 |
|---|---|---|
| protocol v1 → v2 | 旧 EGL 基线不可复现(自身状态相同,RGB 有 1 级差异);OSMesa + deterministic math 对 seed 31001 全 24 s original/diagnostic A **bit-identical** | 不能拿旧 EGL 25/100 当配对基线;改为新采 30 A + 90 B/C/D = **120** 场;`PROTOCOL_REVISION.md` 在 B/C/D 结果前写下 |
| 进度 | driver 3039355,run_v2.log;最后对话核实 **2/120**(A/B seed 31001 完成),正式 A31001 与新离屏基线 1200 步逐项一致;旧 failure_v1.json 为归档证据 | 09-10 核查:进程不在、目录已空,2/120 不代表删除时进度,最终结果无法验证 |

(来源: HANDOFF.md 顶部 ABCD 三条批注;docs/STATE.md「Pelvis ABCD protocol v2 ACTIVE」「NEW user-authorized pelvis ABCD」条;docs/scalebfm_outputs_loss_20260910.md)

## 12. 2026-09-09 300 视频小体积导出(已丢失)

200 采集 + 100 评估(含全部 75 次失败),480×400/12.5 fps 侧前方全身跟随,只回放保存的 qpos;ZIP 53,748,107 字节,每条 86–375 KB、中位 163 KB;300 条编码帧数与 ZIP 完整性通过,14 张代表帧已看。这批视频带第 10 节的 mj_camlight 变暗 bug,未重导。路径 `outputs/videos/scalebfm_yplus200_eval100_small/scalebfm_200_collection_100_evaluation.zip`。

(来源: HANDOFF.md「300视频小体积导出」批注;docs/STATE.md「300-video compact export」条)

## 13. 2026-09-10 outputs/ 事故:什么没了,什么还在

原因以 STATE INCIDENT 另一会话记录为准:合并了误跟踪的 outputs 符号链接。现场:outputs 目录为空,ABCD 进程不在,未做块设备级恢复,未启动任何 GPU 工作。

| 状态 | 内容 |
|---|---|
| **丢失** | `scalebfm_pelvis_abcd_20260909/` 的 run.py、PLAN.md、plan.json、progress.json、逐场状态/轨迹/日志/源码快照/结果;`scalebfm_yplus200_s17_20260909/` 的 checkpoint-3000、200 条数据、100 场原始评估;300 视频与 ZIP;`outputs/scene/sim/scene_locomani.xml` 等场景构建;`scalebfm_visual_goal_20260908/` 的 runtime_assets.json、base_model_snapshots.json(整个 Goal 目录含 A/B/C 数据、模型、18 份审计) |
| **保留** | 当时的 `src/teleop_free/scalebfm/` 全部 expert/VLA/ABCD 代码、tests、`configs/gr00t_bfm_visual_2cam.py`、docs/HANDOFF(多数 untracked;09-10 原样提交为 a3e2fee,随后按 §14 重组) |
| **证据边界** | 25/100 及分层、13/17/11 等只剩文字;ABCD 无最终结果;重建的计划不能当原始结果 |
| **恢复顺序(未执行)** | 先查外部备份/用户下载;无 checkpoint 备份则重建场景 + 依赖清单 + 数据、重训后才能重做对照;编排脚本移出 outputs 到 src/scripts/docs;新实验用新 ID |

教训与 SONIC 线同源(memory never-git-add-all-with-symlinks):实验编排脚本放在 outputs 里随产物一起没了。

(来源: docs/scalebfm_outputs_loss_20260910.md;docs/STATE.md「ScaleBFM post-deletion audit」条;HANDOFF.md 顶部事故批注)

## 14. 2026-09-10 代码与文档重组:并入 locomani,删除已裁定分支

**为什么做**:Codex 线程收工(用户 09-10 确认),refactor-plan 决定 6 的冻结条件("到其 Goal 结束")已满足;49 个平铺模块、58 处环境变量旋钮、15 份带日期的独立文档都不合 `docs/reference/standards.md`。先把 Codex 的工作树原样提交(a3e2fee),再在 worktree `refactor/scalebfm` 上分三步做。

| 步 | commit | 内容 | 验证 |
|---|---|---|---|
| 1 移动 + 删除 | f8fc71b | `src/teleop_free/scalebfm/` → `src/locomani/backends/scalebfm/`(network/runtime/action/executor/chunks/reference/state/camera)+ `src/locomani/tasks/bottle_visual/`(round/harness/planner/motion/observation/sensor/scene/geometry/sampling/recovery/visibility/audit/replay)+ `pipeline/`(collect/recording/export/train/preflight/audit_checkpoint/eval/report/summarize/video_check/follow_video/provenance);tests 改名 `tests/test_visual_*.py` 进 unit tier;`src/teleop_free/` 的 8 个 alias stub 一并删除 | 当前 expert(round.py)在 CPU 上重放 seed 18001/18002/18003:trace.npz、actions.npz、commands/observations/result 与 a3e2fee **逐字节相同**;50 项 visual 单元测试通过 |
| 2 旋钮进 config | 51a4d25 | `ScaleBfmBackendConfig`(checkpoint、scene_dir、object_name、grip_kp/kd)与 `BottleVisualTaskConfig`(grasp: GraspConfig + scene_profile/strata/recoveries/max_s/native_codec/action_hz/integral_gain/reference_style/policy_device);默认值 = yplus200 采集 expert;`configs/backends/scalebfm.json`、`configs/tasks/bottle_visual.json`;round.py 的 21 个 flag 变成 `--config/--backend/--set`;BFM_EDGE / 猴子补丁 STAND_D、EDGE_INBOARD、GRIP_KP 全部改为参数;lint_env --strict 通过 | 同 3 个 seed 经新 CLI 重放,仍逐字节相同;`make lint` / `make test` 绿 |
| 3 文档 | 本节 | 本台账(§1–13 由 15 份文档蒸馏)、教程 `docs/scalebfm.md`、6 份设计/协议文档 + 两份 handoff 进 `docs/archive/`,其余 7 份数字已全部进台账后删除 | — |

**删除的代码及裁定**(删除于 f8fc71b):crate spike 的 episode/envelope(run_spike.run_round、run_bend、render_*,§1 已记录);v3 18-d 线(run_bottle.run_round、record_bottle、eval_bfm_bottle、bfm_action、augment_bfm_velocity、`configs/gr00t_bfm_{2cam,vel_2cam,vel4_2cam}.py`、`scripts/bfm_pilot.sh`;§3 三个 pilot 0/60,checkpoint 与数据均已删)、run_bottle_v4(0/3,§3)、corner 场景(只为复现 v3/v4 数据集,§4)、run_visual_comparison(已取消 Goal 的驱动,§7)、visual_pelvis_ablation + 其测试(ABCD 数据已丢,§11)、visual_baseline 与 cohort 臂 A(运行的是 outputs/ 下的冻结快照,§13)、harness.main(v3 探针 CLI)、bottle_probe / probe_visual_mode6 / calibrate_visual_reach(一次性探针,产物已丢)、sonic_envelope + `scripts/run_sonic_envelope.sh`(SONIC 对照,与本线无关)。

**没做、留作后续**(见 docs/ROADMAP.md):network.py 757 行(vendored)与 12 个 >80 行函数的尺寸债;collect/eval/video_check 的 CLI 仍带自己的 flag(只有 round.py 完全走 config);这条线的 collect/export/train/eval 仍是自己的驱动,没有并入 `locomani/pipeline`(需要 pipeline 真正通过 Backend seam 调用);`lockstep_vendor/` 仍在 `src/teleop_free/`(SONIC lockstep 用,不属本线)。

(来源: 本仓库 commit a3e2fee / f8fc71b / 51a4d25;重放对照脚本与输出在做这次重组的 session scratchpad,未入库)

## 15. 2026-09-10 更正:§0 第 5 条与 §10 的"无法恢复"写过头(Codex 复审,docs/ledger/refactor.md)

§0 第 5 条写"学生无法从下蹲恢复",§10 也把 q99 截断当成恢复失败的机制。已知事实只有:checkpoint 的相对骨盆高度动作 q99 = +0.01836 m,单次修正最高 +1.836 cm;100 场里 38 场骨盆低于 0.60 m。**不能由此推出学生无法多步恢复**,也没有证据说明最初下沉的原因;能回答这两点的 pelvis ABCD 对照(§11)没有最终结果。正确表述:恢复幅度受单步截断限制,恢复能力与下沉成因**均未因果验证**。§0、§10 原文按 append-only 规则不改,以本节为准。

(来源: docs/ledger/refactor.md「Standards 2」;§10 本身的数字)

## 16. 2026-09-10 用户取消抓起后的回收动作

用户明确要求先取消“向身体方向收回约 7 cm”。`GraspConfig.carry_back` 默认与 `configs/tasks/bottle_visual.json` 同步从 0.07 改为 0；正式 collect 与 video_check 使用同一默认任务配置。lift→carry 后以当时右手水平位置为目标，不再主动后收；12 cm 抬手、可见高度补偿和稳定 1 s 成功判据保留。参数仍允许显式覆盖，以便复现历史协议；新默认不再等同旧 yplus200 全配置，旧数据和结果不能冒充本次变更的验证。

验证：默认/JSON 均 carry_back=0、lift_height=carry_clearance=0.12；已有 planner/motion/video_check/config 测试 21 passed，配置字段文档一致性通过。命令：`CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -m pytest tests/test_visual_planner.py tests/test_visual_motion.py tests/test_visual_video_check.py tests/test_configs.py -q -o addopts=''`。未跑 GPU/动作视频；运动质量与闭环收益尚未验证。教程流程图与生成配置参考已同步。

## 17. 2026-09-10 成功条件与 SONIC evaluation 对齐（用户要求）

专家 round 与 VLA eval 共享 `tasks/bottle_visual/success.py`：瓶子相对本轮初始高度抬升 **>0.06 m**，从第一次超过阈值起持续 **>2 s** 才成功，回落到 ≤0.06 m 重置计时。runtime.fallen 与采集 physics audit 的骨盆阈值统一为 **<0.40 m**。不再把 carry 阶段、合爪命令、倾斜 ≤30°、手速 <0.06 m/s 或恢复触发作为成功条件。碰瓶前移仍记 first_move，不再提前判失败；专家旧的走路阶段瓶高变化终止也取消，以免隐藏额外评分门。MuJoCo 异常、非有限状态、恢复执行错误和采集首帧不可见仍会拒绝无效运行。

每集 result 新增 `success_protocol=sonic_lift6cm_hold2s_v1`，避免新旧数字混用。此次只改任务评分，不改 planner 的 8 cm lift→carry 转换和 12 cm 抬手目标；上一节 carry_back=0 保留。采集动作质量 audit（侧倾、扭腰、腕连续性、可见性等）仍独立筛数据，不是 VLA success 条件。预算保留 24 s，评分每 20 ms 使用仿真时间；SONIC async 的 90 s 预算、约 0.5 s 墙钟检查未迁移，因此不宣称整个评估协议逐项相同。新旧成功率、SONIC/ScaleBFM 成功率比较需注明预算和时钟差异。

验证：阈值严格边界、回落重置、无效仿真拒绝、骨盆 0.399/0.40/0.44 m 边界新增 6 项 CPU 测试；与 planner/action 测试共 **17 passed**。`CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 make check` 退出 0，日志 `/tmp/codex-scalebfm-success-check.log`。未跑 GPU 闭环，旧 25/100 不重新标记或推算。下一步在新实验 ID 下采集/评估并检查新口径结果。

## 18. 2026-09-10 新 200 集成功数据采集与转换（已准备、排队）

用户授权按当前 ScaleBFM 逻辑采集 200 集成功数据并转换，不包含训练。新目录 `outputs/experiments/scalebfm200_noretract_sonic_20260910`，runner 源文件 `scripts/experiments/scalebfm200_20260910.py`，后台 PID 2452516。GPU0 有 SONIC visible200 任务，遵守 STATE 的资源约束：runner 每约 33 s 检查，无相关采集进程、GPU0 显存 <500 MiB、CPU 剩余 ≥6 核且无 throttling 才启动，无超时强行启动。使用 OSMesa 和单线程 CPU 配置，ScaleBFM 网络用 GPU0；不启停任何 SONIC stack。

方案：B 普通专家、yplus_center_v1、carry_back=0、sonic_lift6cm_hold2s_v1、24 s、25 Hz，seed 从 4001 递增，轮流 high_near/low_near/low_deep，最多 800 次尝试，直到 200 集同时通过任务与原动作质量审计。失败保留，不足 200 不转换、不用重复样本补数。沿用质量审计意味着抬升成功但侧倾等不合格的样本不入训练集。

准备完成：独立复制当前 scene XML/meta（无 include，引用资产均存在），冻结 locomani/gr00t 源码及 configs，保存完整文件哈希；新 runtime_assets.json 验证 103 项。为避免伪造旧 Goal 资产恢复，collect 新增 --backend / --runtime-assets 指向本次独立配置和清单。已验证冻结源码导入、carry_back=0、资产清单通过且篡改哈希会拒绝；未在 SONIC 运行期间做仿真探针。当前场景为事故后重建场景，不宣称与丢失的 yplus200 场景字节一致。

流水线依次运行 collect -> export -> gr00t.data.stats，产物是 `dataset/`（LeRobot v2.1，200 parquet、400 两相机视频、91D state/29D action、25 fps 与统计）。export 检查每集首帧可见性、审计、有限数值、时间步、视频尺寸/解码帧数。最终 `RESULTS.json` 才表示全部完成；`progress.json` 是实时阶段，`failure.json` 表示异常停止。当前只是排队，200 集结果与耗时尚未知。

命令（仅在确认原 driver 已退出后重启，文件锁防重复）：`python3 scripts/experiments/scalebfm200_20260910.py`。检查：`cat outputs/experiments/scalebfm200_noretract_sonic_20260910/progress.json`；采集中数量见 `collect/train_inputs.json` 与 `collect/summary.json`。入口/统计 CLI 已验证，thread-limited make check 通过，日志 `/tmp/codex-scalebfm200-check.log`。原 Goal 继续取消。

## 19. 2026-09-10 用户改用 GPU1：已实际开始采集

用户询问并要求可用时改在 GPU1 采集。现场 GPU1 总显存 32607 MiB、驻留占用 8977 MiB，5 次间隔 1 s 利用率均为 0%；驻留的是其他 video_grasp 的 3 个 spider_run 进程，未终止或修改。CPU 余量启动前 ≥6 核，启动后采样剩余 12.56 核、无 throttling。基于用户新指示允许不同 GPU 并行，本次保留 OSMesa/单线程和独立场景。

确认仍在资源等待且没有子进程后，停止自己的旧队列 PID 2452516。原 plan/runner 分别归档为 plan_gpu0_before_collection.json / runner_gpu0_before_collection.py（未发生任何物理尝试）；plan 的 gpu 改为 1、allow_shared_gpu=true，冻结代码只更新 runner 与其哈希，专家和场景未变。资源门改为共享 GPU 空闲显存 ≥8192 MiB、利用率 <20%、CPU 余量达标。

新 driver PID 2460752、采集子进程 2460783；nvidia-smi 确认其位于 GPU1，显存约 682 MiB。progress.json 已为 collect，seed4001_y+ 已创建 attempt 和两路录制文件；本次核查时尚未生成首集结果，不能声称已成功收集。GPU0 的 SONIC visible200 不受本任务启停操作影响。采够 200 且通过质量审计后仍自动转换、生成 stats，不训练。


## 20. 本次采集慢的诊断(2026-09-10,Codex)

用户指出之前约半小时完成200集,当前明显慢。前33集wall中位175.94s,第一集150.2s,
最新seed4033仿真13.32s/墙钟207.91s;33/33接受,并非失败重采或转换。慢在SONIC诊断启动前即存在。
确认活跃2460783环境及冻结runner强制MUJOCO_GL=osmesa、LP_NUM_THREADS=1,CUDA_VISIBLE_DEVICES=1
只将策略推理放在GPU,没有将MuJoCo渲染切到EGL。这个配置由Codex启动runner时沿用ABCD逐位
复现诊断要求,不适合作为默认高吞吐采集配置;此前ABCD记录明确旧基线为EGL。

静态同scene/head_camera320x240,3次预热+20帧计时(不运行物理/专家,不是完整episode性能对照):
OSMesa单线程RGB345.9ms/帧、seg36.5ms;EGL_DEVICE_ID=1 RGB0.314ms、seg0.755ms。
两路RGB按25Hz写入,感知还为每个相机渲染有/无遮挡mask,使软件渲染代价累计。
原始脚本及结果:outputs/diagnostics/scalebfm_speed_20260910/{render_bench.py,result.json}。
基准复跑环境:CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1,
分别MUJOCO_GL=osmesa或MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1,用.venv_sim/bin/python执行脚本。
早先EGL_DEVICE_ID=0也做了静态采样,最终报告使用明确device1结果。

结论:错误沿用软件渲染是已证实的重大性能瓶颈;微基准加速比不能直接当整集加速比。
旧数据/运行原始日志已删除,未独立复核用户所述半小时数值。未修改当前采集环境/冻结代码,
未中断任务。下一步对EGL做短episode性能与成功/可见性/图像检查,再以明确新渲染配置断点续采,
保留已有结果和provenance;不把本次采集冒充ABCD逐位复现数据。


## 21. 用户授权切换EGL并固化规则(2026-09-10,Codex)

停止旧driver2460752/child2460783,保留34完整集。seed4035中途停止保留partial与attempt,
续采按既有中断记账逻辑跳过该seed,不覆盖重采。冻结专家/仿真源码不改,只更新runner环境与plan;
旧plan/runner带时间戳归档,runner新hash入plan,rendering_history.json列出旧35个started case归OSMesa,
之后新case归EGL。collect/config原始source不改,渲染变更由顶层历史文件显式补足。

**短测完成**:同一冻结代码和Python环境,CUDA_VISIBLE_DEVICES=1,MUJOCO_GL=egl,
PYOPENGL_PLATFORM=egl,MUJOCO_EGL_DEVICE_ID=1。独立12集pilot(高近/低近/低深各4),12/12成功且
全部质量审计通过,24/24视频完整解码,抽看低深场景头相机RGB可见瓶子且无黑屏。
wall中位5.497s,范围4.82–6.46s;相较旧采集175.94s中位约32倍(不同seed,不是逐帧配对速度基准)。
证据根目录下egl_preflight/及egl_preflight_pass.json。pilot不计入200集训练配额。

**续采运行中**:driver2516911/child2518813,GPU1推理+EGL设备1;已从seed4036继续。
首3个新训练集5.74/5.63/5.11s,均入选;检查时37/200 accepted(38次attempt,其中1次切换中断)。
之后继续collect->export->stats;当前未训练。资源约束未放宽,录像规格及审计规则未降低。

**固定规则落地**:
- docs/scalebfm.md新增「正式采集必须遵循的渲染与吞吐规则」,STATE固定规则同步。
- 通用pipeline.collect默认EGL并记录render_backend/render_device;非EGL必须显式
  --allow-software-rendering用于独立诊断。已实际调用OSMesa不带开关,按预期在采集前拒绝。
- 当前200runner只允许正式EGL;首次EGL续采先自动pilot,要求12入选+24video解码+median<=60s,
  否则不放行批量采集。plan保存EGL设备;RESULTS包括rendering_history,不声称全部200集同渲染方式。
- Ruff和线程受限make check通过(exit0,/tmp/scalebfm_egl_check.log)。冻结collect仍是启动时旧版本,
  本次由runner强制EGL;新的通用入口默认/防误用检查适用于之后新建采集。

## 22. 2026-09-10 无回撤 200 集 → seed 17/3000 步 → 100 场闭环 = 41/100(旧判据下 ≤38/100)

**为什么跑**:用户要求在第 19–21 节采集的 `scalebfm200_noretract_sonic_20260910`(200 集 / 49,387 帧 /
400 视频,EGL)上训练并做 100 场闭环。目录 `outputs/experiments/scalebfm200_noretract_s17_20260910/`,
GPU1 独占,全程不占 instance/端口(这条线是进程内独立 MuJoCo,无 SONIC deploy)。

**启动前修的两个阻塞**(重构搬家 + 09-09 outputs 事故的后遗症,这条线搬家后从未跑过):

| 症状 | 原因 | 修法 |
|---|---|---|
| `pipeline/train.py` 所有 ROOT 相对路径指向不存在的目录 | 搬进 `tasks/bottle_visual/pipeline/` 后 `parents[3]` = `src/locomani` | 改用 `locomani.paths.REPO`(与 eval.py 一致) |
| 冻结的 `base_model_snapshots.json` 不存在 | 随 `scalebfm_visual_goal_20260908/` 一起被删 | 重建为 `configs/scalebfm/base_model_snapshots.json`,指向幸存 HF cache 的同两个快照(Cosmos-Reason2-2B@9ce19a1、GR00T-N1.7-3B@2fc962b),权重 sha256 记在 `_digests.json`;**是重建件,不是原件** |
| `verify_runtime_assets()` 默认清单不存在 | 同上 | `eval.py` 加 `--runtime-assets`(照抄 collect.py),指向采集目录幸存的清单;**103/103 资产哈希未变** |

场景核对:eval 默认场景 `outputs/scene/sim/` 与采集冻结副本 `scene_locomani.xml` / `scene_meta.json` **逐字节相同**。

| 变量 | 数字 |
|---|---|
| 训练 | seed 17,3000 步,GPU1,15:34–17:55 UTC,2.8 s/it;batch 4×accum 16 = 64,LR 1e-4,AdamW8bit,冻结 vision/LLM |
| 闭环 | 100 普通 reset,seeds 31001:31101,`yplus_center_v1`,`--recovery-episodes 0`,17:55–18:08 UTC(745 s 推理墙钟,推理中位 66 ms/次) |
| **结果(新判据 sonic_lift6cm_hold2s_v1)** | **41/100**:高桌近瓶 **25/34**、低桌近瓶 **11/33**、低桌深瓶 **5/33** |
| 失败 | physics warning/fall 34、timeout 25;**"闭合前碰瓶"这一类已不存在**(判据里被删掉了) |
| 其他 | 95/100 至少合过一次爪;0 场 MuJoCo 警告;0 场原地不动不合爪;可见度中位 1.00、最低 0.52 |

**判据变了,所以 41 不能直接和 §9 的 25 相减。** 提交 `2ba1a89` 同时改了两处:删掉"闭合前碰瓶 / 伸手前瓶
高度变化"两个提前中止,并把成功从「carry 阶段 rise ≥ 0.08 m、tilt ≤ 30°、|v| < 0.06,持续 1.0 s」换成
「任意阶段 rise > 0.06 m 持续 > 2.0 s」。用同一批保存的 trace 按旧判据重算(脚本
`scripts/experiments/scalebfm200_rescore_20260910.py`,结果 `RESCORE.json`):

| 打分方式 | 总数 | 高 / 低近 / 低深 |
|---|---|---|
| 新判据(本次正式数) | 41/100 | 25/34, 11/33, 5/33 |
| **旧判据重算(上界)** | **38/100** | 24/34, 10/33, 4/33 |
| 只换升举阈值、不加碰瓶中止 | 39/100 | 24/34, 11/33, 4/33 |
| §9 参照(旧判据) | 25/100 | 10/34, 10/33, 5/33 |

旧判据重算是**上界**:object_audit 没存瓶速,`|v| < 0.06` 这一项没法复现,而它只会让数变小。
100 场里 24 场"合爪前碰到瓶",但其中只有 1 场最终算成功——**删掉碰瓶中止几乎没有送分**,
16 分的差距里判据只占 3 分。

**裁定**:同 seed、同步数、同 100 场池,像素对像素地比,这个臂比 §9 高约 **13 分,几乎全部来自高桌近瓶
(10/34 → 24–25/34)**;低桌近瓶(10 → 10–11)和低桌深瓶(5 → 4–5)**没动**。Fisher 精确检验双侧:
41 vs 25 p = 0.024,像素对齐的 38 vs 25 p = 0.067。§9 的每场结果已在事故中删除,所以只能做非配对比较;
§11 也早就写过旧 EGL 基线不可逐比特复现。因此这是"高桌那一层明显变好、深瓶完全没变"的一次跨实验观测,
不是一个配对显著结果。

**视频**:`outputs/videos/scalebfm200_s17_20260910/reel_10.mp4`(8 场,三层 × 成功/超时/摔倒,130 s)。
顺带修了 §10 记录但一直没修的导出变暗 bug:`follow_video.py` 对新 MjData 只调 `mj_kinematics`,
现补 `mj_camlight`(本片中位亮度 90.2)。`export.py` 没有这个调用点,数据集视频不受影响。

**开放问题**:低桌深瓶 5/33 与 §9 完全一致,两次训练、两套数据、两个判据都没动过它——深瓶层的失败
不像是数据量或判据问题;34 场 physics warning/fall 里 21 场在低深层,与 §10"屈膝持续下沉"的描述一致,
但本次没做骨盆干预对照(ABCD 在事故中丢失)。
