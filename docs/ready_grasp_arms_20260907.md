# ready-grasp 采集协议的可学性实验(2026-09-07/08)

六个模型、同一评测协议、逐局配对。问题是:codex 重构后的 ready-grasp 专家(07)训出来的 VLA 为什么只有 30%,
采集侧改什么才能让**策略**(不是专家)变好。结论:**站近 5 cm + 去掉 shuffle 段**(07d)把闭环从 30% 提到 48%
(配对 McNemar p = 0.052),机制已验证;"没蹲"这一失败类别是训练 seed 方差,不是数据问题。

代码:`8c621fb` 为准,加上本轮之后未提交的四处改动(见 §7)。所有产物在
`/workspace/code/locomanipulation/outputs/experiments/ready07*_2026090[78]/`。

## 1. 起点:creep 的隐变量

cover400(16/100)对 flow200(50/100)的失败 77% 是"到桌边、90 s 内从不闭爪"。离线所有幅值指标正常,
唯一退化的是**时序**指标 `close_frame_err_abs`(10.25 vs 3.85 帧)。原因:STAND_COVER 每局随机抽一个站位格子,
creep 的停止判据是"当前位置 ≈ 抽中的格子"。格子不在相机里,于是同一物理位姿、同一画面,label 一半是"停下闭爪"
一半是"继续挪"。BC 回归到条件均值——半开夹爪慢慢蹭——闭环里没有任何观测能跨过那个不存在的阈值,成为吸收态。
creep 只占 2.8% 的帧,但坐在 phase 闸门上。

原则:**专家可以用特权信息计算动作,但动作必须是学生能观测到的量的函数。** 修法是把停止判据换成
深度带 [0.18, 0.27] m + 方位锥 ±36.5°(都从相对位姿算),验证后 ready07 的闭爪时序 1.83 帧、87 局里 1 局"立着不闭"。
详见 memory `creep-hidden-stop-target`。

## 2. 实验设计

| 项 | 设定 |
|---|---|
| 场景 | 冻结场景 `outputs/diag/grasp_mode_coverage_20260906/scene`,`SCENE_RANDOM=0` |
| 采集 | 每臂 60 集成功示范,spawn 用闭环评测分布(edge 1.1–1.3 m、yaw 20°、lateral 0.20、jitter 0.20),每 10 局重启 stack |
| 转换 | 25 Hz,`--trim-verification-tail`,holdout 6 → 54 train / 6 val,~22k 帧 |
| 训练 | GR00T-N1.7-3B slot 11,3000 步,batch 8×8,单卡,单 checkpoint |
| 评测 | `eval_closed_loop.py --fresh 100 --seed 6000 --client async --replan-interval 0.4 --budget 90`,ready token 初始化,**60 局自动停**,第 i 局各臂出生条件相同 → paired McNemar |
| 离线 | `eval_open_loop.py` on val,h=16 的 `close_frame_err_abs` |
| 一键 | `outputs/experiments/run_learn_pilot.py`(采→转→训→评);`summarize_arms.py` 出总表;`compare_arms.py` 配对 |

## 3. 六臂结果(配对,前 60 局)

| 臂 | 相对 07 的变量 | 专家 yield | 成功 | 不一致对 | p | 闭爪时序(帧)| 失败分类 |
|---|---|---|---|---|---|---|---|
| 07 | 基线(advance 0.03,有 shuffle)| 60/89 | 18/60 = 30% | — | — | 1.83 | far 16 · 扫倒 24 · 掉 1 |
| 07b | 站近 5 cm(advance 0.08)| 60/93 | 26/59 = 44% | 10 : 18 | 0.185 | 3.50 | 没蹲 15 · far 6 · 扫倒 8 · 掉 4 |
| 07c | 07 站位 + pregrasp 横向 jitter ±3 cm | 60/87 | 20/60 = 33% | 14 : 16 | 0.86 | 1.83 | far 15 · 扫倒 11 · 没蹲 8 · 掉 4 |
| **07d** | **07b 站位 + 去 shuffle**(`GYM_SHUFFLE_BAND=0`)| **60/78** | **29/60 = 48%** | **8 : 19** | **0.052** | **1.17** | 没蹲 13 · 扫倒 11 · 掉 4 · far 3 |
| s2 | 07 数据,训练 seed 42→7 | (同 07) | 18/60 = 30% | 13 : 13 | 1.00 | 3.33 | 扫倒 24 · far 11 · 没蹲 6 |
| 07e | 07d + 停靠带放宽(advance 0.04 ± 0.04)| 60/90 | 19/60 = 32% | 11 : 12 | 1.00 | 3.50 | 没蹲 18 · 扫倒 17 · far 3 |

07e vs 07d:9 : 19,p = 0.087。各臂训练 loss 0.068–0.078,数据量 21.7–22.6k 帧,推理延迟中位 70 ms(三臂核对)。

失败类别定义(`compare_arms.py`):far = 蹲下时骨盆离瓶 > 0.43 m 手够不着;扫倒 = 伸手时瓶子倒(按有无采到手指接触分两类);
没蹲 = 到桌边始终未下蹲;掉 = 闭爪后倒/掉。

## 4. 机制:策略的停位由"蹭"决定,不由专家停位决定

走近段基座速度中位(m/s)按骨盆→瓶距离分段:

| 距离段 | 07 专家 | 07 策略 | 07b 专家 | 07b 策略 |
|---|---|---|---|---|
| 0.60–0.70 | 0.26 | 0.27 | 0.26 | 0.22 |
| 0.45–0.50 | 0.30 | 0.07 | 0.04 | 0.003 |
| 0.40–0.45 | 0.41 | 0.01 | 0.08 | 0.02 |
| 0.35–0.40 | 0.04 | 0.04 | 0.22 | 0.007 |

专家在 0.40–0.50 还在 0.3–0.4 m/s 走、最后 5 cm 刹车;策略从 0.5 m 起就以 1–7 cm/s 蹭——这是 APPR_SHUFFLE
(0.20 → 1–2 cm/步)被平均后的产物,和 dart200 的"桌前发呆"同源。结果:专家 first_stop 中位挪 4 cm(0.364 → 0.324),
策略最近停靠只挪 7 mm(0.347 → 0.340)。去掉 shuffle(07d)后策略停位才真的挪进来(0.348 → 0.310),far 16 → 3。

各模型最近停靠中位 / P(蹲):07 0.348 / 1.00;07b 0.339 / 0.75;07c 0.334 / 0.87;07d 0.310 / 0.78;s2 0.352 / 0.90。
成功率随停靠距离单调下降(07:[0.25,0.30) 67%,[0.34,0.38) 18%,≥0.38 0%),专家自己也是([0.30,0.34) 89% → [0.37,0.43) 69%)。

## 5. 方差底线与"没蹲"

s2 只换训练 seed:总成功率纹丝不动(18/60 vs 18/60),但类别构成 far 16→11、没蹲 0→6、闭爪时序 1.83→3.33。
因此 ≤6 局的类别差和 ±1.5 帧的时序差是噪声;站得住的效应:站近把扫倒 24→8、jitter 把扫倒 24→11、去 shuffle 把 far 16→3、
07d 的 +18 点。

"没蹲"的三个数据侧解释全部被否定:窄带覆盖(07e 放宽后反而 18 局)、放宽覆盖(8 局里 4 局停在带内仍不蹲)、刹车静止隐门
(四臂训练数据里到桌→蹲的静止帧结构相同:最长静止段 0.60–0.66 s,静止帧 29–39%)。按 seed 对照,当作训练方差处理。

## 6. 结论与下一步

1. **最佳采集协议 = 07d**:`arrival_advance 0.08`、`recovery_timeout 15`、`GYM_SHUFFLE_BAND=0`、评测分布 spawn。
   两个成分都必要:只站近(07b)44%,站近 + 放宽(07e)32%,站近 + 去 shuffle(07d)48%。
2. 横向 jitter(07c)单独用没有总收益,但它把扫倒减半,可以叠在 07d 上作下一个单变量。
3. 下一杠杆是**数据量与 seed 数**:用 07d 协议采 200 集,每配置训两个 seed;不要再调站位带。
4. 几何上"从上方下压"不可行:dex1 指长 18.5 cm,近站位下 pregrasp 抬高 14 cm 会把指尖压在瓶顶(烟测 3/10),codex 的水平进爪是对的。

## 7. 代码改动与运行记录

已提交(`8c621fb`):`reach_config.py` 新字段(pregrasp_rise/gap/lateral_jitter/min_forward、local_reach_limit,
arrival_advance 上限 0.10);`coupled_grasp_policy.py`/`measured_grasp.py` 消费这些字段(默认 = 07 行为);
`grasp_gym.py` `GYM_SHUFFLE_BAND`;`tools/run_learn_pilot.py`;`convert_to_hf_format.py` `left_gripper`。
**未提交**(本文写作时):`reach_config.py` `arrival_advance_jitter`;`grasp_gym.py` 每局 advance 抽样;
`run_learn_pilot.py` `SQUAT_MAX` 默认 4096;`src/gr00t/experiment/launch_finetune.py` `GR00T_DATA_SEED`。

运行中踩过、已修的坑(细节在 memory):runner 用 `max(mtime)` 选 exporter session 目录在 teardown 变慢后错位一格
(改为等待新目录);批间 `stop_stack.sh` 全局 pkill 杀了别的 instance(改 `stop_instance.sh 0`);instance 1 的 keyboard
端口 5590 与 `serve_policy` 撞(烟测改 instance 3);`/workspace` 写满(单 checkpoint);其他 instance 的 MuJoCo EGL 上下文
落在 GPU0 触发 2048 MiB 显存守卫(`SQUAT_MAX=4096`);`pgrep -f` 路径片段在等待循环里自匹配死锁(改 `.done` 标记)。

保留的权重:`ready07_learn_20260907/model`(基线)与 `ready07d_decisive_20260907/model`(最佳),各 24 G;
07b/07c/07e/s2 的权重已删,其评测 trace、`paired_compare.txt`、`open_loop_val.json` 保留。

## 8. 07d 协议 × 200 集(2026-09-08 追加)

`outputs/experiments/ready07d_200_20260908/`:同一 reach_config、scene、评测分布 spawn、`GYM_SHUFFLE_BAND=0`,seed0 28000,
28 批 281 轮接受 200 集;其中 3 个 session(10:09-10:40 UTC)因同卡跑 ScaleBFM 仿真导致 deploy 自愈重启、exporter
episode 被切成 1-6 s 残段(convert 的顺序对齐看不出来,audit 的 grasp-causality 抓到),剔除后 **176 集训练、10 集 val**,
3000 steps,100 轮闭环(seed 6000,与 07d decisive 逐轮配对)。

| 臂 | 训练集 | 闭环 | 前 60 轮配对 | discordant(新赢 : 旧赢) | McNemar p |
|---|---|---|---|---|---|
| 07d decisive | 60/78 | 29/60 = 48% | 29/60 | | |
| **07d-200** | 176 | **57/100 = 57%** | **37/60 = 62%** | 22 : 14 | 0.24 |

结论:多 3 倍数据配对上涨 8 个点(前 60 轮),但 p = 0.24 仍不显著——60 轮配对分辨不了 10 个点以内的差距(memory:
`vla-baseline-round1-results`),要 2 个 seed 或 200 轮才能定。失败结构变了:07d-200 的 reach 失败停得更近(min dist
中位 0.24 m,decisive 0.31-0.42 m)但"到了桌边不合爪"仍是最大类(34/43),合爪没抬 9/43。第二个 seed 的训练是下一步。
