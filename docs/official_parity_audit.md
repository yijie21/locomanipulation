# 官方 tutorials 逐步对照审计(2026-08-28)

起因:用户从 `vla_inference.html` 的 *Customizing the Initial Pose* 找到了 stand-start
drift 的正解,由此怀疑"官方已经设好、我们自己另造或漏掉"的细节还有更多。本次把
**全部 10 篇 tutorial + 1 篇 user guide** 与仓库代码逐步对照。

**证据标准**:每条结论双侧引证 —— 官方侧引教程原文或 `third_party/` 源码,我们侧引
`file:line` 或实测值。凭记忆的结论一律不写(本次就推翻了两条我自己的记忆)。

**分类**:
- **A** 自研,但官方有标准做法
- **B** 纯遗漏(官方有,我们没有对应物)
- **C** 有意偏离(有理由,理由钉在此处防止将来被当疏漏"修掉")
- **D** 版本漂移

覆盖范围:`data_collection` / `vla_workflow` / `vla_inference` / `zmq` / `manager` /
`keyboard` / `gamepad` / `isaac_teleop_publisher_setup` / `vr_wholebody_teleop` /
`live_camera_teleop` / `user_guide/teleoperation`。提取件存于 `/workspace/tmp/audit/`。

---

## 结论速览

| # | 类 | 条目 | 建议 |
|---|---|---|---|
| A1 | A | 初始位姿保持(已于本次采纳) | ✅ 已做 |
| A2 | A | `vla_utils.py` 的延迟补偿/触发判据被我们重写,且有半帧偏置 | 建议采纳官方函数 |
| A3 | A | 官方 `process_dataset.py` vs 我们的转换器 | 维持(我们做得更多) |
| A4 | A | 官方 `run_gr00t_server.py` vs 我们的 `serve_policy.py` | 维持(已交叉验证) |
| B1 | B | `--color-jitter-params` 未传 | **需实验裁决** |
| B2 | B | `--max-steps` 10000 vs 官方 20000 | 已在 ROADMAP |
| B3 | B | SONIC `sonic_v1_1` / `low_latency` checkpoint 未采用 | **需实验裁决(见下)** |
| C1 | C | 脚本专家替代 VR 遥操作 | 维持(项目核心命题) |
| C2 | C | 30 Hz 发布 vs 官方 50 Hz | 维持(自洽),但见 C5 |
| C3 | C | 65 维 action vs 官方 78 维 | 维持(线上扩展回 78) |
| C4 | C | 2 相机 + 增广 state vs 官方 1 相机 | 维持 |
| C5 | C | `NO_ENCODER=1` —— **我们注释里写的机制不成立** | **需实验裁决** |
| C6 | C | 保留重复帧 | 维持(**官方明文背书**) |
| C7 | C | cadence(削减停顿) | 维持(**官方明文背书**) |
| C8 | C | `--set-compliance` / compliance 惰性 | 维持(**已定性证明惰性**) |
| C9 | C | 四元数序 wxyz | 维持(**已逐环节证明一致**) |
| C10 | C | 29-DoF 关节序换算表 | 维持(**与官方常量逐元素相等**) |
| D1 | D | third_party 落后上游 8 个提交 | 无需升级(纯文档) |
| E1 | — | 官方**没有** VLA 训练超参指引可对照 | 记录空白 |

---

## A 类:自研但官方有标准做法

### A1 初始位姿保持 —— 已采纳

官方 `vla_inference.html`:按 `i` 键 → 切 C++ 控制环到 **POSE 模式** + 把机器人
从当前构型平滑 blend 到预置的 64 维 latent token
(`gear_sonic/utils/inference/initial_poses.py:LATENT_INITIAL_MOTION_TOKEN`),
`--initial-pose-blend-duration` 默认 `1.0` s。

我们此前在 planner IDLE 通道里干等推理,踩出 stand-start drift,并一度用 sim 特权
二次 teleport 绕开。本次已改为官方原语(`vla_round.py`,commit `bbc207d`):局首即灌
预置站姿 token + `enter_streamed()`,keepalive 线程持有,1 s 线性 blend 交接。

对照确认:我们的 `enter_streamed()`(`agent_server.py:306`)发的是
`command(start=True, planner=False)`,与官方 `send_cpp_control_command(start=True,
planner=False)` 逐字等价 —— 命名不同(我们叫 STREAMED,官方叫 POSE),**指令相同**。

### A2 延迟补偿与推理触发:官方有工具函数,我们重写了,且有半帧偏置

官方 `gear_sonic/utils/inference/vla_utils.py` 提供:

```python
def calculate_latency_compensated_index(inference_delay, control_freq, action_horizon):
    raw_index = np.round(inference_delay * control_freq)      # 四舍五入
    return int(np.clip(raw_index, 0, action_horizon - 1))

def should_trigger_new_inference(cached_chunk_exists, inference_thread_running,
                                 time_since_last_inference, inference_interval): ...
```

我们的等价实现(`vla_round.py:566` 附近):

```python
idx = min(int((time.time() - chunk_t0) * RATE), len(tok) - 1)   # 截断
```

**差异是真的**:官方 `np.round`,我们 `int()` 截断 —— 系统性落后半帧,30 Hz 下约
**17 ms** 的恒定滞后。量级小,但它是免费可修的,且方向单一(永远滞后,不会超前)。
`vla_utils` 零引用(`grep -rn vla_utils src/` 无命中)。

**已采纳(2026-08-28,不等配对 A/B)**。`vla_round.py:69` 直接 import 官方
`calculate_latency_compensated_index`;集成路径用不带 clamp 的 `int(round(...))`,
因为官方那个会把耗尽的 chunk 夹回最后一帧、拖进混合里。

**为什么这条可以豁免"配置类改动必须过全预算配对"的规矩**(见
[ROADMAP](ROADMAP.md)):那条规矩防的是**符号未知**的超参——50 轮 A/B 把 FSQ 不吸附
报成 +12 个百分点、全预算配对却是 p=0.755,就是这类。A2 不是超参搜索,是**符号已知
的偏置修正**:

* 它每个 tick 都在把一串定义在 `i/rate` 上的离散动作重采样到 wall-clock 上,取整方式
  决定的是**相位**。`floor` 的误差落在 [0, T),均值 **+T/2 恒为滞后**;`round` 落在
  [−T/2, +T/2],**均值 0**。峰值误差一样,但 `floor` 多出一个纯延迟的直流项 —— 而纯
  延迟吃闭环相位裕度,零均值抖动被下游(50 Hz decoder + 本体)低通掉。
* 这个量存在的**目的**就是抵消推理延迟,选一个系统性再加回半帧延迟的估计量自相矛盾。
* 训练标签本身是 `convert_to_hf_format.py:632` 用 `np.round` 采出来的,**最近邻语义**。
  部署端用 `floor` 会让部署相位和训练相位系统性差半帧,是可免费消除的 train/deploy 不一致。
* 反方唯一的理由是因果性(`round` 会提前最多半帧发出属于未来 tick 的指令),但 action
  chunk 本来就是模型已经承诺的**未来规划**,不是对未观测量的反应,所以不成立。

方向由理论定死,A/B 只能测量级 —— 而量级是 ~17 ms,100 轮配对分辨不出来。**跑它得不到
信息**,所以不跑。

### A3 数据集后处理:官方 `process_dataset.py`,我们自研转换器

官方 `gear_sonic/scripts/process_dataset.py` 做三件事:剔除 `discarded_episode_indices`
标记的集、清理 stale SMPL 帧、合并多个 session。`vla_workflow.html` 明确要求:

> "After collection, run the processing script to remove discarded episodes…
> This ensures only successful demonstrations are used for fine-tuning."

我们零引用它,用自研的 `tools/convert_to_hf_format.py`。**但对照后结论是维持**:我们的
转换器覆盖官方全部能力(剔除 discarded、合并 session)并额外做:50→30 fps 重采样、
视频帧精确重编码、以及**与 grasp_gym manifest 的交叉校验**(`convert_to_hf_format.py:91`
—— 逐集比对"盘上是否 discarded"与"manifest 里 ok 是否为 False",不一致直接报错)。
最后这条官方没有,是更强的保证。

### A4 policy server:官方 `run_gr00t_server.py`,我们 `serve_policy.py`

官方 `vla_workflow.html` 用 `gr00t/eval/run_gr00t_server.py --port 5550`;我们用
71 行的 `tools/serve_policy.py`(端口 5590)。已在 `scripts/crosscheck_open_loop.sh`
里与官方 `open_loop_eval.py` 做过排名一致性验证(Spearman ρ=+0.988,best checkpoint
一致)。**维持**。

---

## B 类:遗漏

### B1 `--color-jitter-params` 未传 —— 需实验裁决

官方 `vla_workflow.html` 的微调命令明确带:

```
--color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

我们的 `scripts/train_bottle.sh:207-234` 不传此参数。

**但要点在这里**(这条差点被写成"我们完全没有增强"):
`third_party/Isaac-GR00T/gr00t/configs/finetune_config.py:70` 的默认是 `None`,而
docstring 写:

> "If None, applying the **default color jitter augmentation from the pretrained model**."

所以不是"零增强",而是"用预训练默认值 vs 教程指定值"。教程的值是否比预训练默认更适合
我们这个单场景任务,无法从文档判定。

**建议**:排一次配对 A/B(同数据同步数,只改这一个参数)。我们的场景已有 appearance
随机化,增强收益可能被摊薄;但视觉塔冻结的情况下,jitter 仍会改变喂给 projector 的
特征分布,值得一测。

### B2 `--max-steps` 10000 vs 官方 20000

官方教程用 `--max-steps 20000`、`--save-steps 5000`,并在部署示例里引用
`checkpoint-20000`。我们默认 `MAX_STEPS=10000`(`train_bottle.sh:152`)。

round 4 的开环曲线在 10000 步仍在下降(0.1429→0.1424→0.1394),这条本来就在 ROADMAP
里。官方默认值为它提供了外部佐证。

### B3 SONIC checkpoint:`sonic_v1_1` / `low_latency` 未采用 —— 需实验裁决

官方提供三个控制器 checkpoint:

| checkpoint | 特性(教程原文) |
|---|---|
| `policy/release` | 默认,10 帧参考视界(~200 ms lookahead) |
| `policy/sonic_v1_1` | **robot-heading-normalized**,wrist-pose 增强训练 |
| `policy/low_latency` | 4 帧参考 @50 Hz(~80 ms lookahead) |

我们用 `policy/release`(`restart_deploy.sh:26`)。本地已下载 `sonic_v1_1`。

**这条与我们花了一整天的 drift 直接相关**。`start_stack.sh:104-134` 记录了 v1.1 的
实测:IDLE hold 漂移 **0.004 m**,而 release 是 **0.03-0.12 m** —— v1.1 几乎消灭了
本次修复对抗的那个病。当时不采用的理由是抓取成功率 2/6 vs 4/6,而注释自己写了
"both samples far too small to separate"。**我们现在有了配对序贯 A/B harness
(`seq_ab.sh`),那个 6 局判决应该重做。**

**但有一个决定性的前置约束**,教程原文:

> "Use the `sonic_v1_1/` checkpoint **when the VLA policy was trained against the
> robot-heading-normalized SONIC controller.**"

我们的 200-250 集训练数据全部是在 `policy/release` 下采的。换 v1.1 只换评测端会造成
**训练/部署控制器不匹配** —— token→动作的映射变了。所以正确的实验是:要么
(a) 只做诊断性对照,不进梯子;要么 (b) 用 v1.1 重采数据 + 重训,成本是整条流水线重跑。
考虑到用户已决定不再重采集重训,**建议记录待办,暂不动**。

`low_latency` 同理,且它面向遥操作响应性,与 VLA token 流关系更弱。

---

## C 类:有意偏离(理由已钉)

### C1 脚本专家替代 VR 遥操作 —— 项目核心命题

官方全部数据采集路径都是人类遥操作(VR / 手柄 / 相机)。我们用脚本专家 + grasp_gym。
这不是疏漏,是研究命题本身(见 `docs/scripted_expert_innovation_discussion.md`)。
5 篇遥操作教程逐篇扫过,**没有任何数据格式或 schema 层面的要求是我们缺失的** ——
它们规定的是操作者行为,不是数据结构。唯一有实质内容的是 C7。

### C2 30 Hz 发布 vs 官方 50 Hz

官方 `run_vla_inference.py:72` `action_publish_rate: int = 50`;我们
`vla_round.py:505` `RATE = 30.0`。理由(commit `b7f220f`):我们的数据集是 30 fps
(转换器 `SRC_FPS, DST_FPS = 50, 30`),官方两侧都是 50,各自自洽,混用会让动作回放变速。

推理频率两侧一致:官方 `rate: float = 1/0.4`(2.5 Hz),我们 `inference_interval=0.4`。
chunk 长度两侧一致:官方 `action_horizon: int = 40`,我们 `delta_indices=list(range(40))`。

**补充事实**(本次 seq 序号落地后测得):parquet 行是按 **50 Hz 墙钟**滴答的,但其中
38-40% 是 CONFLATE 复用的重复内容行,sim 只以 ~0.65× 墙钟推进 —— 所以"50 fps 数据集"
里独立 sim 帧只有 ~33 Hz。我们的 50→30 重采样因此比字面看起来更接近"每帧都是新的"。

### C3 65 维 action vs 官方 78 维

官方 `embodiment_configs.py` 的 `unitree_g1_sonic`:
`["motion_token", "left_hand_joints", "right_hand_joints"]` = 64+7+7 = 78。
我们 `configs/gr00t_sonic_2cam.py:107`:`["motion_token", "right_gripper"]` = 64+1 = 65,
在发送时由 `expand_hands()` 扩展回协议 v4 的 7+7。理由:数据集里两只手的 jaw 分量
恒等,左手恒定,学 1 个标量比学 14 个恒等维度更省。**维持**。

### C4 2 相机 + 增广 state vs 官方 1 相机

| | 官方 | 我们 |
|---|---|---|
| video | `["ego_view"]` | `["ego_view", "right_wrist"]` |
| state | 8 组(46 维) | 10 组(53 维),加 `right_wrist_pos` / `right_wrist_abs_quat` |

相机分辨率两侧一致(480×640,`base_sim.py:56` 的官方默认 = 我们
`run_sim_locomani.py:154`)。腕相机是官方 exporter 支持的选项
(`--record-wrist-cameras`)。config 采用 `deepcopy(MODALITY_CONFIGS[_TAG])` 再覆盖,
language 段原样继承 —— 继承方式正确。**维持**。

### C5 `NO_ENCODER=1` —— 我们注释里的机制不成立

我们的 `start_stack.sh:137` 注释称:带 encoder 时 `initial_encoder_mode_` 是 0,
deploy 的 token 路径"keeps flipping back so an external token stream never holds"
(引 `g1_deploy_onnx_ref.cpp:3096`)。官方 VLA 流程则是**带 encoder 跑**
(`deploy.sh` 恒传 `--encoder-file`)。

**读源码后,我们的机制描述站不住**:

```cpp
// g1_deploy_onnx_ref.cpp:3022  收到外部 token 就永久关掉 encoder
if (has_ext_token_data) { ...; is_using_encoder_ = false; ... }
// :3054  只有"本帧没有外部 token 且 encoder 已关"才会翻回去
else if (!has_ext_token_data && !is_using_encoder_) {
    if (initial_encoder_mode_ == -1) { /* 等外部 token */ }
    else { is_using_encoder_ = true; }        // <- 我们担心的那一行
}
```

而 `has_ext_token_data` 来自 `input_interface.hpp:149` 的 `GetExternalTokenState()`,
底层是 **`DataBuffer` 锁存**(`utils.hpp:59`,保存最后一次值),`has_external_token_state_`
只在三种情况下被置回 false(`zmq_endpoint_interface.hpp:253/291/341`):协议错误退出
流模式、interface manager 的 safety reset、切回预载参考动作。**普通的帧间隔不会。**

所以第一个 v4 token 到达后,encoder 会被永久关闭,翻回去的分支不可达。我们观察到的
"翻回去"更可能发生在**首个 token 到达之前**(我们每局重启 deploy)或某次 safety reset。

**这不是说 NO_ENCODER 有害** —— 它当前工作正常,而且省掉一份 TensorRT engine。但
"必须如此"的论证是错的,注释应该改。**可选实验**:带 encoder 跑一局闭环,看 token 流
是否稳住;若稳,则与官方部署形态更一致(真机上少一个偏离项)。

### C6 保留重复帧 —— 官方明文背书

我们保留了 parquet 里 38-40% 的重复内容行。官方 `process_dataset.py:65` 的
`build_stale_mask` docstring:

> "Frozen runs that do **NOT** lead into a zero row are **left untouched** — those
> occur naturally when the SMPL stream publishes at a slightly lower rate than the
> collection loop."

即:速率不匹配造成的 frozen run 是正常现象,官方明确不清理。我们的重复帧同因(privileged
流 CONFLATE 复用),处置一致。**这条从"待查隐患"降级为"已背书的正常现象"。**

### C7 cadence(削减专家停顿)—— 官方明文背书

这是本次审计最有价值的一条**正向验证**。`user_guide/teleoperation.html` 原文:

> **"Try to be as natural as possible.** The GEAR-SONIC policy is trained on natural
> human motion, so moving naturally gives the best results. **Hesitating when moving
> actually lead to more stumbling!**"
>
> - ❌ **Overtly slow movements** — Don't slow down too much!
> - ✅ **Use moderate speeds** — start with walking at a comfortable pace.

我们在 round 3 靠实验独立发现了同一件事:把脚本专家的停顿削掉(到位→伸手中位
4.6 s → 3.0 s),同样 200 集把成功率从 13.7% 翻到 28.0%。**官方把它写在一个我们从没
读过的页面上。**

**前向含义**:官方的指引比我们做到的更宽 —— 不只是"别停顿",还有"整体动作要自然、
速度适中、别刻意迁就机器人"。我们的脚本专家目前只对"停顿"做过优化;运动剖面的其余
部分(加减速、转身方式、伸手轨迹形状)是否"自然",从未被检查过。这是一条尚未开采的
改进方向,且有官方文档背书。

---

## D 类:版本漂移 —— 良性

我们的 `third_party/GR00T-WholeBodyControl` 停在 `1983e88`(2026-08-06),上游已多出
**8 个提交**,其中包含一个标签为 `v1.1` 的提交。逐文件核对
(`git diff --stat HEAD origin/main`):

```
README.md | 14 +-          docs/source/tutorials/live_camera_teleop.md | 423 +++
docs/source/_static/*.gif/mp4 | Bin   docs/source/model_card.md | 15 +-
docs/source/getting_started/download_models.md | 2 +-
docs/source/index.rst | 3 +-  docs/source/tutorials/vr_wholebody_teleop.md | 4 +-
gear_sonic/pyproject.toml | 2 +
```

**全部是文档与静态资源,零代码变更**(`gear_sonic/scripts/`、`gear_sonic/utils/` 无
任何 diff),唯一的功能性改动是 `pyproject.toml` 声明了两个 training extra 依赖。

结论:本次读的教程**确实描述我们这份 checkout 的代码**,审计结论不受版本漂移影响;
也没有升级的必要(升级会作废全部已测基线,收益为零)。

### C8 compliance 惰性 —— 已定性证明,不是错过的旋钮

`references/deployment_code.html` 记载 deploy 有 `--set-compliance`(默认
`0.5,0.5,0.0` = 左腕/右腕/头,0.01 刚性、0.5 柔顺,运行时 `g/h`、`b/v` ±0.1)。
我们从没用过它,而我们的主导失败恰好是伸手精度(81/88 mid-reach),所以这条一度像是
"错过的旋钮"。

**核实后是惰性的,证据是决定性的**:C++ 在 `g1_deploy_onnx_ref.cpp:1890-1911` 显式
检查观测列表里是否有 `vr_3point_compliance`,注释写明"This affects whether keyboard
compliance controls (g/h/b/v) have any effect";而
`policy/release/observation_config.yaml` 与 `policy/sonic_v1_1/observation_config.yaml`
**都不含该观测项**(grep 零命中)。故 `has_vr_3point_compliance_obs_` 恒 false,
CLI 与按键均无效。

这与既有结论[[sonic-reach-envelope]]("compliance 和 height 字段是惰性的")一致,并
把理由从"planner 消息字段被忽略"升级为"checkpoint 的观测空间里根本没有这一项"。
**换 checkpoint 也不会解锁它** —— v1.1 同样没有。

### C9 四元数序 —— 逐环节证明一致(wxyz)

我们代码全线假设 scalar-first(wxyz):`agent_server.py:27/323`、
`convert_to_hf_format.py:181/186`、`vla_round.py:244`。逐环节对照官方权威实现:

| 环节 | 官方实现 | 序 |
|---|---|---|
| `observation.root_orientation` | `transforms.py:91` `R.from_quat(base_quat, scalar_first=True)` | **wxyz** |
| `observation.eef_state` | `run_data_exporter.py:701` `as_quat(scalar_first=True)` | **wxyz** |
| 我们的 `yaw_of_quat_wxyz` / `quat_mul_wxyz` | — | **wxyz** ✓ |

审计过程中我曾从 parquet 数据看到"xyzw 解释的 yaw 曲线略平滑",一度怀疑序搞反;
上表的权威代码证明那是噪声。**这条记录下来正是为了防止将来有人重复这个误判。**

### C10 关节序换算 —— 与官方常量逐元素相等

`references/conventions.html` 规定:motion PKL 的 `dof`/`pose_aa` 是 **MuJoCo order**,
Isaac Lab 仿真用 **IsaacLab order**,两者由 `G1_ISAACLAB_TO_MUJOCO_DOF` /
`G1_MUJOCO_TO_ISAACLAB_DOF` 互换(页面本身不列索引表)。

我们两侧都在用这个换算(`agent_server.pose_chunk` 收 IsaacLab 序、
`privileged_pub` 按 `MUJOCO_JOINT_NAMES` 读 MuJoCo 序),而 `g1_joints.py:39/44`
的两张表是**我们自己写的**。关节序错位属于最凶的一类静默失败(数值全在合理范围,
动作却是别人的),所以做了逐元素比对:

```
src/teleop_free/g1_joints.py     vs  gear_sonic/envs/manager_env/robots/g1.py:61/93
ISAACLAB_TO_MUJOCO == 官方 G1_ISAACLAB_TO_MUJOCO_DOF   -> True (29/29)
MUJOCO_TO_ISAACLAB == 官方 G1_MUJOCO_TO_ISAACLAB_DOF   -> True (29/29)
```

**完全一致**。另注:`g1_joints.py:48` 本来就有一条自洽断言
(`MUJOCO_TO_ISAACLAB[ISAACLAB_TO_MUJOCO] == arange(29)`),现在再加一条与官方的
外部一致性证明。

其余 conventions 页要点,逐条核对均一致或不适用:
- **quaternion**:SONIC/Isaac Lab/**MuJoCo** = wxyz;SciPy/ROS = xyzw。原文加粗
  "**Always use `w_last=False`**"。我们全线 wxyz(见 C9),且不调用带 `w_last` 的
  Isaac 工具函数。
- **坐标系**:Isaac Lab 与 MuJoCo 同为 **Z-up、右手系、X 前 / Y 左**,重力 -Z。
  SMPL/BVH 是 Y-up(由 `smpl_y_up` 自动转)。我们只在 MuJoCo 侧工作,不涉及 SMPL。
- **观测命名** `{base}_{N}frame_step{S}`,S 以 **50 Hz** control tick 计 ——
  与 C2/deployment 页的 50 Hz 控制环互证。
- `observation_config.yaml` 只有 `observations`(按列出顺序拼接)与可选 `encoder`
  (`dimension` / `use_fp16` / `encoder_observations` / `encoder_modes`)。**schema 里
  没有 normalization / clipping / scaling 键**,也没有 `encode_mode` 标识符 ——
  后者只存在于 C++ 侧(见 C5),文档不覆盖。

---

## E 类:官方文档的空白(不是我们的遗漏,但影响可对照性)

审计中额外抓了 `user_guide/training_data.html`、`user_guide/training.html`、
`user_guide/troubleshooting.html`、`references/deployment_code.html`,结论是它们
**都不覆盖 VLA 这一侧**:

- `training_data.html` 讲的是 **BONES-SEED 动捕数据集**(142,220 motions @120 fps,
  用于训练 SONIC 控制器本身),与 LeRobot episode 数据集无关;整页无 admonition,
  且上游文档在 "After downloading, extract the motion archives:" 处被截断,冒号后
  没有内容。
- `training.html` 讲的是**用 PPO 训练 SONIC 控制器**(`num_envs=4096`、推荐 64+ GPU、
  `sonic_release`/`sonic_v1_1`/`sonic_bones_seed` 三个 exp config),不是 VLA 微调。
- `deployment_code.html` 确认 **控制环 50 Hz、planner 重规划 ~10 Hz**,但
  `encode_mode` 一词全页不出现,外部 token 的注入协议未文档化。

**含义**:官方对 VLA 微调的全部处方就是 `vla_workflow.html` 的那一条命令
(见 B1/B2)。我们训练脚本的其余选择(lr schedule、`--tune-projector` /
`--no-tune-visual` 的取舍、8-bit optimizer、`--save-only-model`)**没有官方对照物**,
既不能说我们偏离了标准,也不能引用官方为其背书。这一点在写论文时要注意措辞。

---

## 未采纳/无需动作的对照(完整性记录)

逐项核对一致、无需改动的:

- **sim 启动 flag**:官方 `--enable-image-publish --enable-offscreen`(教程称 required)
  = 我们 `start_stack.sh:93`。
- **deploy 命令行**:我们直接调 binary 而非 `deploy.sh`,但参数与 `deploy.sh:238-253`
  组装的默认值逐项一致(decoder / `reference/example/` / obs-config / planner-file /
  `--input-type zmq_manager` / `--output-type all` / `--zmq-host localhost`),
  额外项 `--disable-crc-check` 是 sim 需要。
- **exporter 频率**:官方默认 `--data-collection-frequency 50`,我们不传即用默认。
- **exporter 按键协议**:官方只实现 `c`(三态循环)与 `x`(discard),我们用的正是这两个。
  (`run_vla_inference.py` 里的 `s`/`f` 只是 print,非功能键。)
- **数据量**:官方建议 "at least 50–100 demonstrations",我们 200-250,超出。
- **language 标注**:官方 key `annotation.human.task_description`,我们继承未改;
  `tasks.jsonl` 内容 `"pick up the water bottle"`,格式正确。
- **protocol v4 语义**:`frame_index` 在 v4 中是 "for logging only, does not affect
  playback",所以我们 keepalive 期间恒发 `frame_index=0` 无害。
- **planner 指令范围**:官方 keyboard 教程给出 Slow Walk `0.2–0.8 m/s`、height
  `0.2–0.8 m`(仅 squat 类模式生效)。与我们的实测记忆一致(height 只在 IDEL_SQUAT
  生效,单位是绝对米)。

---

## 建议的后续动作(等勾选)

| 优先级 | 动作 | 成本 |
|---|---|---|
| ~~高~~ **已做** | ~~A2:延迟补偿索引 `int()` → `round()`~~ 已采纳官方 `vla_utils`,并按上文理由豁免配对 A/B | 已完成 |
| 高 | C5:改正 `NO_ENCODER` 注释的机制描述;可选带 encoder 跑一局验证 | 10 分钟 + 20 分钟探针 |
| 中 | B1:`--color-jitter-params` 配对 A/B | 一次重训(用户已决定暂不重训) |
| 中 | C7:按"自然运动"指引审视脚本专家的运动剖面 | 调研性质,无既定成本 |
| 低 | B3:v1.1 checkpoint 重新裁决(需重采数据,与当前决定冲突) | 整条流水线 |
| — | B2:20000 步训练已在 ROADMAP | — |
| — | C8/C9:已证结论,只需保持记录不被后人推翻 | — |

## 审计覆盖清单(共 15 篇)

tutorials(10):`data_collection` `vla_workflow` `vla_inference` `zmq` `manager`
`keyboard` `gamepad` `isaac_teleop_publisher_setup` `vr_wholebody_teleop`
`live_camera_teleop`
user_guide(4):`teleoperation` `training_data` `training` `troubleshooting`
references(1):`deployment_code`
未抓(判定低相关):`user_guide/configuration`(SONIC 控制器训练超参)、
`user_guide/new_embodiments`、`references/planner_onnx`、`references/motion_reference`、
`references/conventions` 与 `references/observation_config`(抓取中,四元数与观测
空间的问题已由源码直接证明,见 C9 与 C8)。
