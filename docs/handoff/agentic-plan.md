# Agentic expert:设计决策记录(2026-09-03)

与用户逐条确认后的共同理解。上游背景见 `agentic-scripted-policy.md`;
本文只记**决定**和它们之间的依赖,不记证据(证据在 memory 与 docs 里)。
未写代码。

## 0. 目标

**任务扩展**(不是数据质量、不是省人力、不是论文 claim)。抓瓶子单任务
不需要 agent;agent 的杠杆只在任务族变大之后出现。因此 agent 放在
**task 规划层**,不放在抓取控制层(reach map / IK / 50 Hz 伺服不碰)。

## 1. 决策树

| # | 决定 | 选择 | 被排除的选项与理由 |
|---|---|---|---|
| Q1 | agent 换来什么 | 任务扩展 | 数据质量(B)、省人力(A)、论文(D) |
| Q2 | 第一批任务族 | **跨台面 pick-and-place**:"把 X 放到 counter / plate 旁" | 单桌序列任务(4 个物体上很快枯竭);开放式家务(判据无法定义) |
| Q3 | 决策粒度 | **相位边界闭环**:每个 primitive 结束后把结果回给 agent,它决定下一步或重规划 | 每局一次开环计划(是闭环的退化);agent 写脚本(留作以后) |
| Q4 | agent 观测 | **特权状态文本**(:5581 位姿/接触/可见性 + reach map 查表结果),**几何查表、agent 不猜几何** | VLM 看图(延迟 ×3,位姿误差叠在 reach map 上,失败归因变浆糊) |
| Q5 | 决策延迟 | **提前规划为主**(primitive 执行中就规划下一步,失败才真停)+ **转换时切掉思考窗口**兜底;窗口在 manifest 保留标记 | 原样录入(放大"到桌边不下手"的失败模式,明确反对);lockstep 暂停世界(绑死另一条线) |
| Q6 | 任务与判据 | **受限模板文法** `put <obj> on <surface \| next to obj2>`,判据从模板编译、由 sim 特权状态计算,**agent 说了不算**;**失败局也录**,manifest 标失败相位,转换默认只取成功局 | 自由语言 + agent 自判(最静默的检查器);agent 生成判据代码(错了静默) |
| Q7 | 工程顺序 | **并行开发,串行验收**:primitive 补齐(carry/place 迁到新控制逻辑)与 agent 框架并行;agent 接真 stack 之前,脚本化完整任务必须 **20 局 ≥ 10 通过** 且每个失败相位可被 failure_anatomy 归类 | agent 先上、primitive 后补(失败归因分不开) |
| Q8 | 模型 | **Claude Code 本身**(`claude -p` headless,用现有登录,无需 API key;实测 4 s/次) | Claude API(要 key);本地模型占 GPU0 |
| Q9 | 调用形态 | **纯规划器,JSON 进出**:`claude -p --output-format json` + schema;每局一个 session,`--resume` 续;不开任何工具;全部输入输出落盘 | 带工具的执行者 / Agent SDK(agent 掌握控制流,破坏 wall-clock 原则,能碰到 seam) |
| Q10 | 场景生成 | **本阶段不参与**,沿用随机采样;场景生成 agent 是下一阶段 | 联合生成 / 主动课程 |
| Q11 | 第一阶段验收 | **失败恢复**:注入失败(teleport 物体、搬运中施力、放置点被占、走偏),同一组 seed 上 paired 比较,agent OK-rate 显著高于脚本 expert,恢复段可归类 | 任务多样性(Q10a 下无空间);"不比脚本差"(什么都没证明) |
| Q12 | 基础设施 | ~~production Aug 17 deploy~~ → **nvdef 二进制**(wall-clock,见 §7);`INSTANCE=2` + GPU0,不与闭环评测同跑;独立 worktree `../locomanipulation-agentic`(branch `agentic`,base `b41f9b2`);fork `grasp_gym_agentic.py`,`grasp_gym.py` 原样;carry/place 作为加法进 `primitives.py`;共享树未提交改动不碰 | — |

## 2. 一个必须意识到的后果

Q6(模板)+ Q10(不生成场景)之后,正常局里 agent 的选择一个枚举器也能做。
**第一阶段 agent 唯一不可替代的贡献是失败后的决策**,所以 Q11 的验收只看恢复。
自然失败率在 Q7 门槛下太低,20 局凑不出统计量,故失败必须注入。

## 3. 接口草图(待实现时细化)

```
gym --(JSON: 任务模板实例, 状态快照, 历史, 可达性查表)--> claude -p
claude -p --(JSON: {next: primitive, args, rationale})--> gym
gym 校验 args(在模板/可达集合内)→ 执行 primitive → 结果回传 → 循环
```

- primitive 集合:`walk_to / turn_to / grasp / carry_to / place / retreat / abort`
- 每次调用:输入、输出、耗时、是否落在"提前规划"里,全部写进 manifest 同目录的 `agent_trace.jsonl`
- 思考窗口:`[t0, t1]` 写进 manifest,`convert_to_hf_format.py` 按标记切帧

## 4. 里程碑

1. **M0 脚本化完整任务**:carry/place 迁移;首先量"抱着物体走路会不会甩出去"(`PALM_DRIFT_MAX` 是静止判据);20 局 ≥ 10。
2. **M1 agent 框架离线**:状态序列化、模板文法、判据编译、`claude -p` 封装、trace 落盘;用录好的状态回放测,不起 stack。
3. **M2 接真 stack**:agent 跑 M0 的同一组 seed,不比脚本差(sanity,不是验收)。
4. **M3 注入失败 paired 比较**:Q11 的验收。
5. **M4 转换 + 审计**:切窗口、`check_dataset.py` 四段审计、reel。

## 5. 新分支之外已知会影响它的事

- planner 命令之后 deploy 走路 token 幅度 1/3(memory `planner-phase-degrades-token-walking`)。
  现 gym 走路靠 planner,flow200 即如此采;carry 段走路走不走 planner在 M0 里一起量。
- `main` 不含 flow200 之后的未提交改动(RTC / lockstep / `SONIC_DEPLOY_BIN`);
  目前看不出新分支需要,若需要请 training/eval 线先提交。

## 6. M0 状态(2026-09-03 03:45,暂停点)

已做:worktree `agentic` + `third_party` 软链;`SCENE_PRESET=bottle_plate`
(flow200 的 water_bottle + counter 上的 plate);`grasp_gym_agentic.py` =
grasp_gym.py + 主树未提交的 flow200 改动(one-shot / mid-band / RAISE_FINISH 0.30)
+ carry / place / retreat 阶段;`stream_reach.py` / `primitives.py` 同步了主树
未提交的 off-hand guard 与 LIFT_HEIGHT 0.10;`scripts/agentic/smoke_m0.sh`。

5 局 smoke(seed 7000 ×2,7100 ×3),0 局到达 place:

| 局 | 结果 | 阶段 |
|---|---|---|
| s2-1 | expert 抓取规划失败 | [grasp] |
| s2-2 | 抓起、hold 过,**第一步就掉**(手臂伸直走) | [carry] |
| s3-1 | 抓起,**streamed 收臂过程中脱手** | [carry] |
| s3-2 | expert 站位失败 | [grasp] |
| s3-3 | 抓起、收臂成功、第 1 段(到桌角)漂移 0.6 cm,**第 2 段中途掉** | [carry] |

结论:`carry_pose` 在 frozen upper-body 模式下是空操作(VR-3pt 腕通道死);
streamed IK 收臂可行(s3-3 palm pf [0.20 -0.16 0.12]),但 dex1 夹持 0.30 kg 瓶子
在步态下仍会脱手。下一步:carry 期间 10 Hz 记录掌系漂移与接触,区分"逐渐蠕
变"与"瞬间撞脱";对照量夹持力(squeeze)、走速(CARRY_SPEED_CAP)、收臂高度。

同步记录:2026-09-03 04:10 把主树当时未提交的 pick 段改动(MARGIN_STOP 0.04、
GYM_ONE_SHOT、闭环侧转、CROUCH_BACK、x_min 带、servo 遥测)三方合并进
`grasp_gym_agentic.py`,`stream_reach.py` / `primitives.py` 与主树工作副本逐字一致。
下一次同步节点:M0 跑 20 局之前;之后到 M3 之前冻结。smoke 脚本改用
`stop_instance.sh 2`,不再杀别人的 stack。

## 7. deploy 二进制的更正(2026-09-03,来自 training/eval 线)

`/workspace/deps/gear_sonic_deploy_ours_lockstep_nvdef` **不是 lockstep 模式**,是"干净
评测协议"的 wall-clock 二进制:WBC 网络 / ONNX / kp-kd / 观测格式与 production 完全一致,
区别是 (a) 支持 `SONIC_AUTO_START=1`,评测可以不发 planner 命令,避开 production 的
"走过 planner 之后行走 token 幅度只剩 1/3"的进程级退化态;(b) 还原了 NVIDIA 原始
default_angles(SIMPLE 移植把肩 roll 0.2 改成 0,直走 0.68 m 掉到 0.44 m)。
自 clean100 起主线所有正式数字都在它上面;production + planner settle 的数字不可迁移。

对本分支:**采集侧 = planner 模式 + nvdef 二进制**(`SONIC_DEPLOY_BIN` 指向它,不切
streamed,不加 `--sim-lockstep`);**评测侧 = `CLEAN_DEPLOY=1` + `--clean-deploy`**。
`scripts/agentic/smoke_m0.sh` 已默认指向 nvdef;`start_stack.sh` / `restart_deploy.sh`
同步了主树的 `SONIC_DEPLOY_BIN` / `CLEAN_DEPLOY` 分支。§6 之前的 5 局 smoke 是
production 二进制跑的,carry 脱手的结论要在 nvdef 上重新确认一次。

## 8. carry 脱手的根因与修法(2026-09-03 09:30)

**根因:MuJoCo 默认软接触下的摩擦爬行,不是控制器。** 10 Hz 掌系遥测(`carry_trace.jsonl`):
无论收臂还是走路,掌-瓶偏移都以约 0.5 cm/s 单调增长,手指位置不变(hand_q 停在瓶宽),
到 2.5–3 cm 先失去一根手指的接触然后掉落。静力学余量 3 倍(每指 5–8 N,摩擦 0.95–1.2,
瓶重 2.9 N)。追加夹紧无效:抓取结束时指令常已在行程终点 -0.02。场景无 `<option>`,
即 pyramidal cone / impratio 1 / noslip 0。

| 变体(同 4 个 seed) | 抓起局 | carry 存活 | 副作用 |
|---|---|---|---|
| 默认物理 ×2 轮 | 4 | 0/4(收臂 3 次、走路第 2 段 1 次) | — |
| 全局 `noslip_iterations=10` | 0 | — | **原地转身超时 3/4**,脚被钉住,不可用 |
| 指面 4 geom `priority=1 condim=4 solref 0.004 solimp 0.99/0.999` | 1 | **1/1,全程漂移 0.8 cm** | 无(转身正常) |

那一局是分支上第一局完整成功:pick → carry 24.7 s → place(离盘心 7.1 cm,竖直)→
retreat,共 120 s。参数已写进 `build_sim_scene.py`(`PAD_HARDEN=0` 可关),带编译后
断言;`bottle_plate` 场景已重建。

**待议(需和 99 协调)**:这是共享的夹爪物理问题,他们的 2 s hold 检查看不到它,
但任何超过几秒的持物动作都会遇到;`make_dex1_model.py` 里"gear 150 → 250 因罐子搬运
滑出"是同一现象的早期形态。进主线要重验 pick 段与走路。

**其它观察**:同步 99 的 `MARGIN_STOP` 0.04 之后,我这台 stack 的 approach 停在 0.39–0.44,
带宽 ≤0.36/0.38,one-shot 几乎每局丢弃(刹车后动量滑行为 0,他们量到 5–18 cm);A/B
运行应去掉 `--one-shot` 提高抓起率。新 stack 的第 1 局两次出现"抬臂阶段手臂 8 s 不动",
非挪核所致,待查。

## 9. repair8(2026-09-03 09:35–10:00,指面硬化 + 无 one-shot,seed 7200)

| 局 | 结果 | 备注 |
|---|---|---|
| 1 | [grasp] 首局抬臂停滞 | `NO_ENCODER=1` 起的首个 deploy 丢上肢参考(3/3);脚本已改为 encoder 开 |
| 2 | [carry] 到达前原地转身时掉 | 累计蠕变:收臂 1.9 + 两段 1.3 cm,越过 3 cm |
| 3 | **OK** 127 s | 放置误差 1.7 cm |
| 4 | [place] 对不中 7.2 cm | goto 残差 4.2 / 5.6 cm |
| 5 | [place] 对不中 22.1 cm | goto 残差 4.3 / 9.3 cm |
| 6 | **OK** 132 s | 放置误差 4.4 cm |
| 7 | [place] 倒下,17.4 cm | goto 残差 5.6 / 6.3 cm |
| 8 | [place] 对不中 22.5 cm | goto 残差 8.5 / 9.8 cm |

- carry 存活 6/7(蠕变 0.16–0.28 cm/s,到达时 0.9–1.5 cm),**sim RTF 0.977–0.984**:
  隔离模式下 sim 是实时的,时序假设排除。
- 放置成了主要失败面:goto 残差与最终误差单调相关,是可达域边缘形态。改动:
  站位横向偏 `PLACE_LATERAL=0.12`(盘子对着右手)、走进距离 0.42 → 0.38、放置前打印
  drop 点骨盆系坐标与 reach map 余量。待下一轮验证。

## 10. place8(10:00–10:12,站位横向偏 0.12 + 走进 0.38):中止

4 局后中止:pick 1 次 expert 碰倒(首局,encoder 开后手臂正常动了,**首局停滞假设验证**),
放置 0/3(8.7 / 35.1 / 21.3 cm,goto 残差 5.9–6.0 cm)。横向偏移**有害**:走进循环本来
就朝 drop 点转向,偏移只让接近变成斜线,站位处方位 -19..-20°。撤回横向偏移;走进改
0.34(抓取带宽 0.22–0.36),并在走进后打印掌心目标的骨盆系坐标。

同步待办(2026-09-03 12:05,99 通知):主树 pick 段已改为蹲前闭环 creep(`_creep_to_band`,
shuffle 速度边走边量 grip x,进 [0.18, 0.27] 即蹲,无 IDLE 帧),`MARGIN_STOP` 回到 0.075。
下次同步 `stream_reach.py` / `grasp_gym.py` 时一起拿。非 one-shot 的 nudge-重蹲-重试
路径会丢上肢参考(arm not tracking),99 已记录,暂不修;A/B 运行用 `ONE_SHOT=0` 时
首局/重试局可能因此失败,统计时按 [grasp] 计。

## 11. 放置段收敛(2026-09-03 12:00–13:25,四轮 plan8 → servo8d)

放置失败的真相链:goto 诊断显示腰交付正常,**上肢整体跟踪滞后**(腕 pitch 11–38°、
腕 roll 16–43°、肘 14–18°、腰 pitch 8–11°)让离散 goto 停在 4–10 cm 外,"竖直释放"
也不竖直 → 改为 50 Hz 连续伺服(位置 + lead、每拍校正竖直、只用手臂);伺服收敛
0.4–1.7 cm 后瓶子仍被扫倒 → 逐步取证:释放、三步撤手全干净,**最后的关节空间收臂
把手从上方扫穿瓶子**(bottle 比 can 高 4 cm、counter 比桌高 4 cm)→ 收臂挪到退后
0.5 m 之后。站位 walk_to + turn_to 误差 2–17 cm → 站位伺服(shuffle,朝向锁定)。
另:收臂遥测线程并发读 ZMQ socket 导致 libzmq abort,改为读主线程缓存。

servo8d(seed 7200,无 one-shot,GPU1,录像开):

| 局 | 结果 | 放置误差 |
|---|---|---|
| 1 | [grasp] 重试路径停滞 | — |
| 2 | [place] 伺服中脱手(站位偏 17 cm) | — |
| 3 | **OK** 127 s | 1.2 cm |
| 4 | **OK** 132 s | 3.6 cm |
| 5 | [grasp] 进刀碰倒 | — |
| 6 | **OK** 121 s | 0.7 cm |
| 7 | [place] 伺服中脱手(站位准,悬停/下降 12 s 蠕变累计) | — |
| 8 | [grasp] 站位错 | — |

抓起 5 局里 carry+place 通过 3 局,放置误差 0.7–3.6 cm,全部竖直。两次伺服脱手
的对策已加:伺服 10 Hz 遥测(`place_trace.jsonl`)+ 下降超过 4 s 且 xy < 3 cm 即释放;
站位伺服替换校正腿(下轮生效)。录像开着时 carry 段 RTF 降到 0.85–0.90,统计跑
应关录像。

## 12. stand8(14:33–14:47,站位伺服 + 限时释放):1/8,退步

站位伺服本身准(2–5 cm,一次 13 cm 超时),但它把持物时间拉长:3 次抓起在放置前
的下蹲 / staging 就丢了(到达时漂移已 1.9 cm)。放置伺服遥测:起始倾斜 25°,伺服 4 s
内 slip 1.2 cm——把瓶子扭正也在消耗额度。对策(pad2_8 起):指面第二档
(solref 0.002、solimp 0.995/0.9999)、放置前**不做 planner 下蹲**(站立高度下 drop 点
在完整可达域内余量满,9 cm 后退冲击不值)、站位伺服限时 8 s。

## 13. up8(15:16–15:35):**5/8**,分支最好成绩

配置:指面 L2、无下蹲、站位伺服(0.35/0.25,8 s)、放置伺服 anti-windup + 4 s 限时释放、
撤手先上抬 8 cm。抓起 7 局:OK 5(放置误差 1.1–7.9 cm,全部竖直),撤手上抬时倒 1
(r4),伺服中瓶子绕手指接触线转到 47° 滑出 1(r7)。carry 7/7,到达漂移 0.4–1.3 cm。
已加(下轮生效):撤手慢速 0.1 + 姿态锁定实测掌心;伺服倾斜 > 10° 不下降、到位且
倾斜 < 12° 立即释放、下降中倾斜 > 25° 回抬。
