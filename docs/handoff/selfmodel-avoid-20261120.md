# 自身模型线 2026-11-20：桌面抓取卡桌沿、cuRobo 感知避障、归因、grasp_kit，以及多路点自身模型方案

分支：本地 `sonic-selfmodel`，推到 `origin/selfwm`。这一天在 Mac 上开发，在 4090 和 vastgpu 上运行。接下来的工作换到另一台机器上做，本文是交接材料：做了什么、为什么这样选、哪些方案被否决了、下一步从哪里接。

## 0. 接手前先读

- **用户偏好：**
  - 每次回复都用中文。
  - 给领导看的报告只讲一条主线，用白话，不用实验室黑话，旁支分析删掉。
  - 只在用户要求时才提交和推送。暂存时写明具体路径，不用 `git add -A`。提交信息末尾加 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。
- **实验纪律**（Mac 上有 `~/research-notebook` 和启动拦截 hook，新机器上没有，需要手动照做）：
  - 启动会产出结论的运行之前，先写好假设、带数字的预测和判据。
  - 结果贴原始输出，不转述。
  - 要报告的结论先找一个独立的 agent 复核：只给它结果和数据路径，不给推理过程。
  - beliefs 只做增量修改。
  - 本次相关的笔记条目原样复制在 [notebook-20261120/](notebook-20261120/) 里。
- **共享机器上不要动别人的东西：**
  - 4090 的 GPU 0 经常有别人的任务，用 GPU 1–3；11-20 还有一组 `handover/lgrab_plan_test.py` 进程不是我们的。
  - vastgpu 只用 GPU 1。那里的 SAM 3 服务（127.0.0.1:18832）和几个深度服务不是我们的；`/tmp/.tmp*` 下约 11 GB 的旧目录也不是我们的，不要删。
  - a800 的 tmux `data_scaling_spin17` 不要碰。
  - 停进程按 pid 停。
- **远程长任务：**用户在 Mac 上用 CodeIsland 看进度。预计超过 5 分钟的远程任务要登记到 `~/.claude/longjobs/`，格式见那里的 README。新机器上如果没有这套，就跳过。

## 1. 这一天的经过

下面引号里是用户原话，后面是我们做了什么。

### 1.1 TabletopGrasp 为什么这么差

- 用户问："现在为什么 tabletopgrasp 上的指标为什么这么差？"
- 用户要求："把 TableTopGrasp 的 30 局视频都导出到这台 macbook 上，我看一下失败都是怎么产生的"
- 用户看完视频："这些所有的失败的视频都是因为伸手的时候，手被桌子挡住了导致手抬不上去。"
  - 他接着问：初始站位是不是 SIMPLE 固定的？是，评测时就定下来了。
  - 他又问：能不能用 cuRobo 避障来避开碰撞？
- 追查结论见 [SIMPLE Isaac 画面记录](simple-isaac-eyes-20261120.md) 里"TabletopGrasp 为什么这么差"一节。
  - 失败不是手腕姿态造成的，是手被桌子前沿挂住：机器人站得近，右手从身侧往上往前伸，食指根部顶着桌沿往上爬。
  - 评测默认去掉了桌沿外那个路点（`--clear`）。

### 1.2 是不是在刷榜

- 用户的顾虑："这样做的话其实还是有点像是在 hack 榜单，我想的是我们的自身模型这个方法是不是有能力直接通过 perception 的方式让自身模型做出避障这个动作"
- 用户比较了两条路：
  - 一是专门构造绕障碍的自身模型，把障碍物当作输入；
  - 二是"把机器人看到的当前画面转化成点云，然后点云转化为 curobo 可以读取的障碍物格式"。
- 他问哪种更合适："学习的方式有点像是把 curobo 的能力学到一个网络里……会不会显得多此一举？"
- **当时的回答：**先做第二条，用感知加 cuRobo。把 cuRobo 的避障能力本身学进网络确实多此一举。
- 用户："好，从第 1 步开始做"。

### 1.3 cuRobo 感知避障（curobo_avoid_20261120）

- 全文：[notebook-20261120/curobo_avoid_20261120.md](notebook-20261120/curobo_avoid_20261120.md)
- 代码：[`scripts/experiments/curobo_avoid_20261120/`](../../scripts/experiments/curobo_avoid_20261120/)
  - `obstacles.py`、`obstacles_box.py`：第 1 步，点云转障碍；
  - `prep2.py`、`plan2.py`、`check2.py`、`via2.py`：第 2 步，规划；
  - `step3.py`：第 3 步，统计；
  - `run_box.sh`：三个盒子任务的流水线。
  - 4090 上 `loco_outputs/simple_bench_20261119/curobo/` 有同样的副本，运行脚本调的是那里的副本。
- **以复核后的修订为准：**
  - **感知：**用机器人自己的模型渲染出哪些像素是自己的身体，把它们去掉。桌沿误差中位数 0.20 cm。亮面或玻璃桌的 4 / 30 局桌沿处深度塌下去，误差 8–19 cm，机器人自己察觉不到。
  - **TabletopGrasp：**
    - 卡住次数 direct 18 → 1，model 19 → 1。
    - 45 s 口径成功 direct 2 → 12，model 3 → 17；SIMPLE 官方 16 s 口径 direct 2 → 5，model 3 → 2（路线长一倍）。
    - 有路线的 28 局里，27 局越过桌沿时仍然碰桌约 1 s。
  - **盒子任务：**锁腰、只动右臂的规划器在 83 / 90 局够不到预抓取位姿，等于没检验到。
  - **撤回的说法：**"这是考虑自身模型的规划的直接证据"。按预注册判据（被挡住 ≤ 4 局），路径版自身模型当时搁置了。
  - **复核留下的 E2 和 E3 都还没做：**
    - E2：model 是否真的优于 direct（多扰动重复，加"direct 加常数偏置"的对照）；
    - E3：激活距离改成 5 cm，能不能消除碰桌。

### 1.4 归因实验（curobo_attrib_20261120，复核的 E1）

- 全文：[notebook-20261120/curobo_attrib_20261120.md](notebook-20261120/curobo_attrib_20261120.md)
- 代码：`vias_e1.py`、`run_attrib.sh`、`attrib_sum.py`，都在同一个目录。
- 设置：28 局，站位左右挪一点，一共 84 次伸手；只用 direct，只改手走的路线。

| 组 | 手走的路线 | 卡住（/ 84） |
|---|---|---|
| A0 | 直线（原做法） | 49 |
| A1 | cuRobo 路线 | 0 |
| B | 直线，切成同样多段、放慢到同样时长 | 49 |
| C | cuRobo 的位置，手腕按直线时的方式转 | 1 |
| D | 直线的位置，手腕按 cuRobo 的方式转 | 41 |
| E | 固定模板：后撤 8 cm，抬到预抓取高度，再伸过去 | 0 |
| F | 借用另一局的 cuRobo 路线 | 3 |

- **复核的追加实验**（在卡住的 16 局上）：
  - 只竖直抬到预抓取高度、不后撤：0 / 16 卡住；
  - 只后撤、不抬：11 / 16 卡住。
- **结论（B46）：**
  - 原因是手从桌板下面往前伸，还没抬过桌面高度。
  - 这个基准说明不了感知规划有必要：28 局是同一张桌子，障碍永远在目标下方。
- **撤回的说法：**"E 的成功率高于 cuRobo"。差距主要是 E 的路程短了约 1 s。
- **证据强度：**站位挪动只让起点移动 3–4 mm，84 次伸手实际约等于 28 个独立局。

### 1.5 grasp_kit（已提交 `992427ab`）

- 用户："现在把机器人感知，然后得到抓取姿势这一整个过程打包成一个完整的 package，我想发给另一个有 g1 机器人的人"
- 范围（用户确认"就这么打包"）：
  - Dex3 右手；
  - RGB-D 深度，双目可选；
  - 只做到给出抓取姿势；
  - 权重用脚本下载，不打包；
  - 带示例。
- **验证（vastgpu GPU 1）：**按 README 从零安装，6 个示例 × 两种深度全部 `all ok`；在 Linux 上解开打包文件再跑一遍，也通过。
- 中途修过三个问题：
  - LingBot-Depth 需要 xformers；
  - xformers 0.0.31 在 RTX 50 系列上会选错内核，于是感知环境改用 torch 2.11 + xformers 0.0.35；
  - 下载脚本原来多下了约 3 GB。
- **只在 Mac 上、不在 git 里的两样东西：**
  - 打包文件 `outputs/g1_grasp_kit_20261120.tar.gz`（6.2 MB，sha256 `d5ee36df88ce0d70278d0a2c25aed9d28e47b67d81688828d4c612bca1c7480e`）；
  - 示例数据 `outputs/grasp_kit_build/examples/`。
  
  重新打包的命令在 `grasp_kit/dev/make_examples.py` 的说明里。
- vastgpu 上的测试目录已删除（用户："删掉 g1kit 吧"）。

### 1.6 新实验的讨论：从"桌上放随机障碍物"到"几何管障碍、自身模型管身体"

1. **第一版设计：桌上放随机障碍物**（被否，见 DECISIONS）。
   - 做法：桌上随机放箱子和横梁，比较直线、先抬手、高弧、看一眼挑动作、看见后规划、上帝视角六种做法。
   - 用户的否决理由："如果是在桌子上摆放障碍物的话，其实还是跟桌子相关的。但我们其实是想要一个通用的绕遮挡的方法。而且……遮挡的长宽高，大小，位置……要完全覆盖的话其实需要 sample 巨量的数据"。
2. **拆法：障碍物交给几何，身体交给自身模型。**
   - cuRobo 直接用点云做碰撞检查，不需要训练，对任何形状都一样，通用性来自几何而不是数据。
   - 自身模型只学"给控制器一串命令，身体实际怎么走"，在空地上采数据，和障碍物无关。
   - 文献佐证"学障碍物到路线的映射需要海量数据"：
     - MπNets（CoRL 2022）用了 50 万多个环境、300 多万条轨迹；
     - Neural MP（2024）用了 100 万个场景，部署时还要额外优化。
3. **用户问："怎么保证可以覆盖所有的可能性？会不会被 reviewer 攻击说是比较特化的处理"。**回答有三点：
   - 方法里学到的部分不依赖障碍物分布。需要覆盖的只有身体的命令空间，它有界、低维，可以覆盖。
   - 用空地数据事先预测"多窄的缝能过"，再在障碍场景里检验转折点，这样结论和场景无关。
   - 测试场景尽量用别人的：MotionBenchMaker（RA-L 2022，8 类环境）、MπNets 的程序生成场景、其他人形基准里的家具。按 G1 能够着的范围重新摆放，摆放规则事先定好。
4. **核对现有数据：有整条轨迹。**
   - 每段按 10 Hz 存全身状态。用正运动学从关节角和骨盆位姿重建手的位置，误差中位数 0.003 cm。
   - 缺的是"连续经过多个路点"的命令。
   - 现有数据里手偏离参考路线：站着 3.3 cm，蹲低 7.1 cm；蹲低、低目标、远伸手时 9.7 cm。
   - 见 [方案条目](notebook-20261120/selfmodel_via_20261120.md) 的背景一节。
5. **补采方案：**空地上随机生成多路点路线，加覆盖检查、整片形状留出、35 条真实 cuRobo 路线回放。见方案条目。
6. **用户问测试时怎么避障、和直接用 cuRobo 比有什么优势。**
   - 流程：看 → cuRobo 规划 → 自身模型预测整条手臂实际会走到哪 → 对照点云检查 → 在自身模型上微调路点，或者放弃 → 一次性下发。
   - 优势：
     - cuRobo 默认机器人会精确执行它规划的关节轨迹，而我们只下发手的路点，肘和腰由控制器自己解 IK，控制器跟踪还有误差。
     - 直接用 cuRobo 只能统一加厚障碍：加得薄，窄缝里会撞；加得厚，窄缝被堵死。
     - 自身模型能按预测给余量、提前补偿。
   - 没有优势的情况：
     - 宽敞场景；
     - 空地上身体本身偏得很少；
     - "把 cuRobo 的关节轨迹直接作为控制器参考"这个不用学习的对手已经够好。
7. **演示 case（用户："可以，把这个也记进方案"）：**蹲下，从塞满瓶子的矮柜底层，穿过前排瓶子之间的窄缝，取后排的一罐饮料。三联画面加半透明手臂叠加，最后缝越推越窄。详见方案条目的"后续"一节。
8. **时间估计（4090 全力跑）：**
   - 写代码 2–3 小时；
   - 机器时间 45–60 分钟：试跑 3 分钟、正式采集 17–25 分钟、标签 5 分钟、训练 10–20 分钟、评测 5 分钟；
   - 依据：11-19 采 10200 次开场用了 14.4 分钟，训练两个种子用了 8.2 分钟。

## 2. 还没解决的问题

- **"路径版自身模型"这次重新打开了**，但 curobo_avoid 的预注册判据当时把它搁置了。重新打开的理由是用户要的是通用绕障，不是 TabletopGrasp。那次复核的 E3（激活距离 5 cm）还没跑：如果 5 cm 就能消除碰桌，而且多花的时间不超过 1 s，那么 TabletopGrasp 本身不需要它。
- **4.1 cm 这个数：**direct 全程手偏离规划路线的中位数是 4.1 cm，最大 11.7 cm；只看规划段，两种方法都是 3.82 cm。这些都是在碰桌的情况下测的。接触处规划余量中位数约 5.6 cm，手点偏差约 3.4 cm，接触原因还没找到，候选是手腕姿态误差，或者碰到的是手指和手臂而不是手点。这正是方案要预测整条手臂、而且要做空地回放的原因。
- **现有自身模型执行避障路线时预抓取点更差：**中位数 6.5 cm，direct 是 3.4 cm。它没见过多路点命令。
- **查新还没做：**"学身体的动态，再用几何检查障碍"用在冻结的人形全身控制器上，是否已经有人做过。
- **Isaac 画面里的罐子偏上约 3 个像素**（见 SIMPLE Isaac 画面记录），相机位姿和内参还没核对。

## 3. 机器和路径

- **4090：**`ssh -p 27984 root@14.103.60.188`
  - 仓库 `R=/wuji-vefps-D/wuji-il/yijie/ws/code/loco`，停在 `4344165c`。`motor.py` 和 `simple_bench.py` 有同步过去、没提交的改动，已确认（11-20，md5）和 `d8993b35` 里的完全相同，可以 `git checkout` 掉这两个文件再快进到 `origin/selfwm`。
  - 仿真 Python：`$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python`
  - SIMPLE 和 cuRobo 的 Python：`/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE/.venv/bin/python`
  - 避障和归因的产出：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_bench_20261119/curobo/`
    - `step1/ step2/ via_tabletop/ step3_tabletop/ box/ attrib/ review_attrib/`
    - 真实路线文件在 `via_tabletop/`（28 条）和 `box/via/`（7 条）；回放用的起点在 `step2/prep_*.npz`、`box/step2/prep_*.npz`。
  - 自身模型数据和模型：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_selfmodel_20261119/`。仓库里的 `$R/outputs/reachwm_20261030` 是指向它的符号链接（已确认），每份数据以目录里的 `all.npz` 为准。
    - 运行脚本 `run_wrist_4090.sh`，数据集定义 `datasets_wrist.sh`；
    - 采集场景 `$R/outputs/scene/simple_g1_sonic`；
    - 环境变量 `BFM_ROOT=odo REACH_HAND_OFF=0.0415,0,0`。
  - 分析脚本副本：`/wuji-vefps-D/wuji-il/yijie/loco_outputs/{fk_check,dev_map,via_stats}.py`；仓库里在 [`scripts/experiments/selfmodel_via_20261120/`](../../scripts/experiments/selfmodel_via_20261120/)。
  - 重建时要注意：
    - 关节顺序按 `runtime.META` 的 `joint_names`，不是 `MUJOCO_JOINT_NAMES`；
    - 场景用 `simple_g1_sonic`，用 `walk_mars` 会有 0.7 cm 的偏差。
- **vastgpu：**只用 GPU 1。它渲染 SIMPLE 的 Isaac 画面，见 SIMPLE Isaac 画面记录。
- **Mac：**
  - 完整对话记录只在 Mac 上；
  - 实验笔记本是 `~/research-notebook`，不在 git 里，本次的相关条目已复制到 [notebook-20261120/](notebook-20261120/)；
  - grasp_kit 的打包文件和示例见 1.5。

### 3.1 4090 上的命令（2026-11-20 从 STATE.md 移到这里）

```bash
# 4090
ssh -p 27984 root@14.103.60.188
R=/wuji-vefps-D/wuji-il/yijie/ws/code/loco                                   # 仓库
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_selfmodel_20261119         # 自身模型数据、模型、运行脚本
C=/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_bench_20261119/curobo      # 避障和归因的产出、35 条真实路线
PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python                 # 仿真
SP=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE/.venv/bin/python     # SIMPLE 和 cuRobo，有 numpy

# 把 4090 的仓库更新到这次推送的版本（两个未提交文件和 d8993b35 相同，丢掉后快进）
cd $R && git status --short && git checkout -- src/locomani/backends/scalebfm/motor.py \
  scripts/experiments/simple_eval_20261112/simple_bench.py && git fetch origin selfwm && git merge --ff-only origin/selfwm

# 重建检查和偏差统计（在 $R 下跑）
$PY -I scripts/experiments/selfmodel_via_20261120/fk_check.py outputs/reachwm_20261030/data_graspr_wrist_odo 200 simple_g1_sonic
$SP -I scripts/experiments/selfmodel_via_20261120/dev_map.py outputs/reachwm_20261030/data_graspr_wrist_odo
$SP -I scripts/experiments/selfmodel_via_20261120/via_stats.py $C/via_tabletop $C/step2 $C/box/via $C/box/step2
```

- **现有抓取数据的采集方式**（新模式照这个加开关）：
  - 数据集定义：`$O/datasets_wrist.sh` 里的 `graspr_wrist_odo` 一行；
  - 环境变量：`REACH_GRASP=1 REACH_GRASP_ROOM=1 REACH_GRASP_SPEED=0.08 REACH_GRASP_CONT=0.5 REACH_GRASP_IKREF=1 REACH_WRIST=1`；
  - 分机器采集：`$O/gen_part.sh MACHINE NAME RESETS SEED ID0 [VAR=VAL ...]`；
  - 合并：`$PY scripts/experiments/reachwm_20261030/gen_reach.py merge`；
  - 整套流程：`$O/run_wrist_4090.sh`。
- **其他机器：**
  - vastgpu 只用 GPU 1；它的 SAM 3 服务、深度服务和 `/tmp/.tmp*` 都不是我们的。
  - a800 的 tmux `data_scaling_spin17` 不要碰。

## 4. 下一步

按 [TODO](../../TODO.md) 第 38 节的顺序来。

**2026-11-20 在 vastgpu 上已经做了的**（细节在方案条目、TODO 38.5 和 DECISIONS 的“续”一节）：

- 查新（38.5.4）：这个分工本身不新，最接近 Dynamic Tube MPC；用户定了两处修改。
- `gen_reach.py` 的两个开关（38.5.5）、覆盖检查（38.5.6）、全臂标签（38.5.7）和偏差分析都写完，冒烟测试通过。命令在 [运行手册](../selfwm.md) 第 7 节末尾。

**接下来：**

1. 把 35 条真实路线和它们的起点文件拷到跑试跑的机器（用户已同意）。
2. 试跑，按方案条目里的判据决定要不要正式采集。预测和判据在方案条目里，启动前可以再改；第一次启动以后就不再改，只在"预测修订"一节追加。
