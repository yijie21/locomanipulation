# STATE — 2026-09-25（会话：VLA 闭环的 Isaac 按需镜像；MuJoCo 2²⁰ 网格门槛；协作者 deploy 对齐与 setup 钉版本；gate8 bank 入库）

工作目录 `/workspace/code/locomanipulation`，分支 `main`。已推送到 `c7f06c6`…`b3b645f`；**`b8f6c82`（镜像状态索引）和本次 handoff 提交未推送**（TODO 4.8.8）。
没有运行中的栈或镜像。理由与被否决方案在 DECISIONS 最后两节（2026-09-24 镜像、2026-09-25 deploy 对齐）。

## 工作树

- 无未提交的代码改动。未跟踪（与本会话无关，不要提交）：`.serena/`、`node_modules/`、`package*.json`、`uv.lock`、
  `assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`。
- `third_party/GR00T-WholeBodyControl` 挂着两个 worktree（4.8.5 定位用，勿删）：`outputs/deploy_docbuild_20260924/`（上游 b042411 + 旧文档补丁，二进制 472ed2b）、
  `outputs/deploy_pinned_20260925/`（1983e888 + 新三补丁序列，二进制 fb546c2，源码 = 5cab2fb 的源码）。

## 当前进度

- **Isaac 镜像（已确认，main）**：`eval.image_source: isaac` → `scripts/run_isaac_mirror.sh <run> --gpu 1`；真实 eval 栈冒烟 50/50、状态年龄 p50 11 ms、
  渲染 p50 98 ms（栈与镜像同在 GPU1）。做了 / 没做 / 分支对账的索引：`docs/handoff/isaac-vla-mirror-20260925.md`。**真实策略闭环未跑**（无 warehouse checkpoint）。
- **MuJoCo 慢的根因（已确认）**：单块网格 3×面数 > 2²⁰ 渲染慢约 12 倍；`warehouse_textured` 已拆 9 块，离线测三路 3.2 ms。实时栈 RTF 未测（4.7.7）。
- **deploy 对齐（已确认）**：本机二进制全来自本地 `lockstep-port` 分支；setup.md 现钉 1983e888 + 三补丁，在干净 checkout 上重建出 5cab2fb 源码（指纹 `c1360628…`）。
  本机在用：bottle / warehouse = 5cab2fb（`outputs/bottle_continuous_20260914/deploy`），crate = 4add301（默认树，无夹爪补丁）。
- **文档版 vs 现用（已确认，台账 bottle §38）**：旧文档编法（上游 main + 两补丁）bottle 0 摔倒，但抓取 10/17 对 20/23、髋碰桌 5/17 对 1/23（p≈0.07）；原因未定位。
- **gate8（已确认）**：bank 入库 `configs/cases/`，新版本 `bottle_simple_gate8_20260925.yaml`，case bank schema 2 可跨 checkout。

## 卡点 / 待验证

- **协作者重编后仍 `initial_stance_unstable`**：init token 假设已排除（已确认）。原因**待验证**，等他的日志；候选（假设）：停栈后不足 20 s 重起致 deploy 未接管；
  mujoco 3.14 或 3.14 下建的场景；仿真不满实时。开局判定失败时现在只记一个 `false`，没有细节（4.8.2.1 待用户定）。
- 待验证：crate 换成带夹爪补丁的二进制是否等价（读代码推断无差别，未跑）。
- 待验证：按需跳着渲的 Isaac 图与离线逐帧渲是否一致（RTX 跨帧状态，4.7.9.8）；训练图仍是 MuJoCo，与 Isaac 评测图的域差未测（4.7.9.9）。
- 协作者从零按新 setup 装一遍的结果未知（4.6）。

## 下一步（编号见 `TODO.md`，等用户定顺序）

- **4.8.2** 收协作者日志，判断 `initial_stance_unstable` 属于哪一类；**4.8.2.1** 若用户同意，先加开局失败诊断再请他重跑。
- **4.8.8** 推送 `b8f6c82` 与 handoff 提交。
- **4.8.4.3** crate 20 局验证后全线统一 5cab2fb（用户暂缓实测）。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"   # 本机 bottle/warehouse 的 5cab2fb；crate 不设（默认树 4add301）
bash scripts/cpu_headroom.sh; nvidia-smi; find /workspace/tmp -maxdepth 3 -name stack.json   # 有输出 = 有栈在跑
# 停栈后等 ≥20 s 再起；pkill -f 会匹配自己的命令行，杀进程写脚本文件

# Isaac 镜像（评测；实验文件设 eval.image_source: isaac）
bash scripts/start_stack.sh eval <exp.yaml|run> --gpu 0
bash scripts/run_isaac_mirror.sh outputs/runs/<run> --gpu 1          # --dry-run 看场景/端口
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src "$VENV_SIM_PY" scripts/experiments/isaac_mirror_smoke_20260924.py outputs/runs/<run>

# deploy 二进制指纹与能力标记（对比协作者）
(cd <deploy_root>/src && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum)   # 期望 c1360628…
for m in POSE_UPPER_OWNERSHIP_V2 PLANNER_HAND_OWNERSHIP_V1 SONIC_AUTO_START SONIC_DDS_DOMAIN; do grep -aq $m <binary> && echo "有 $m" || echo "缺 $m"; done

# 提交：make check → git commit --no-verify → 按文件名 add（钩子会被别人的 WIP 干扰）
make check
```

- 测速命令与证据目录见 `docs/handoff/isaac-vla-mirror-20260925.md`；文档版对比：`outputs/deploy_docbuild_20260924/compare_arms.py`（台账 bottle §38）。
