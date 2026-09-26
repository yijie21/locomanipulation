# STATE — 2026-09-26（会话：warehouse 采集 → Isaac 渲染数据集相机 → 200 集 → 训练停在 5000 步）

工作目录 `/workspace/code/locomanipulation`，分支 `main`，已推送到 `4655254`；本地未推送：`e887520`、`9056a43`、`c0b97ff` 和本 handoff 提交。
没有运行中的栈、训练或渲染；GPU0 空闲；GPU1 上的 `depth_server.py`、SimpleRoboJev vision worker 是别的项目的（占用 3.7–16.6 GB，时多时少）。理由与被否决的方案见 DECISIONS 最后两节。

## 工作树

- 未跟踪、与本项目无关、不要提交：`.serena/`、`node_modules/`、`package*.json`、`uv.lock`、`assets/blender/{fonts,textures}/`、`assets/crates/nor_cal_box/visual/`；`third_party/GR00T-WholeBodyControl` 挂着 worktree `outputs/deploy_{docbuild_20260924,pinned_20260925}/`（删时用 `git worktree remove`）。

## 当前进度（均已确认）

- warehouse 10 集（TODO 2.9）：`box_room_grasp_10ep_20260925` 30 局保留 11；视频预览在 `outputs/videos/box_room_grasp_10ep_20260925{/,.zip}`。
- Isaac 渲染数据集相机（4.7.9.9，`9056a43`）：`scripts/render_dataset_cameras_isaac.sh` 在转换前用镜像同一个 `IsaacView` 填占位帧，只渲成功局和 25 Hz 取到的行；convert 按 `eval.image_source` 拒绝渲染器不一致的 session。
  实测每局 62–90 s，与采集并跑不拖慢采集。用户定：只用三路相机，第三视角不作为输入。
- 200 集（4.7.6.1）：`box_room_grasp_200_cap120_20260925` 共 950 局，保留 204 集（含补采 1 批），产出率 0.21；转换成训练 194 集 + 验证 10 集，审计 32/32 两边都过；
  训练图全部来自 Isaac（81 个 session 标 `isaac`，其余 14 个没有成功局）。
- exporter 丢每批最后一集（2/94 批）已修（`c0b97ff`，`batches.wait_saved`）；本运行记录已更正，原件 `.orig`，脚本在 `<run>/corrections/`。
- 训练按用户要求停在第 5000 步：`<run>/model/checkpoint-5000` 完整（3 个分片 + 索引，1030 个张量，loss 0.0437）。**`summary.json` 没有 train 条目**，评测没跑。
- bottle：guard 走近是默认专家（10/10，`e3d49d9`）；用户定 bottle 专家不再微调。

## 卡点 / 待验证

- 待验证：Isaac 镜像 + 真实 checkpoint 的闭环从没跑过（4.7.6.1.2）；`eval.initial_token` 沿用 crate 的，未核对（4.7.6.2）；数据集逐帧渲染与镜像按需渲染是否同图（4.7.9.8）。
- 风险（假设）：镜像在 GPU1 约需 23 GB，别的项目占得多时会放不下；`run_overnight` 会在"mirror never listened"处停下，不会带坏镜像跑完。
- 已确认：没存优化器状态（`--save-only-model`），"续训到 10000"只能从 checkpoint-5000 的权重重开优化器（4.7.6.1.3）。
- 已确认（1013 局）：站位深度决定成败，≥ 0.52 m 的 309 局只成 3%；`stand_off` 对照待用户定（2.10）。
- bottle 遗留（均待用户定）：5.4.1.1 抬手前扫、5.8.1 / 5.8.2 gate8 基线与旧走法删除、5.10 整条线、4.8.2.2 协作者摔倒、4.8.4.3 crate 仍是 4add301。

## 下一步（编号见 `TODO.md`，顺序等用户定）

- **4.7.6.1.2**：checkpoint-5000 跑 100 局闭环（或 **4.7.6.1.3** 续训）。
- **2.10**：`stand_off` 0.44 的 20 局对照。
- **5.10**：bottle 新默认跑整条线。

## 关键路径与命令

```bash
cd /workspace/code/locomanipulation && . scripts/paths.sh && export PYTHONPATH="$LOCOMANI_ROOT/src" TMPDIR=/workspace/tmp
export SONIC_DEPLOY_ROOT="$LOCOMANI_OUTPUTS/bottle_continuous_20260914/deploy"   # 本机 bottle/warehouse 用；crate 不设
bash scripts/cpu_headroom.sh; nvidia-smi; find /workspace/tmp -maxdepth 3 -name stack.json   # 有输出 = 有栈在跑；停栈后等 ≥20 s 再起
RUN=$PWD/outputs/runs/box_room_grasp_200_cap120_20260925; EXP=configs/experiments/box_room_grasp_200_cap120_20260925.yaml

# 4.7.6.1.2：先补 train 条目（评测靠它找 checkpoint），dry-run，再评测（栈/策略 GPU0，镜像 GPU1，结果 $RUN/eval/closed_loop.json）
python3 -c "import json,sys,time; p=sys.argv[1]+'/summary.json'; s=json.load(open(p)); s['train']=dict(checkpoint=sys.argv[1]+'/model/checkpoint-5000', steps=5000, dataset=sys.argv[1]+'/train_dataset', stopped_early='user 2026-09-26: 5000 of 10000', finished=time.strftime('%Y-%m-%d %H:%M:%S')); json.dump(s, open(p,'w'), indent=1, sort_keys=True)" "$RUN"
bash scripts/bimanual/run_vla_inference.sh "$RUN" --dry-run
bash scripts/run_overnight.sh $EXP --gpu 0 --isaac-gpu 1 --line bimanual --from eval

# 整条线无人值守 / 单独补渲；采集中也可以在另一张卡上反复跑补渲
bash scripts/run_overnight.sh <实验文件> --gpu 0 --isaac-gpu 1 --line bimanual
bash scripts/render_dataset_cameras_isaac.sh outputs/runs/<run> --gpu 1 [--dry-run]
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1 "$VENV_SIM_PY" scripts/render_trace_videos.py outputs/runs/<run>   # 第三视角录像

# 本会话脚本（原在 scratchpad，已复制到 outputs/box_room_isaac_20260925/）
python3 outputs/box_room_isaac_20260925/export_warehouse_previews.py outputs/runs/<run> outputs/videos/<run>   # ≤500 KB 预览并核对
python3 outputs/box_room_isaac_20260925/stand_depth_yield.py <run> [<run> ...]                                # 按停位深度分组的成功率
PYTHONPATH=src "$VENV_SIM_PY" outputs/box_room_isaac_20260925/isaac_vs_mujoco_rows.py <run> <session> <ep> <out.jpg> 0,500,1000   # 另有本次编排/停训脚本（pid 写死，只作记录）

# 提交：make check（STATE ≤ 60 行，README 100，warehouse 150）→ 按文件名 add → git commit --no-verify
make check
```
