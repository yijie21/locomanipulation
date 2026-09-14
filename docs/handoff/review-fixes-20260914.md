# 2026-09-14 外部提交审查后的修复

本次补丁基于远端 main `347da0154de46a7fbca46c40ee9c16712c06dbdb`，修复
`4c15d42`、`35ad898`、`347da01` 的审查问题。保留原作者的提交和功能。
工作分支为 `fix/review-followups-20260914`；日常修改仍从 [change map](../reference/change-map.md) 进入。

## 改了什么

- 渲染生命周期由 `pipeline/render_job.py` 管理。启动时生成任务标识，子进程继承；
  输出目录锁防止并发覆盖，孤立 worker 仍阻止目录复用。停止只选择同一用户、同一任务的进程，
  关键字仅筛选输出目录，匹配多个任务时拒绝。用 pidfd 绑定已核对的进程身份，避免 PID 复用和
  进程组广播误伤；暂停并枚举后再终止。仿真 Python 缺少 pidfd 包装时通过 libc 调用同一 Linux API。
- Isaac 多相机读取不再保留前一次步进的旧图。三路相机均需预热，每帧必须更新且具有相同
  `rendering_frame`；缺失或过期的相机使该集明确失败。稳定性检查覆盖三路相机。
  批处理在渲染器失败时立即退出，不再继续合成视频或打印 `RENDER_ALL_DONE`。
- `crate_dy_range_m` 拒绝非有限值、非数值、布尔值、错误长度和逆序区间。
  配置从 CLI 显式传给每轮 `WallRoundContext`，实际 reset 直接消费，不再新增模块全局变量。
- 从 `df2a80b` 恢复被删除的 Carry motion continuity、Table-height validation、Whole-body
  low-reach pilot 和 ghost hands 解释；保留新画中画功能说明，修正停止流程和旧资产安装指引。
  恢复低桌文档锚点，更新 change map；修正 `test_head_camera_spec.py` 的格式。

## 验证与证据

原始审查复现保存在 `outputs/review_347da01_20260914/`。例如旧 `_render` 在右手腕相机
始终无图时仍写出 2/2 帧并报告 `done`；本次针对同一消费者增加失败回归。
本次所有日志及原始录像保存在 `outputs/review_fixes_20260914/`。

- `make check` 通过：格式、lint、配置文档新鲜度、仓库导航和完整默认 CPU 测试。
  默认跳过的仿真测试不能算作通过；真实 GPU 验证见下文。
- `test_render_job.py` 用测试自己启动的进程检查两任务隔离、同组无关进程、独立进程组 worker、
  孤立 worker、输出目录占用、PID 身份变化，以及批处理传递渲染错误。5 项通过。
- `test_isaac_capture.py` 和 `test_isaac_inset_render.py` 覆盖旧图、帧号不同、预热缺相机、
  运行中丢相机、三路稳定性和最终像素布局/源帧顺序。与既有 inset 测试一起 13 项通过。
- `test_crate_task_config.py` 经真实 CLI → round → reset 检查 +3.5 / −2.5 / +3.5 cm
  连续三次配置交付，另覆盖默认、序列化、合法覆盖和非法输入。

GPU 0 / instance 1 的真实采集命令（从候选工作树执行，`LOCOMANI_ROOT` 指向该工作树）：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u -m locomani.pipeline.run_crate_pilot \
  --expert wall --rounds 1 --seed 41 --gpu 0 --instance 1 \
  --scene /workspace/code/locomanipulation/outputs/scene/crate_bimanual \
  --set crate_dy_range_m=0.035,0.035 \
  --out /workspace/code/locomanipulation/outputs/review_fixes_20260914/wall_sampling
```

`wall_sampling/task_config.json` 保存 `[0.035, 0.035]`；`manifest_00.jsonl` 的实际抽样
`crate_dy=0.035`，本集 `ok=true`、`rise=0.12634 m`，双手接触保持，deploy 重启为 0。
这是默认 ready-stand 诊断，不是走入专家或 VLA 数据集验证；单集成功不能证明成功率提升。
起栈、expert、停栈均退出 0，sim/deploy 日志另存于 `wall_sampling/stack_logs/`。

随后用 Isaac Sim 4.5.0 对该集完整轨迹验证两个渲染器：

```bash
LOCOMANI_SIM_DIR="$V/crate_bimanual" CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  MUJOCO_RENDER_ARGS=--insets ISAAC_RENDER_ARGS=--insets \
  bash scripts/render_wall_pilot.sh "$V" "$R" 1 wall_s41
```

其中 `$V=outputs/review_fixes_20260914/wall_sampling`、`$R=outputs/review_fixes_20260914/render`，
`RENDER_PY` 为已安装 Isaac 4.5 的 SIMPLE venv。候选工作树临时引用主工作树已构建的 Isaac 资产；
该临时链接已移除，源资产保留。`render/r00/render.log` 记录 622 状态、stride 4，主视角
156/156 帧、三路画中画 156/156 帧，批处理正常结束。首尾画面检查确认房间、桌子、机器人及
三路 inset 都存在。预览 `previews/sbs_round00.mp4` 为 409,188 字节、156 帧、12.48 秒，
完整解码通过，与原片时长相同；原始 3,282,305 字节并排视频保留。

`validation_identity.json` 记录基线、改变的源代码/测试、场景、配置、机器人 USD、SONIC 二进制的 SHA256。
改变的代码/测试哈希表（按 key 排序的 JSON）的 SHA256：
`2ee8f5b493c9929d6eb0d43ed59306a32a3343a6a08ed85e1223a743d36ea2a9`。

## 范围与后续

旧版或直接启动的未标记渲染进程不在新停止命令的管理范围，见
[停止渲染](../crate_collection.md#中途停止渲染--清理)。当前验证环境为 Linux + Isaac 4.5，
未声明覆盖所有 Isaac 版本。相机间帧号一致不等于与 MuJoCo 像素完全一致。
本次未修改物理控制时序、接触/成功判据或 VLA 模态，没有运行新的 bottle 行为比较。
已有 crate 成功率、动作稳定性和正式 VLA 导出限制保留原结论。
GPU 0 的工作已结束，instance 1 锁已释放；GPU 1 的其它任务未触碰。
