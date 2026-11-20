# 用到的第三方代码、模型和数据

这个包本身只有胶水代码（`g1grasp/`、`perceive.py`、`plan_grasp.py`、`tools/`、`setup/`）。下面这些都是安装时从原始来源下载的，包里不带；使用前请自己阅读各自的许可。

| 名称 | 用途 | 版本 | 许可 |
|---|---|---|---|
| [SAM 3](https://github.com/facebookresearch/sam3)（Meta） | 按物体名分割目标 | 代码 commit `f92a5ef`；权重 Hugging Face `facebook/sam3` | SAM License。权重需要每个使用者在 Hugging Face 上自己同意许可后下载，**不要转发权重文件** |
| [LingBot-Depth](https://github.com/robbyant/lingbot-depth) | 深度补全 | 代码 commit `f3a237e`；权重 `robbyant/lingbot-depth-pretrain-vitl-14-v0.5` | 代码 Apache-2.0；权重见其模型页 |
| [GraspGen-X](https://github.com/NVlabs/GraspGenX)（NVIDIA） | 生成抓取位姿 | 代码 commit `b942909`；权重 `adithyamurali/GraspGenXModel` | 代码 Apache-2.0；权重 NVIDIA Open Model License |
| GraspGen-X 手模型库 `adithyamurali/gripper_descriptions` | 其中的 `unitree_g1`（Dex3 右手） | — | 该手模型来自 Unitree 的 unitree_ros，BSD-3-Clause |
| Unitree G1 URDF（`g1_29dof_with_hand.urdf`） | `g1grasp/frames.py` 里的相机安装位置、腰部关节、手掌偏移、手指关节限位 | — | Unitree Robotics，BSD-3-Clause |

**示例数据（`examples/`）**
- 来源：SIMPLE 仿真基准的场景，用 Isaac Sim 渲染的 G1 头部相机画面。
- 物体：来自 YCB（饼干盒）和 GraspNet-1Billion 物体库，后者是 CC BY-NC-SA 4.0。
- 用途：只用于检查安装，仅限非商业研究。
