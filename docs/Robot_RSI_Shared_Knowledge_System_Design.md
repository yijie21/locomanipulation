# Robot RSI 跨任务共享知识系统设计

Shared Schema and Local Adaptation  |  SSLA

设计提案 v0.1  •  2026 年 11 月 11 日

## 1 设计结论与研究目标

采用两层知识架构：共享层保存带适用条件的程序性知识，局部层保存物体、任务和控制器条件下的具体适配。现有 code skills 继续执行动作；知识层决定选择什么技能、如何组合、如何验证，以及失败后优先修改什么。

核心研究问题是自动学习“哪些经验值得共享，哪些经验必须局部保留”。两层划分本身不是充分的 novelty；候选贡献在于利用 action-conditioned 局部 scene graph 的执行证据，自动提升、收窄和拆分知识，并验证是否降低新任务的总学习成本。该收益是待检验假设，尚非实验结论。

| 层级 | 保存内容 | 跨任务复用方式 |
| --- | --- | --- |
| Shared knowledge | 阶段目标、前后置条件、依赖与恢复分支；参数含义、约束及初始化规律 | 匹配物理条件和子目标后实例化，不直接复制世界坐标 |
| Local adaptation | 角色绑定、局部步骤、具体参数、失败配置、例外和效果记录 | 在兼容的物体、场景及控制器上下文中复用 |
| Execution substrate | 有版本的技能代码、控制器、仿真和评估接口 | 执行已选计划，不替代两层知识 |

### 范围与不变项

首版在仿真中测试刚体 pick、transport、place 等技能的跨物体和跨任务组合。冻结基础模型权重、低层控制器及技能 API；允许更新知识、绑定和动作参数。代码进化作为后续独立实验，避免把代码改善混入知识共享收益。

agent 不得修改任务目标、成功判定、数据划分或预算统计。Privileged state 仅用于仿真学习及验证；真机迁移必须另行检验可观测性。此系统首先实现 knowledge-based self-improvement，不据此直接宣称模型权重自进化或严格递归改进。

## 2 组件和知识边界

| 组件 | 输入与输出 | 写入权限 |
| --- | --- | --- |
| Graph extractor | 动作和状态 → 局部对象、关系、事件及变化 | 追加 episode evidence |
| Retriever | 目标和局部图 → 条件兼容的 schema 候选 | 只读已发布版本 |
| Instantiator | schema 和对象角色 → 有阶段判定的执行计划 | 本次计划 |
| Local adapter | 失败证据 → 参数、绑定或局部步骤候选 | 局部知识分支 |
| Executor and verifier | 计划 → 分阶段结果和标准化失败 | 追加不可变日志 |
| Consolidator | 多个局部案例 → 可共享规则及适用范围 | 待验证 candidate |
| Validator and registry | 对照与回归结果 → 发布、拒绝或回滚 | 原子发布共享版本 |

在线路径依次为检索、实例化、局部适配、执行与验证；离线路径汇总证据并验证共享候选。模块是逻辑职责，可由一个 LLM 加确定性程序实现，无需多个独立 agent。

### 共享的单位是有条件的子过程

| 子过程 | 阶段目标或终止条件 | 局部适配示例 |
| --- | --- | --- |
| Approach | 末端到达可达且无碰撞的预抓取位姿 | 抓取部位、接近方向、距离 |
| Grasp and verify | 有效接触；试提后物体跟随且不滑落 | 夹爪开度、力上限、试提高度 |
| Transport | 保持稳定抓取并到达目标入口 | 路径、速度、物体姿态 |
| Place and retreat | 物体被目标支撑；释放后保持稳定 | 放置深度、释放时机、退出方向 |

“所有 container 都从上往下抓”不是可靠共享规则。应显式检查开口可达、外形与夹爪兼容、无阻挡及接触稳定性。带把手容器和无把手容器可以共享“验证抓取”的子过程，而采用不同的抓取分支。物体类别用于召回，物理条件决定适用性。

## 3 两层知识的数据契约

共享层存步骤的目的和约束；局部层绑定实例并填入数值。参数既有可共享的语义和初始化规律，也有上下文相关的具体值。以下为可解析的最小 JSON 示例，数值只是示意，不是已验证的抓取规律。

```json
{
  "shared_schema": {
    "id": "grasp_verify",
    "version": "1.0",
    "status": "candidate",
    "roles": {
      "object": "rigid_object",
      "effector": "gripper"
    },
    "applicability": [
      "reachable(grasp_part)",
      "fits_gripper(grasp_part)"
    ],
    "stages": [
      {
        "id": "grasp",
        "skill_ref": "grasp@1",
        "goal": "valid_contact",
        "requires": [
          "collision_free(approach)"
        ],
        "ensures": [
          "contact(object)"
        ]
      },
      {
        "id": "verify",
        "skill_ref": "test_lift@1",
        "goal": "stable_grasp",
        "requires": [
          "contact(object)"
        ],
        "ensures": [
          "follows_hand(object)",
          "no_slip(object)"
        ]
      }
    ],
    "parameter_contract": {
      "lift_m": {
        "unit": "m",
        "source": "local_adapter"
      }
    },
    "recovery": {
      "verify_failed": "rebind_or_retune_then_retry"
    },
    "evidence_refs": [],
    "counterexample_refs": []
  },
  "local_adaptation": {
    "id": "bottle_07_pick_03",
    "schema_ref": "grasp_verify@1.0",
    "context": {
      "task": "pick",
      "object": "bottle_07",
      "controller": "ctrl@1"
    },
    "bindings": {
      "object": "obj_07",
      "grasp_part": "obj_07.body"
    },
    "parameters": {
      "lift_m": {
        "value": 0.03,
        "bounds": [
          0.01,
          0.05
        ]
      }
    },
    "overrides": [],
    "trial_refs": [
      "episode_014"
    ],
    "status": "unvalidated"
  }
}
```

生产记录还必须包含机器人和仿真版本、生成者、时间、参数搜索历史、适用条件的测试覆盖及证据计数。所有 predicate 均调用有版本的确定性评估器；阈值和时间窗从不可被 agent 修改的评估配置读取。

### 索引与冲突处理

先用技能能力、对象物理属性、目标关系及控制器版本过滤，再做语义检索排序。共享规则设置默认约束；局部 override 必须记录原因并通过检查，不能静默取消安全不变量。条件不满足时返回不适用，不把未知值当作 true。

## 4 动作条件局部 Scene Graph

每条记录只围绕一次动作或阶段，包含目标物体、相关部位、末端、支撑面、邻近障碍及会影响该动作的关系；不维护一个不断膨胀的全局 JSON。局部快照写入 episode 日志，两层知识通过 evidence_refs 引用它。

| 内容 | 字段与用途 |
| --- | --- |
| 节点 | instance_id、语义与物理类别、部位、位姿、尺寸、质量、摩擦、关节状态及来源 |
| 空间与接触关系 | inside、supported_by、contact、clearance、relative_pose；含单位和坐标系 |
| 时间与动作 | 仿真时间、阶段、skill/version/parameters、接触与滑移事件、持续时间 |
| 状态变化与结果 | before/after、关系增删、阶段判定、标准失败码及评估器版本 |
| 数据可信度 | privileged 或 estimated、valid/unknown、估计置信度、对象稳定 ID |

```json
{
  "episode_id": "episode_014",
  "action_id": "verify_01",
  "time": {
    "start_s": 2.0,
    "end_s": 2.4,
    "clock": "sim"
  },
  "action": {
    "skill": "test_lift",
    "version": "1",
    "params": {
      "lift_m": 0.03
    }
  },
  "graph_before": {
    "frame": "world",
    "units": "SI",
    "source": "privileged",
    "nodes": [
      {
        "id": "obj_07",
        "type": "container",
        "rigid": true
      },
      {
        "id": "hand",
        "type": "gripper"
      }
    ],
    "edges": [
      {
        "s": "hand",
        "r": "contact",
        "o": "obj_07"
      }
    ]
  },
  "events": [
    {
      "t_s": 2.23,
      "type": "slip",
      "object": "obj_07"
    }
  ],
  "graph_after": {
    "nodes": [
      {
        "id": "obj_07"
      },
      {
        "id": "hand"
      }
    ],
    "edges": []
  },
  "outcome": {
    "success": false,
    "failure_code": "GRASP_SLIP",
    "evaluator": "verify_grasp@1"
  },
  "snapshot_ref": "snap_014",
  "seed": 14
}
```

样本只列出本动作需要的字段；真实采集器按 skill 的 field contract 补充相对位姿、速度和接触力等。before/after 无法单独反映瞬时滑移或碰撞，因此必须保留执行中事件及时间。由图的变化生成失败假设，再用受控回放验证，不能仅凭相关性宣称原因。

## 5 改进循环与共享知识发布

### 局部学习

1. 为一次学习 campaign 固定共享版本、控制器和评估配置。实例化计划后，先测试从兼容案例检索到的参数初值，并记录复用来源。

2. 失败后按阶段生成诊断候选。区分参数不合适、角色绑定错误、步骤结构缺失和执行能力不足；每次修改尽量限制在一种因素。

3. 从同一动作前快照进行配对回放，保持随机种子和外部条件一致；通过后再测试完整任务。记录全部尝试和成本，失败候选不得覆盖已验证版本。

| 观察到的失败 | 优先修订 | 候选共享经验 |
| --- | --- | --- |
| 抓住后运输中滑落 | 试提、力或姿态；核查接触与负载 | 运输前增加稳定抓取验证 |
| 杯口边缘被误当作把手 | 部位绑定和几何过滤 | 部位选择需要可达与尺寸条件 |
| 深箱内释放后无法退出 | 局部退避路径或增加退出检查 | 放置前验证退避可行性 |
| 夹爪无法覆盖物体 | 能力判定或切换技能 | 缩小适用范围；停止无效调参 |

### 从局部修订到共享候选

4. Consolidator 将重复有效的局部修订抽象为对象角色、关系条件和子目标，保留原始案例及反例。参数值不能直接取平均推广；优先提出依赖尺寸、负载或几何关系的约束或初始化函数。

5. 在未参与候选生成的开发集上，对比旧版和候选版。既测试预期受益任务，也测试边界条件及旧任务。用配对种子、相同预算和预先设定的最小收益及允许退化阈值判定；阈值必须在实验前固定。

6. 通过后原子发布新版本；失败则拒绝、收窄条件或拆分 schema。运行中的 campaign 保持旧版本，下一次检索再迁移。发现回归时撤销 registry 指针并保留日志，不删除反例。

### 约束与停止条件

每个任务配置试验数、仿真步数和 token 上限。达到预算、能力不满足或连续没有改善时停止并保留最好候选。新知识不能修改成功标准。共享提升需要受控验证，不能只因 LLM 自评可信或在一个物体上成功就发布。

## 6 实验验证与实现计划

### 收益假设及实验设置

| 待验证假设 | 对应测量 |
| --- | --- |
| 共享阶段知识减少重新规划 | 新任务达到固定成功率所需总仿真步数 |
| 条件化参数知识缩小搜索 | 有效参数维度、失败尝试及 LLM token 成本 |
| 图变化和事件改善失败定位 | 修订一次后的恢复率及达到成功的试验次数 |
| 收窄与拆分减少负迁移 | 新任务收益、旧任务回归、错误适用率 |

设置 10 个学习任务及独立开发集、最终测试集，评测换物体、换任务及两者联合变化。按物体家族与几何来源划分，避免 mesh 轻微缩放造成泄漏；使用多个任务顺序和随机种子。

比较 code-only、平面经验记忆、人工双层知识和自动双层知识，匹配模型、技能、privileged state 及总预算。消融局部图、条件过滤、知识提升和回归验证；图与文本对照包含等价信息。

总成本包括源任务学习、回放、验证和回归。报告零样本效果及适配曲线；最终测试不参与更新。固定协议后检验收益，不预设改善百分比。

### 最小可实现接口与持久化

| 接口或存储 | 约定 |
| --- | --- |
| retrieve / instantiate | 输入 task_spec 和局部图；返回 schema_ref、bindings、stage_plan |
| run_trial / propose_patch | 输入版本化计划；返回 trace_ref、metrics 或带证据的 patch |
| validate / publish | 返回开发集对照报告；仅 registry 可发布 approved 版本 |
| 持久化 | 数据库保存版本与发布事务；对象存储保存 trace、snapshot、视频和哈希 |

实现顺序：A 打通局部图与固定评估器；B 实现双层检索和局部搜索；C 实现提升、收窄及回滚；D 完成跨任务实验，再考虑代码进化和真机感知。

验收要求：结果可重放、版本可追溯、不适用规则被拒绝、失败候选不发布、旧版本可恢复。研究成功以总成本下降且无不可接受的负迁移为准。
