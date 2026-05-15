# PRD：Diffusion Policy 分析任务

## Task 4：分析

### Q1：超参数分析

选择至少一个 Diffusion Policy 配置中的超参数做控制实验，例如：

- diffusion steps
- learning rate
- 网络宽度或深度
- inference steps

保持其他设置不变，分析该超参数对以下方面的影响：

- 训练稳定性
- 收敛速度
- 最终任务表现，例如 success rate 或 prediction error
- rollout 质量，例如轨迹平滑性和长时程稳定性

### Q2：为什么使用 Diffusion Policy？

结合实验结果，对比 Diffusion Policy 和 Behavior Cloning baseline。

重点讨论：

- 训练稳定性
- 处理多模态行为的能力
- 长时程 rollout 表现

说明 DP 在哪些场景优于 BC，以及二者差距在哪些情况下最明显。

### Q3：失败案例分析

选取代表性失败 rollout，分析失败类型和原因。

常见失败类型：

- 感知错误
- 误差累积或轨迹漂移
- 动作不匹配
- 接触失败、滑落、碰撞等任务相关错误

进一步判断失败主要来自：

- 数据不足或覆盖不全
- 模型能力不足
- rollout 动态导致的分布偏移或误差累积

最后提出可能的改进方向。

## Task 5：高级扩展

任选一个方向完成。

### Option A：Horizon 影响（不选）

系统分析 observation horizon 和 action horizon 对 DP 表现的影响。

要求：

- observation horizon 至少比较两个设置，例如 `To = 1-2` 和 `To = 5-10`
- action horizon 至少比较两个设置，例如短 horizon 和长 horizon
- 分析时间上下文、动作分块对决策质量、轨迹平滑性和长时程稳定性的影响

证据要求：

- success rate 对比，或 final position error plot
- action horizon 分析需要包含 rollout 对比视频
- 可选：jerk、velocity variance 等平滑性指标

### Option B：数据效率分析

比较 DP 在不同训练数据量下的表现。

数据设置：

- 10% 数据
- 50% 数据
- 100% 数据

要求在相同评估设置下比较：

- 训练稳定性
- 最终 success rate
- 泛化能力
- 数据量与性能是否近似线性增长
- 数据不足时的 rollout 质量和失败模式

## 交付检查

- Task 4 的三个问题都有实验或案例支撑
- Task 5 完成 Option A 或 Option B
- 包含定量结果，例如 success rate、prediction error 或 final position error
- 包含定性分析，例如 rollout 观察、视频或失败案例
- 结论需要联系训练表现、模型行为和 rollout 动态
