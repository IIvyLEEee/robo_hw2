# 最小实验计划

基础配置：`diffusion_policy/diffusion_policy/config/image_m7_star1.yaml`

已有 DP 基础 checkpoint 可以复用：

- DP-100：`max_train_episodes=null`，`num_train_timesteps=100`，`num_inference_steps=100`，`n_layer=8`

如果不复用已有 checkpoint，则需要按基础配置再训练一次 DP-100。

## 必训实验

| ID | 用途 | 关键参数覆盖 |
|---|---|---|
| DP-base | baseline, 纯净apple box数据 | 只使用0-365这些episode |
| DP-mix | baseline, apple box数据 | 只使用0-449这些episode |
| DP-full | baseline, 所有数据 | 使用完整0-490这些episode |
| DP-10 | Option B，10% 数据 | `task.dataset.max_train_episodes=49` |
| DP-50 | Option B，50% 数据 | `task.dataset.max_train_episodes=246` |
| DP-trn50 | Q1，diffusion steps | `policy.noise_scheduler.num_train_timesteps=50` |
| DP-inf50 | Q1, infer steps | `policy.num_inference_steps=50` |
| DP-depth4 | Q1，网络深度 | `policy.n_layer=4` |
| BC-100 | Q2，BC baseline | M7 BC 配置，`max_train_episodes=null` |

复用已有 DP-100 时，新训练 5 个实验；不复用时，新训练 6 个实验。

## 仅评估实验

使用 DP-100 checkpoint 做 inference steps 消融：

| ID | 用途 | 关键参数覆盖 |
|---|---|---|
| DP-infer25 | Q1，inference steps | `policy.num_inference_steps=25` |
| DP-infer50 | Q1，inference steps | `policy.num_inference_steps=50` |
| DP-infer100 | Q1，inference steps | 基础设置，`policy.num_inference_steps=100` |

Inference steps 只影响 eval 采样步数，不需要重新训练。

评估步骤

```
CUDA_VISIBLE_DEVICES=1 python script/xbot_policy_server.py -c data/outputs/2026.05.13/08.20.11_image_m7_star1_m7_star1/checkpoints/epoch=1300-train_action_mse_error=0.000.ckpt -d cuda:0 --bind tcp://*:8003
```

```
python scripts/vla_test_xbot.py task=Xbot/XbotPAP vla.server=tcp://127.0.0.1:8003 +app.device=cuda:2 +task.object_select='[1,3]' task.task.params.placement_strategy.params.num_top_objects=1 task.observation.world_cam.camera.use_3dgs=false task.observation.left_cam.camera.use_3dgs=false task.observation.right_cam.camera.use_3dgs=false
```

## 备注

- M7 数据集有 491 条 episode；`49/246/null` 近似对应 10% / 50% / 100%。
- 除表中参数外，其余保持不变：`horizon=10`，`n_obs_steps=2`，`n_action_steps=8`，`learning_rate=1e-4`，`seed=42`。
- BC 复用 `TrainRobomimicImageWorkspace` / `RobomimicImagePolicy`，但 `shape_meta` 和 dataset 要指向 M7（`M7Star1Dataset`，`data/m7.zarr`），保证对比公平。
