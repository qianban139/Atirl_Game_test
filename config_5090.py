"""RTX 5090 32GB optimized config — scales up parallelism and capacity."""
import torch


class Config5090:
    # ── Environment ──
    env_name = "ALE/DonkeyKong-v5"
    frame_stack = 4
    image_size = 84
    num_actions = 18
    num_envs = 24              # 8→24, 更多并行采样

    # ── PPO ──
    gamma = 0.99
    lam = 0.95
    clip_epsilon = 0.1
    rollout_steps = 256        # 128→256, 每批 6144 条转移
    ppo_epochs = 4
    minibatch_size = 1024      # 256→1024, 更稳定的梯度估计
    lr = 2.5e-4
    value_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5

    # ── ICM ──
    intrinsic_scale = 0.01
    icm_lr_mult = 0.1
    forward_loss_weight = 0.8
    inverse_loss_weight = 0.2

    # ── Training ──
    total_timesteps = 50_000_000
    save_interval = 500
    eval_interval = 1_000_000
    log_interval = 100

    # ── Network ──
    feature_dim = 512          # 256→512, 更多视觉容量

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Derived ──
    @property
    def batch_size(self):
        return self.num_envs * self.rollout_steps  # 6144
