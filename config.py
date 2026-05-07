import torch


class Config:
    # ── Environment ──
    env_name = "ALE/DonkeyKong-v5"
    frame_stack = 4
    image_size = 84
    num_actions = 18          # full Atari action space
    num_envs = 8              # parallel envs for PPO

    # ── PPO ──
    gamma = 0.99
    lam = 0.95                # GAE lambda
    clip_epsilon = 0.1
    rollout_steps = 128       # T: steps per env per batch
    ppo_epochs = 4            # K: update epochs per batch
    minibatch_size = 256      # within each epoch
    lr = 2.5e-4
    value_coef = 0.5          # c1
    entropy_coef = 0.01       # c2 (initial, annealed over training)
    max_grad_norm = 0.5

    # ── ICM ──
    intrinsic_scale = 0.01    # beta
    icm_lr_mult = 0.1         # eta: icm lr = lr * icm_lr_mult
    forward_loss_weight = 0.8
    inverse_loss_weight = 0.2

    # ── Training ──
    total_timesteps = 50_000_000
    save_interval = 500       # episodes between checkpoints
    eval_interval = 1_000_000 # timesteps between evaluations
    log_interval = 100        # episodes between logging

    # ── Network ──
    feature_dim = 256

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Derived ──
    @property
    def batch_size(self):
        return self.num_envs * self.rollout_steps  # 1024
