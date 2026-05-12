"""DreamerV3 hyperparameters for Atari Beam Rider — aligned with paper spec."""
import torch


class DreamerConfig:
    # ── Environment ──
    env_name = "ALE/BeamRider-v5"
    image_size = 84
    num_actions = 9

    # ── RSSM ──
    rssm_hidden = 1024
    rssm_stoch_categories = 32
    rssm_stoch_classes = 16
    encoder_feat = 1024
    rssm_input = rssm_stoch_categories * rssm_stoch_classes + num_actions  # 521

    @property
    def combined_feat(self):
        return self.rssm_hidden + self.rssm_stoch_categories * self.rssm_stoch_classes  # 1536

    # ── Training ──
    total_env_steps = 5_000_000
    batch_size = 16             # paper default (8 was too small)
    seq_len = 64
    seed_steps = 5000
    wm_updates = 5
    ac_updates = 5
    imagination_horizon = 15
    imagination_starts = 1024

    # ── World Model Loss weights ──
    beta_recon = 1.0
    beta_reward = 1.0
    beta_cont = 1.0
    beta_kl = 0.03             # reduced: let z encode more freely
    kl_alpha = 0.8
    free_nats = 3.0            # 3x: force stochastic state to carry more information
    unimix = 0.01

    # ── Optimizers ──
    wm_lr = 1e-4
    ac_lr = 3e-5
    max_grad_norm = 100.0
    adam_eps = 1e-8

    # ── TwoHot Critic ──
    critic_bins = 255
    critic_low = -20.0
    critic_high = 20.0
    slow_critic_tau = 0.02          # EMA decay per step

    # ── Lambda-return ──
    gamma = 0.997
    lam = 0.95

    # ── Actor-Critic ──
    entropy_eta = 3e-3              # tuned for stability with small action space

    # ── Logging ──
    log_interval = 10
    save_interval = 100

    # ── Buffer ──
    buffer_capacity = 100_000

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def critic_bin_width(self):
        return (self.critic_high - self.critic_low) / (self.critic_bins - 1)
