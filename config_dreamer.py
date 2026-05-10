"""DreamerV3 hyperparameters for Atari Seaquest."""
import torch


class DreamerConfig:
    # ── Environment ──
    env_name = "ALE/Seaquest-v5"
    image_size = 84
    num_actions = 18

    # ── RSSM ──
    rssm_hidden = 512
    rssm_stoch_categories = 32
    rssm_stoch_classes = 16
    encoder_feat = 512
    rssm_input = rssm_stoch_categories * rssm_stoch_classes + num_actions  # 530

    # ── Training ──
    total_env_steps = 1_000_000
    batch_size = 8
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
    beta_kl = 0.1
    kl_alpha = 0.8
    free_nats = 1.0
    unimix = 0.01

    # ── Optimizers ──
    wm_lr = 3e-4
    ac_lr = 3e-5
    max_grad_norm = 0.5

    # ── Lambda-return ──
    gamma = 0.997
    lam = 0.95

    # ── Actor-Critic ──
    entropy_eta = 3e-4

    # ── Logging ──
    log_interval = 10
    save_interval = 100

    # ── Buffer ──
    buffer_capacity = 100_000

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
