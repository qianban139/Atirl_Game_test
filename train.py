"""DreamerV3 training loop for Atari Seaquest."""
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from config import DreamerConfig
from env_wrapper import make_env
from networks import (
    CNNEncoder, GRUWithLN, Prior, Posterior, CNNDecoder,
    RewardHead, ContinueHead, ActorHead, CriticHead,
)
from rssm import RSSM
from replay_buffer import ReplayBuffer
from imagination import imagine_and_train
from utils import seed_everything, Logger


def evaluate(rssm, actor, config, num_eps=3):
    """Quick evaluation during training."""
    env = make_env(config.env_name, done_on_life_loss=False, render_mode=None)()
    total_rewards = []
    for _ in range(num_eps):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        h = torch.zeros(1, config.rssm_hidden, device=config.device)
        z = torch.zeros(1, config.rssm_stoch_categories, config.rssm_stoch_classes, device=config.device)
        prev_action = 0
        while not done:
            obs_t = torch.from_numpy(obs[-1:].astype(np.float32) / 255.0).unsqueeze(0).to(config.device)
            with torch.no_grad():
                h, z = rssm.forward_step(h, z, torch.tensor([prev_action], device=config.device), obs_t)
                features = torch.cat([h, z.reshape(1, -1)], dim=-1)
                action = torch.argmax(actor(features), dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            prev_action = action
        total_rewards.append(ep_reward)
    env.close()
    return np.mean(total_rewards)


def train(resume_from=None):
    config = DreamerConfig()
    seed_everything(42)
    print(f"[Setup] Device: {config.device} | Env: {config.env_name}")

    # ── Networks ──
    encoder = CNNEncoder(feat_dim=config.encoder_feat).to(config.device)
    gru = GRUWithLN(config.rssm_input, config.rssm_hidden).to(config.device)
    prior = Prior(hidden=config.rssm_hidden, cats=config.rssm_stoch_categories, classes=config.rssm_stoch_classes).to(config.device)
    post = Posterior(hidden=config.rssm_hidden, feat=config.encoder_feat, cats=config.rssm_stoch_categories, classes=config.rssm_stoch_classes).to(config.device)
    decoder = CNNDecoder().to(config.device)
    reward_head = RewardHead().to(config.device)
    continue_head = ContinueHead().to(config.device)
    actor = ActorHead().to(config.device)
    critic = CriticHead().to(config.device)

    rssm = RSSM(config, encoder, gru, prior, post, decoder, reward_head, continue_head)
    wm_optimizer = torch.optim.Adam(rssm.parameters(), lr=config.wm_lr)
    ac_optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=config.ac_lr)

    # ── Resume from checkpoint ──
    if resume_from:
        print(f"[Resume] Loading checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=config.device, weights_only=True)
        rssm.load_state_dict(ckpt["rssm"])
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        if "wm_optimizer" in ckpt:
            wm_optimizer.load_state_dict(ckpt["wm_optimizer"])
        if "ac_optimizer" in ckpt:
            ac_optimizer.load_state_dict(ckpt["ac_optimizer"])
        total_env_steps_prev = ckpt.get("total_env_steps", 0)
        print(f"[Resume] Restored at step {total_env_steps_prev:,}")
    else:
        total_env_steps_prev = 0

    # ── Buffer ──
    buffer = ReplayBuffer(config.buffer_capacity, (1, 84, 84), config.rssm_hidden)

    # ── Logger ──
    logger = Logger("logs/dreamer")
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    # ── Seed collection (skip if resuming — buffer refills from on-policy data) ──
    if not resume_from:
        print(f"[Seed] Collecting {config.seed_steps} random steps...")
        env = make_env(config.env_name, done_on_life_loss=False)()
        obs, _ = env.reset()
        h = torch.zeros(1, config.rssm_hidden, device=config.device)
        z = torch.zeros(1, config.rssm_stoch_categories, config.rssm_stoch_classes, device=config.device)
        for step in range(config.seed_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs_t = torch.from_numpy(obs[-1:].astype(np.float32) / 255.0).unsqueeze(0).to(config.device)
            with torch.no_grad():
                h, z = rssm.forward_step(h, z, torch.tensor([action], device=config.device), obs_t)
            buffer.add(obs[-1:], action, reward, done, h[0].cpu().numpy(), z[0].cpu().numpy())
            obs = next_obs
            if done:
                obs, _ = env.reset()
                h = torch.zeros(1, config.rssm_hidden, device=config.device)
                z = torch.zeros(1, config.rssm_stoch_categories, config.rssm_stoch_classes, device=config.device)
        env.close()
        print(f"[Seed] Buffer size: {len(buffer)}")

    # ── Main training loop ──
    total_env_steps = max(config.seed_steps, total_env_steps_prev)
    iteration = 0
    start_time = time.time()
    print(f"[Training] Starting: target {config.total_env_steps:,} env steps...")

    while total_env_steps < config.total_env_steps:
        iteration += 1

        # (1) Collect 1 episode
        env = make_env(config.env_name, done_on_life_loss=False)()
        obs, _ = env.reset()
        done = False
        h_col = torch.zeros(1, config.rssm_hidden, device=config.device)
        z_col = torch.zeros(1, config.rssm_stoch_categories, config.rssm_stoch_classes, device=config.device)
        ep_reward = 0.0
        prev_action = 0  # initial NOOP before first step
        while not done:
            obs_t = torch.from_numpy(obs[-1:].astype(np.float32) / 255.0).unsqueeze(0).to(config.device)
            with torch.no_grad():
                h_col, z_col = rssm.forward_step(h_col, z_col, torch.tensor([prev_action], device=config.device), obs_t)
                features = torch.cat([h_col, z_col.reshape(1, -1)], dim=-1)
                logits = actor(features)
                action = torch.distributions.Categorical(logits=logits).sample().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs[-1:], action, reward, done, h_col[0].cpu().numpy(), z_col[0].cpu().numpy())
            total_env_steps += 1
            ep_reward += reward
            obs = next_obs
            prev_action = action
            if done:
                h_col = torch.zeros(1, config.rssm_hidden, device=config.device)
                z_col = torch.zeros(1, config.rssm_stoch_categories, config.rssm_stoch_classes, device=config.device)
                prev_action = 0
        env.close()

        # (2) Train world model (K_wm updates)
        wm_stats = {"recon": 0.0, "reward": 0.0, "cont": 0.0, "kl": 0.0}
        for _ in range(config.wm_updates):
            obs_b, act_b, rew_b, done_b = buffer.sample_sequences(
                config.batch_size, config.seq_len, config.device)
            loss_wm, stats = rssm.compute_world_model_loss(obs_b, act_b, rew_b, done_b)
            wm_optimizer.zero_grad()
            loss_wm.backward()
            nn.utils.clip_grad_norm_(rssm.parameters(), config.max_grad_norm)
            wm_optimizer.step()
            for k in wm_stats: wm_stats[k] += stats[k] / config.wm_updates

        # (3) Train actor-critic (K_ac updates)
        ac_total, ac_policy, ac_critic = 0.0, 0.0, 0.0
        for _ in range(config.ac_updates):
            start_h, start_z = buffer.sample_starts(config.imagination_starts, config.device)
            loss_ac, loss_p, loss_c = imagine_and_train(rssm, actor, critic, ac_optimizer, start_h, start_z, config)
            ac_total += loss_ac / config.ac_updates
            ac_policy += loss_p / config.ac_updates
            ac_critic += loss_c / config.ac_updates

        # (4) Logging
        if iteration % config.log_interval == 0:
            elapsed = time.time() - start_time
            eval_rew = evaluate(rssm, actor, config)
            print(f"[{iteration:>5} iters | {total_env_steps:>10,} steps] "
                  f"R={eval_rew:>5.0f} ep={ep_reward:>5.0f} "
                  f"recon={wm_stats['recon']:.3f} kl={wm_stats['kl']:.1f} "
                  f"ac_p={ac_policy:.3f} ac_c={ac_critic:.3f} "
                  f"t={elapsed/3600:.1f}h")
            logger.log_scalar("eval/reward", eval_rew, total_env_steps)
            logger.log_scalar("train/ep_reward", ep_reward, total_env_steps)
            logger.log_scalar("train/wm_recon", wm_stats["recon"], total_env_steps)
            logger.log_scalar("train/wm_kl", wm_stats["kl"], total_env_steps)
            logger.log_scalar("train/ac_total", ac_total, total_env_steps)

        # (5) Save checkpoint
        if iteration % config.save_interval == 0:
            ckpt = {
                "rssm": rssm.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "wm_optimizer": wm_optimizer.state_dict(),
                "ac_optimizer": ac_optimizer.state_dict(),
                "total_env_steps": total_env_steps,
            }
            torch.save(ckpt, f"checkpoints/dreamer_{total_env_steps}.pt")
            print(f"[Checkpoint] Saved at step {total_env_steps:,}")

    envs_total_time = (time.time() - start_time) / 3600
    print(f"[Training] Complete. Total time: {envs_total_time:.1f}h, steps: {total_env_steps:,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(resume_from=args.resume)
