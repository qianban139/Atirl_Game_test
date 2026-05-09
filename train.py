"""PPO + ICM training script for Atari Donkey Kong.

Usage:
    python train.py                     # PPO+ICM, base config
    python train.py --profile 5090      # PPO+ICM, RTX 5090 config
    python train.py --no-icm            # Ablation: PPO only
    python train.py --resume ckpt_*.pt  # Resume from checkpoint
"""
import argparse
import time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from config import Config
from env_wrapper import make_vec_env
from networks import CNNEncoder, Actor, Critic, InverseDynamics, ForwardDynamics
from ppo_agent import RolloutBuffer, PPOTrainer
from icm_agent import ICMTrainer
from utils import seed_everything, Logger


def train(profile: str = "base", use_icm: bool = True, resume_from: str | None = None):
    # ── Config ──
    if profile == "5090":
        config = Config.preset_5090()
    else:
        config = Config()
    seed_everything(42)
    obs_shape = (4, 84, 84)

    print(f"[Setup] Profile: {profile} | ICM: {use_icm} | Device: {config.device}")
    print(f"[Setup] {config.num_envs} envs x {config.rollout_steps} steps = {config.batch_size} batch")

    # ── Environments ──
    envs = make_vec_env(config.env_name, config.num_envs)

    # ── Networks ──
    encoder = CNNEncoder(feature_dim=config.feature_dim).to(config.device)
    actor = Actor(feature_dim=config.feature_dim, num_actions=config.num_actions).to(config.device)
    critic = Critic(feature_dim=config.feature_dim).to(config.device)
    inverse_model = InverseDynamics(feature_dim=config.feature_dim, num_actions=config.num_actions).to(config.device)
    forward_model = ForwardDynamics(feature_dim=config.feature_dim, num_actions=config.num_actions).to(config.device)

    # ── Trainers ──
    ppo = PPOTrainer(config, encoder, actor, critic)
    icm = ICMTrainer(config, encoder, inverse_model, forward_model) if use_icm else None

    # ── Buffer ──
    buffer = RolloutBuffer(config.rollout_steps, config.num_envs, obs_shape, config.device)

    # ── Logger ──
    logger = Logger()
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    # ── Training state ──
    obs = torch.from_numpy(envs.reset()[0]).float().to(config.device)
    episode_rewards = np.zeros(config.num_envs)
    raw_episode_rewards = np.zeros(config.num_envs)
    episode_lengths = np.zeros(config.num_envs)
    episode_count = 0
    reward_history = deque(maxlen=100)
    total_timesteps = 0

    # ── Resume from checkpoint ──
    if resume_from:
        print(f"[Resume] Loading checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=config.device, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        if use_icm and "inverse_model" in ckpt:
            inverse_model.load_state_dict(ckpt["inverse_model"])
            forward_model.load_state_dict(ckpt["forward_model"])
            icm.optimizer.load_state_dict(ckpt["icm_optimizer"])
        ppo.optimizer.load_state_dict(ckpt["ppo_optimizer"])
        total_timesteps = ckpt.get("total_timesteps", 0)
        episode_count = ckpt.get("episode_count", 0)
        print(f"[Resume] Restored at step {total_timesteps:,}, episode {episode_count}")

    start_time = time.time() - (total_timesteps / 750 if total_timesteps > 0 else 0)  # approximate elapsed
    print(f"[Training] Starting at step {total_timesteps:,}, target {config.total_timesteps:,} timesteps...")

    while total_timesteps < config.total_timesteps:
        # ── Collect rollout ──
        for _ in range(config.rollout_steps):
            with torch.no_grad():
                action, log_prob, value, _ = ppo.get_action_and_value(obs)

            action_np = action.cpu().numpy()
            next_obs, raw_reward, terminated, truncated, _ = envs.step(action_np)
            done = terminated | truncated
            extrinsic_reward = np.clip(raw_reward, -1.0, 1.0)

            next_obs_tensor = torch.from_numpy(next_obs).float().to(config.device)
            reward_tensor = torch.from_numpy(extrinsic_reward).float().to(config.device)
            done_tensor = torch.from_numpy(done).float().to(config.device)

            buffer.insert(obs, action, reward_tensor, value, log_prob, done_tensor, next_obs_tensor)

            episode_rewards += extrinsic_reward
            raw_episode_rewards += raw_reward
            episode_lengths += 1

            for i in range(config.num_envs):
                if done[i]:
                    reward_history.append(raw_episode_rewards[i])
                    logger.log_episode(episode_count, raw_episode_rewards[i],
                                       episode_lengths[i], 0, 0.0, 0.0)
                    episode_rewards[i] = 0
                    raw_episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    episode_count += 1

            obs = next_obs_tensor
            total_timesteps += config.num_envs

        # ── Intrinsic rewards (ICM) ──
        flat_obs = buffer.obs.view(-1, *obs_shape)
        flat_actions = buffer.actions.view(-1)
        flat_next_obs = buffer.next_obs.view(-1, *obs_shape)

        if use_icm:
            intrinsic_rewards = icm.compute_intrinsic_reward(flat_obs, flat_actions, flat_next_obs)
            intrinsic_rewards = intrinsic_rewards.view(config.rollout_steps, config.num_envs)
            buffer.rewards = buffer.rewards + intrinsic_rewards
        else:
            intrinsic_rewards = torch.zeros(config.rollout_steps, config.num_envs, device=config.device)

        # ── GAE ──
        with torch.no_grad():
            last_values = ppo.get_value(obs)
        buffer.compute_gae(last_values, config.gamma, config.lam)

        # ── PPO Update ──
        ppo_stats = ppo.update(buffer)

        # ── ICM Update ──
        if use_icm:
            torch.cuda.empty_cache()
            icm_loss = icm.update(flat_obs, flat_actions, flat_next_obs)
            torch.cuda.empty_cache()
        else:
            icm_loss = 0.0

        # ── Logging ──
        if episode_count > 0 and episode_count % config.log_interval == 0:
            elapsed = time.time() - start_time
            fps = total_timesteps / max(elapsed, 0.001)
            reward_avg = np.mean(reward_history) if reward_history else 0
            icm_str = f"icm={icm_loss:.4f} " if use_icm else ""
            print(f"[{total_timesteps:>10,} steps | {episode_count:>5} eps] "
                  f"reward={reward_avg:>7.0f} "
                  f"policy_loss={ppo_stats['policy_loss']:.4f} "
                  f"value_loss={ppo_stats['value_loss']:.4f} "
                  f"entropy={ppo_stats['entropy']:.4f} "
                  f"{icm_str}"
                  f"fps={fps:.0f}")

            logger.log_scalar("train/policy_loss", ppo_stats["policy_loss"], total_timesteps)
            logger.log_scalar("train/value_loss", ppo_stats["value_loss"], total_timesteps)
            logger.log_scalar("train/entropy", ppo_stats["entropy"], total_timesteps)
            logger.log_scalar("train/fps", fps, total_timesteps)
            if use_icm:
                logger.log_scalar("train/icm_loss", icm_loss, total_timesteps)
                logger.log_scalar("train/intrinsic_reward_mean", intrinsic_rewards.mean().item(), total_timesteps)

        # ── Save checkpoint ──
        if episode_count > 0 and episode_count % config.save_interval == 0:
            ckpt_path = Path("checkpoints") / f"ckpt_{total_timesteps}.pt"
            ckpt_data = {
                "encoder": encoder.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "ppo_optimizer": ppo.optimizer.state_dict(),
                "total_timesteps": total_timesteps,
                "episode_count": episode_count,
            }
            if use_icm:
                ckpt_data["inverse_model"] = inverse_model.state_dict()
                ckpt_data["forward_model"] = forward_model.state_dict()
                ckpt_data["icm_optimizer"] = icm.optimizer.state_dict()
            torch.save(ckpt_data, ckpt_path)
            print(f"[Checkpoint] Saved to {ckpt_path}")

        buffer.clear()

    envs.close()
    elapsed = time.time() - start_time
    print(f"[Training] Complete. Total time: {elapsed / 3600:.1f}h, steps: {total_timesteps:,}")
    print(f"[Training] Episodes: {episode_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["base", "5090"], default="base")
    parser.add_argument("--no-icm", action="store_true", help="Disable ICM (PPO-only ablation)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint.pt to resume from")
    args = parser.parse_args()
    train(profile=args.profile, use_icm=not args.no_icm, resume_from=args.resume)
