"""Ablation: PPO only (beta = 0, ICM disabled)."""
import time
from pathlib import Path
import numpy as np
import torch
from config import Config
from env_wrapper import make_vec_env
from networks import CNNEncoder, Actor, Critic
from ppo_agent import RolloutBuffer, PPOTrainer
from utils import seed_everything, Logger


def train():
    config = Config()
    config.intrinsic_scale = 0.0  # Disable ICM → PPO-only baseline
    seed_everything(42)

    print(f"[Ablation] PPO only (ICM disabled, beta=0)")
    print(f"[Setup] Device: {config.device}")
    print(f"[Setup] Creating {config.num_envs} parallel environments...")

    envs = make_vec_env(config.env_name, config.num_envs)
    obs_shape = (4, 84, 84)

    encoder = CNNEncoder(feature_dim=config.feature_dim).to(config.device)
    actor = Actor(feature_dim=config.feature_dim, num_actions=config.num_actions).to(config.device)
    critic = Critic(feature_dim=config.feature_dim).to(config.device)

    ppo = PPOTrainer(config, encoder, actor, critic)
    buffer = RolloutBuffer(config.rollout_steps, config.num_envs, obs_shape, config.device)
    logger = Logger()

    obs = torch.from_numpy(envs.reset()[0]).float().to(config.device)
    episode_rewards = np.zeros(config.num_envs)
    episode_lengths = np.zeros(config.num_envs)
    episode_count = 0

    total_timesteps = 0
    start_time = time.time()

    print(f"[Training] Starting {config.total_timesteps:,} timesteps...")

    while total_timesteps < config.total_timesteps:
        for _ in range(config.rollout_steps):
            with torch.no_grad():
                action, log_prob, value, _ = ppo.get_action_and_value(obs)

            action_np = action.cpu().numpy()
            next_obs, extrinsic_reward, terminated, truncated, _ = envs.step(action_np)
            done = terminated | truncated

            extrinsic_reward = np.clip(extrinsic_reward, -1.0, 1.0)

            next_obs_tensor = torch.from_numpy(next_obs).float().to(config.device)
            reward_tensor = torch.from_numpy(extrinsic_reward).float().to(config.device)
            done_tensor = torch.from_numpy(done).float().to(config.device)

            buffer.insert(obs, action, reward_tensor, value, log_prob, done_tensor, next_obs_tensor)

            episode_rewards += extrinsic_reward
            episode_lengths += 1

            for i in range(config.num_envs):
                if done[i]:
                    logger.log_episode(episode_count, episode_rewards[i],
                                       episode_lengths[i], 0, 0.0, 0.0)
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    episode_count += 1

            obs = next_obs_tensor
            total_timesteps += config.num_envs

        # Skip ICM — PPO only uses extrinsic reward directly
        with torch.no_grad():
            last_values = ppo.get_value(obs)
        buffer.compute_gae(last_values, config.gamma, config.lam)

        ppo_stats = ppo.update(buffer)

        if episode_count > 0 and episode_count % config.log_interval == 0:
            elapsed = time.time() - start_time
            fps = total_timesteps / elapsed
            print(f"[{total_timesteps:>10,} steps | {episode_count:>5} eps] "
                  f"policy_loss={ppo_stats['policy_loss']:.4f} "
                  f"value_loss={ppo_stats['value_loss']:.4f} "
                  f"entropy={ppo_stats['entropy']:.4f} "
                  f"fps={fps:.0f}")

            logger.log_scalar("train/policy_loss", ppo_stats["policy_loss"], total_timesteps)
            logger.log_scalar("train/value_loss", ppo_stats["value_loss"], total_timesteps)
            logger.log_scalar("train/entropy", ppo_stats["entropy"], total_timesteps)
            logger.log_scalar("train/fps", fps, total_timesteps)

        if episode_count > 0 and episode_count % config.save_interval == 0:
            ckpt_path = Path("checkpoints") / f"ckpt_ablation_{total_timesteps}.pt"
            torch.save({
                "encoder": encoder.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "ppo_optimizer": ppo.optimizer.state_dict(),
                "total_timesteps": total_timesteps,
                "episode_count": episode_count,
            }, ckpt_path)
            print(f"[Checkpoint] Saved to {ckpt_path}")

        buffer.clear()

    envs.close()
    print(f"[Training] Complete. Total time: {(time.time() - start_time) / 3600:.1f}h")
    print(f"[Training] Episodes: {episode_count}")


if __name__ == "__main__":
    train()
