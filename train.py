import time
from pathlib import Path
import numpy as np
import torch
from config import Config
from env_wrapper import make_vec_env
from networks import CNNEncoder, Actor, Critic, InverseDynamics, ForwardDynamics
from ppo_agent import RolloutBuffer, PPOTrainer
from icm_agent import ICMTrainer
from utils import seed_everything, Logger


def train():
    config = Config()
    seed_everything(42)

    print(f"[Setup] Device: {config.device}")
    print(f"[Setup] Creating {config.num_envs} parallel environments...")

    envs = make_vec_env(config.env_name, config.num_envs)
    obs_shape = (4, 84, 84)  # [stack, H, W]

    # ── Networks ──
    encoder = CNNEncoder(feature_dim=config.feature_dim).to(config.device)
    actor = Actor(feature_dim=config.feature_dim, num_actions=config.num_actions).to(config.device)
    critic = Critic(feature_dim=config.feature_dim).to(config.device)
    inverse_model = InverseDynamics(feature_dim=config.feature_dim, num_actions=config.num_actions).to(config.device)
    forward_model = ForwardDynamics(feature_dim=config.feature_dim, num_actions=config.num_actions).to(config.device)

    # ── Trainers ──
    ppo = PPOTrainer(config, encoder, actor, critic)
    icm = ICMTrainer(config, encoder, inverse_model, forward_model)

    # ── Buffer ──
    buffer = RolloutBuffer(config.rollout_steps, config.num_envs, obs_shape, config.device)

    # ── Logger ──
    logger = Logger()

    # ── Training state ──
    obs = torch.from_numpy(envs.reset()[0]).float().to(config.device)
    episode_rewards = np.zeros(config.num_envs)
    episode_lengths = np.zeros(config.num_envs)
    episode_count = 0
    best_mean_reward = -float("inf")

    total_timesteps = 0
    start_time = time.time()

    print(f"[Training] Starting {config.total_timesteps:,} timesteps...")

    while total_timesteps < config.total_timesteps:
        # ── Collect rollout ──
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

            # Reset terminated envs
            for i in range(config.num_envs):
                if done[i]:
                    logger.log_episode(
                        episode_count,
                        episode_rewards[i],
                        episode_lengths[i],
                        0,  # stage (tracked via info if available)
                        0.0,  # will be logged during update
                        0.0,
                    )
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    episode_count += 1

            obs = next_obs_tensor
            total_timesteps += config.num_envs

        # ── Compute intrinsic rewards and total rewards ──
        flat_obs = buffer.obs.view(-1, *obs_shape)
        flat_actions = buffer.actions.view(-1)
        flat_next_obs = buffer.next_obs.view(-1, *obs_shape)

        intrinsic_rewards = icm.compute_intrinsic_reward(flat_obs, flat_actions, flat_next_obs)
        intrinsic_rewards = intrinsic_rewards.view(config.rollout_steps, config.num_envs)

        buffer.rewards = buffer.rewards + intrinsic_rewards

        # ── GAE ──
        with torch.no_grad():
            last_values = ppo.get_value(obs)
        buffer.compute_gae(last_values, config.gamma, config.lam)

        # ── PPO Update ──
        ppo_stats = ppo.update(buffer)

        # ── ICM Update ──
        icm_loss = icm.update(flat_obs, flat_actions, flat_next_obs)

        # ── Logging ──
        if episode_count > 0 and episode_count % config.log_interval == 0:
            elapsed = time.time() - start_time
            fps = total_timesteps / elapsed
            print(f"[{total_timesteps:>10,} steps | {episode_count:>5} eps] "
                  f"policy_loss={ppo_stats['policy_loss']:.4f} "
                  f"value_loss={ppo_stats['value_loss']:.4f} "
                  f"entropy={ppo_stats['entropy']:.4f} "
                  f"icm_loss={icm_loss:.4f} "
                  f"fps={fps:.0f}")

            logger.log_scalar("train/policy_loss", ppo_stats["policy_loss"], total_timesteps)
            logger.log_scalar("train/value_loss", ppo_stats["value_loss"], total_timesteps)
            logger.log_scalar("train/entropy", ppo_stats["entropy"], total_timesteps)
            logger.log_scalar("train/icm_loss", icm_loss, total_timesteps)
            logger.log_scalar("train/intrinsic_reward_mean", intrinsic_rewards.mean().item(), total_timesteps)
            logger.log_scalar("train/fps", fps, total_timesteps)

        # ── Save checkpoint ──
        if episode_count > 0 and episode_count % config.save_interval == 0:
            ckpt_path = Path("checkpoints") / f"ckpt_{total_timesteps}.pt"
            torch.save({
                "encoder": encoder.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "inverse_model": inverse_model.state_dict(),
                "forward_model": forward_model.state_dict(),
                "ppo_optimizer": ppo.optimizer.state_dict(),
                "icm_optimizer": icm.optimizer.state_dict(),
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
