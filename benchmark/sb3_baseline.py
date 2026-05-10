"""SB3 PPO baseline on Seaquest for DreamerV3 comparison."""
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper


def train_sb3(total_timesteps=1_000_000):
    def make():
        env = gym.make("ALE/Seaquest-v5")
        return AtariWrapper(env, terminal_on_life_loss=False)

    env = DummyVecEnv([make for _ in range(8)])

    model = PPO("CnnPolicy", env, n_steps=128, batch_size=256, n_epochs=4,
                learning_rate=2.5e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.1, ent_coef=0.01, vf_coef=0.5,
                max_grad_norm=0.5, tensorboard_log="logs/", verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("checkpoints/sb3_seaquest_ppo")
    print("[SB3] Saved to checkpoints/sb3_seaquest_ppo.zip")
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.train:
        train_sb3(args.timesteps)
