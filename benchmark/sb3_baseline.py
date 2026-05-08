"""Train a Stable-Baselines3 PPO agent on Donkey Kong as a benchmark baseline."""
import numpy as np
import ale_py
import shimmy
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper


def make_sb3_env(env_name: str, n_envs: int = 8):
    def _init():
        env = gym.make(env_name)
        env = AtariWrapper(env, terminal_on_life_loss=True, clip_reward=False)
        return env

    env = DummyVecEnv([_init for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)
    return env


def train_sb3_baseline(total_timesteps: int = 10_000_000):
    envs = make_sb3_env("ALE/DonkeyKong-v5", n_envs=8)

    model = PPO(
        "CnnPolicy",
        envs,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="logs/",
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save("checkpoints/sb3_ppo_dk")
    print(f"[SB3] Saved to checkpoints/sb3_ppo_dk.zip")

    envs.close()


def evaluate_sb3(num_episodes: int = 10):
    from stable_baselines3.common.atari_wrappers import AtariWrapper
    model = PPO.load("checkpoints/sb3_ppo_dk")
    env = AtariWrapper(gym.make("ALE/DonkeyKong-v5"), terminal_on_life_loss=False)

    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        rewards.append(ep_reward)
        print(f"  Episode {ep + 1}: {ep_reward:.0f}")

    print(f"\n[SB3 Benchmark] Mean: {np.mean(rewards):.0f} ± {np.std(rewards):.0f}")
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    if args.train:
        train_sb3_baseline(args.timesteps)
    if args.eval:
        evaluate_sb3()
