import argparse
from pathlib import Path
import numpy as np
import torch
import ale_py
import shimmy
import gymnasium as gym
from config import Config
from config_5090 import Config5090
from env_wrapper import make_env
from networks import CNNEncoder, Actor


def evaluate(checkpoint_path: str, num_episodes: int = 10, record: bool = False,
             use_5090: bool = False):
    config = Config5090() if use_5090 else Config()
    device = config.device

    # Load model
    encoder = CNNEncoder(feature_dim=config.feature_dim).to(device)
    actor = Actor(feature_dim=config.feature_dim, num_actions=config.num_actions).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    encoder.eval()
    actor.eval()

    print(f"[Eval] Loaded checkpoint from step {ckpt.get('total_timesteps', '?')}")

    render_mode = "rgb_array" if record else None
    env = make_env(config.env_name, render_mode=render_mode)()
    if record:
        env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda e: True)

    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                phi = encoder(obs_tensor)
                logits = actor(phi)
                action = torch.argmax(logits, dim=1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        print(f"  Episode {ep + 1}: reward={ep_reward:.0f}, length={ep_length}")

    env.close()

    print(f"\n[Results] Mean reward: {np.mean(episode_rewards):.0f} ± {np.std(episode_rewards):.0f}")
    print(f"[Results] Mean length: {np.mean(episode_lengths):.0f} ± {np.std(episode_lengths):.0f}")

    return episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--5090", action="store_true", dest="use_5090",
                        help="Use 5090 config (feature_dim=512)")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.episodes, args.record, args.use_5090)
