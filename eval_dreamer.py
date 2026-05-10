"""Evaluate a trained DreamerV3 checkpoint on Seaquest."""
import argparse
import numpy as np
import torch
import gymnasium as gym
from config_dreamer import DreamerConfig
from env_wrapper import make_env
from networks_dreamer import CNNEncoder, GRUWithLN, Prior, Posterior, ActorHead
from rssm import sample_categorical


def evaluate(checkpoint_path, num_episodes=10, record=False):
    config = DreamerConfig()
    device = config.device

    # Load models
    encoder = CNNEncoder(feat_dim=config.encoder_feat).to(device)
    gru = GRUWithLN(config.rssm_input, config.rssm_hidden).to(device)
    prior = Prior(hidden=config.rssm_hidden, cats=config.rssm_stoch_categories, classes=config.rssm_stoch_classes).to(device)
    post = Posterior(hidden=config.rssm_hidden, feat=config.encoder_feat, cats=config.rssm_stoch_categories, classes=config.rssm_stoch_classes).to(device)
    actor = ActorHead().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Load RSSM sub-modules from the saved rssm state dict
    encoder.load_state_dict({k.replace('encoder.',''): v for k,v in ckpt['rssm'].items() if k.startswith('encoder.')})
    gru.load_state_dict({k.replace('gru.','',1): v for k,v in ckpt['rssm'].items() if k.startswith('gru.')})
    prior.load_state_dict({k.replace('prior.',''): v for k,v in ckpt['rssm'].items() if k.startswith('prior.')})
    post.load_state_dict({k.replace('posterior.',''): v for k,v in ckpt['rssm'].items() if k.startswith('posterior.')})
    actor.load_state_dict(ckpt['actor'])
    encoder.eval(); gru.eval(); prior.eval(); post.eval(); actor.eval()

    render_mode = "rgb_array" if record else None
    env = make_env(config.env_name, done_on_life_loss=False, render_mode=render_mode)()
    if record:
        env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda e: True)

    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset(); done = False; ep_r = 0.0
        h = torch.zeros(1, config.rssm_hidden, device=device)
        z = torch.zeros(1, config.rssm_stoch_categories, config.rssm_stoch_classes, device=device)
        while not done:
            obs_t = torch.from_numpy(obs[-1:].astype(np.float32)/255.0).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = encoder(obs_t)
                z_flat = z.reshape(1, -1)
                a0 = torch.zeros(1, dtype=torch.long, device=device)
                a_onehot = torch.zeros(1, config.num_actions, device=device)
                h = gru(torch.cat([z_flat, a_onehot], dim=-1), h)
                logits = post(h, feat)
                z, _ = sample_categorical(logits, config.unimix)
                features = torch.cat([h, z.reshape(1,-1)], -1)
                action = torch.argmax(actor(features), -1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated; ep_r += reward
        rewards.append(ep_r)
        print(f"  Episode {ep+1}: {ep_r:.0f}")

    env.close()
    print(f"\n[Results] Mean: {np.mean(rewards):.0f} +/- {np.std(rewards):.0f}")
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.episodes, args.record)
