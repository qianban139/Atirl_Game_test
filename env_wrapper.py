import ale_py
import shimmy
import gymnasium as gym
import numpy as np
import torch


def make_env(env_name: str, seed: int = 0, render_mode: str | None = None):
    """Create a single Atari environment with standard preprocessing."""
    def _init():
        env = gym.make(env_name, render_mode=render_mode)

        # Atari preprocessing wrappers (gymnasium built-in)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,        # random no-ops at start
            frame_skip=4,       # standard frameskip
            screen_size=84,     # resize to 84x84
            terminal_on_life_loss=True,  # life loss = episode boundary
            grayscale_obs=True,
        )
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env
    return _init


def make_vec_env(env_name: str, num_envs: int, seed: int = 0):
    """Create vectorized Atari environments."""
    envs = gym.vector.AsyncVectorEnv(
        [make_env(env_name, seed + i) for i in range(num_envs)]
    )
    return envs
