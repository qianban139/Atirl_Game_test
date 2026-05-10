import ale_py
import shimmy
import gymnasium as gym


def make_env(env_name: str, seed: int = 0, render_mode: str | None = None,
             terminal_on_life_loss: bool = True, done_on_life_loss: bool = None,
             frame_stack: int = 4):
    """Create a single Atari environment with standard preprocessing.
    done_on_life_loss: if provided, overrides terminal_on_life_loss (DreamerV3 needs False).
    frame_stack: number of frames to stack (DreamerV3 uses 1 — RSSM handles temporal)."""
    if done_on_life_loss is not None:
        terminal_on_life_loss = done_on_life_loss
    def _init():
        env = gym.make(env_name, frameskip=1, render_mode=render_mode)

        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=terminal_on_life_loss,
            grayscale_obs=True,
        )
        if frame_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, frame_stack)
        return env
    return _init


def make_vec_env(env_name: str, num_envs: int, seed: int = 0):
    """Create vectorized Atari environments."""
    envs = gym.vector.AsyncVectorEnv(
        [make_env(env_name, seed + i) for i in range(num_envs)]
    )
    return envs
