import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class RunningMeanStd:
    """Running mean and std for intrinsic reward normalization."""
    def __init__(self, momentum=0.99, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.momentum = momentum
        self.epsilon = epsilon

    def update(self, x: torch.Tensor):
        batch_mean = x.mean().item()
        batch_var = x.var(unbiased=False).item()
        self.mean = self.momentum * self.mean + (1 - self.momentum) * batch_mean
        self.var = self.momentum * self.var + (1 - self.momentum) * batch_var

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / (np.sqrt(self.var) + self.epsilon)


class Logger:
    def __init__(self, log_dir: str = "logs"):
        path = Path(log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(str(path))
        print(f"[Logger] Writing to {path}")

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_episode(self, episode: int, reward: float, length: int,
                    stage: int, entropy: float, intrinsic: float):
        self.writer.add_scalar("episode/reward", reward, episode)
        self.writer.add_scalar("episode/length", length, episode)
        self.writer.add_scalar("episode/stage", stage, episode)
        self.writer.add_scalar("episode/entropy", entropy, episode)
        self.writer.add_scalar("episode/intrinsic_reward", intrinsic, episode)
