# Donkey Kong PPO+ICM Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PPO+ICM agent from scratch in PyTorch to play Donkey Kong at competitive performance for DTS307TC coursework.

**Architecture:** Shared CNN encoder → Actor (action distribution) + Critic (value) + ICM (intrinsic reward). 8 parallel Atari environments → rollout buffer → GAE → PPO clipped update + ICM inverse/forward dynamics update. 50M timesteps on GPU.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, Gymnasium with ALE, Stable-Baselines3 (benchmark only), NumPy, Matplotlib, TensorBoard

---

## File Structure

```
ATRI/
├── config.py              # All hyperparameters in one dataclass
├── env_wrapper.py         # Atari env creation + preprocessing wrappers
├── networks.py            # CNNEncoder, Actor, Critic, InverseDynamics, ForwardDynamics
├── ppo_agent.py           # RolloutBuffer + PPOTrainer (GAE, clipped update)
├── icm_agent.py           # ICMTrainer (intrinsic reward, running stats)
├── train.py               # Main training loop
├── eval.py                # Evaluation + video recording
├── utils.py               # Logger, RunningStats, seed_everything
├── benchmark/
│   └── sb3_baseline.py    # Stable-Baselines3 PPO baseline
├── checkpoints/           # Saved .pt files
├── logs/                  # TensorBoard logs
└── requirements.txt       # Dependencies
```

---

### Task 1: Project Setup & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `config.py`
- Create: `utils.py`

- [ ] **Step 1: Initialize git and write requirements.txt**

```bash
cd /d/projects/ATRI
git init
```

`requirements.txt`:
```
gymnasium[atari]>=1.0.0
gymnasium[accept-rom-license]>=1.0.0
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tensorboard>=2.13.0
stable-baselines3>=2.0.0
opencv-python>=4.8.0
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 3: Write config.py**

```python
import torch


class Config:
    # ── Environment ──
    env_name = "ALE/DonkeyKong-v5"
    frame_stack = 4
    image_size = 84
    num_actions = 18          # full Atari action space
    num_envs = 8              # parallel envs for PPO

    # ── PPO ──
    gamma = 0.99
    lam = 0.95                # GAE lambda
    clip_epsilon = 0.1
    rollout_steps = 128       # T: steps per env per batch
    ppo_epochs = 4            # K: update epochs per batch
    minibatch_size = 256      # within each epoch
    lr = 2.5e-4
    value_coef = 0.5          # c1
    entropy_coef = 0.01       # c2 (initial, annealed over training)
    max_grad_norm = 0.5

    # ── ICM ──
    intrinsic_scale = 0.01    # beta
    icm_lr_mult = 0.1         # eta: icm lr = lr * icm_lr_mult
    forward_loss_weight = 0.8
    inverse_loss_weight = 0.2

    # ── Training ──
    total_timesteps = 50_000_000
    save_interval = 500       # episodes between checkpoints
    eval_interval = 1_000_000 # timesteps between evaluations
    log_interval = 100        # episodes between logging

    # ── Network ──
    feature_dim = 256

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Derived ──
    @property
    def batch_size(self):
        return self.num_envs * self.rollout_steps  # 1024
```

- [ ] **Step 4: Write utils.py**

```python
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
```

- [ ] **Step 5: Verify utils import works**

```bash
python -c "from config import Config; c = Config(); print(f'Device: {c.device}, Batch: {c.batch_size}')"
```
Expected: `Device: cuda, Batch: 1024` (or `cpu` if no GPU)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt config.py utils.py
git commit -m "feat: project setup with config and utilities"
```

---

### Task 2: Environment Wrapper

**Files:**
- Create: `env_wrapper.py`

- [ ] **Step 1: Write env_wrapper.py**

```python
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
```

- [ ] **Step 2: Verify environment output shapes**

```bash
python -c "
from env_wrapper import make_env
import numpy as np

env = make_env('ALE/DonkeyKong-v5')()
obs, info = env.reset()
print(f'Observation shape: {obs.shape}')  # Expected: (4, 84, 84)
print(f'Action space: {env.action_space}')  # Expected: Discrete(18)

# Test step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f'Step obs shape: {obs.shape}')
print(f'Reward: {reward}, Terminated: {terminated}')
env.close()
"
```

- [ ] **Step 3: Verify vectorized environment**

```bash
python -c "
from env_wrapper import make_vec_env
envs = make_vec_env('ALE/DonkeyKong-v5', num_envs=4)
obs, info = envs.reset()
print(f'Vec obs shape: {obs.shape}')  # Expected: (4, 4, 84, 84)
actions = envs.action_space.sample()
obs, rewards, terms, truncs, infos = envs.step(actions)
print(f'Rewards shape: {rewards.shape}')  # Expected: (4,)
print(f'Rewards: {rewards}')
envs.close()
"
```

- [ ] **Step 4: Commit**

```bash
git add env_wrapper.py
git commit -m "feat: add Atari environment wrapper with standard preprocessing"
```

---

### Task 3: Network Definitions

**Files:**
- Create: `networks.py`

- [ ] **Step 1: Write networks.py**

```python
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Shared feature extractor: [B,4,84,84] → [B,256]."""
    def __init__(self, input_channels=4, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        # Compute conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            conv_out = self.conv(dummy)
            conv_out_size = conv_out.shape[1]
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 4, 84, 84] or [B, C, H, W]
        return self.fc(self.conv(x))


class Actor(nn.Module):
    """Policy head: φ(s) → action logits."""
    def __init__(self, feature_dim=256, num_actions=18):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class Critic(nn.Module):
    """Value head: φ(s) → V(s)."""
    def __init__(self, feature_dim=256):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features).squeeze(-1)


class InverseDynamics(nn.Module):
    """ICM inverse model: [φ(s), φ(s')] → predicted action."""
    def __init__(self, feature_dim=256, num_actions=18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_actions),
        )

    def forward(self, phi_s: torch.Tensor, phi_s_next: torch.Tensor) -> torch.Tensor:
        x = torch.cat([phi_s, phi_s_next], dim=1)
        return self.net(x)


class ForwardDynamics(nn.Module):
    """ICM forward model: [φ(s), one_hot(a)] → predicted φ(s')."""
    def __init__(self, feature_dim=256, num_actions=18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + num_actions, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, phi_s: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([phi_s, action_onehot], dim=1)
        return self.net(x)
```

- [ ] **Step 2: Verify network forward passes**

```bash
python -c "
import torch
from networks import CNNEncoder, Actor, Critic, InverseDynamics, ForwardDynamics

B, C, H, W = 4, 4, 84, 84

# Encoder
encoder = CNNEncoder()
x = torch.randn(B, C, H, W)
phi = encoder(x)
print(f'Encoder output: {phi.shape}')  # Expected: [4, 256]

# Actor
actor = Actor()
logits = actor(phi)
print(f'Actor output: {logits.shape}')  # Expected: [4, 18]

# Critic
critic = Critic()
v = critic(phi)
print(f'Critic output: {v.shape}')  # Expected: [4]

# Inverse dynamics
inv = InverseDynamics()
phi_next = torch.randn(B, 256)
a_hat = inv(phi, phi_next)
print(f'Inverse output: {a_hat.shape}')  # Expected: [4, 18]

# Forward dynamics
fwd = ForwardDynamics()
a_onehot = torch.zeros(B, 18); a_onehot[:, 0] = 1
phi_hat = fwd(phi, a_onehot)
print(f'Forward output: {phi_hat.shape}')  # Expected: [4, 256]

# Verify trainable params
total = sum(p.numel() for p in encoder.parameters())
print(f'Encoder params: {total:,}')  # Expected: ~1.2M
print('All shape checks passed.')
"
```

- [ ] **Step 3: Commit**

```bash
git add networks.py
git commit -m "feat: add CNN encoder, actor, critic, and ICM networks"
```

---

### Task 4: PPO Agent (Rollout Buffer + GAE + Update)

**Files:**
- Create: `ppo_agent.py`

- [ ] **Step 1: Write ppo_agent.py — RolloutBuffer**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config


class RolloutBuffer:
    """Stores T×N transitions. Computes GAE on demand."""

    def __init__(self, num_steps: int, num_envs: int, obs_shape: tuple, device: torch.device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.obs_shape = obs_shape

        # Pre-allocate buffers
        self.obs = torch.zeros(num_steps, num_envs, *obs_shape, device=device)
        self.actions = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.next_obs = torch.zeros(num_steps, num_envs, *obs_shape, device=device)

        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

        self.pos = 0
        self.full = False

    def insert(self, obs, actions, rewards, values, log_probs, dones, next_obs):
        idx = self.pos
        self.obs[idx].copy_(obs)
        self.actions[idx].copy_(actions)
        self.rewards[idx].copy_(rewards)
        self.values[idx].copy_(values)
        self.log_probs[idx].copy_(log_probs)
        self.dones[idx].copy_(dones)
        self.next_obs[idx].copy_(next_obs)
        self.pos += 1

    def is_full(self) -> bool:
        return self.pos >= self.num_steps

    def compute_gae(self, last_values: torch.Tensor, gamma: float, lam: float):
        """Compute GAE advantages and returns."""
        advantages = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_values = self.values[t + 1]

            delta = (self.rewards[t]
                     + gamma * next_values * next_non_terminal
                     - self.values[t])
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae

        self.advantages = advantages
        self.returns = advantages + self.values

    def get_minibatches(self, minibatch_size: int):
        """Yield random minibatches as dicts."""
        total = self.num_steps * self.num_envs
        indices = torch.randperm(total, device=self.device)

        # Flatten all buffers to [total, ...]
        flat_obs = self.obs.view(total, *self.obs_shape)
        flat_actions = self.actions.view(total)
        flat_log_probs = self.log_probs.view(total)
        flat_advantages = self.advantages.view(total)
        flat_returns = self.returns.view(total)

        # Advantage normalization per batch
        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std() + 1e-8
        flat_advantages = (flat_advantages - adv_mean) / adv_std

        for start in range(0, total, minibatch_size):
            end = min(start + minibatch_size, total)
            idx = indices[start:end]
            yield {
                "obs": flat_obs[idx],
                "actions": flat_actions[idx],
                "log_probs_old": flat_log_probs[idx],
                "advantages": flat_advantages[idx],
                "returns": flat_returns[idx],
            }

    def clear(self):
        self.pos = 0


class PPOTrainer:
    """Handles PPO clipped surrogate update."""

    def __init__(self, config: Config, encoder: nn.Module, actor: nn.Module,
                 critic: nn.Module):
        self.config = config
        self.encoder = encoder
        self.actor = actor
        self.critic = critic

        self.optimizer = torch.optim.Adam(
            list(encoder.parameters())
            + list(actor.parameters())
            + list(critic.parameters()),
            lr=config.lr,
        )

    def get_action_and_value(self, obs: torch.Tensor):
        """Sample action and return action, log_prob, value."""
        phi = self.encoder(obs)
        logits = self.actor(phi)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(phi)
        entropy = dist.entropy()
        return action, log_prob, value, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        phi = self.encoder(obs)
        return self.critic(phi)

    def update(self, buffer: RolloutBuffer) -> dict:
        """Run K epochs of PPO updates. Returns loss stats."""
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        num_updates = 0

        for _ in range(self.config.ppo_epochs):
            for batch in buffer.get_minibatches(self.config.minibatch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                log_probs_old = batch["log_probs_old"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Forward pass
                phi = self.encoder(obs)
                logits = self.actor(phi)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                values = self.critic(phi)

                # PPO clipped objective
                ratio = torch.exp(log_probs - log_probs_old)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon,
                                    1.0 + self.config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = F.mse_loss(values, returns)

                # Combined loss
                loss = (policy_loss
                        + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy)

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters())
                    + list(self.actor.parameters())
                    + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                num_updates += 1

        for k in stats:
            stats[k] /= num_updates
        return stats
```

- [ ] **Step 2: Quick CartPole smoke test (verifies PPO works on a simple env)**

```bash
python -c "
import gymnasium as gym
import torch
import numpy as np
from config import Config
from networks import CNNEncoder, Actor, Critic
from ppo_agent import RolloutBuffer, PPOTrainer

# Adapt networks for CartPole (MLP instead of CNN for small obs)
class MLPEncoder(torch.nn.Module):
    def __init__(self, obs_dim=4, feature_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, feature_dim), torch.nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)

env = gym.make('CartPole-v1')
obs_dim = 4
n_actions = 2

encoder = MLPEncoder(obs_dim, 64)
actor = Actor(64, n_actions)
critic = Critic(64)

class CartPoleConfig:
    num_envs = 4
    rollout_steps = 32
    ppo_epochs = 4
    minibatch_size = 64
    lr = 3e-4
    gamma = 0.99
    lam = 0.95
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    device = torch.device('cpu')

cfg = CartPoleConfig()
trainer = PPOTrainer(cfg, encoder, actor, critic)
buffer = RolloutBuffer(cfg.rollout_steps, cfg.num_envs, (obs_dim,), cfg.device)

# Quick training loop over a few batches
obs = torch.randn(cfg.num_envs, obs_dim)
total_reward = 0
for step in range(200):
    action, log_prob, value, entropy = trainer.get_action_and_value(obs)
    next_obs = torch.randn(cfg.num_envs, obs_dim)
    reward = torch.zeros(cfg.num_envs)
    done = torch.zeros(cfg.num_envs)
    buffer.insert(obs, action, reward, value, log_prob, done, next_obs)
    obs = next_obs
    if buffer.is_full():
        last_val = trainer.get_value(obs)
        buffer.compute_gae(last_val, cfg.gamma, cfg.lam)
        stats = trainer.update(buffer)
        buffer.clear()

print(f'PPO update stats: {stats}')
print('CartPole smoke test passed.')
"
```

- [ ] **Step 3: Commit**

```bash
git add ppo_agent.py
git commit -m "feat: add PPO trainer with RolloutBuffer, GAE, and clipped update"
```

---

### Task 5: ICM Module

**Files:**
- Create: `icm_agent.py`

- [ ] **Step 1: Write icm_agent.py**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from utils import RunningMeanStd


class ICMTrainer:
    """Intrinsic Curiosity Module: inverse + forward dynamics models."""

    def __init__(self, config: Config, encoder: nn.Module,
                 inverse_model: nn.Module, forward_model: nn.Module):
        self.config = config
        self.encoder = encoder
        self.inverse_model = inverse_model
        self.forward_model = forward_model

        self.optimizer = torch.optim.Adam(
            list(inverse_model.parameters()) + list(forward_model.parameters()),
            lr=config.lr * config.icm_lr_mult,
        )

        self.running_stats = RunningMeanStd(momentum=0.99)

    def compute_intrinsic_reward(self, obs: torch.Tensor,
                                 actions: torch.Tensor,
                                 next_obs: torch.Tensor) -> torch.Tensor:
        """Compute normalized intrinsic reward without computing gradients on encoder."""
        with torch.no_grad():
            phi_s = self.encoder(obs)
            phi_s_next = self.encoder(next_obs)

        # Forward dynamics: predict next state encoding
        action_onehot = F.one_hot(actions, num_classes=self.config.num_actions).float()
        phi_hat_next = self.forward_model(phi_s, action_onehot)

        # Intrinsic reward = prediction error
        pred_error = ((phi_hat_next - phi_s_next) ** 2).mean(dim=1)

        # Normalize with running stats
        self.running_stats.update(pred_error)
        intrinsic_reward = self.config.intrinsic_scale * self.running_stats.normalize(pred_error)

        return intrinsic_reward

    def compute_loss(self, obs: torch.Tensor, actions: torch.Tensor,
                     next_obs: torch.Tensor) -> torch.Tensor:
        """Compute ICM loss (inverse + forward) with gradients."""
        phi_s = self.encoder(obs)
        phi_s_next = self.encoder(next_obs)

        # Inverse dynamics loss
        pred_actions = self.inverse_model(phi_s, phi_s_next)
        inverse_loss = F.cross_entropy(pred_actions, actions)

        # Forward dynamics loss
        action_onehot = F.one_hot(actions, num_classes=self.config.num_actions).float()
        phi_hat_next = self.forward_model(phi_s.detach(), action_onehot)
        forward_loss = F.mse_loss(phi_hat_next, phi_s_next.detach())

        loss = (self.config.inverse_loss_weight * inverse_loss
                + self.config.forward_loss_weight * forward_loss)

        return loss

    def update(self, obs: torch.Tensor, actions: torch.Tensor,
               next_obs: torch.Tensor) -> float:
        """Single ICM update step. Returns loss value."""
        loss = self.compute_loss(obs, actions, next_obs)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.inverse_model.parameters()) + list(self.forward_model.parameters()),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        return loss.item()
```

- [ ] **Step 2: Verify ICM forward pass and intrinsic reward shape**

```bash
python -c "
import torch
from config import Config
from networks import CNNEncoder, InverseDynamics, ForwardDynamics
from icm_agent import ICMTrainer

cfg = Config()
encoder = CNNEncoder()
inv = InverseDynamics()
fwd = ForwardDynamics()
icm = ICMTrainer(cfg, encoder, inv, fwd)

B = 8
obs = torch.randn(B, 4, 84, 84)
actions = torch.randint(0, 18, (B,))
next_obs = torch.randn(B, 4, 84, 84)

# Test intrinsic reward
r_i = icm.compute_intrinsic_reward(obs, actions, next_obs)
print(f'Intrinsic reward shape: {r_i.shape}')  # Expected: [8]
print(f'Intrinsic reward values: {r_i}')
print(f'Running mean: {icm.running_stats.mean:.6f}, std: {icm.running_stats.var**0.5:.6f}')

# Test loss and update
loss = icm.update(obs, actions, next_obs)
print(f'ICM loss: {loss:.4f}')

print('ICM smoke test passed.')
"
```

- [ ] **Step 3: Commit**

```bash
git add icm_agent.py
git commit -m "feat: add ICM trainer with inverse/forward dynamics and intrinsic reward"
```

---

### Task 6: Main Training Loop

**Files:**
- Create: `train.py`

- [ ] **Step 1: Write train.py**

```python
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
```

- [ ] **Step 2: Sanity check — run for 5000 steps to verify no crashes**

```bash
python -c "
# Monkey-patch config for a quick test
from config import Config
Config.total_timesteps = 5000
Config.num_envs = 2
Config.rollout_steps = 32
Config.log_interval = 1
Config.save_interval = 1000000  # don't save during quick test

from train import train
train()
"
```
Expected: Training runs without error, prints loss values, completes in < 2 minutes.

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add main training loop with PPO+ICM integration"
```

---

### Task 7: Evaluation Script

**Files:**
- Create: `eval.py`

- [ ] **Step 1: Write eval.py**

```python
import argparse
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from config import Config
from env_wrapper import make_env
from networks import CNNEncoder, Actor


def evaluate(checkpoint_path: str, num_episodes: int = 10, record: bool = False):
    config = Config()
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
    args = parser.parse_args()

    evaluate(args.checkpoint, args.episodes, args.record)
```

- [ ] **Step 2: Verify eval script loads checkpoint correctly**

```bash
python -c "
from config import Config
from networks import CNNEncoder, Actor
import torch
# Create a dummy checkpoint for testing
encoder = CNNEncoder()
actor = Actor()
dummy = {'encoder': encoder.state_dict(), 'actor': actor.state_dict(), 'total_timesteps': 1000}
torch.save(dummy, 'checkpoints/dummy.pt')
print('Dummy checkpoint saved.')
"

python eval.py checkpoints/dummy.pt --episodes 2
```
Expected: Loads successfully, runs 2 episodes (random actions since untrained), prints results.

- [ ] **Step 3: Commit**

```bash
git add eval.py
git commit -m "feat: add evaluation script with video recording support"
```

---

### Task 8: Benchmark Baseline

**Files:**
- Create: `benchmark/sb3_baseline.py`

- [ ] **Step 1: Write benchmark/sb3_baseline.py**

```python
"""Train a Stable-Baselines3 PPO agent on Donkey Kong as a benchmark baseline."""
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
    import argparse, numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    if args.train:
        train_sb3_baseline(args.timesteps)
    if args.eval:
        evaluate_sb3()
```

- [ ] **Step 2: Verify SB3 baseline imports correctly**

```bash
python -c "from benchmark.sb3_baseline import make_sb3_env; print('SB3 import OK')"
```

- [ ] **Step 3: Commit**

```bash
git add benchmark/sb3_baseline.py
git commit -m "feat: add SB3 PPO baseline for benchmark comparison"
```

---

### Task 9: Full Training Run & Results

- [ ] **Step 1: Start full training**

```bash
python train.py
```

- [ ] **Step 2: Monitor with TensorBoard**

```bash
tensorboard --logdir logs/
```

- [ ] **Step 3: Evaluate best checkpoint after training completes**

```bash
python eval.py checkpoints/ckpt_<best_step>.pt --episodes 100
```

- [ ] **Step 4: Run SB3 baseline (in parallel or after)**

```bash
python benchmark/sb3_baseline.py --train --timesteps 10000000
python benchmark/sb3_baseline.py --eval
```

---

### Task 10: Ablation — PPO only (β=0)

**Files:**
- Create: `train_ablation.py`

- [ ] **Step 1: Write train_ablation.py (copy of train.py with β=0)**

The ablation script is identical to `train.py` except:
```python
config.intrinsic_scale = 0.0  # Disable ICM → PPO-only baseline
```

- [ ] **Step 2: Run ablation training**

```bash
python train_ablation.py
```

- [ ] **Step 3: Generate comparison plot**

```bash
python -c "
import matplotlib.pyplot as plt
# Plot both training curves from TensorBoard logs
# (Manual step: extract CSV from TensorBoard or log data during training)
print('Generate comparison plot from saved metrics.')
"
```

---

### Task 11: Generate Report Figures

**Files:**
- Create: `plot_results.py`

- [ ] **Step 1: Write plot_results.py**

```python
"""Generate publication-quality figures for the coursework report."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_curve(ppo_rewards, icm_rewards, save_path: str):
    """Training curve with rolling mean and std."""
    fig, ax = plt.subplots(figsize=(8, 5))

    def rolling(x, window=100):
        means = np.convolve(x, np.ones(window)/window, mode='valid')
        return means

    ax.plot(rolling(ppo_rewards), label='PPO only', alpha=0.7)
    ax.plot(rolling(icm_rewards), label='PPO + ICM', alpha=0.7)

    ax.set_xlabel('Episodes')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Donkey Kong: PPO vs PPO+ICM')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f'Saved to {save_path}')


def plot_entropy_decay(entropies, save_path: str):
    """Policy entropy over training."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(entropies, color='purple', alpha=0.8)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Policy Entropy')
    ax.set_title('Policy Entropy Decay')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


def plot_stage_progress(stages: list[int], save_path: str):
    """Histogram of stages reached."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(stages, bins=range(1, max(stages)+2), align='left',
            color='steelblue', edgecolor='white')
    ax.set_xlabel('Stage')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Stages Reached')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


if __name__ == "__main__":
    Path("report/figures").mkdir(parents=True, exist_ok=True)
    print("Run after training completes with collected metric data.")
```

---

## Verification Checklist

Before submitting coursework, verify:

- [ ] `python train.py` runs without error for 1M+ steps
- [ ] TensorBoard logs show decreasing policy/value/ICM loss
- [ ] Episode rewards show upward trend (even if slow)
- [ ] `python eval.py checkpoints/<best>.pt --episodes 10` works
- [ ] SB3 baseline produces comparable/worse results than our agent
- [ ] Ablation (β=0) shows ICM contribution
- [ ] All figures generated for report
- [ ] `checkpoints/` contains saved .pt files
- [ ] `requirements.txt` is complete
- [ ] No Stable-Baselines3 code in core algorithm files
