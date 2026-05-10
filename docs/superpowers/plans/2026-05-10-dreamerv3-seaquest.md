# DreamerV3-Lite for Atari Seaquest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement DreamerV3 (world-model-based RL) from scratch for Atari Seaquest — ~1M environment steps, RTX 3090, ~4-8 GPU hours.

**Architecture:** RSSM (GRU+LN + categorical latents) as world model → train on 64-step sequences → imagine 15-step trajectories in latent space → train actor-critic with REINFORCE on imagined data.

**Tech Stack:** Python 3.10+, PyTorch 2.x, Gymnasium 1.x (Atari ALE), Stable-Baselines3 (benchmark only)

---

## File Structure

```
ATRI/
├── config_dreamer.py      # DreamerV3 hyperparameters (NEW)
├── networks_dreamer.py    # CNN Encoder, RSSM, Decoder, all heads (NEW)
├── rssm.py                # RSSM forward pass + world model training (NEW)
├── replay_buffer.py       # Sequence storage + episode-boundary-safe sampling (NEW)
├── imagination.py         # Latent rollout + actor-critic (NEW)
├── train_dreamer.py       # Full training loop (NEW)
├── eval_dreamer.py        # Evaluation + video recording (NEW)
├── utils.py               # REUSE: seed_everything, RunningMeanStd, Logger
└── env_wrapper.py         # MODIFY: add Seaquest + done_on_life_loss=False support
```

---

### Task 1: Project Setup & Configuration

**Files:**
- Create: `config_dreamer.py`

- [ ] **Step 1: Write config_dreamer.py**

```python
"""DreamerV3 hyperparameters for Atari Seaquest."""
import torch


class DreamerConfig:
    # ── Environment ──
    env_name = "ALE/Seaquest-v5"
    image_size = 84
    num_actions = 18

    # ── RSSM ──
    rssm_hidden = 512
    rssm_stoch_categories = 32
    rssm_stoch_classes = 16
    encoder_feat = 512
    rssm_input = rssm_stoch_categories * rssm_stoch_classes + num_actions  # 530

    # ── Training ──
    total_env_steps = 1_000_000
    batch_size = 8            # sequences per WM training batch
    seq_len = 64              # timesteps per sequence
    seed_steps = 5000         # random exploration to seed buffer
    collect_steps = 500       # env steps per iteration
    wm_updates = 5            # world model updates per iteration
    ac_updates = 5            # actor-critic updates per iteration
    imagination_horizon = 15
    imagination_starts = 1024 # number of (h,z) start states

    # ── World Model Loss weights ──
    beta_recon = 1.0
    beta_reward = 1.0
    beta_cont = 1.0
    beta_kl = 0.1
    kl_alpha = 0.8            # prior gets 4x gradient
    free_nats = 1.0           # minimum KL per latent dimension
    unimix = 0.01             # 1% uniform mixture

    # ── Optimizers ──
    wm_lr = 3e-4              # world model (Adam)
    ac_lr = 3e-5              # actor-critic (Adam)
    max_grad_norm = 0.5

    # ── Lambda-return ──
    gamma = 0.997
    lam = 0.95

    # ── Actor-Critic ──
    entropy_eta = 3e-4        # entropy bonus coefficient

    # ── Logging ──
    log_interval = 10         # iterations between logging
    save_interval = 100       # iterations between checkpoints

    # ── Buffer ──
    buffer_capacity = 100_000 # transitions

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- [ ] **Step 2: Verify config imports**

```bash
python -c "from config_dreamer import DreamerConfig; c = DreamerConfig(); print(f'RSSM input: {c.rssm_input}, Device: {c.device}')"
```
Expected: `RSSM input: 530, Device: cuda`

- [ ] **Step 3: Commit**

```bash
git add config_dreamer.py && git commit -m "feat: add DreamerV3 config for Seaquest"
```

---

### Task 2: Environment Wrapper (Seaquest)

**Files:**
- Modify: `env_wrapper.py`

- [ ] **Step 1: Update env_wrapper.py to accept done_on_life_loss**

Replace the `make_env` function signature:

```python
def make_env(env_name: str, seed: int = 0, render_mode: str | None = None,
             terminal_on_life_loss: bool = True, done_on_life_loss: bool = None):
    """Create a single Atari environment with standard preprocessing.
    done_on_life_loss: if provided, overrides terminal_on_life_loss (dreamerv3 needs False)"""
    if done_on_life_loss is not None:
        terminal_on_life_loss = done_on_life_loss
    ...
```

- [ ] **Step 2: Verify Seaquest environment**

```bash
python -c "
import ale_py; import shimmy
from env_wrapper import make_env
env = make_env('ALE/Seaquest-v5', done_on_life_loss=False)()
obs, _ = env.reset()
print(f'Seaquest obs: {obs.shape}, Action space: {env.action_space}')
env.close()
"
```
Expected: `Seaquest obs: (4, 84, 84), Action space: Discrete(18)`

- [ ] **Step 3: Commit**

```bash
git add env_wrapper.py && git commit -m "feat: add done_on_life_loss flag for DreamerV3 Seaquest"
```

---

### Task 3: Neural Networks (Encoder, RSSM, Decoder, Heads)

**Files:**
- Create: `networks_dreamer.py`

- [ ] **Step 1: Write CNNEncoder**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """Encode [B,1,84,84] → [B,512]."""
    def __init__(self, feat_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            out = self.conv(torch.zeros(1, 1, 84, 84))
            conv_dim = out.shape[1]
        self.fc = nn.Sequential(
            nn.Linear(conv_dim, 1024), nn.SiLU(),
            nn.Linear(1024, feat_dim),
        )

    def forward(self, x):
        return self.fc(self.conv(x))
```

- [ ] **Step 2: Write GRUCell + LayerNorm wrapper**

```python
class GRUWithLN(nn.Module):
    """GRUCell with LayerNorm on output."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x, h):
        return self.ln(self.gru(x, h))
```

- [ ] **Step 3: Write Prior and Posterior**

```python
class Prior(nn.Module):
    """p(z_t | h_t). 32×16 categorical logits from h [512]."""
    def __init__(self, hidden=512, cats=32, classes=16):
        super().__init__()
        self.output_dim = cats * classes
        self.net = nn.Sequential(
            nn.Linear(hidden, 256), nn.SiLU(),
            nn.Linear(256, self.output_dim),
        )

    def forward(self, h):
        logits = self.net(h)  # [B, 512]
        return logits.view(-1, 32, 16)  # [B, 32, 16]


class Posterior(nn.Module):
    """q(z_t | h_t, x_t). h [512] + x [512] → categorical logits."""
    def __init__(self, hidden=512, feat=512, cats=32, classes=16):
        super().__init__()
        self.output_dim = cats * classes
        self.net = nn.Sequential(
            nn.Linear(hidden + feat, 256), nn.SiLU(),
            nn.Linear(256, self.output_dim),
        )

    def forward(self, h, x):
        logits = self.net(torch.cat([h, x], dim=-1))
        return logits.view(-1, 32, 16)
```

- [ ] **Step 4: Write Decoder**

```python
class CNNDecoder(nn.Module):
    """Decode [h+z_flat] [B,1024] → [B,1,84,84]."""
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(feat_dim, 3136), nn.SiLU(),
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 1), nn.ReLU(),   # 7→9
            nn.ConvTranspose2d(64, 32, 4, 2), nn.ReLU(),   # 9→20
            nn.ConvTranspose2d(32, 1, 8, 4), nn.Sigmoid(), # 20→84
        )

    def forward(self, features):
        x = self.linear(features)           # [B, 3136]
        x = x.view(-1, 64, 7, 7)           # [B, 64, 7, 7]
        return self.convt(x)                # [B, 1, 84, 84]
```

- [ ] **Step 5: Write prediction heads**

```python
class RewardHead(nn.Module):
    """Predict symlog reward from [h+z]."""
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 1)
    def forward(self, features): return self.fc(features).squeeze(-1)


class ContinueHead(nn.Module):
    """Predict continue logit (BCEWithLogitsLoss) from [h+z]."""
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 1)
    def forward(self, features): return self.fc(features).squeeze(-1)


class ActorHead(nn.Module):
    """Predict action logits from [h+z]."""
    def __init__(self, feat_dim=1024, num_actions=18):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_actions)
    def forward(self, features): return self.fc(features)


class CriticHead(nn.Module):
    """Predict symlog value from [h+z]."""
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 1)
    def forward(self, features): return self.fc(features).squeeze(-1)
```

- [ ] **Step 6: Verify forward pass dimensions**

```bash
python -c "
import torch
from networks_dreamer import *

B = 4
x = torch.randn(B, 1, 84, 84)
h = torch.randn(B, 512)

enc = CNNEncoder()
prior = Prior()
post = Posterior()
dec = CNNDecoder()
rew = RewardHead()
cont = ContinueHead()
actor = ActorHead()
critic = CriticHead()

feat = enc(x)
print(f'Encoder: {x.shape} -> {feat.shape}')           # [4,1,84,84] -> [4,512]

p = prior(h)
print(f'Prior: {h.shape} -> {p.shape}')                 # [4,32,16]

q = post(h, feat)
print(f'Posterior: -> {q.shape}')                       # [4,32,16]

# Straight-through sample
probs = 0.99 * torch.softmax(p, -1) + 0.01*(1/16)
sample = torch.zeros_like(probs).scatter_(-1, probs.argmax(-1, keepdim=True), 1)
z = sample + probs - probs.detach()
z_flat = z.reshape(B, -1)
features = torch.cat([h, z_flat], -1)
print(f'Features: {features.shape}')                    # [4, 1024]

recon = dec(features)
print(f'Decoder: {features.shape} -> {recon.shape}')    # [4, 1, 84, 84]
print(f'Reward: {rew(features).shape}')                 # [4]
print(f'Continue: {cont(features).shape}')              # [4]
print(f'Actor: {actor(features).shape}')                # [4, 18]
print(f'Critic: {critic(features).shape}')              # [4]
print('All shapes OK.')
"
```

- [ ] **Step 7: Commit**

```bash
git add networks_dreamer.py && git commit -m "feat: add DreamerV3 networks (encoder, RSSM, decoder, heads)"
```

---

### Task 4: RSSM Core + World Model Training

**Files:**
- Create: `rssm.py`

- [ ] **Step 1: Write RSSM class with open-loop unroll**

```python
"""RSSM core: forward pass, open-loop unroll, world model training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks_dreamer import Prior, Posterior


def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def sample_categorical(logits, unimix=0.01):
    """Sample from 32 independent categoricals with straight-through and unimix."""
    probs = (1 - unimix) * torch.softmax(logits, dim=-1) + unimix * (1 / logits.shape[-1])
    sample = torch.zeros_like(probs).scatter_(-1, probs.argmax(-1, keepdim=True), 1.0)
    return sample + probs - probs.detach(), probs  # straight-through


class RSSM(nn.Module):
    """Recurrent State-Space Model for DreamerV3."""
    def __init__(self, config, encoder, gru, prior, posterior, decoder, reward_head, continue_head):
        super().__init__()
        self.cfg = config
        self.encoder = encoder
        self.gru = gru
        self.prior = prior
        self.posterior = posterior
        self.decoder = decoder
        self.reward_head = reward_head
        self.continue_head = continue_head

    def initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.cfg.rssm_hidden, device=device)
        z = torch.zeros(batch_size, self.cfg.rssm_stoch_categories, self.cfg.rssm_stoch_classes, device=device)
        return h, z

    def forward_step(self, h, z, a, x=None):
        """Single RSSM step. If x provided, uses posterior (training). Otherwise prior (imagination)."""
        z_flat = z.reshape(z.shape[0], -1)
        a_onehot = F.one_hot(a, self.cfg.num_actions).float()
        h_next = self.gru(torch.cat([z_flat, a_onehot], dim=-1), h)

        if x is not None:
            feat = self.encoder(x)
            logits = self.posterior(h_next, feat)
        else:
            logits = self.prior(h_next)

        z_next, probs = sample_categorical(logits, self.cfg.unimix)
        return h_next, z_next

    def observe_sequence(self, obs, actions, rewards, dones):
        """Open-loop RSSM unroll for world model training. Returns all states and predictions."""
        B, T = obs.shape[0], obs.shape[1]
        device = obs.device

        h = torch.zeros(B, self.cfg.rssm_hidden, device=device)
        z = torch.zeros(B, self.cfg.rssm_stoch_categories, self.cfg.rssm_stoch_classes, device=device)

        hs, zs = [], []
        prior_logits_list, post_logits_list = [], []
        recon_list, reward_pred_list, cont_pred_list = [], [], []

        for t in range(T):
            feat = self.encoder(obs[:, t])
            z_flat = z.reshape(B, -1)
            a_onehot = F.one_hot(actions[:, t], self.cfg.num_actions).float()

            h = self.gru(torch.cat([z_flat, a_onehot], dim=-1), h)
            prior_logits = self.prior(h)
            post_logits = self.posterior(h, feat)

            z, _ = sample_categorical(post_logits, self.cfg.unimix)

            features = torch.cat([h, z.reshape(B, -1)], dim=-1)
            recon = self.decoder(features)
            reward_pred = self.reward_head(features)
            cont_pred = self.continue_head(features)  # raw logit

            hs.append(h)
            zs.append(z)
            prior_logits_list.append(prior_logits)
            post_logits_list.append(post_logits)
            recon_list.append(recon)
            reward_pred_list.append(reward_pred)
            cont_pred_list.append(cont_pred)

        return {
            "h": torch.stack(hs, 1), "z": torch.stack(zs, 1),
            "prior_logits": torch.stack(prior_logits_list, 1),
            "post_logits": torch.stack(post_logits_list, 1),
            "recon": torch.stack(recon_list, 1),
            "reward_pred": torch.stack(reward_pred_list, 1),
            "cont_pred": torch.stack(cont_pred_list, 1),
        }

    def compute_world_model_loss(self, obs, actions, rewards, dones):
        """Compute world model loss for a batch of sequences."""
        out = self.observe_sequence(obs, actions, rewards, dones)
        B, T = obs.shape[0], obs.shape[1]

        # Reconstruction loss
        L_recon = F.mse_loss(out["recon"], obs, reduction='mean')

        # Reward loss (symlog space)
        reward_target = symlog(rewards)
        L_reward = F.mse_loss(out["reward_pred"], reward_target, reduction='mean')

        # Continue loss (BCE with logits, target = 1 - done)
        cont_target = (1 - dones.float())
        L_cont = F.binary_cross_entropy_with_logits(out["cont_pred"], cont_target)

        # KL divergence with balancing and free bits
        q_logits = out["post_logits"]
        p_logits = out["prior_logits"]
        q_dist = torch.distributions.Categorical(logits=q_logits)
        p_dist = torch.distributions.Categorical(logits=p_logits)

        # KL(q||p) per category, summed over categories
        kl = torch.distributions.kl_divergence(q_dist, p_dist).sum(-1)  # [B, T, 32]

        # KL balancing: 80% gradient to prior, 20% to posterior
        kl_prior = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=q_logits.detach()), p_dist
        ).sum(-1)
        kl_post = torch.distributions.kl_divergence(
            q_dist, torch.distributions.Categorical(logits=p_logits.detach())
        ).sum(-1)
        kl_balanced = self.cfg.kl_alpha * kl_prior + (1 - self.cfg.kl_alpha) * kl_post

        # Free bits: minimum KL = 1.0 nat
        kl_clipped = torch.clamp(kl_balanced, min=self.cfg.free_nats)
        L_kl = kl_clipped.mean()

        # Combined loss
        L_wm = (self.cfg.beta_recon * L_recon +
                self.cfg.beta_reward * L_reward +
                self.cfg.beta_cont * L_cont +
                self.cfg.beta_kl * L_kl)

        return L_wm, {"recon": L_recon.item(), "reward": L_reward.item(),
                       "cont": L_cont.item(), "kl": L_kl.item(), "total": L_wm.item()}
```

- [ ] **Step 2: Verify RSSM forward pass**

```bash
python -c "
import torch
from config_dreamer import DreamerConfig
from networks_dreamer import *
from rssm import RSSM

cfg = DreamerConfig()
B, T = 2, 8
obs = torch.randn(B, T, 1, 84, 84)
act = torch.randint(0, 18, (B, T))
rew = torch.randn(B, T)
done = torch.zeros(B, T); done[:, -1] = 1

encoder = CNNEncoder(feat_dim=cfg.encoder_feat)
gru = GRUWithLN(cfg.rssm_input, cfg.rssm_hidden)
prior = Prior(hidden=cfg.rssm_hidden, cats=cfg.rssm_stoch_categories, classes=cfg.rssm_stoch_classes)
posterior = Posterior(hidden=cfg.rssm_hidden, feat=cfg.encoder_feat, cats=cfg.rssm_stoch_categories, classes=cfg.rssm_stoch_classes)
decoder = CNNDecoder()
reward_head = RewardHead()
continue_head = ContinueHead()

rssm = RSSM(cfg, encoder, gru, prior, posterior, decoder, reward_head, continue_head)
loss, stats = rssm.compute_world_model_loss(obs, act, rew, done)
print(f'WM Loss: {stats}')
print('RSSM forward pass OK.')
"
```

- [ ] **Step 3: Commit**

```bash
git add rssm.py && git commit -m "feat: add RSSM core with world model training (open-loop unroll + losses)"
```

---

### Task 5: Replay Buffer

**Files:**
- Create: `replay_buffer.py`

- [ ] **Step 1: Write ReplayBuffer with episode boundary safety**

```python
"""Replay buffer with episode-boundary-safe sequence sampling."""
import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, latent_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.h = np.zeros((capacity, latent_dim), dtype=np.float32)
        self.z = np.zeros((capacity, 32, 16), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, done, h, z):
        idx = self.pos
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.h[idx] = h
        self.z[idx] = z
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.pos

    def sample_sequences(self, batch_size, seq_len, device):
        """Sample sequences that do NOT cross episode boundaries."""
        size = len(self)
        obs_batch = np.zeros((batch_size, seq_len, 1, 84, 84), dtype=np.float32)
        act_batch = np.zeros((batch_size, seq_len), dtype=np.int64)
        rew_batch = np.zeros((batch_size, seq_len), dtype=np.float32)
        done_batch = np.zeros((batch_size, seq_len), dtype=np.float32)

        for b in range(batch_size):
            while True:
                start = random.randint(0, size - seq_len - 1)
                valid = True
                for t in range(seq_len - 1):
                    if self.dones[(start + t) % self.capacity]:
                        valid = False
                        break
                if valid:
                    break

            for t in range(seq_len):
                idx = (start + t) % self.capacity
                obs_batch[b, t] = self.obs[idx].astype(np.float32) / 255.0
                act_batch[b, t] = self.actions[idx]
                rew_batch[b, t] = self.rewards[idx]
                done_batch[b, t] = self.dones[idx]

        return (torch.from_numpy(obs_batch).to(device),
                torch.from_numpy(act_batch).to(device),
                torch.from_numpy(rew_batch).to(device),
                torch.from_numpy(done_batch).to(device))

    def sample_starts(self, n, device):
        """Sample (h, z) pairs from non-terminal states for imagination."""
        size = len(self)
        indices = []
        while len(indices) < n:
            idx = random.randint(0, size - 1)
            if not self.dones[idx]:
                indices.append(idx)

        h_batch = torch.from_numpy(np.stack([self.h[i] for i in indices])).float().to(device)
        z_batch = torch.from_numpy(np.stack([self.z[i] for i in indices])).float().to(device)
        return h_batch, z_batch
```

- [ ] **Step 2: Verify buffer operations**

```bash
python -c "
from replay_buffer import ReplayBuffer
import numpy as np

buf = ReplayBuffer(1000, (1, 84, 84), 512)
h = np.zeros(512, dtype=np.float32); z = np.zeros((32, 16), dtype=np.float32)
for i in range(200):
    buf.add(np.zeros((1,84,84), dtype=np.uint8), 0, 0.0, (i%50==49), h, z)
print(f'Buffer size: {len(buf)}')

obs, act, rew, done = buf.sample_sequences(4, 16, 'cpu')
print(f'Sequence: obs {obs.shape}, act {act.shape}')
print(f'No mid-sequence dones: {(done[:,:-1]==0).all()}')

h_starts, z_starts = buf.sample_starts(8, 'cpu')
print(f'Starts: h {h_starts.shape}, z {z_starts.shape}')
print('Buffer test OK.')
"
```

- [ ] **Step 3: Commit**

```bash
git add replay_buffer.py && git commit -m "feat: add replay buffer with episode-boundary-safe sequence sampling"
```

---

### Task 6: Imagination + Actor-Critic Training

**Files:**
- Create: `imagination.py`

- [ ] **Step 1: Write imagination loop and actor-critic trainer**

```python
"""Imagination: latent rollout + actor-critic training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from rssm import sample_categorical, symlog


def compute_lambda_return(rewards, continues, values, gamma, lam):
    """Compute lambda-return in symlog space. c_hat is continue PROBABILITY (after sigmoid)."""
    H = len(rewards)
    returns = torch.zeros(H, device=rewards.device)
    G = values[-1]
    for t in reversed(range(H)):
        G = rewards[t] + gamma * continues[t] * ((1 - lam) * values[t] + lam * G)
        returns[t] = G
    return returns


def imagine_and_train(rssm, actor, critic, start_h, start_z, config):
    """Imagine H-step trajectories from (start_h, start_z) and train actor-critic."""
    N, H = start_h.shape[0], config.imagination_horizon
    device = start_h.device

    h, z = start_h, start_z

    # Freeze world model params
    wm_params = {}
    for name, p in rssm.named_parameters():
        wm_params[name] = p.requires_grad
        p.requires_grad = False

    # ── Imagination rollout ──
    hs, zs = [h], [z]
    actions, log_probs, rewards_hat, continues_hat, values_hat = [], [], [], [], []

    for t in range(H):
        features = torch.cat([h, z.reshape(N, -1)], dim=-1)
        logits = actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        lp = dist.log_prob(a)

        rew = rssm.reward_head(features)
        cont_logit = rssm.continue_head(features)
        c_hat = torch.sigmoid(cont_logit)
        v = critic(features)

        # RSSM transition: h first, then z from prior
        z_flat = z.reshape(N, -1)
        a_onehot = F.one_hot(a, config.num_actions).float()
        h = rssm.gru(torch.cat([z_flat, a_onehot], dim=-1), h)
        prior_logits = rssm.prior(h)
        z, _ = sample_categorical(prior_logits, config.unimix)

        hs.append(h); zs.append(z)
        actions.append(a); log_probs.append(lp)
        rewards_hat.append(rew); continues_hat.append(c_hat)
        values_hat.append(v)

    # Stack all
    rewards_hat = torch.stack(rewards_hat)  # [H, N]
    continues_hat = torch.stack(continues_hat)
    values_hat = torch.stack(values_hat)
    log_probs = torch.stack(log_probs)
    actions = torch.stack(actions)

    # ── Compute lambda-returns ──
    N_last = torch.cat([h, z.reshape(N, -1)], dim=-1)
    v_last = critic(N_last)
    all_values = torch.cat([values_hat, v_last.unsqueeze(0)])  # [H+1, N]

    returns = torch.zeros(H, N, device=device)
    for b in range(N):
        returns[:, b] = compute_lambda_return(
            rewards_hat[:, b], continues_hat[:, b], all_values[:, b],
            config.gamma, config.lam
        )

    # ── Actor loss (REINFORCE + entropy) ──
    advantages = returns - values_hat.detach()
    L_policy = -(log_probs * advantages.detach()).mean()
    dist_entropy = torch.distributions.Categorical(logits=actor(
        torch.cat([hs[-1], zs[-1].reshape(N, -1)], dim=-1)
    )).entropy().mean()  # approximate; full implementation computes per-step
    L_entropy = -config.entropy_eta * dist_entropy
    L_actor = L_policy + L_entropy

    # ── Critic loss (MSE in symlog space, target detached) ──
    L_critic = F.mse_loss(values_hat, returns.detach())

    # ── Update ──
    ac_optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=config.ac_lr)
    L_ac = L_actor + L_critic
    ac_optimizer.zero_grad()
    L_ac.backward()
    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), config.max_grad_norm)
    ac_optimizer.step()

    # ── Restore world model params ──
    for name, p in rssm.named_parameters():
        p.requires_grad = wm_params[name]
    assert all(p.requires_grad for p in rssm.parameters()), "WM params not restored!"

    return L_ac.item(), L_actor.item(), L_critic.item()
```

- [ ] **Step 2: Verify imagination pipeline**

```bash
python -c "
import torch
from config_dreamer import DreamerConfig
from networks_dreamer import *
from rssm import RSSM
from imagination import imagine_and_train

cfg = DreamerConfig()
cnn_enc = CNNEncoder(feat_dim=cfg.encoder_feat)
gru = GRUWithLN(cfg.rssm_input, cfg.rssm_hidden)
prior = Prior(hidden=cfg.rssm_hidden, cats=cfg.rssm_stoch_categories, classes=cfg.rssm_stoch_classes)
post = Posterior(hidden=cfg.rssm_hidden, feat=cfg.encoder_feat, cats=cfg.rssm_stoch_categories, classes=cfg.rssm_stoch_classes)
decoder = CNNDecoder()
rew_h = RewardHead(); cont_h = ContinueHead()
actor = ActorHead(); critic = CriticHead()

rssm = RSSM(cfg, cnn_enc, gru, prior, post, decoder, rew_h, cont_h)

N = 4
h = torch.randn(N, 512)
z = torch.randn(N, 32, 16)
loss, policy_loss, critic_loss = imagine_and_train(rssm, actor, critic, h, z, cfg)
print(f'Total loss: {loss:.4f}, Policy: {policy_loss:.4f}, Critic: {critic_loss:.4f}')
print('Imagination test OK.')
"
```

- [ ] **Step 3: Commit**

```bash
git add imagination.py && git commit -m "feat: add imagination loop and actor-critic training on latent rollouts"
```

---

### Task 7: Full Training Loop

**Files:**
- Create: `train_dreamer.py`

- [ ] **Step 1: Write train_dreamer.py**

```python
"""DreamerV3 training loop for Atari Seaquest."""
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from config_dreamer import DreamerConfig
from env_wrapper import make_env
from networks_dreamer import (
    CNNEncoder, GRUWithLN, Prior, Posterior, CNNDecoder,
    RewardHead, ContinueHead, ActorHead, CriticHead,
)
from rssm import RSSM, sample_categorical, symlog
from replay_buffer import ReplayBuffer
from imagination import imagine_and_train
from utils import seed_everything, Logger


def evaluate(rssm, actor, config, num_eps=3):
    """Quick evaluation during training."""
    env = make_env(config.env_name, done_on_life_loss=False, render_mode=None)()
    total_rewards = []
    for _ in range(num_eps):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        h = torch.zeros(1, config.rssm_hidden, device=config.device)
        z = torch.zeros(1, config.rssm_stoch_categories, config.rssm_stoch_classes, device=config.device)
        while not done:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(config.device) / 255.0
            h, z = rssm.forward_step(h, z, torch.tensor([0], device=config.device), obs_t)
            features = torch.cat([h, z.reshape(1, -1)], dim=-1)
            with torch.no_grad():
                logits = actor(features)
                action = torch.argmax(logits, dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_rewards.append(ep_reward)
    env.close()
    return np.mean(total_rewards)


def train():
    config = DreamerConfig()
    seed_everything(42)
    print(f"[Setup] Device: {config.device} | Env: {config.env_name}")

    # ── Networks ──
    encoder = CNNEncoder(feat_dim=config.encoder_feat).to(config.device)
    gru = GRUWithLN(config.rssm_input, config.rssm_hidden).to(config.device)
    prior = Prior(hidden=config.rssm_hidden, cats=config.rssm_stoch_categories, classes=config.rssm_stoch_classes).to(config.device)
    post = Posterior(hidden=config.rssm_hidden, feat=config.encoder_feat, cats=config.rssm_stoch_categories, classes=config.rssm_stoch_classes).to(config.device)
    decoder = CNNDecoder().to(config.device)
    reward_head = RewardHead().to(config.device)
    continue_head = ContinueHead().to(config.device)
    actor = ActorHead().to(config.device)
    critic = CriticHead().to(config.device)

    rssm = RSSM(config, encoder, gru, prior, post, decoder, reward_head, continue_head)
    wm_optimizer = torch.optim.Adam(list(rssm.parameters()), lr=config.wm_lr)

    # ── Buffer ──
    buffer = ReplayBuffer(config.buffer_capacity, (1, 84, 84), config.rssm_hidden)

    # ── Logger ──
    logger = Logger("logs/dreamer")
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    # ── Seed collection (random policy) ──
    print(f"[Seed] Collecting {config.seed_steps} random steps...")
    env = make_env(config.env_name, done_on_life_loss=False)()
    obs, _ = env.reset()
    h = np.zeros(config.rssm_hidden, dtype=np.float32)
    z = np.zeros((config.rssm_stoch_categories, config.rssm_stoch_classes), dtype=np.float32)
    for step in range(config.seed_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs_np = obs.astype(np.float32) / 255.0
        # Get posterior latent for this step
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(config.device)
            feat = encoder(obs_t)
            h_t = torch.tensor(h).unsqueeze(0).to(config.device)
            z_t = torch.tensor(z).unsqueeze(0).to(config.device)
            h_t, z_t = rssm.forward_step(h_t, z_t, torch.tensor([action], device=config.device), feat)
        buffer.add(obs, action, reward, done, h_t[0].cpu().numpy(), z_t[0].cpu().numpy())
        obs = next_obs
        if done:
            obs, _ = env.reset()
            h = np.zeros(config.rssm_hidden, dtype=np.float32)
            z = np.zeros((config.rssm_stoch_categories, config.rssm_stoch_classes), dtype=np.float32)
        else:
            h = h_t[0].cpu().numpy()
            z = z_t[0].cpu().numpy()
    env.close()
    print(f"[Seed] Buffer size: {len(buffer)}")

    # ── Main training loop ──
    total_env_steps = config.seed_steps
    iteration = 0
    start_time = time.time()

    print(f"[Training] Starting: target {config.total_env_steps:,} env steps...")

    while total_env_steps < config.total_env_steps:
        iteration += 1

        # ① Collect data (1 episode)
        env = make_env(config.env_name, done_on_life_loss=False)()
        obs, _ = env.reset()
        done = False
        h = torch.zeros(1, config.rssm_hidden, device=config.device)
        z = torch.zeros(1, config.rssm_stoch_categories, config.rssm_stoch_classes, device=config.device)
        ep_reward = 0.0
        while not done:
            obs_t = torch.from_numpy(obs.astype(np.float32)/255.0).unsqueeze(0).to(config.device)
            with torch.no_grad():
                h, z = rssm.forward_step(h, z, torch.tensor([0], device=config.device), obs_t)
                features = torch.cat([h, z.reshape(1, -1)], dim=-1)
                logits = actor(features)
                action = torch.distributions.Categorical(logits=logits).sample().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, reward, done, h[0].cpu().numpy(), z[0].cpu().numpy())
            total_env_steps += 1
            ep_reward += reward
            obs = next_obs
            if done:
                h = torch.zeros(1, config.rssm_hidden, device=config.device)
                z = torch.zeros(1, config.rssm_stoch_categories, config.rssm_stoch_classes, device=config.device)
        env.close()

        # ② Train world model (K_wm updates)
        wm_stats = {"recon": 0, "reward": 0, "cont": 0, "kl": 0}
        for _ in range(config.wm_updates):
            obs_b, act_b, rew_b, done_b = buffer.sample_sequences(config.batch_size, config.seq_len, config.device)
            _, stats = rssm.compute_world_model_loss(obs_b, act_b, rew_b, done_b)
            wm_optimizer.zero_grad()
            (stats["recon"]*config.beta_recon + stats["reward"]*config.beta_reward +
             stats["cont"]*config.beta_cont + stats["kl"]*config.beta_kl).backward()
            nn.utils.clip_grad_norm_(rssm.parameters(), config.max_grad_norm)
            wm_optimizer.step()
            for k in wm_stats: wm_stats[k] += stats[k] / config.wm_updates

        # ③ Train actor-critic (K_ac updates)
        ac_losses = 0.0
        for _ in range(config.ac_updates):
            start_h, start_z = buffer.sample_starts(config.imagination_starts, config.device)
            ac_loss, _, _ = imagine_and_train(rssm, actor, critic, start_h, start_z, config)
            ac_losses += ac_loss / config.ac_updates

        # ④ Logging
        if iteration % config.log_interval == 0:
            elapsed = time.time() - start_time
            eval_rew = evaluate(rssm, actor, config)
            print(f"[{iteration:>5} iters | {total_env_steps:>10,} steps] "
                  f"recon={wm_stats['recon']:.4f} kl={wm_stats['kl']:.2f} "
                  f"ac={ac_losses:.4f} eval_reward={eval_rew:.0f} "
                  f"eps_rew={ep_reward:.0f} time={elapsed/3600:.1f}h")
            logger.log_scalar("train/wm_recon", wm_stats['recon'], total_env_steps)
            logger.log_scalar("train/wm_kl", wm_stats['kl'], total_env_steps)
            logger.log_scalar("train/ac_loss", ac_losses, total_env_steps)
            logger.log_scalar("eval/reward", eval_rew, total_env_steps)

        # ⑤ Save checkpoint
        if iteration % config.save_interval == 0:
            torch.save({
                "rssm": rssm.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "total_env_steps": total_env_steps,
            }, f"checkpoints/dreamer_{total_env_steps}.pt")

    print(f"[Training] Complete. Total time: {(time.time()-start_time)/3600:.1f}h")


if __name__ == "__main__":
    train()
```

- [ ] **Step 2: Smoke test (10 iterations)**

```bash
python -c "
from config_dreamer import DreamerConfig as C; C.total_env_steps = 100; C.seed_steps = 50
from train_dreamer import train; train()
"
```
Expected: Training runs without error, prints log lines.

- [ ] **Step 3: Commit**

```bash
git add train_dreamer.py && git commit -m "feat: add DreamerV3 full training loop with collect-train-imagine cycle"
```

---

### Task 8: Evaluation Script

**Files:**
- Create: `eval_dreamer.py`

- [ ] **Step 1: Write eval_dreamer.py**

```python
"""Evaluate a trained DreamerV3 checkpoint on Seaquest."""
import argparse
import numpy as np
import torch
import gymnasium as gym
from config_dreamer import DreamerConfig
from env_wrapper import make_env
from networks_dreamer import CNNEncoder, GRUWithLN, Prior, Posterior, ActorHead
from rssm import RSSM


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
    encoder.load_state_dict({k.replace('encoder.',''): v for k,v in ckpt['rssm'].items() if k.startswith('encoder.')})
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
            obs_t = torch.from_numpy(obs.astype(np.float32)/255.0).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = encoder(obs_t)
                h = gru(torch.cat([z.reshape(1,-1), torch.zeros(1,18,device=device)], -1), h)
                logits = post(h, feat)
                from rssm import sample_categorical
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
```

- [ ] **Step 2: Commit**

```bash
git add eval_dreamer.py && git commit -m "feat: add DreamerV3 evaluation script"
```

---

### Task 9: SB3 Baseline Comparison

**Files:**
- Create: `benchmark/sb3_dreamer_baseline.py`

- [ ] **Step 1: Write SB3 PPO baseline for Seaquest**

```python
"""SB3 PPO baseline on Seaquest for DreamerV3 comparison."""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper


def train_sb3(total_timesteps=1_000_000):
    def make():
        env = gym.make("ALE/Seaquest-v5")
        return AtariWrapper(env, terminal_on_life_loss=False)

    env = DummyVecEnv([make])
    env = VecFrameStack(env, n_stack=4)

    model = PPO("CnnPolicy", env, n_steps=128, batch_size=256, n_epochs=4,
                learning_rate=2.5e-4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.1, ent_coef=0.01, vf_coef=0.5,
                max_grad_norm=0.5, tensorboard_log="logs/", verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("checkpoints/sb3_seaquest_ppo")
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.train:
        train_sb3(args.timesteps)
```

- [ ] **Step 2: Commit**

```bash
git add benchmark/sb3_dreamer_baseline.py && git commit -m "feat: add SB3 PPO baseline for Seaquest comparison"
```

---

### Task 10: Deploy & Train on GPU Server

- [ ] **Step 1: SCP all files to server**

```bash
scp -P 51822 D:/projects/ATRI/config_dreamer.py D:/projects/ATRI/networks_dreamer.py D:/projects/ATRI/rssm.py D:/projects/ATRI/replay_buffer.py D:/projects/ATRI/imagination.py D:/projects/ATRI/train_dreamer.py D:/projects/ATRI/eval_dreamer.py D:/projects/ATRI/env_wrapper.py root@i-2.gpushare.com:/hy-tmp/ATRI/
```

- [ ] **Step 2: On server, install deps and verify**

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# GPU test
python -c "
from config_dreamer import DreamerConfig
from networks_dreamer import *
print('All imports OK.')
print(f'Device: {DreamerConfig().device}')
"
```

- [ ] **Step 3: Start training**

```bash
tmux new -s dreamer
python train_dreamer.py
```

---

### Task 11: Results & Analysis

- [ ] **Step 1: Evaluate best checkpoint**

```bash
python eval_dreamer.py checkpoints/dreamer_XXXXXX.pt --episodes 100
```

- [ ] **Step 2: Compare with SB3 baseline**

```bash
python benchmark/sb3_dreamer_baseline.py --train --timesteps 1000000
```

- [ ] **Step 3: Generate report figures**

World model reconstructions, training curves, latent space PCA, SB3 comparison.

---

## Verification Checklist

- [ ] `config_dreamer.py` imports without error
- [ ] `networks_dreamer.py` all forward passes produce correct shapes
- [ ] `rssm.py` world model loss decreases over training
- [ ] `replay_buffer.py` sequences respect episode boundaries
- [ ] `imagination.py` actor-critic loss decreases
- [ ] `train_dreamer.py` full training loop runs without crash for 10+ iterations
- [ ] Evaluation produces scores > random baseline
- [ ] SB3 baseline trained for comparison
- [ ] All checkpoints saved successfully
