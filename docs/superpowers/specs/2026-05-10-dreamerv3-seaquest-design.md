# DreamerV3-Lite for Atari Seaquest — Design Document

## Overview

Implement a simplified DreamerV3 (Hafner et al., 2023) — a world-model-based RL agent — for the Atari game Seaquest. Built from scratch in PyTorch. DreamerV3 learns an internal "world model" of the game and trains the agent entirely inside its own imagination, achieving sample efficiency 100-500× higher than model-free methods like PPO.

## Why Seaquest

Seaquest is an ideal testbed for world models because the RSSM's latent state must independently encode multiple physically unrelated variables:
- Oxygen level (continuous, monotonic decrease)
- Submarine position (2D continuous)
- Enemy count and types (discrete)
- Diver positions and rescue status (discrete)

These variables form a clean set of "ground truth" labels for probing the latent space — you can train linear classifiers on z_t to verify what the world model learned without supervision. This is the core analysis contribution for the coursework report.

## Why DreamerV3 Over PPO

| Concern | PPO+ICM (previous attempt) | DreamerV3 |
|---------|---------------------------|-----------|
| Sample efficiency | 50M steps | **1M steps** |
| GPU time | ~35 hours | **~4-8 hours** |
| Entropy collapse | Recurring problem | KL balancing + free bits prevent it |
| Exploration | ICM auxiliary bonus | World model instrinsic motivation |
| Report novelty | Standard method | Cutting-edge (2023) |

---

## Architecture

### Component Overview

```
Seaquest Atari frame [1, 84, 84]
         │
    CNN Encoder → x_t [512]
         │
    ┌────┴──────────────────────────┐
    │  RSSM (Recurrent State-Space) │
    │                               │
    │  h_t = GRU+LN(h_{t-1}, z_{t-1}, a_{t-1})  [512]  │
    │  z_t ~ q(z | h_t, x_t)        [32 cats × 16 classes] │
    │                               │
    │  Prior:     p(ẑ | h)          │
    │  Posterior: q(z | h, x)       │
    └────────────┬──────────────────┘
                 │  (h_t, z_t flat) [1024]
    ┌────────────┼──────────────────┐
    │            │                  │
    ▼            ▼            ▼     ▼     ▼
  Decoder    Reward MLP   Cont MLP  Actor  Critic
  [1,84,84]  [1] symlog   [1] sig   [18]   [1] symlog
```

### CNN Encoder

```
Input: [B, 1, 84, 84]
Conv2d(1→32, k=8, s=4) → ReLU → [B, 32, 20, 20]
Conv2d(32→64, k=4, s=2) → ReLU → [B, 64, 9, 9]
Conv2d(64→64, k=3, s=1) → ReLU → [B, 64, 7, 7]
Flatten → [B, 3136]
Linear(3136→1024) → SiLU → [B, 1024]
Linear(1024→512) → [B, 512]  ← x_t
```

### RSSM Core

| Component | Architecture | Input | Output |
|-----------|-------------|-------|--------|
| GRU + LayerNorm | GRUCell(530, 512) + LN | [z_{t-1} flat (512) + a_{t-1} one-hot (18)] | h_t [512] |
| Prior | Linear(512→256) → SiLU → Linear(256→32×16) | h_t [512] | logits [32,16] |
| Posterior | Linear(1024→256) → SiLU → Linear(256→32×16) | [h_t (512) + x_t (512)] | logits [32,16] |

### Straight-Through Categorical Sampling

```python
# 32 categories, each independently 16-way categorical
# unimix: 1% uniform prevents log-prob saturation
probs = 0.99 * softmax(logits, dim=-1) + 0.01 * (1/16)
sample = one_hot(categorical(probs))           # [B, 32, 16]
z = sample + probs - probs.detach()            # straight-through
z_flat = z.reshape(B, 512)                     # flatten for concatenation
```

### Prediction Heads

| Head | Architecture | Input | Output |
|------|-------------|-------|--------|
| Decoder | Linear(1024→3136) → Reshape(64,7,7) → ConvT(64→64,k=3,s=1)→[9,9] → ConvT(64→32,k=4,s=2)→[20,20] → ConvT(32→1,k=8,s=4)→[84,84] | [h+z_flat] | [1,84,84] |
| Reward | Linear(1024→1) | [h+z_flat] | scalar (symlog) |
| Continue | Linear(1024→1) → Sigmoid | [h+z_flat] | ĉ ∈ [0,1] |
| Actor | Linear(1024→18) | [h+z_flat] | action logits |
| Critic | Linear(1024→1) | [h+z_flat] | scalar (symlog) |

### Parameter Count

| Component | Approx. params |
|-----------|---------------|
| CNN Encoder | ~3.8M |
| GRU + LN | ~1.6M |
| Prior + Posterior MLPs | ~0.8M |
| Decoder CNN | ~1.5M |
| 4 prediction heads | ~0.01M |
| **Total** | **~7.7M** |

---

## World Model Training

### Replay Buffer

- Stores sequences of T=64 consecutive transitions
- Pool size: 100,000 sequences (rolling FIFO)
- Each stored element: (obs, action, reward, done)

### Open-Loop RSSM Unroll

For a batch of B=8 sequences × T=64 steps:

```
for t = 0 to 63:
    h_t = GRU+LN(h_{t-1}, z_{t-1}, a_{t-1})
    p_prior  = Prior(h_t)
    p_post   = Posterior(h_t, Encoder(x_t))
    z_t ~ p_post                                   # training: posterior
    x̂_t     = Decoder(h_t, z_t)
    r̂_t     = RewardMLP(h_t, z_t)
    ĉ_t     = ContinueMLP(h_t, z_t)
```

### World Model Losses

```
L_recon  = (1/BT) Σ MSE(x̂_t, x_t)                          # pixel reconstruction
L_reward = (1/BT) Σ MSE(r̂_t, symlog(r_t_actual))           # reward in symlog
L_cont   = (1/BT) Σ BCE(ĉ_t, done_t)                       # terminal prediction

KL_balanced = 0.8 * KL(sg[q] || p) + 0.2 * KL(q || sg[p]) # KL balancing
KL_clipped  = max(KL_balanced, 1.0)                        # free bits

L_wm = β_recon·L_recon + β_reward·L_reward + β_cont·L_cont + β_kl·KL_clipped
```

| Hyperparameter | Value | Justification |
|---------------|-------|---------------|
| β_recon | 1.0 | Primary signal |
| β_reward | 1.0 | Equal weight |
| β_cont | 1.0 | Helps value prediction |
| β_kl | 0.1 | KL dominates numerically (~90x recon); free bits prevent collapse |
| KL α | 0.8 | Asymmetric: prior gets 4× gradient |
| Free bits | 1.0 nat | Guarantees z carries information |
| Unimix | 1% | Prevents categorical saturation |
| B (WM batch) | 8 | Reduced from 16 to avoid OOM on 24GB |
| T (sequence) | 64 | Standard DreamerV3 |

---

## Imagination and Actor-Critic

### Imagination Loop (H=15 steps)

Starting from 1024 posterior states (h_0, z_0) sampled from the replay buffer:

```
for t = 0 to 14:
    a_t ~ Actor(h_t, z_t)                   # With gradient
    # World model forward pass — run normally so gradients flow through a_t into h_{t+1}
    # WM parameters are frozen via requires_grad=False, NOT via torch.no_grad()
    z_{t+1} ~ Prior(h_t)
    h_{t+1} = GRU+LN(h_t, z_t, a_t)        # GRADIENT FLOWS: V(h_{t+1}) → h_{t+1} → a_t → Actor
    r̂_t = RewardMLP(h_t, z_t)
    ĉ_t = ContinueMLP(h_t, z_t)
    v̂_t = Critic(h_t, z_t)                 # With gradient
```

**Critical**: DO NOT use `torch.no_grad()` — it would block the gradient chain from Critic loss → h_t → a_t → Actor. Instead, freeze world model parameters before imagination by setting `requires_grad=False` on Encoder, GRU, Prior, Decoder, Reward, and Continue parameters, then restore after. Only Actor and Critic parameters remain trainable during imagination. The GRU forward pass records the gradient path through `a_t` to `h_{t+1}` without updating WM weights.

### λ-Return Computation

```
G_H = V̂(h_H, z_H)                                  # bootstrap from last step
for t = H-1 down to 0:
    G_t = r̂_t + γ · ĉ_t · ((1-λ)·V̂(h_{t+1},z_{t+1}) + λ·G_{t+1})

Advantage_t = G_t - V̂(h_t, z_t)
```

**Key difference from PPO GAE**: Uses `ĉ_t` (continue probability) instead of binary `(1-done)`. This softens credit assignment boundaries. All values in symlog space.

### Actor-Critic Losses

```python
# Actor (PPO clipped surrogate, same structure as original PPO code)
ratio = exp(log_prob_new - log_prob_old)
L_actor = -min(ratio * advantage, clip(ratio, 0.98, 1.02) * advantage)

# Critic (MSE in symlog space)
L_critic = MSE(V̂(h,z), G)

# Combined
L_ac = L_actor + L_critic
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| H (imagination) | 15 | Standard DreamerV3 |
| N (start states) | 1024 | Good coverage of buffer diversity |
| λ (TD-λ) | 0.95 | Same as GAE |
| γ (discount) | 0.997 | Atari standard |
| PPO clip ε | 0.02 | Tighter than real-env PPO (imagined data is "cleaner") |

---

## Training Loop

```
1. Seed: Collect 5000 random steps → Replay Buffer

2. For each iteration (target: ~2000 iterations for 1M env steps):
   
   a. Collect: Play ~500 steps (≈1 Seaquest episode) in real environment
      using current Actor. Store in buffer.
   
   b. Train World Model (K_wm = 5):
      - Sample B=8 × T=64 sequences from buffer
      - 64-step RSSM open-loop unroll
      - Compute L_wm → update Encoder, GRU, Prior, Posterior, Decoder, heads
   
   c. Imagine + Train Actor-Critic (K_img = 5):
      - Sample 1024 (h,z) start states from buffer (posterior)
      - Imagine H=15 trajectories using prior
      - Compute λ-returns → update Actor + Critic
   
   d. Log: Every 10 iterations, record reward, losses, reconstruction

3. Target: 1,000,000 environment steps (~4-8 GPU hours on RTX 3090)
```

### Reward Normalization

```python
# EMA percentile tracking (P5, P95) of episode returns
S5  = ema(S5,  percentile_5(recent_returns),  momentum=0.99)
S95 = ema(S95, percentile_95(recent_returns), momentum=0.99)
normalized_return = (return - S95) / max(1.0, S95 - S5)
```

This prevents the critic's prediction target from shifting over training as the agent improves.

---

## Implementation Order

### Phase 1: World Model (verify reconstruction quality)
1. `config.py` — hyperparameters for Seaquest + DreamerV3
2. `env_wrapper.py` — update for Seaquest (frameskip=4, no frame stack)
3. `networks.py` — CNN Encoder, GRU+LN, Prior, Posterior, Decoder, heads
4. `replay_buffer.py` — sequence storage + sampling
5. `world_model.py` — RSSM open-loop unroll + loss computation
6. **Test**: Can the model reconstruct Seaquest frames?

### Phase 2: Imagination + Agent (verify policy learns)
7. `imagination.py` — latent rollout + λ-return + PPO update
8. `train.py` — full training loop integration
9. **Test**: Does the agent score > 0 after 100k steps?

### Phase 3: Analysis
10. `eval.py` — evaluation + video recording
11. `latent_analysis.py` — PCA, t-SNE, linear probes of z_t
12. SB3 PPO baseline for comparison

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Posterior collapse | KL balancing (α=0.8) + free bits (min 1.0 nat) |
| NaN losses | LayerNorm in GRU + gradient clipping |
| WM batch OOM | B=8 (reduced from 16); gradients flow through smaller batches |
| Imagination gradient loss | h_t computed WITHOUT no_grad; only WM params excluded from optimizer |
| Reward scale instability | Symlog + EMA percentile normalization |
| Diagnostic dead zone | Seaquest gives gradual score improvement (unlike Breakout's binary signal) |
