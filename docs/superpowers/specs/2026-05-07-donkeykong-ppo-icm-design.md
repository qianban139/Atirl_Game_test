# Donkey Kong PPO + ICM Agent Design

## Overview

Implement a deep reinforcement learning agent for the Atari game Donkey Kong using **PPO (Proximal Policy Optimization)** augmented with **ICM (Intrinsic Curiosity Module)**. Built from scratch in PyTorch (no RL-specific libraries for core algorithm). Targeted at the DTS307TC coursework.

**Key constraint:** Algorithm must be implemented manually. Stable-Baselines3 is allowed only for benchmark comparison.

## Why Donkey Kong

Donkey Kong is one of the most challenging Atari 2600 games for RL agents due to:

- **Sparse rewards:** Points only from jumping barrels, collecting items, completing stages
- **Multi-stage gameplay:** 4 distinct stages (barrel, conveyor, elevator, rivet) with different mechanics
- **Precision requirements:** Tight jump timing, ladder navigation, hammer usage
- **Hierarchical behaviors:** Agent must chain sub-goals (climb → avoid → jump → climb)
- **Life system:** Death resets position, creating partial observability at the frame level

Standard DQN agents typically score <1,000 points, while human players average ~8,000.

## Algorithm Choice: PPO + ICM

### Why PPO over DQN-family

| Concern | DQN | PPO |
|---------|-----|-----|
| Exploration | ε-greedy (undirected) | Stochastic policy (directed) |
| Sparse reward handling | Poor TD propagation | Better with GAE + stochastic exploration |
| Stability | Target network needed | Clipped surrogate objective |
| Multi-stage generalization | Struggles | Policy can learn hierarchical patterns |

### Why add ICM

Donkey Kong's extrinsic rewards are too sparse for PPO alone to explore effectively. ICM provides:

- **Intrinsic reward** = MSE between predicted and actual next-state encoding
- Encourages visiting **unfamiliar states** (novel screens, new stage layouts)
- Forward model prediction error naturally decreases as agent masters predictable transitions

### Alternatives considered and rejected

| Method | Reason rejected |
|--------|----------------|
| DQN + extensions | Known poor performance on DK; ε-greedy insufficient for exploration |
| Rainbow DQN | Too complex to implement from scratch; 6 interacting components |
| PPO only | Baseline included for ablation; insufficient exploration alone |
| RND | More hyperparameters than ICM; harder to tune for a first implementation |
| A2C | Less stable than PPO; no clipped surrogate |

## Architecture

### Data Flow

```
Atari DK (ALE) → EnvWrapper → [4×84×84] → CNN Encoder → φ(s) [256]
                                                              │
                                          ┌───────────────────┼───────────────────┐
                                          ▼                   ▼                   ▼
                                      Actor Head          Critic Head          ICM
                                      → π(a|s)            → V(s)              → rⁱ
                                          │                                       │
                                          ▼                                       │
                                    PPO Loss ← GAE (r_total = rᵉ + β·rⁱ) ←──────┘
```

### Components

1. **Environment Wrapper** (`env_wrapper.py`)
   - Grayscale conversion, resize to 84×84
   - Frame stack (k=4)
   - 18 discrete actions (full Atari action space)
   - Extrinsic reward clipping to [-1, +1]
   - Life-loss as episode boundary (helps agent learn to avoid death)
   - 8 parallel environments via AsyncVectorEnv

2. **CNN Encoder** (`networks.py`)
   - Shared feature extractor for Actor, Critic, and ICM
   - Architecture: Conv(4→32,8,4) → Conv(32→64,4,2) → Conv(64→64,3,1) → FC(3136→256)
   - LayerNorm (not BatchNorm — poor with RL minibatch sizes)
   - Output: φ(s) ∈ R²⁵⁶

3. **Actor-Critic (PPO)** (`ppo_agent.py`)
   - Actor: Linear(256→18) → Categorical distribution
   - Critic: Linear(256→1) → scalar value
   - Rollout buffer: T=128 steps × 8 envs = 1024 transitions per batch
   - GAE: λ=0.95, γ=0.99
   - PPO update: K=4 epochs, ε=0.1 clip, Adam lr=2.5e-4
   - Loss: L = L_policy + 0.5·L_value - 0.01·L_entropy

4. **ICM** (`icm_agent.py`)
   - Inverse Model: [φ(s)||φ(s')] → FC(512→256) → FC(256→18) → â
   - Forward Model: [φ(s)||one_hot(a)] → FC(274→256) → FC(256→256) → φ̂(s')
   - Intrinsic reward: rⁱ = β · ||φ̂(s') - φ(s')||², normalized by running std
   - Loss: L_icm = 0.2·CE(â, a) + 0.8·MSE(φ̂(s'), φ(s'))
   - ICM learning rate: 2.5e-5 (×0.1 of base lr)

### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| γ (discount) | 0.99 | Standard Atari |
| λ (GAE) | 0.95 | Balance bias-variance |
| ε (PPO clip) | 0.1 | Original paper; prevents overly large updates |
| T (rollout steps) | 128 | Original PPO paper |
| N (parallel envs) | 8 | Break sample correlation |
| K (PPO epochs) | 4 | Avoid overfitting to single batch |
| lr (Adam) | 2.5e-4 | Standard for Atari PPO |
| c₁ (value coeff) | 0.5 | Value loss weight |
| c₂ (entropy coeff) | 0.01 | Initial; decays over training |
| β (intrinsic scale) | 0.01 | Prevent intrinsic reward dominating extrinsic |
| η (ICM lr multiplier) | 0.1 | ICM learns slower than policy |
| grad clip (max norm) | 0.5 | Prevent gradient explosion |
| total timesteps | 50M | ~5 days on RTX 3080 |

## Training Loop

```
for timestep in 1..50M (across 8 parallel envs):
    1. Collect: sample aₜ ~ π(·|sₜ), observe sₜ₊₁, rₜᵉ, done
    2. Store (sₜ, aₜ, rₜᵉ, sₜ₊₁, done) in Rollout Buffer
    3. When buffer full (1024 transitions):
       a. Compute rⁱ via ICM for all transitions
       b. Compute r_total = rᵉ + β·rⁱ
       c. Compute GAE advantages and returns
       d. PPO update (K=4 epochs over buffer)
       e. ICM update (single pass over buffer)
       f. Clear buffer
    4. Every 100 episodes: log metrics, save checkpoint
    5. Every 1M steps: evaluation run + video recording
```

## Evaluation

### Metrics

- **Primary:** Episode extrinsic reward (raw game score, unclipped)
- **Secondary:** Episode length, max stage reached, policy entropy, value loss, intrinsic reward mean

### Benchmarks

| Benchmark | Purpose |
|-----------|---------|
| Random agent | Sanity check — proves agent learned |
| SB3 PPO (10M steps) | PPO-only baseline for ICM ablation comparison |
| Human average (~8,000) | Reference point for "competitive" |

### Ablation

Same seed, same hyperparameters:
- **PPO only** (β=0): Isolates ICM contribution
- **PPO + ICM** (β=0.01): Full method

### Visualizations

1. Training curve: extrinsic reward vs timesteps (rolling mean ± std, window=100 episodes)
2. Stage progression histogram
3. Policy entropy decay curve
4. Ablation comparison plot (PPO vs PPO+ICM)

## Project Structure

```
ATRI/
├── config.py              # All hyperparameters
├── env_wrapper.py         # Atari environment wrapper
├── networks.py            # CNN Encoder, Actor, Critic, ICM models
├── ppo_agent.py           # PPO rollout buffer, GAE, update logic
├── icm_agent.py           # ICM intrinsic reward computation
├── train.py               # Main training loop
├── eval.py                # Evaluation + video recording
├── utils.py               # Logger, RunningStats, seed setup
├── benchmark/
│   └── sb3_baseline.py    # SB3 PPO baseline
├── checkpoints/           # Saved model weights
├── logs/                  # TensorBoard logs
└── requirements.txt       # Dependencies
```

## Implementation Order

| Step | Scope | Verification |
|------|-------|-------------|
| 1 | Environment wrapper + ALE config | Render frame stack, verify [4,84,84] shape |
| 2 | Network definitions (Encoder, Actor, Critic, ICM) | Forward pass dim check |
| 3 | PPO (rollout, GAE, clipped loss) | Quick CartPole convergence test (~5 min) |
| 4 | ICM module (intrinsic reward, normalization) | No reward scale explosion |
| 5 | Full training loop integration | 1M step sanity check, loss trends |
| 6 | Evaluation, SB3 benchmark, charts | Full 50M training + ablation runs |

## Dependencies

```
gymnasium[atari]
gymnasium[accept-rom-license]
torch>=2.0
numpy
matplotlib
tensorboard
stable-baselines3>=2.0  # benchmark only
opencv-python           # video recording
```

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| ICM intrinsic reward explodes | Running mean/std normalization; gradient clipping |
| PPO policy collapses (entropy → 0) | Entropy bonus with annealing, not constant |
| DK too hard, agent never passes stage 1 | Accept partial success; rich analysis of failure modes is valid report material |
| Training too slow for 50M steps | Profile with 1M steps first; consider reducing ICM update frequency |
