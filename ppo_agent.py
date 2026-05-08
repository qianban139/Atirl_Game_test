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

        self.advantages = advantages.detach()
        self.returns = (advantages + self.values).detach()

    def get_minibatches(self, minibatch_size: int):
        """Yield random minibatches as dicts."""
        total = self.num_steps * self.num_envs
        indices = torch.randperm(total, device=self.device)

        # Flatten all buffers to [total, ...], detach to break computation graph
        flat_obs = self.obs.view(total, *self.obs_shape)
        flat_actions = self.actions.view(total)
        flat_log_probs = self.log_probs.view(total)
        flat_advantages = self.advantages.view(total).detach()
        flat_returns = self.returns.view(total).detach()

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
        """Sample action and return action, log_prob, value, entropy."""
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

                # Value loss
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
