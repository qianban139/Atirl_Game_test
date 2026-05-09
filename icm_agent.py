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
        """Compute ICM loss (inverse + forward). Encoder is frozen here — ICM optimizer
        does not include encoder params, so we use no_grad to avoid storing its
        computation graph (saves ~10+ GB VRAM on 1024-sample batches)."""
        with torch.no_grad():
            phi_s = self.encoder(obs)
            phi_s_next = self.encoder(next_obs)

        # Inverse dynamics loss
        pred_actions = self.inverse_model(phi_s, phi_s_next)
        inverse_loss = F.cross_entropy(pred_actions, actions)

        # Forward dynamics loss
        action_onehot = F.one_hot(actions, num_classes=self.config.num_actions).float()
        phi_hat_next = self.forward_model(phi_s, action_onehot)
        forward_loss = F.mse_loss(phi_hat_next, phi_s_next)

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
