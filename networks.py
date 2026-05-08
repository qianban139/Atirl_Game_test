import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """Shared feature extractor: [B,4,84,84] -> [B,256]."""
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
        return self.fc(self.conv(x))


class Actor(nn.Module):
    """Policy head: phi(s) -> action logits."""
    def __init__(self, feature_dim=256, num_actions=18):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_actions)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class Critic(nn.Module):
    """Value head: phi(s) -> V(s)."""
    def __init__(self, feature_dim=256):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features).squeeze(-1)


class InverseDynamics(nn.Module):
    """ICM inverse model: [phi(s), phi(s')] -> predicted action."""
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
    """ICM forward model: [phi(s), one_hot(a)] -> predicted phi(s')."""
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
