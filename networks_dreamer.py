import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """Encode [B,1,84,84] -> [B,512]."""
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


class GRUWithLN(nn.Module):
    """GRUCell with LayerNorm on output."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x, h):
        return self.ln(self.gru(x, h))


class Prior(nn.Module):
    """p(z_t | h_t). 32x16 categorical logits from h [512]."""
    def __init__(self, hidden=512, cats=32, classes=16):
        super().__init__()
        self.output_dim = cats * classes
        self.net = nn.Sequential(
            nn.Linear(hidden, 256), nn.SiLU(),
            nn.Linear(256, self.output_dim),
        )

    def forward(self, h):
        logits = self.net(h)
        return logits.view(-1, 32, 16)


class Posterior(nn.Module):
    """q(z_t | h_t, x_t). h [512] + x [512] -> categorical logits."""
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


class CNNDecoder(nn.Module):
    """Decode [h+z_flat] [B,1024] -> [B,1,84,84]."""
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(feat_dim, 3136), nn.SiLU(),
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 8, 4), nn.Sigmoid(),
        )

    def forward(self, features):
        x = self.linear(features)
        x = x.view(-1, 64, 7, 7)
        return self.convt(x)


class RewardHead(nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 1)
    def forward(self, features): return self.fc(features).squeeze(-1)


class ContinueHead(nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 1)
    def forward(self, features): return self.fc(features).squeeze(-1)


class ActorHead(nn.Module):
    def __init__(self, feat_dim=1024, num_actions=18):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_actions)
    def forward(self, features): return self.fc(features)


class CriticHead(nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 1)
    def forward(self, features): return self.fc(features).squeeze(-1)
