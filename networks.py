"""DreamerV3 networks — paper-aligned: SiLU, RMSNorm pre-norm, TwoHot critic."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — DreamerV3 paper default."""
    def __init__(self, normalized_shape, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(x.dtype)


def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# ─── Encoder ────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """Encode symlog-squashed [B,1,84,84] → [B,512]."""
    def __init__(self, feat_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            RMSNorm([1, 84, 84]),
            nn.Conv2d(1, 32, 8, 4), nn.SiLU(),
            RMSNorm([32, 20, 20]),
            nn.Conv2d(32, 64, 4, 2), nn.SiLU(),
            RMSNorm([64, 9, 9]),
            nn.Conv2d(64, 64, 3, 1), nn.SiLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            out = self.conv(torch.zeros(1, 1, 84, 84))
            conv_dim = out.shape[1]
        self.fc = nn.Sequential(
            RMSNorm(conv_dim),
            nn.Linear(conv_dim, 1024), nn.SiLU(),
            RMSNorm(1024),
            nn.Linear(1024, feat_dim),
        )

    def forward(self, x):
        return self.fc(self.conv(symlog(x)))


# ─── GRU with LayerNorm ─────────────────────────────────

class GRUWithLN(nn.Module):
    """GRUCell with LayerNorm on output."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.ln = RMSNorm(hidden_size)

    def forward(self, x, h):
        return self.ln(self.gru(x, h))


# ─── RSSM Prior / Posterior ─────────────────────────────

class Prior(nn.Module):
    """p(z_t | h_t)."""
    def __init__(self, hidden=512, cats=32, classes=16):
        super().__init__()
        out = cats * classes
        self.net = nn.Sequential(
            RMSNorm(hidden),
            nn.Linear(hidden, 256), nn.SiLU(),
            RMSNorm(256),
            nn.Linear(256, out),
        )

    def forward(self, h):
        return self.net(h).view(h.shape[0], -1, 16)


class Posterior(nn.Module):
    """q(z_t | h_t, x_t)."""
    def __init__(self, hidden=512, feat=512, cats=32, classes=16):
        super().__init__()
        out = cats * classes
        self.net = nn.Sequential(
            RMSNorm(hidden + feat),
            nn.Linear(hidden + feat, 256), nn.SiLU(),
            RMSNorm(256),
            nn.Linear(256, out),
        )

    def forward(self, h, x):
        return self.net(torch.cat([h, x], dim=-1)).view(h.shape[0], -1, 16)


# ─── Decoder (NO sigmoid — outputs raw symlog) ──────────

class CNNDecoder(nn.Module):
    """Decode features [B,1024] → symlog-space pixels [B,1,84,84]."""
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.linear = nn.Sequential(
            RMSNorm(feat_dim),
            nn.Linear(feat_dim, 3136), nn.SiLU(),
        )
        self.convt = nn.Sequential(
            RMSNorm([64, 7, 7]),
            nn.ConvTranspose2d(64, 64, 3, 1), nn.SiLU(),
            RMSNorm([64, 9, 9]),
            nn.ConvTranspose2d(64, 32, 4, 2), nn.SiLU(),
            RMSNorm([32, 20, 20]),
            nn.ConvTranspose2d(32, 1, 8, 4),
        )

    def forward(self, features):
        x = self.linear(features)
        x = x.view(-1, 64, 7, 7)
        return self.convt(x)


# ─── Prediction Heads ───────────────────────────────────

class RewardHead(nn.Module):
    """TwoHot reward predictor: features → 255 bin logits."""
    def __init__(self, feat_dim=1024, bins=255):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(feat_dim),
            nn.Linear(feat_dim, 256), nn.SiLU(),
            RMSNorm(256),
            nn.Linear(256, bins),
        )

    def forward(self, features):
        return self.net(features)


class ContinueHead(nn.Module):
    """Continue predictor: features → logit."""
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(feat_dim),
            nn.Linear(feat_dim, 256), nn.SiLU(),
            RMSNorm(256),
            nn.Linear(256, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class ActorHead(nn.Module):
    """Policy: features → action logits (2 hidden layers)."""
    def __init__(self, feat_dim=1024, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(feat_dim),
            nn.Linear(feat_dim, 512), nn.SiLU(),
            RMSNorm(512),
            nn.Linear(512, 256), nn.SiLU(),
            RMSNorm(256),
            nn.Linear(256, num_actions),
        )

    def forward(self, features):
        return self.net(features)


class CriticHead(nn.Module):
    """TwoHot distributional critic: features → 255 bin logits."""
    def __init__(self, feat_dim=1024, bins=255):
        super().__init__()
        self.bins = bins
        self.net = nn.Sequential(
            RMSNorm(feat_dim),
            nn.Linear(feat_dim, 512), nn.SiLU(),
            RMSNorm(512),
            nn.Linear(512, 256), nn.SiLU(),
            RMSNorm(256),
            nn.Linear(256, bins),
        )

    def forward(self, features):
        return self.net(features)


# ─── TwoHot helpers ─────────────────────────────────────

def twohot_encode(value_symlog, bins=255, low=-20.0, high=20.0):
    """Encode scalar symlog value as two-hot target over bins."""
    bin_width = (high - low) / (bins - 1)
    pos = (value_symlog - low) / bin_width
    pos = torch.clamp(pos, 0.0, bins - 1.001)
    lo = pos.long()
    hi = torch.clamp(lo + 1, max=bins - 1)
    hi_w = pos - lo.float()
    lo_w = 1.0 - hi_w
    target = torch.zeros(*value_symlog.shape, bins, device=value_symlog.device)
    target.scatter_add_(-1, lo.unsqueeze(-1), lo_w.unsqueeze(-1))
    target.scatter_add_(-1, hi.unsqueeze(-1), hi_w.unsqueeze(-1))
    return target


def twohot_decode(logits, low=-20.0, high=20.0):
    """Decode TwoHot logits → expected raw value via symexp."""
    bins = logits.shape[-1]
    bin_centers = torch.linspace(low, high, bins, device=logits.device)
    probs = torch.softmax(logits, dim=-1)
    value_symlog = (probs * bin_centers).sum(dim=-1)
    return symexp(value_symlog)
