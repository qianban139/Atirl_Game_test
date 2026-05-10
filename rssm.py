"""RSSM core: forward pass, open-loop unroll, world model training."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        """Open-loop RSSM unroll for world model training."""
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
            cont_pred = self.continue_head(features)

            hs.append(h); zs.append(z)
            prior_logits_list.append(prior_logits); post_logits_list.append(post_logits)
            recon_list.append(recon); reward_pred_list.append(reward_pred); cont_pred_list.append(cont_pred)

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

        L_recon = F.mse_loss(out["recon"], obs, reduction='mean')
        L_reward = F.mse_loss(out["reward_pred"], symlog(rewards), reduction='mean')
        cont_target = (1 - dones.float())
        L_cont = F.binary_cross_entropy_with_logits(out["cont_pred"], cont_target)

        q_logits = out["post_logits"]
        p_logits = out["prior_logits"]
        q_dist = torch.distributions.Categorical(logits=q_logits)
        p_dist = torch.distributions.Categorical(logits=p_logits)

        kl_prior = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=q_logits.detach()), p_dist
        ).sum(-1)
        kl_post = torch.distributions.kl_divergence(
            q_dist, torch.distributions.Categorical(logits=p_logits.detach())
        ).sum(-1)
        kl_balanced = self.cfg.kl_alpha * kl_prior + (1 - self.cfg.kl_alpha) * kl_post
        kl_clipped = torch.clamp(kl_balanced, min=self.cfg.free_nats)
        L_kl = kl_clipped.mean()

        L_wm = (self.cfg.beta_recon * L_recon +
                self.cfg.beta_reward * L_reward +
                self.cfg.beta_cont * L_cont +
                self.cfg.beta_kl * L_kl)

        return L_wm, {"recon": L_recon.item(), "reward": L_reward.item(),
                       "cont": L_cont.item(), "kl": L_kl.item(), "total": L_wm.item()}
