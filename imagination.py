"""Imagination: latent rollout + actor-critic training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from rssm import sample_categorical


def compute_lambda_return(rewards, continues, values, gamma, lam):
    """Compute lambda-return in symlog space. continues = continue PROBABILITY (after sigmoid)."""
    H = len(rewards)
    G = values[-1]
    for t in reversed(range(H)):
        G = rewards[t] + gamma * continues[t] * ((1 - lam) * values[t] + lam * G)
    return G  # scalar at t=0, or full sequence if accumulated


def compute_lambda_returns(rewards, continues, values, gamma, lam):
    """Compute lambda-returns for all timesteps."""
    H, N = rewards.shape
    returns = torch.zeros(H, N, device=rewards.device)
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

    # Freeze world model params (NOT no_grad — h must carry gradient through a_t)
    wm_params = {}
    for name, p in rssm.named_parameters():
        wm_params[name] = p.requires_grad
        p.requires_grad = False

    # ── Imagination rollout ──
    all_features = []
    all_actions = []
    all_log_probs = []
    all_rewards = []
    all_continues = []
    all_values = []

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

        # RSSM transition: h first, then z from Prior(h)
        z_flat = z.reshape(N, -1)
        a_onehot = F.one_hot(a, config.num_actions).float()
        h = rssm.gru(torch.cat([z_flat, a_onehot], dim=-1), h)
        prior_logits = rssm.prior(h)
        z, _ = sample_categorical(prior_logits, config.unimix)

        all_features.append(features)
        all_actions.append(a)
        all_log_probs.append(lp)
        all_rewards.append(rew)
        all_continues.append(c_hat)
        all_values.append(v)

    # Stack
    all_rewards = torch.stack(all_rewards)    # [H, N]
    all_continues = torch.stack(all_continues)  # [H, N]
    all_values = torch.stack(all_values)      # [H, N]
    all_log_probs = torch.stack(all_log_probs)  # [H, N]

    # Bootstrap value from final state
    final_features = torch.cat([h, z.reshape(N, -1)], dim=-1)
    v_last = critic(final_features)
    all_values_with_last = torch.cat([all_values, v_last.unsqueeze(0)])  # [H+1, N]

    # ── Compute lambda-returns ──
    returns = compute_lambda_returns(all_rewards, all_continues, all_values_with_last, config.gamma, config.lam)

    # ── Actor loss (REINFORCE + entropy) ──
    advantages = returns - all_values.detach()
    L_policy = -(all_log_probs * advantages.detach()).mean()

    # Entropy: compute from the stacked features
    features_stacked = torch.stack(all_features)  # [H, N, 1024]
    logits_all = actor(features_stacked)  # [H, N, 18]
    dist_all = torch.distributions.Categorical(logits=logits_all)
    entropy = dist_all.entropy().mean()
    L_entropy = -config.entropy_eta * entropy
    L_actor = L_policy + L_entropy

    # ── Critic loss (MSE in symlog space, target detached) ──
    L_critic = F.mse_loss(all_values, returns.detach())

    # ── Update ──
    ac_optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=config.ac_lr)
    L_ac = L_actor + L_critic
    ac_optimizer.zero_grad()
    L_ac.backward()
    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), config.max_grad_norm)
    ac_optimizer.step()

    # ── RESTORE world model params ── (CRITICAL: WM stops learning if forgotten)
    for name, p in rssm.named_parameters():
        p.requires_grad = wm_params[name]
    assert all(p.requires_grad for p in rssm.parameters()), "WM params not restored!"

    return L_ac.item(), L_actor.item(), L_critic.item()
