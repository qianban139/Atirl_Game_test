"""Imagination: latent rollout + actor-critic training with TwoHot critic + EMA."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from rssm import sample_categorical
from networks import symlog, twohot_encode, twohot_decode


def compute_lambda_returns(rewards, continues, values, gamma, lam):
    """Lambda-returns in symlog space for all timesteps."""
    H, N = rewards.shape
    returns = torch.zeros(H, N, device=rewards.device)
    G = values[-1]
    for t in reversed(range(H)):
        G = rewards[t] + gamma * continues[t] * ((1 - lam) * values[t] + lam * G)
        returns[t] = G
    return returns


def imagine_and_train(rssm, actor, critic, slow_critic, ac_optimizer, start_h, start_z, config):
    """Imagine H-step trajectories and train actor-critic.
    slow_critic: EMA copy used for value bootstrap (not updated here)."""
    N, H = start_h.shape[0], config.imagination_horizon
    device = start_h.device
    h, z = start_h, start_z

    # Freeze world model params
    wm_params = {name: p.requires_grad for name, p in rssm.named_parameters()}
    for p in rssm.parameters():
        p.requires_grad = False

    # ── Imagination rollout ──
    all_features = []
    all_log_probs = []
    all_rewards = []
    all_continues = []
    all_values_symlog = []        # decoded critic values (scalar symlog)
    all_critic_logits = []        # raw critic logits for loss

    for t in range(H):
        features = torch.cat([h, z.reshape(N, -1)], dim=-1)
        logits = actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        lp = dist.log_prob(a)

        # Reward and continue from world model
        rew_logits = rssm.reward_head(features)
        rew_probs = torch.softmax(rew_logits, dim=-1)
        bin_centers = torch.linspace(config.critic_low, config.critic_high,
                                     config.critic_bins, device=device)
        rew_symlog = (rew_probs * bin_centers).sum(-1)

        cont_logit = rssm.continue_head(features)
        c_hat = torch.sigmoid(cont_logit)

        # Critic: use online critic for loss, slow critic for bootstrap
        critic_logits = critic(features)
        v_symlog = (torch.softmax(critic_logits, dim=-1) * bin_centers).sum(-1)

        # RSSM transition
        z_flat = z.reshape(N, -1)
        a_onehot = F.one_hot(a, config.num_actions).float()
        h = rssm.gru(torch.cat([z_flat, a_onehot], dim=-1), h)
        prior_logits = rssm.prior(h)
        z, _ = sample_categorical(prior_logits, config.unimix)

        all_features.append(features)
        all_log_probs.append(lp)
        all_rewards.append(rew_symlog)
        all_continues.append(c_hat)
        all_values_symlog.append(v_symlog)
        all_critic_logits.append(critic_logits)

    # Stack
    all_rewards = torch.stack(all_rewards)
    all_continues = torch.stack(all_continues)
    all_values_symlog = torch.stack(all_values_symlog)
    all_log_probs = torch.stack(all_log_probs)
    all_critic_logits = torch.stack(all_critic_logits)

    # Bootstrap with SLOW critic (EMA)
    final_features = torch.cat([h, z.reshape(N, -1)], dim=-1)
    slow_logits = slow_critic(final_features)
    bin_centers = torch.linspace(config.critic_low, config.critic_high,
                                 config.critic_bins, device=device)
    v_last = (torch.softmax(slow_logits, dim=-1) * bin_centers).sum(-1)
    all_values_with_last = torch.cat([all_values_symlog, v_last.unsqueeze(0)])

    # ── Lambda-returns (in symlog space) ──
    returns = compute_lambda_returns(all_rewards, all_continues, all_values_with_last,
                                     config.gamma, config.lam)

    # ── Critic loss: TwoHot cross-entropy ──
    returns_symlog = symlog(twohot_decode(
        torch.zeros_like(all_critic_logits[0])[None, None],  # placeholder, actual target below
    ))
    # Actually encode returns directly as TwoHot targets
    returns_clipped = torch.clamp(returns, config.critic_low, config.critic_high)
    critic_target = twohot_encode(returns_clipped, config.critic_bins,
                                  config.critic_low, config.critic_high)
    L_critic = -(critic_target.detach() * F.log_softmax(all_critic_logits, dim=-1)).sum(-1).mean()

    # ── Actor loss: REINFORCE + entropy ──
    advantages = returns - all_values_symlog.detach()
    S = torch.quantile(advantages, 0.95) - torch.quantile(advantages, 0.05)
    advantages = advantages / max(1.0, S)
    L_policy = -(all_log_probs * advantages.detach()).mean()

    features_stacked = torch.stack(all_features)
    logits_all = actor(features_stacked)
    dist_all = torch.distributions.Categorical(logits=logits_all)
    entropy = dist_all.entropy().mean()
    L_entropy = -config.entropy_eta * entropy
    L_actor = L_policy + L_entropy

    # ── Update ──
    L_ac = L_actor + L_critic
    ac_optimizer.zero_grad()
    L_ac.backward()
    nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()),
                             config.max_grad_norm)
    ac_optimizer.step()

    # ── Update slow critic (EMA) ──
    with torch.no_grad():
        for sp, p in zip(slow_critic.parameters(), critic.parameters()):
            sp.data.lerp_(p.data, config.slow_critic_tau)

    # ── Restore WM params ──
    for name, p in rssm.named_parameters():
        p.requires_grad = wm_params[name]

    return L_ac.item(), L_actor.item(), L_critic.item()
