import math

import torch
import torch.nn.functional as F

from MetaOpt.models import Critic, TransformerActor
from MetaOpt.perf import unwrap_compiled


class SACAgent:
    def __init__(self, state_dim, action_dim, cfg, device):
        self.device = device
        self.action_dim = action_dim

        self.actor = TransformerActor(
            n_params=action_dim,
            max_action=cfg.rl.max_action,
            log_std_min=cfg.rl.log_std_min,
            log_std_max=cfg.rl.log_std_max,
        ).to(device)

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.target1 = Critic(state_dim, action_dim).to(device)
        self.target2 = Critic(state_dim, action_dim).to(device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.rl.actor_lr)
        self.opt_c1 = torch.optim.Adam(self.critic1.parameters(), lr=cfg.rl.critic_lr)
        self.opt_c2 = torch.optim.Adam(self.critic2.parameters(), lr=cfg.rl.critic_lr)

        self.log_alpha = torch.tensor(-3.0, requires_grad=True, device=device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.rl.alpha_lr)

        self.target_entropy = -float(action_dim)

    def _sample_action(self, state, deterministic=False):
        mean, log_std = self.actor(state)

        if deterministic:
            u = mean
        else:
            std = log_std.exp()
            eps = torch.randn_like(std)
            u = mean + std * eps

        a = torch.tanh(u)
        max_action = float(unwrap_compiled(self.actor).max_action)
        a_scaled = a * max_action

        log_prob = None
        if not deterministic:
            std = log_std.exp()
            log_prob_gauss = -0.5 * (((u - mean) / (std + 1e-8)) ** 2 + 2 * log_std + math.log(2 * math.pi))
            log_prob_gauss = log_prob_gauss.sum(dim=-1, keepdim=True)
            log_prob_tanh = torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            log_prob = log_prob_gauss - log_prob_tanh

        return a_scaled, log_prob

    def select_action(self, state, deterministic=False):
        action, _ = self._sample_action(state, deterministic=deterministic)
        return action

    def update(self, replay, cfg):
        if len(replay) < cfg.rl.batch_size:
            return

        s, a, r, s2, d = replay.sample(cfg.rl.batch_size, self.device)

        with torch.no_grad():
            a2, logp2 = self._sample_action(s2, deterministic=False)
            q1_t = self.target1(s2, a2)
            q2_t = self.target2(s2, a2)
            q_min = torch.min(q1_t, q2_t)
            alpha = self.log_alpha.exp()
            v2 = q_min - alpha * logp2.squeeze(-1)
            target_q = r + cfg.rl.gamma * (1.0 - d) * v2

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)

        loss_q1 = F.mse_loss(q1, target_q)
        loss_q2 = F.mse_loss(q2, target_q)

        self.opt_c1.zero_grad(set_to_none=True)
        loss_q1.backward()
        self.opt_c1.step()

        self.opt_c2.zero_grad(set_to_none=True)
        loss_q2.backward()
        self.opt_c2.step()

        a_pi, logp_pi = self._sample_action(s, deterministic=False)
        alpha = self.log_alpha.exp()
        q1_pi = self.critic1(s, a_pi)
        q2_pi = self.critic2(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (alpha * logp_pi.squeeze(-1) - q_pi).mean()

        self.opt_a.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_a.step()

        logp_detached = logp_pi.detach()
        alpha_loss = -(self.log_alpha * (logp_detached + self.target_entropy)).mean()

        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()

        tau = cfg.rl.tau
        with torch.no_grad():
            for p, pt in zip(self.critic1.parameters(), self.target1.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)
            for p, pt in zip(self.critic2.parameters(), self.target2.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)

    def state_dict(self):
        return {
            "actor": unwrap_compiled(self.actor).state_dict(),
            "critic1": unwrap_compiled(self.critic1).state_dict(),
            "critic2": unwrap_compiled(self.critic2).state_dict(),
            "target1": unwrap_compiled(self.target1).state_dict(),
            "target2": unwrap_compiled(self.target2).state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dict(self, state):
        unwrap_compiled(self.actor).load_state_dict(state["actor"])
        unwrap_compiled(self.critic1).load_state_dict(state["critic1"])
        unwrap_compiled(self.critic2).load_state_dict(state["critic2"])
        unwrap_compiled(self.target1).load_state_dict(state["target1"])
        unwrap_compiled(self.target2).load_state_dict(state["target2"])
        self.log_alpha = state["log_alpha"].to(self.device).requires_grad_(True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=self.opt_alpha.param_groups[0]["lr"])

    def alpha_value(self):
        return self.log_alpha.exp().item()
