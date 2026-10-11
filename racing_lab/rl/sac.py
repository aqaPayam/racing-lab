"""SAC adapted from Remonda/Watanabe's author source; see THIRD_PARTY_NOTICES.md."""

from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .config import ADAPTER_VERSION, SOURCE_REVISION, SACConfig


def linear_network(input_dim, output_dim, hidden_units):
    layers = []
    for units in hidden_units:
        layers.extend([nn.Linear(input_dim, units), nn.ReLU()])
        input_dim = units
    layers.append(nn.Linear(input_dim, output_dim))
    network = nn.Sequential(*layers)
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return network


class GaussianPolicy(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_units):
        super().__init__()
        self.net = linear_network(observation_dim, 2 * action_dim, hidden_units)

    def forward(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        normals = Normal(means, log_stds.clamp(-20, 2).exp())
        samples = normals.rsample()
        actions = samples.tanh()
        log_probs = normals.log_prob(samples) - torch.log(1 - actions.square() + 1e-6)
        return actions, -log_probs.sum(dim=1, keepdim=True), means.tanh()

    def deterministic(self, states):
        return self.net(states).chunk(2, dim=-1)[0].tanh()


class StateActionFunction(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_units):
        super().__init__()
        self.net = linear_network(observation_dim + action_dim, 1, hidden_units)

    def forward(self, x):
        return self.net(x)


class TwinQ(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_units):
        super().__init__()
        self.net1 = StateActionFunction(observation_dim, action_dim, hidden_units)
        self.net2 = StateActionFunction(observation_dim, action_dim, hidden_units)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.net1(x), self.net2(x)


class SACAgent:
    def __init__(self, observation_dim, action_dim=3, config=None, *, seed=0, device="cpu"):
        self.config = config or SACConfig()
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("SAC supports cpu or cuda devices")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu")
        torch.manual_seed(seed)
        self.observation_dim, self.action_dim = observation_dim, action_dim
        args = (observation_dim, action_dim, self.config.hidden_units)
        self.actor = GaussianPolicy(*args).to(self.device)
        self.critic = TwinQ(*args).to(self.device)
        self.target_critic = TwinQ(*args).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.q_lr)
        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.entropy_lr)
        self.target_entropy = -float(action_dim)
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.detach().exp()

    def act(self, observation, *, deterministic=False):
        observation = np.asarray(observation, np.float32)
        if observation.shape != (self.observation_dim,) or not np.isfinite(observation).all():
            raise ValueError("Observation is incompatible with the SAC policy")
        states = torch.as_tensor(observation[None], device=self.device)
        with torch.no_grad():
            action = self.actor.deterministic(states) if deterministic else self.actor(states)[0]
        result = action.cpu().numpy()[0]
        if not np.isfinite(result).all():
            raise RuntimeError("SAC produced a nonfinite action")
        return result

    def update(self, batch):
        states, actions, rewards, next_states, bootstrap = (
            torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in batch
        )
        # Preserve author order: actor -> entropy -> critics -> target critics.
        sampled, entropies, _ = self.actor(states)
        q1, q2 = self.critic(states, sampled)
        actor_loss = (-torch.minimum(q1, q2) - self.alpha * entropies).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        entropy_loss = -(self.log_alpha * (self.target_entropy - entropies.detach())).mean()
        self.alpha_optimizer.zero_grad()
        entropy_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            next_actions, next_entropies, _ = self.actor(next_states)
            target1, target2 = self.target_critic(next_states, next_actions)
            targets = rewards + bootstrap * (
                torch.minimum(target1, target2) + self.alpha * next_entropies
            )
        current1, current2 = self.critic(states, actions)
        critic_loss = (current1 - targets).square().mean() + (current2 - targets).square().mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        with torch.no_grad():
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.copy_(target * (1 - self.config.tau) + source * self.config.tau)
        self.updates += 1
        stats = dict(
            policy_loss=float(actor_loss.detach()),
            q_loss=float(critic_loss.detach()),
            entropy_loss=float(entropy_loss.detach()),
            alpha=float(self.alpha.item()),
        )
        if not all(np.isfinite(v) for v in stats.values()):
            raise RuntimeError("SAC update produced nonfinite losses")
        return stats

    def save(self, path, metadata=None):
        """Save inference/optimizer state; replay and simulator state are not resumed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            format_version=1,
            adapter_version=ADAPTER_VERSION,
            source_revision=SOURCE_REVISION,
            algorithm="SAC-3step",
            config=asdict(self.config),
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            target_critic=self.target_critic.state_dict(),
            log_alpha=self.log_alpha.detach(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(),
            alpha_optimizer=self.alpha_optimizer.state_dict(),
            updates=self.updates,
            metadata=metadata or {},
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=True)
        if (
            payload.get("format_version") != 1
            or payload.get("algorithm") != "SAC-3step"
            or payload.get("adapter_version") != ADAPTER_VERSION
        ):
            raise ValueError("Unsupported SAC checkpoint format or task adapter")
        config = dict(payload["config"])
        config["hidden_units"] = tuple(config["hidden_units"])
        agent = cls(
            payload["observation_dim"], payload["action_dim"], SACConfig(**config), device=device
        )
        agent.actor.load_state_dict(payload["actor"])
        agent.critic.load_state_dict(payload["critic"])
        agent.target_critic.load_state_dict(payload["target_critic"])
        with torch.no_grad():
            agent.log_alpha.copy_(payload["log_alpha"])
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        agent.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        agent.updates = payload["updates"]
        return agent, payload["metadata"]
