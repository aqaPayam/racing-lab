"""TD3 learning rule following Evans et al.'s pinned F1TENTH source.

Modified Apache-2.0 source: configurable dimensions, device, safe serialization,
separate exploration RNG, explicit bootstrap discounts and update counters.
See THIRD_PARTY_NOTICES.md and TD3_LICENSE.txt. No expert/controller input.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TD3_SOURCE_REVISION = "c9730789dd0fa85fc18dd4b600c5246e134ebe22"
TD3_PAPER_URL = "https://arxiv.org/abs/2402.18558"


@dataclass(frozen=True)
class TD3Config:
    hidden_units: tuple[int, int] = (100, 100)
    batch_size: int = 100
    gamma: float = 0.99
    tau: float = 0.005
    policy_lr: float = 0.001
    q_lr: float = 0.001
    replay_capacity: int = 100_000
    start_steps: int = 100
    exploration_noise: float = 0.1
    target_noise: float = 0.2
    noise_clip: float = 0.5
    critic_steps_per_call: int = 2
    policy_delay: int = 2
    eval_interval: int = 100_000
    eval_episodes: int = 1
    checkpoint_interval: int = 200_000
    n_step: int = 1

    def __post_init__(self):
        if len(self.hidden_units) != 2 or any(
            type(v) is not int or v < 1 for v in self.hidden_units
        ):
            raise ValueError("TD3 requires two positive hidden-layer sizes")
        for name in ("batch_size", "replay_capacity", "critic_steps_per_call", "policy_delay"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("start_steps", "eval_interval", "checkpoint_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.n_step != 1 or self.eval_episodes != 1:
            raise ValueError(
                "This TD3 profile requires one-step targets and one evaluation rollout"
            )
        if self.replay_capacity < self.batch_size or self.start_steps < self.batch_size:
            raise ValueError("Replay and learning start must accommodate a full batch")
        for name in ("gamma", "tau"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be in (0,1)")
        for name in ("policy_lr", "q_lr"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("exploration_noise", "target_noise", "noise_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc_mu = nn.Linear(hidden[1], action_dim)

    def forward(self, state):
        return self.fc_mu(F.relu(self.fc2(F.relu(self.fc1(state))))).tanh()


class QFunction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc_out = nn.Linear(hidden[1], 1)

    def forward(self, state, action):
        return self.fc_out(F.relu(self.fc2(F.relu(self.fc1(torch.cat((state, action), 1))))))


class TD3Agent:
    algorithm = "TD3-F1TENTH"

    def __init__(self, observation_dim, action_dim=3, config=None, *, seed=0, device="cpu"):
        self.config = config or TD3Config()
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("TD3 supports cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.observation_dim, self.action_dim = observation_dim, action_dim
        args = (observation_dim, action_dim, self.config.hidden_units)
        # Preserve the author's initialization order, including target RNG consumption.
        self.actor = Policy(*args).to(self.device)
        self.actor_target = Policy(*args).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1 = QFunction(*args).to(self.device)
        self.critic_target_1 = QFunction(*args).to(self.device)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_2 = QFunction(*args).to(self.device)
        self.critic_target_2 = QFunction(*args).to(self.device)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        for target in (self.actor_target, self.critic_target_1, self.critic_target_2):
            target.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.policy_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=self.config.q_lr,
        )
        self.updates = self.actor_updates = self.training_calls = 0

    def act(self, observation, *, deterministic=False):
        observation = np.asarray(observation, np.float32)
        if observation.shape != (self.observation_dim,) or not np.isfinite(observation).all():
            raise ValueError("Observation is incompatible with the TD3 policy")
        with torch.no_grad():
            result = self.actor(torch.as_tensor(observation[None], device=self.device))
        result = result.cpu().numpy()[0]
        if not deterministic:
            result = result + self.rng.normal(0, self.config.exploration_noise, self.action_dim)
        if not np.isfinite(result).all():
            raise RuntimeError("TD3 produced a nonfinite action")
        return np.clip(result, -1, 1).astype(np.float32)

    def update(self, sample_batch):
        """Two independent batches; actor/targets update after local iteration zero.

        The source resets its delay counter on each call. This deliberately differs
        from implementations that delay the actor until the second global critic step.
        """
        stats = {}
        for local_step in range(self.config.critic_steps_per_call):
            states, actions, rewards, next_states, bootstrap = (
                torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in sample_batch()
            )
            with torch.no_grad():
                noise = torch.normal(torch.zeros_like(actions), self.config.target_noise)
                noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
                next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
                targets = rewards + bootstrap * torch.minimum(
                    self.critic_target_1(next_states, next_actions),
                    self.critic_target_2(next_states, next_actions),
                )
            loss = F.mse_loss(self.critic_1(states, actions), targets) + F.mse_loss(
                self.critic_2(states, actions), targets
            )
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
            self.updates += 1
            stats["q_loss"] = float(loss.detach())
            if local_step % self.config.policy_delay == 0:
                actor_loss = -self.critic_1(states, self.actor(states)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.actor_updates += 1
                stats["policy_loss"] = float(actor_loss.detach())
                with torch.no_grad():
                    for source, target in (
                        (self.actor, self.actor_target),
                        (self.critic_1, self.critic_target_1),
                        (self.critic_2, self.critic_target_2),
                    ):
                        for current, averaged in zip(source.parameters(), target.parameters()):
                            averaged.copy_(
                                averaged * (1 - self.config.tau) + current * self.config.tau
                            )
        self.training_calls += 1
        if not all(np.isfinite(v) for v in stats.values()):
            raise RuntimeError("TD3 update produced nonfinite losses")
        return stats

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            format_version=1,
            algorithm=self.algorithm,
            source_revision=TD3_SOURCE_REVISION,
            config=asdict(self.config),
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            updates=self.updates,
            actor_updates=self.actor_updates,
            training_calls=self.training_calls,
            metadata=metadata or {},
        )
        for name in (
            "actor",
            "actor_target",
            "critic_1",
            "critic_2",
            "critic_target_1",
            "critic_target_2",
            "actor_optimizer",
            "critic_optimizer",
        ):
            payload[name] = getattr(self, name).state_dict()
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("format_version") != 1 or payload.get("algorithm") != cls.algorithm:
            raise ValueError("Unsupported TD3 checkpoint")
        config = dict(payload["config"])
        config["hidden_units"] = tuple(config["hidden_units"])
        agent = cls(
            payload["observation_dim"], payload["action_dim"], TD3Config(**config), device=device
        )
        for name in (
            "actor",
            "actor_target",
            "critic_1",
            "critic_2",
            "critic_target_1",
            "critic_target_2",
            "actor_optimizer",
            "critic_optimizer",
        ):
            getattr(agent, name).load_state_dict(payload[name])
        agent.updates, agent.actor_updates = payload["updates"], payload["actor_updates"]
        agent.training_calls = payload["training_calls"]
        return agent, payload["metadata"]
