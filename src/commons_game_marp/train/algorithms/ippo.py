import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ...env.vec_env import agents_to_rows, rows_to_agents
from .base import Algorithm
from .gae import compute_gae
from ..entropy_control import EntropyController


def orthogonal_init(
    module: nn.Module,
    head_gain: float = 1.0,
    trunk_gain: float = np.sqrt(2),
) -> None:
    """Orthogonally initialize a module with layer-wise gains.

    The final Linear/Conv2d layer is the output head and receives ``head_gain``;
    every preceding layer is trunk and receives ``trunk_gain``. Applying a small
    head gain uniformly -- as this function previously did -- drives the trunk's
    activations toward zero and leaves the network near-dead at initialization.

    Head detection relies on ``module.modules()`` registration order matching
    the network's forward order, so the last Linear/Conv2d layer encountered is
    assumed to be the output head. A module that registers its head submodule
    before its trunk (e.g. assigns ``self.head`` before ``self.conv``) would
    have its gains silently inverted.
    """
    layers = [m for m in module.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    for index, layer in enumerate(layers):
        is_head = index == len(layers) - 1
        nn.init.orthogonal_(layer.weight, gain=head_gain if is_head else trunk_gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class SingleAgentBuffer:
    """Rollout buffer for one agent across `num_envs` parallel environments.

    Every entry carries a leading env axis, so the stored arrays are (T,
    num_envs). `size()` reports T -- per-env timesteps -- which is what
    `n_steps` counts, following SB3's convention; the update batch is
    therefore n_steps * num_envs.
    """

    def __init__(self, num_envs: int = 1) -> None:
        self.num_envs = num_envs
        self.clear()

    def clear(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.logprobs: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.dones: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.next_values: List[np.ndarray] = []

    def add_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        logprob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        next_value: np.ndarray,
    ) -> None:
        """Record one timestep. `obs` is (num_envs, ...); the rest (num_envs,)."""
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.next_values.append(next_value)

    def size(self) -> int:
        """Timesteps per environment, not rows."""
        return len(self.rewards)

    def compute_advantages(
        self, gamma: float, gae_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        return compute_gae(
            rewards=np.stack(self.rewards, axis=0),
            dones=np.stack(self.dones, axis=0),
            values=np.stack(self.values, axis=0),
            next_values=np.stack(self.next_values, axis=0),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )


class MLPActor(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class CNNActor(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int) -> None:
        super().__init__()
        height, width, channels = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_out = self.conv(dummy)
            conv_out_size = int(np.prod(conv_out.shape[1:]))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 4:
            raise ValueError(f"Expected obs with shape (B,H,W,C), got {obs.shape}")
        x = obs.permute(0, 3, 1, 2)
        x = self.conv(x)
        return self.head(x)


class CNNCritic(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int]) -> None:
        super().__init__()
        height, width, channels = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_out = self.conv(dummy)
            conv_out_size = int(np.prod(conv_out.shape[1:]))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 4:
            raise ValueError(f"Expected obs with shape (B,H,W,C), got {obs.shape}")
        x = obs.permute(0, 3, 1, 2)
        x = self.conv(x)
        return self.head(x).squeeze(-1)


class IPPOAlgorithm(Algorithm):
    """
    Independent PPO (IPPO) - True decentralized training with decentralized execution.
    
    Each agent has its own:
    - Actor network (policy)
    - Critic network (value function using local observations only)
    - Optimizer
    - Rollout buffer
    
    All agents train simultaneously, learning from interactions with each other's
    evolving policies (not random/fixed opponents).
    
    Features:
    - Orthogonal weight initialization for stable training
    - One EntropyController per agent (adaptive by default), targeting a
      policy entropy rather than scheduling the coefficient directly
    - Value function clipping for stability
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.agent_ids: List[str] = []
        self.num_envs = 1
        self.num_agents = 0
        self.num_actions = 0
        self.device = torch.device("cpu")
        self.obs_shape: Tuple[int, ...] = ()
        self.obs_dim: int = 0
        
        # Per-agent components
        self.actors: Dict[str, nn.Module] = {}
        self.critics: Dict[str, nn.Module] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.buffers: Dict[str, SingleAgentBuffer] = {}
        self.ent_controllers: Dict[str, EntropyController] = {}

        self._last_step: Dict[str, Dict[str, Any]] = {}
        self._last_metrics: Dict[str, float] = {}
        
        # Training progress tracking for entropy annealing
        self._total_episodes = 0
        self._current_episode = 0

    def uses_external_loop(self) -> bool:
        return True

    def set_total_episodes(self, total: int) -> None:
        """Set total episodes for the anneal-mode entropy schedule."""
        self._total_episodes = total
        for controller in self.ent_controllers.values():
            controller.set_total_episodes(total)

    def on_env_ready(self, env) -> None:
        obs_space = env.observation_space["curr_obs"]
        self.obs_shape = obs_space.shape
        self.num_actions = int(env.action_space.n)
        self.agent_ids = list(env.agent_ids)
        self.num_envs = int(env.num_envs)
        self.num_agents = len(self.agent_ids)

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Create per-agent networks, optimizers, and buffers
        for agent_id in self.agent_ids:
            if self.config.flatten_obs:
                self.obs_dim = int(np.prod(self.obs_shape))
                actor = MLPActor(
                    self.obs_dim, self.num_actions, self.config.hidden_size
                ).to(self.device)
                critic = MLPCritic(self.obs_dim, self.config.hidden_size).to(self.device)
            else:
                actor = CNNActor(self.obs_shape, self.num_actions).to(self.device)
                critic = CNNCritic(self.obs_shape).to(self.device)

            # Apply orthogonal initialization for stable training
            orthogonal_init(actor, head_gain=0.01)
            orthogonal_init(critic, head_gain=1.0)

            optimizer = torch.optim.Adam(
                list(actor.parameters()) + list(critic.parameters()),
                lr=self.config.learning_rate,
                eps=1e-5,  # PPO convention; PyTorch's 1e-8 default is less stable here
            )

            self.actors[agent_id] = actor
            self.critics[agent_id] = critic
            self.optimizers[agent_id] = optimizer
            self.buffers[agent_id] = SingleAgentBuffer(self.num_envs)
            self.ent_controllers[agent_id] = EntropyController(
                self.config, self.num_actions, self.device
            )
            self.ent_controllers[agent_id].set_total_episodes(self._total_episodes)

    def _format_obs(self, obs_batch: np.ndarray) -> np.ndarray:
        """One agent's (num_envs, ...) uint8 rows -> the policy's input dtype."""
        img = obs_batch.astype(np.float32)
        if self.config.normalize_obs:
            img = img / 255.0
        if self.config.flatten_obs:
            return img.reshape(obs_batch.shape[0], -1)
        return img

    def act(self, observations: np.ndarray, step: int) -> np.ndarray:
        per_agent = rows_to_agents(observations, self.num_envs, self.agent_ids)
        actions: Dict[str, np.ndarray] = {}
        self._last_step = {}

        for agent_id in self.agent_ids:
            obs = self._format_obs(per_agent[agent_id])
            obs_tensor = torch.from_numpy(obs).float().to(self.device)

            with torch.no_grad():
                logits = self.actors[agent_id](obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
                value = self.critics[agent_id](obs_tensor)

            actions_np = action.cpu().numpy().astype(np.int64)
            actions[agent_id] = actions_np

            self._last_step[agent_id] = {
                "obs": obs,
                "action": actions_np,
                "logprob": logprob.cpu().numpy().astype(np.float32),
                "value": value.cpu().numpy().astype(np.float32),
            }

        return agents_to_rows(actions, self.num_envs, self.agent_ids)

    def observe(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        step: int,
    ) -> None:
        if not self._last_step:
            return

        next_by_agent = rows_to_agents(next_observations, self.num_envs, self.agent_ids)
        rew_by_agent = rows_to_agents(rewards, self.num_envs, self.agent_ids)
        done_by_agent = rows_to_agents(dones, self.num_envs, self.agent_ids)

        for agent_id in self.agent_ids:
            if agent_id not in self._last_step:
                continue

            last = self._last_step[agent_id]

            # Compute next value
            next_obs = self._format_obs(next_by_agent[agent_id])
            next_obs_tensor = torch.from_numpy(next_obs).float().to(self.device)
            with torch.no_grad():
                next_value = (
                    self.critics[agent_id](next_obs_tensor)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            self.buffers[agent_id].add_step(
                obs=last["obs"],
                action=last["action"],
                logprob=last["logprob"],
                reward=np.asarray(rew_by_agent[agent_id], dtype=np.float32),
                done=np.asarray(done_by_agent[agent_id], dtype=np.float32),
                value=last["value"],
                next_value=next_value,
            )

        self._last_step = {}

        # The `or done_all` trigger the dict interface carried is gone: episodes
        # are lockstep and fixed-length, and `on_episode_end` already flushes
        # whatever is left in the buffers at the boundary.
        min_buffer_size = min(buf.size() for buf in self.buffers.values())
        if min_buffer_size >= self.config.n_steps:
            self._update_all()

    def on_episode_end(self, episode: int) -> Dict[str, Any]:
        # Update any remaining data in buffers
        if any(buf.size() > 0 for buf in self.buffers.values()):
            self._update_all()

        # Track episode for the anneal-mode schedule
        self._current_episode = episode + 1
        for controller in self.ent_controllers.values():
            controller.set_episode(self._current_episode)

        metrics = dict(self._last_metrics)
        per_agent = {
            agent_id: controller.coefficient()
            for agent_id, controller in self.ent_controllers.items()
        }
        metrics["ent_coef_per_agent"] = per_agent
        metrics["ent_coef"] = sum(per_agent.values()) / len(per_agent) if per_agent else 0.0
        if self.ent_controllers:
            metrics["target_entropy"] = next(
                iter(self.ent_controllers.values())
            ).target_entropy
            metrics["ent_coef_saturated"] = sum(
                1 for controller in self.ent_controllers.values() if controller.is_saturated()
            ) / len(self.ent_controllers)
        self._last_metrics = {}
        return metrics

    def _update_all(self) -> None:
        """Update all agents independently."""
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_batches = 0
        entropy_per_agent: Dict[str, float] = {}

        for agent_id in self.agent_ids:
            buffer = self.buffers[agent_id]
            if buffer.size() == 0:
                continue

            metrics = self._update_agent(agent_id)
            total_loss += metrics.get("loss", 0.0)
            total_policy += metrics.get("policy_loss", 0.0)
            total_value += metrics.get("value_loss", 0.0)
            total_entropy += metrics.get("entropy", 0.0)
            entropy_per_agent[agent_id] = metrics.get("entropy", 0.0)
            total_batches += 1

            buffer.clear()

        if total_batches > 0:
            self._last_metrics = {
                "loss": total_loss / total_batches,
                "policy_loss": total_policy / total_batches,
                "value_loss": total_value / total_batches,
                "entropy": total_entropy / total_batches,
                "entropy_per_agent": entropy_per_agent,
            }

    def _update_agent(self, agent_id: str) -> Dict[str, float]:
        """Perform PPO update for a single agent with value clipping and a
        per-agent entropy-targeting coefficient."""
        buffer = self.buffers[agent_id]
        actor = self.actors[agent_id]
        critic = self.critics[agent_id]
        optimizer = self.optimizers[agent_id]

        advantages, returns = buffer.compute_advantages(
            gamma=self.config.gamma, gae_lambda=self.config.gae_lambda
        )

        # (T, num_envs, ...) -> (T * num_envs, ...). Advantages were computed
        # per column first, so no gradient path crosses an environment.
        stacked_obs = np.stack(buffer.obs, axis=0)
        T, E = stacked_obs.shape[0], stacked_obs.shape[1]
        total = T * E
        obs = stacked_obs.reshape(total, *stacked_obs.shape[2:])
        actions = np.stack(buffer.actions, axis=0).reshape(total).astype(np.int64)
        old_logprobs = np.stack(buffer.logprobs, axis=0).reshape(total).astype(np.float32)
        old_values = np.stack(buffer.values, axis=0).reshape(total).astype(np.float32)
        advantages = advantages.reshape(total)
        returns = returns.reshape(total)

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().to(self.device)
        old_logprobs_tensor = torch.from_numpy(old_logprobs).float().to(self.device)
        old_values_tensor = torch.from_numpy(old_values).float().to(self.device)
        advantages_tensor = torch.from_numpy(advantages).float().to(self.device)
        returns_tensor = torch.from_numpy(returns).float().to(self.device)

        batch_size = min(int(self.config.batch_size), total)

        controller = self.ent_controllers[agent_id]
        vf_clip = getattr(self.config, "vf_clip", None)

        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_batches = 0

        for _ in range(int(self.config.update_epochs)):
            indices = np.random.permutation(total)
            for start in range(0, total, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]
                mb_obs = obs_tensor[mb_idx]
                mb_actions = actions_tensor[mb_idx]
                mb_old_logprobs = old_logprobs_tensor[mb_idx]
                mb_old_values = old_values_tensor[mb_idx]
                mb_adv = advantages_tensor[mb_idx]
                mb_returns = returns_tensor[mb_idx]

                # Policy loss
                logits = actor(mb_obs)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_logprobs - mb_old_logprobs).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range
                ) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with optional clipping
                values_pred = critic(mb_obs)
                if vf_clip is not None and vf_clip > 0:
                    # Clipped value loss (PPO-style)
                    values_clipped = mb_old_values + torch.clamp(
                        values_pred - mb_old_values, -vf_clip, vf_clip
                    )
                    value_loss_unclipped = (mb_returns - values_pred).pow(2)
                    value_loss_clipped = (mb_returns - values_clipped).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = (mb_returns - values_pred).pow(2).mean()

                # Total loss with the current entropy coefficient
                current_ent_coef = controller.coefficient()
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - current_ent_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                # Clipped separately: a joint norm lets a critic-loss spike
                # scale the actor's gradient down with it. value_loss went
                # 0.02 -> 0.14 around episode 100 of the analysed run, exactly
                # where entropy first dropped.
                nn.utils.clip_grad_norm_(actor.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), self.config.max_grad_norm)
                optimizer.step()
                controller.observe_entropy(float(entropy.item()))

                total_loss += float(loss.item())
                total_policy += float(policy_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_batches += 1

        if total_batches > 0:
            return {
                "loss": total_loss / total_batches,
                "policy_loss": total_policy / total_batches,
                "value_loss": total_value / total_batches,
                "entropy": total_entropy / total_batches,
            }
        return {}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "agent_ids": self.agent_ids,
            "config": asdict(self.config),
            "agents": {},
        }
        for agent_id in self.agent_ids:
            payload["agents"][agent_id] = {
                "actor_state": self.actors[agent_id].state_dict(),
                "critic_state": self.critics[agent_id].state_dict(),
            }
        torch.save(payload, path)

