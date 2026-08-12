import os
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .base import Algorithm


class RolloutBuffer:
    def __init__(self, num_agents: int) -> None:
        self.num_agents = num_agents
        self.clear()

    def clear(self) -> None:
        self.local_obs: List[np.ndarray] = []
        self.global_obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.logprobs: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.dones: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.next_values: List[np.ndarray] = []

    def add_step(
        self,
        local_obs: np.ndarray,
        global_obs: np.ndarray,
        actions: np.ndarray,
        logprobs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
    ) -> None:
        self.local_obs.append(local_obs)
        self.global_obs.append(global_obs)
        self.actions.append(actions)
        self.logprobs.append(logprobs)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.values.append(values)
        self.next_values.append(next_values)

    def size(self) -> int:
        return len(self.rewards)

    def compute_advantages(
        self, gamma: float, gae_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.stack(self.rewards, axis=0)
        dones = np.stack(self.dones, axis=0).astype(np.float32)
        values = np.stack(self.values, axis=0)
        next_values = np.stack(self.next_values, axis=0)
        T, N = rewards.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        last_adv = np.zeros((N,), dtype=np.float32)
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values[t] * mask - values[t]
            last_adv = delta + gamma * gae_lambda * mask * last_adv
            advantages[t] = last_adv
        returns = advantages + values
        return advantages, returns


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
    def __init__(self, obs_dim: int, num_agents: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_agents),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


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
    def __init__(self, obs_shape: Tuple[int, int, int], num_agents: int) -> None:
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
            nn.Linear(128, num_agents),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 4:
            raise ValueError(f"Expected obs with shape (B,H,W,C), got {obs.shape}")
        x = obs.permute(0, 3, 1, 2)
        x = self.conv(x)
        return self.head(x)


class MAPPOAlgorithm(Algorithm):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.agent_ids: List[str] = []
        self.agent_index: Dict[str, int] = {}
        self.num_actions = 0
        self.device = torch.device("cpu")
        self.actor: nn.Module
        self.critic: nn.Module
        self.optimizer: torch.optim.Optimizer
        self.buffer: RolloutBuffer
        self._last_step: Dict[str, Any] = {}
        self._last_metrics: Dict[str, float] = {}

    def uses_external_loop(self) -> bool:
        return True

    def on_env_ready(self, env) -> None:
        obs_space = env.observation_space["curr_obs"]
        obs_shape = obs_space.shape
        self.num_actions = int(env.action_space.n)
        self.agent_ids = list(env.agents.keys())
        self.agent_index = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}
        num_agents = len(self.agent_ids)

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if self.config.flatten_obs:
            obs_dim = int(np.prod(obs_shape))
            global_dim = obs_dim * num_agents
            self.actor = MLPActor(obs_dim, self.num_actions, self.config.hidden_size).to(self.device)
            self.critic = MLPCritic(global_dim, num_agents, self.config.hidden_size).to(self.device)
        else:
            global_shape = (obs_shape[0], obs_shape[1], obs_shape[2] * num_agents)
            self.actor = CNNActor(obs_shape, self.num_actions).to(self.device)
            self.critic = CNNCritic(global_shape, num_agents).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config.learning_rate,
        )
        self.buffer = RolloutBuffer(num_agents)

    def _format_local_obs(self, obs: Dict[str, Any], agent_id: str) -> np.ndarray:
        img = obs[agent_id]["curr_obs"]
        if self.config.normalize_obs:
            img = (img / 255.0).astype(np.float32)
        else:
            img = img.astype(np.float32)
        if self.config.flatten_obs:
            return img.reshape(-1)
        return img

    def _format_global_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        imgs = []
        for agent_id in self.agent_ids:
            if agent_id not in obs:
                raise RuntimeError(f"Agent '{agent_id}' missing from observations.")
            imgs.append(self._format_local_obs(obs, agent_id))
        if self.config.flatten_obs:
            return np.concatenate(imgs, axis=0)
        return np.concatenate(imgs, axis=2)

    def act(self, observations: Dict[str, Any], step: int) -> Dict[str, int]:
        local_obs = np.stack(
            [self._format_local_obs(observations, agent_id) for agent_id in self.agent_ids], axis=0
        )
        global_obs = self._format_global_obs(observations)
        obs_tensor = torch.from_numpy(local_obs).float().to(self.device)
        global_tensor = torch.from_numpy(global_obs).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.actor(obs_tensor)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            values = self.critic(global_tensor).squeeze(0)

        actions_np = actions.cpu().numpy()
        self._last_step = {
            "local_obs": local_obs,
            "global_obs": global_obs,
            "actions": actions_np,
            "logprobs": logprobs.cpu().numpy(),
            "values": values.cpu().numpy(),
        }
        return {agent_id: int(actions_np[idx]) for idx, agent_id in enumerate(self.agent_ids)}

    def observe(
        self,
        observations: Dict[str, Any],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_observations: Dict[str, Any],
        dones: Dict[str, bool],
        infos: Dict[str, Any],
        step: int,
    ) -> None:
        if not self._last_step:
            return
        done_all = bool(dones.get("__all__", False))
        rewards_arr = np.array([float(rewards[a]) for a in self.agent_ids], dtype=np.float32)
        dones_arr = np.array(
            [bool(dones.get(a, False)) or done_all for a in self.agent_ids], dtype=np.float32
        )
        next_global = self._format_global_obs(next_observations)
        next_global_tensor = torch.from_numpy(next_global).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_values = self.critic(next_global_tensor).squeeze(0).cpu().numpy()

        self.buffer.add_step(
            local_obs=self._last_step["local_obs"],
            global_obs=self._last_step["global_obs"],
            actions=self._last_step["actions"],
            logprobs=self._last_step["logprobs"],
            rewards=rewards_arr,
            dones=dones_arr,
            values=self._last_step["values"],
            next_values=next_values,
        )

        self._last_step = {}
        if self.buffer.size() >= self.config.n_steps or done_all:
            self._update()
            self.buffer.clear()

    def on_episode_end(self, episode: int) -> Dict[str, Any]:
        if self.buffer.size() > 0:
            self._update()
            self.buffer.clear()
        metrics = dict(self._last_metrics)
        self._last_metrics = {}
        return metrics

    def _update(self) -> None:
        advantages, returns = self.buffer.compute_advantages(
            gamma=self.config.gamma, gae_lambda=self.config.gae_lambda
        )
        local_obs = np.stack(self.buffer.local_obs, axis=0)
        global_obs = np.stack(self.buffer.global_obs, axis=0)
        actions = np.stack(self.buffer.actions, axis=0)
        logprobs = np.stack(self.buffer.logprobs, axis=0)
        values = np.stack(self.buffer.values, axis=0)
        T, N = actions.shape
        local_obs = local_obs.reshape(T * N, *local_obs.shape[2:])
        global_obs = np.repeat(global_obs, N, axis=0)
        actions = actions.reshape(T * N)
        logprobs = logprobs.reshape(T * N)
        values = values.reshape(T * N)
        advantages = advantages.reshape(T * N)
        returns = returns.reshape(T * N)
        agent_idx = np.tile(np.arange(N, dtype=np.int64), T)

        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        obs_tensor = torch.from_numpy(local_obs).float().to(self.device)
        global_tensor = torch.from_numpy(global_obs).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).long().to(self.device)
        old_logprobs_tensor = torch.from_numpy(logprobs).float().to(self.device)
        advantages_tensor = torch.from_numpy(advantages).float().to(self.device)
        returns_tensor = torch.from_numpy(returns).float().to(self.device)
        agent_idx_tensor = torch.from_numpy(agent_idx).long().to(self.device)

        batch_size = int(self.config.batch_size)
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_batches = 0

        for _ in range(int(self.config.update_epochs)):
            indices = np.random.permutation(T * N)
            for start in range(0, T * N, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]
                mb_obs = obs_tensor[mb_idx]
                mb_global = global_tensor[mb_idx]
                mb_actions = actions_tensor[mb_idx]
                mb_old_logprobs = old_logprobs_tensor[mb_idx]
                mb_adv = advantages_tensor[mb_idx]
                mb_returns = returns_tensor[mb_idx]
                mb_agent_idx = agent_idx_tensor[mb_idx]

                logits = self.actor(mb_obs)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_logprobs - mb_old_logprobs).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                values_all = self.critic(mb_global)
                values_pred = values_all.gather(1, mb_agent_idx.unsqueeze(1)).squeeze(1)
                value_loss = (mb_returns - values_pred).pow(2).mean()

                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                total_loss += float(loss.item())
                total_policy += float(policy_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_batches += 1

        if total_batches > 0:
            self._last_metrics = {
                "loss": total_loss / total_batches,
                "policy_loss": total_policy / total_batches,
                "value_loss": total_value / total_batches,
                "entropy": total_entropy / total_batches,
            }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "config": asdict(self.config),
            "agent_ids": self.agent_ids,
        }
        torch.save(payload, path)
