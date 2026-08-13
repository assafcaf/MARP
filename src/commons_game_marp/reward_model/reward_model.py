from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


# Default chunk size for processing long sequences
DEFAULT_CHUNK_SIZE: int = 512


class RewardModel(nn.Module):
    """Bradley-Terry reward model over single `(observation, action)` pairs.

    `obs_scale` is applied to the observation *inside* `forward`, on the
    compute device. That lets the whole pipeline -- preference buffer, host to
    device transfer -- carry raw `uint8` frames and pay the 4x float32 cost
    only for the few hundred rows actually inside a forward pass. Set it to
    `1/255` to reproduce the "normalized observation" scale and to `1.0` for
    raw pixel values.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        num_actions: int = 8,
        obs_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.obs_scale = float(obs_scale)
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
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        if actions.ndim == 0:
            actions = actions.unsqueeze(0)
        if obs.ndim != 4:
            raise ValueError(f"Expected obs with shape (B,H,W,C), got {obs.shape}")
        x = obs.permute(0, 3, 1, 2).float()
        if self.obs_scale != 1.0:
            x = x * self.obs_scale
        x = self.conv(x)
        x = self.encoder(x)
        action_feat = F.one_hot(actions.long(), num_classes=self.num_actions).float()
        x = torch.cat([x, action_feat], dim=1)
        reward = self.head(x).squeeze(-1)
        return reward

    def predict(self, obs_img: np.ndarray, action: int) -> float:
        """Score a single `(obs, action)` pair. Prefer `predict_batch`."""
        return float(self.predict_batch(np.asarray(obs_img)[None, ...], [action])[0])

    def predict_batch(self, obs_batch: np.ndarray, actions: Sequence[int]) -> np.ndarray:
        """Score a batch of `(obs, action)` pairs in one forward pass.

        The per-step loop in the trainer used to call `predict` once per agent,
        which meant one kernel launch, one host-to-device copy and one
        `.item()` device synchronisation *per agent per env step*. Batching
        collapses that to one of each per step.
        """
        obs_batch = np.asarray(obs_batch)
        if obs_batch.ndim == 3:
            obs_batch = obs_batch[None, ...]
        dev = self.device
        obs_tensor = torch.from_numpy(np.ascontiguousarray(obs_batch)).to(dev)
        action_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=dev)
        with torch.inference_mode():
            rewards = self.forward(obs_tensor, action_tensor)
        return rewards.float().cpu().numpy()

    def _forward_chunk(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        grad_checkpoint: bool = False,
    ) -> torch.Tensor:
        """Forward pass for a single chunk of observations and actions.

        With `grad_checkpoint`, the chunk's activations are dropped after the
        forward pass and recomputed during backward. Without it, chunking does
        *not* bound training memory: every chunk keeps its activation graph
        alive until `.backward()`, so peak memory tracks the full sequence
        regardless of `chunk_size`.
        """
        if grad_checkpoint and torch.is_grad_enabled():
            return checkpoint(self.forward, obs, actions, use_reentrant=False)
        return self.forward(obs, actions)

    def _to_device(self, array: np.ndarray, dev: torch.device) -> torch.Tensor:
        """Move an observation array to `dev` in its source dtype.

        Deliberately *not* `.float()` on the host: uint8 frames are a quarter
        of the bytes over the bus, and `forward` casts on device anyway.
        """
        return torch.from_numpy(np.ascontiguousarray(array)).to(dev, non_blocking=True)

    def sequence_score(
        self,
        traj: Sequence[Tuple[np.ndarray, int]],
        device: Optional[torch.device] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        grad_checkpoint: bool = False,
    ) -> torch.Tensor:
        """Score a single trajectory, processing in chunks to limit memory."""
        scores = self.batch_sequence_scores(
            [traj], device=device, chunk_size=chunk_size, grad_checkpoint=grad_checkpoint
        )
        return scores[0]

    def batch_sequence_scores(
        self,
        trajectories: Sequence[Sequence[Tuple[np.ndarray, int]]],
        device: Optional[torch.device] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        grad_checkpoint: bool = False,
    ) -> torch.Tensor:
        """
        Score multiple trajectories efficiently using batched forward passes.

        Returns a tensor of shape (num_trajectories,) with the sum of rewards per trajectory.
        Empty trajectories score 0.
        """
        dev = device or self.device

        if len(trajectories) == 0:
            return torch.zeros(0, device=dev)

        # Flatten every step of every trajectory into one array, remembering
        # which trajectory each step came from so the per-step rewards can be
        # summed back per trajectory with a single scatter.
        all_obs: List[np.ndarray] = []
        all_actions: List[int] = []
        traj_indices: List[int] = []
        for idx, traj in enumerate(trajectories):
            for obs, action in traj:
                all_obs.append(obs)
                all_actions.append(action)
                traj_indices.append(idx)

        result = torch.zeros(len(trajectories), device=dev)
        if not all_obs:
            return result

        obs_array = np.stack(all_obs, axis=0)
        actions_array = np.asarray(all_actions, dtype=np.int64)
        indices_array = np.asarray(traj_indices, dtype=np.int64)

        obs_all = self._to_device(obs_array, dev)
        actions_all = torch.from_numpy(actions_array).to(dev, non_blocking=True)
        indices_all = torch.from_numpy(indices_array).to(dev, non_blocking=True)

        total_steps = obs_all.shape[0]
        step = chunk_size if chunk_size and chunk_size > 0 else total_steps

        for start in range(0, total_steps, step):
            end = min(start + step, total_steps)
            chunk_rewards = self._forward_chunk(
                obs_all[start:end], actions_all[start:end], grad_checkpoint
            )
            # float() keeps the accumulator in fp32 even under autocast, where
            # chunk_rewards comes back as fp16 and would not match `result`.
            result = result.index_add(0, indices_all[start:end], chunk_rewards.float())

        return result

    def save(self, path: str) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "obs_shape": self.obs_shape,
            "num_actions": self.num_actions,
            "obs_scale": self.obs_scale,
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str, device: str = "auto") -> "RewardModel":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # weights_only=True: the payload is plain tensors/ints/tuples, so there
        # is no reason to hand a checkpoint file the ability to execute code.
        payload = torch.load(path, map_location=device, weights_only=True)
        model = RewardModel(
            tuple(payload["obs_shape"]),
            int(payload["num_actions"]),
            obs_scale=float(payload.get("obs_scale", 1.0)),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(torch.device(device))
        return model
