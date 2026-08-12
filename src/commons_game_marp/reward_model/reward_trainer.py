from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from .oracle import compute_phi, preference
from .preference_buffer import EpisodeRecord, PreferenceBuffer
from .reward_model import RewardModel

# Check if CUDA AMP is available
_AMP_AVAILABLE = torch.cuda.is_available()


class RewardModelTrainer:
    def __init__(
        self,
        reward_model: RewardModel,
        lr: float = 1e-4,
        device: str = "auto",
        use_amp: bool = True,
        chunk_size: int = 512,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.reward_model = reward_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=lr)

        # Mixed precision training
        self.use_amp = use_amp and _AMP_AVAILABLE and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Chunk size for batched processing
        self.chunk_size = chunk_size

    def _pair_sequences(
        self, buffer: PreferenceBuffer, ep_i: EpisodeRecord, ep_j: EpisodeRecord, mode: str
    ) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]]]:
        if mode == "input_aggregation":
            return buffer.aggregate_episode(ep_i), buffer.aggregate_episode(ep_j)
        if mode == "narrow_view":
            return buffer.sample_agent_trajectory(ep_i), buffer.sample_agent_trajectory(ep_j)
        raise ValueError(f"Unsupported reward model mode: {mode}")

    @staticmethod
    def _weights_from_deltas(deltas: torch.Tensor) -> torch.Tensor:
        if deltas.numel() == 0:
            return deltas
        mean = deltas.mean()
        std = deltas.std(unbiased=False)
        z = (deltas - mean) / (std + 1e-6)
        weights = torch.softmax(z, dim=0)
        return weights

    def _episode_score_corr(
        self,
        buffer: PreferenceBuffer,
        batch_pairs: List[Tuple[EpisodeRecord, EpisodeRecord]],
        phi_key: str,
    ) -> Optional[float]:
        """Compute correlation between model scores and phi values (batched)."""
        episodes = []
        seen = set()
        for ep_i, ep_j in batch_pairs:
            if id(ep_i) not in seen:
                episodes.append(ep_i)
                seen.add(id(ep_i))
            if id(ep_j) not in seen:
                episodes.append(ep_j)
                seen.add(id(ep_j))
        if len(episodes) < 2:
            return None

        # Batch score computation
        sequences = [buffer.aggregate_episode(ep) for ep in episodes]
        phis = [compute_phi(ep.metrics, phi_key) for ep in episodes]

        self.reward_model.eval()
        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    scores_tensor = self.reward_model.batch_sequence_scores(
                        sequences, device=self.device, chunk_size=self.chunk_size
                    )
            else:
                scores_tensor = self.reward_model.batch_sequence_scores(
                    sequences, device=self.device, chunk_size=self.chunk_size
                )
            scores = scores_tensor.float().cpu().numpy()

        phis_arr = np.array(phis, dtype=np.float32)
        if scores.std() == 0 or phis_arr.std() == 0:
            return None
        corr = float(np.corrcoef(scores, phis_arr)[0, 1])
        return corr

    def train(
        self,
        buffer: PreferenceBuffer,
        phi_key: str,
        mode: str,
        batch_pairs: int,
        train_steps: int,
    ) -> Dict[str, float]:
        if len(buffer) < 2:
            return {}
        losses: List[float] = []
        accuracies: List[float] = []
        last_pairs: List[Tuple[EpisodeRecord, EpisodeRecord]] = []

        self.reward_model.train()
        for _ in range(train_steps):
            pairs = buffer.sample_episode_pairs(batch_pairs)
            if not pairs:
                break
            last_pairs = pairs

            # Collect all sequences and metadata
            sequences_i = []
            sequences_j = []
            mus = []
            deltas = []

            for ep_i, ep_j in pairs:
                phi_i = compute_phi(ep_i.metrics, phi_key)
                phi_j = compute_phi(ep_j.metrics, phi_key)
                mu, delta = preference(phi_i, phi_j)
                seq_i, seq_j = self._pair_sequences(buffer, ep_i, ep_j, mode)
                sequences_i.append(seq_i)
                sequences_j.append(seq_j)
                mus.append(mu)
                deltas.append(delta)

            # Batched scoring with optional mixed precision
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    scores_i_t = self.reward_model.batch_sequence_scores(
                        sequences_i, device=self.device, chunk_size=self.chunk_size
                    )
                    scores_j_t = self.reward_model.batch_sequence_scores(
                        sequences_j, device=self.device, chunk_size=self.chunk_size
                    )

                # Compute loss outside autocast for numerical stability
                scores_i_t = scores_i_t.float()
                scores_j_t = scores_j_t.float()
                mu_t = torch.tensor(mus, dtype=torch.float32, device=self.device)
                delta_t = torch.tensor(deltas, dtype=torch.float32, device=self.device)
                weights = self._weights_from_deltas(delta_t)
                prob = torch.sigmoid(scores_i_t - scores_j_t)
                bce = F.binary_cross_entropy(prob, mu_t, reduction="none")
                loss = (bce * weights).sum() / (weights.sum() + 1e-8)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                scores_i_t = self.reward_model.batch_sequence_scores(
                    sequences_i, device=self.device, chunk_size=self.chunk_size
                )
                scores_j_t = self.reward_model.batch_sequence_scores(
                    sequences_j, device=self.device, chunk_size=self.chunk_size
                )

                mu_t = torch.tensor(mus, dtype=torch.float32, device=self.device)
                delta_t = torch.tensor(deltas, dtype=torch.float32, device=self.device)
                weights = self._weights_from_deltas(delta_t)
                prob = torch.sigmoid(scores_i_t - scores_j_t)
                bce = F.binary_cross_entropy(prob, mu_t, reduction="none")
                loss = (bce * weights).sum() / (weights.sum() + 1e-8)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            acc = ((prob >= 0.5).float() == mu_t).float().mean().item()
            losses.append(float(loss.item()))
            accuracies.append(float(acc))

        corr = None
        if last_pairs:
            corr = self._episode_score_corr(buffer, last_pairs, phi_key)

        metrics: Dict[str, float] = {}
        if losses:
            metrics["loss"] = float(np.mean(losses))
        if accuracies:
            metrics["pref_accuracy"] = float(np.mean(accuracies))
        if corr is not None:
            metrics["score_phi_corr"] = float(corr)
        return metrics
