import contextlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from .oracle import compute_phi, preference
from .preference_buffer import EpisodeRecord, PreferenceBuffer
from .reward_model import RewardModel

# Check if CUDA AMP is available
_AMP_AVAILABLE = torch.cuda.is_available()

# Minimum distinct episodes before `score_phi_corr` is meaningful.
_MIN_EPISODES_FOR_CORR = 4


def _mean(values: List[torch.Tensor]) -> float:
    """Mean of per-step scalars, with a single device-to-host transfer."""
    return float(torch.stack(values).float().mean())


class RewardModelTrainer:
    def __init__(
        self,
        reward_model: RewardModel,
        lr: float = 1e-4,
        device: str = "auto",
        use_amp: bool = True,
        chunk_size: int = 512,
        weight_decay: float = 1e-4,
        max_grad_norm: Optional[float] = 1.0,
        delta_temperature: float = 1.0,
        grad_checkpoint: bool = False,
        tie_tolerance: float = 0.0,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.reward_model = reward_model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.reward_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Mixed precision training
        self.use_amp = use_amp and _AMP_AVAILABLE and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Chunk size for batched processing
        self.chunk_size = chunk_size
        self.max_grad_norm = max_grad_norm
        self.delta_temperature = delta_temperature
        self.grad_checkpoint = grad_checkpoint
        self.tie_tolerance = tie_tolerance

    def _autocast(self):
        if self.use_amp:
            return torch.amp.autocast("cuda")
        return contextlib.nullcontext()

    def _pair_sequences(
        self, buffer: PreferenceBuffer, ep_i: EpisodeRecord, ep_j: EpisodeRecord, mode: str
    ) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]]]:
        if mode == "input_aggregation":
            return buffer.aggregate_episode(ep_i), buffer.aggregate_episode(ep_j)
        if mode == "narrow_view":
            return buffer.sample_agent_trajectory(ep_i), buffer.sample_agent_trajectory(ep_j)
        raise ValueError(f"Unsupported reward model mode: {mode}")

    def _weights_from_deltas(self, deltas: torch.Tensor) -> torch.Tensor:
        """Preference-magnitude weights: `softmax(delta / (std(delta) + tau))`.

        `tau` is not cosmetic. The previous implementation used `tau = 1e-6`,
        which divides by the batch's own standard deviation and so pins the
        softmax to a fixed +-1 sigma temperature no matter how the deltas are
        actually distributed. Whenever the distribution is skewed -- the normal
        case for phi, a *product* of social metrics, where most episodes bunch
        near zero and a few stand well clear -- the tail runs many sigma out
        and takes almost all the weight. Measured effective sample sizes
        (`1 / sum(w^2)`) with `tau = 1e-6`: 7.5 of 64 pairs for a log-normal
        delta spread, 1.5 of 16 when one pair is an outlier, and 1.6 of 16 when
        the deltas are near-identical and only float jitter separates them.
        The reference MARP implementation uses `delta / (delta.std() + 1)`, a
        soft floor that keeps the weighting a gentle tilt rather than a hard
        arg-max; the same three cases give 58, 15.7 and 16.0. `tau = 1.0`
        restores that. `effective_pairs` is reported alongside the loss so the
        collapse is visible if it ever comes back.

        (Mean-centering the deltas first, as the old code did, is a no-op:
        softmax is invariant to a constant shift.)
        """
        if deltas.numel() == 0:
            return deltas
        std = deltas.std(unbiased=False)
        return torch.softmax(deltas / (std + self.delta_temperature), dim=0)

    def _score_sequences(
        self, sequences: Sequence[Sequence[Tuple[np.ndarray, int]]]
    ) -> torch.Tensor:
        return self.reward_model.batch_sequence_scores(
            sequences,
            device=self.device,
            chunk_size=self.chunk_size,
            grad_checkpoint=self.grad_checkpoint,
        )

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
        # A 2-point Pearson correlation is +-1 by construction, and a 3-point
        # one is barely better; reporting either would log confident-looking
        # noise for the first few updates.
        if len(episodes) < _MIN_EPISODES_FOR_CORR:
            return None

        # Batch score computation
        sequences = [buffer.aggregate_episode(ep) for ep in episodes]
        phis = [compute_phi(ep.metrics, phi_key) for ep in episodes]

        was_training = self.reward_model.training
        self.reward_model.eval()
        try:
            with torch.no_grad(), self._autocast():
                scores_tensor = self._score_sequences(sequences)
            scores = scores_tensor.float().cpu().numpy()
        finally:
            self.reward_model.train(was_training)

        phis_arr = np.array(phis, dtype=np.float32)
        if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(phis_arr)):
            return None
        if scores.std() == 0 or phis_arr.std() == 0:
            return None
        corr = float(np.corrcoef(scores, phis_arr)[0, 1])
        if not np.isfinite(corr):
            return None
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
        # Per-step scalars are accumulated as detached device tensors and moved
        # to host once, after the loop. Reading them inline with `.item()` costs
        # a pipeline flush per statistic per step -- four per step, 200 per
        # default update -- purely to build a number that is averaged at the end
        # anyway. Detaching is what keeps the 50 autograd graphs from piling up.
        losses: List[torch.Tensor] = []
        accuracies: List[torch.Tensor] = []
        effective_pairs: List[torch.Tensor] = []
        grad_norms: List[torch.Tensor] = []
        tie_fractions: List[float] = []
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
                mu, delta = preference(phi_i, phi_j, tie_tolerance=self.tie_tolerance)
                seq_i, seq_j = self._pair_sequences(buffer, ep_i, ep_j, mode)
                sequences_i.append(seq_i)
                sequences_j.append(seq_j)
                mus.append(mu)
                deltas.append(delta)

            num_pairs = len(pairs)
            # Both sides go through one call: the two halves are independent
            # rows of the same flattened batch, so scoring them together halves
            # the number of chunked forward passes.
            with self._autocast():
                scores = self._score_sequences(list(sequences_i) + list(sequences_j))

            scores = scores.float()
            scores_i_t = scores[:num_pairs]
            scores_j_t = scores[num_pairs:]

            mu_t = torch.tensor(mus, dtype=torch.float32, device=self.device)
            delta_t = torch.tensor(deltas, dtype=torch.float32, device=self.device)
            weights = self._weights_from_deltas(delta_t)

            # Logit form, not sigmoid-then-BCE. A sequence score is a sum over
            # up to `max_steps_per_sequence` unbounded per-step rewards, so the
            # difference routinely saturates a sigmoid to exactly 0.0 or 1.0 --
            # at which point plain `binary_cross_entropy` clamps to its -100
            # floor and the gradient through the saturated tail is destroyed
            # (and under fp16 it produces inf/NaN outright).
            logits = scores_i_t - scores_j_t
            bce = F.binary_cross_entropy_with_logits(logits, mu_t, reduction="none")
            loss = (bce * weights).sum() / (weights.sum() + 1e-8)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                # unscale_ before clipping, otherwise the norm is measured on
                # loss-scaled gradients and the threshold means nothing. The
                # scaler records the overflow at unscale_ time, so a step whose
                # gradients went inf is still skipped rather than applied.
                self.scaler.unscale_(self.optimizer)
                grad_norms.append(
                    torch.nn.utils.clip_grad_norm_(
                        self.reward_model.parameters(), self.max_grad_norm
                    ).detach()
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                # Ties (mu == 0.5) carry no ground truth to be right or wrong
                # about, so they are excluded from accuracy instead of being
                # scored against an arbitrary side. The mask is built from the
                # host-side `mus` list so deciding whether to score accuracy at
                # all costs no device round-trip.
                decisive = [index for index, mu in enumerate(mus) if mu != 0.5]
                if decisive:
                    index = torch.tensor(decisive, dtype=torch.long, device=self.device)
                    correct = (logits[index] >= 0).float() == mu_t[index]
                    accuracies.append(correct.float().mean())
                tie_fractions.append(1.0 - len(decisive) / num_pairs)
                # 1 / sum(w^2): how many pairs the weighted loss really used.
                effective_pairs.append(1.0 / weights.detach().pow(2).sum().clamp_min(1e-12))

            losses.append(loss.detach())

        corr = None
        if last_pairs:
            corr = self._episode_score_corr(buffer, last_pairs, phi_key)

        metrics: Dict[str, float] = {}
        if losses:
            metrics["loss"] = _mean(losses)
        if accuracies:
            metrics["pref_accuracy"] = _mean(accuracies)
        if effective_pairs:
            metrics["effective_pairs"] = _mean(effective_pairs)
        if tie_fractions:
            metrics["tie_fraction"] = float(np.mean(tie_fractions))
        if grad_norms:
            norms = torch.stack(grad_norms).float().cpu().numpy()
            # A non-finite norm is the AMP loss scaler probing for its working
            # scale (that step is skipped by the scaler, not applied), not a
            # real gradient magnitude. Averaging it in would pin `grad_norm` to
            # NaN for the whole run, so report the two separately.
            finite = np.isfinite(norms)
            if finite.any():
                metrics["grad_norm"] = float(norms[finite].mean())
            if not finite.all():
                metrics["grad_overflow_rate"] = float(1.0 - finite.mean())
        if corr is not None:
            metrics["score_phi_corr"] = float(corr)
        return metrics
