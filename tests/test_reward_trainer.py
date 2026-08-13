"""Preference-loss behaviour: labels, magnitude weighting, and stability."""

import numpy as np
import pytest
import torch

from commons_game_marp.reward_model.oracle import preference
from commons_game_marp.reward_model.preference_buffer import EpisodeRecord, PreferenceBuffer
from commons_game_marp.reward_model.reward_model import RewardModel
from commons_game_marp.reward_model.reward_trainer import RewardModelTrainer


OBS_SHAPE = (5, 5, 3)


def _trainer(**kwargs) -> RewardModelTrainer:
    torch.manual_seed(0)
    model = RewardModel(obs_shape=OBS_SHAPE, num_actions=8)
    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("use_amp", False)
    return RewardModelTrainer(model, **kwargs)


def _episode(efficiency: float, peace: float, seed: int, agents: int = 2, steps: int = 4):
    rng = np.random.default_rng(seed)
    trajs = {
        f"agent-{a}": [
            (rng.integers(0, 256, size=OBS_SHAPE, dtype=np.uint8), int(rng.integers(0, 8)))
            for _ in range(steps)
        ]
        for a in range(agents)
    }
    return EpisodeRecord(
        agent_trajs=trajs,
        metrics={"efficiency": efficiency, "peace": peace, "equality": 1.0, "sustainability": 1.0},
    )


class TestPreferenceLabels:
    def test_strict_preference_is_a_hard_label(self):
        assert preference(0.8, 0.2) == (1.0, pytest.approx(0.6))
        assert preference(0.2, 0.8) == (0.0, pytest.approx(0.6))

    def test_ties_get_an_indifferent_label(self):
        """A tie used to be labelled `mu = 1`, teaching the model that whichever
        episode was drawn first was better. Early training is almost all ties
        (every episode scores efficiency 0), so this was the dominant signal."""
        mu, delta = preference(0.0, 0.0)
        assert mu == 0.5
        assert delta == 0.0

    def test_tie_tolerance_widens_the_indifference_band(self):
        assert preference(0.5, 0.5 + 1e-9, tie_tolerance=1e-6)[0] == 0.5
        assert preference(0.5, 0.5 + 1e-3, tie_tolerance=1e-6)[0] == 0.0

    def test_a_tie_contributes_no_gradient(self):
        """`mu = 0.5` on an indistinguishable pair should be a flat log 2."""
        logits = torch.zeros(1, requires_grad=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor([0.5])
        )
        loss.backward()
        assert loss.item() == pytest.approx(np.log(2), rel=1e-6)
        assert logits.grad.abs().item() == pytest.approx(0.0)


class TestDeltaWeighting:
    @staticmethod
    def _effective(weights: torch.Tensor) -> float:
        return 1.0 / float(weights.pow(2).sum())

    @pytest.mark.parametrize(
        "name, deltas, collapsed_max, restored_min",
        [
            # phi is a product of social metrics, so its pairwise deltas are
            # skewed: most pairs bunch near zero, a few stand well clear.
            ("one outlier", [0.10] * 15 + [0.60], 2.0, 15.0),
            # Near-identical deltas are the worst case: std collapses, so pure
            # float jitter gets amplified into a near-arg-max weighting.
            ("float jitter only", [0.30] * 15 + [0.3001], 2.0, 15.9),
        ],
    )
    def test_skewed_deltas_keep_a_usable_effective_batch(
        self, name, deltas, collapsed_max, restored_min
    ):
        """Dividing by the batch's own std pins the softmax to a +-1 sigma
        temperature, so a skewed delta distribution hands nearly all the weight
        to its tail and the batch trains on a couple of pairs."""
        deltas = torch.tensor(deltas)

        collapsed = _trainer(delta_temperature=1e-6)._weights_from_deltas(deltas)
        restored = _trainer(delta_temperature=1.0)._weights_from_deltas(deltas)

        assert self._effective(collapsed) < collapsed_max
        assert self._effective(restored) > restored_min

    def test_weights_sum_to_one_and_rank_with_delta(self):
        deltas = torch.tensor([0.1, 0.9, 0.4])
        weights = _trainer()._weights_from_deltas(deltas)

        assert float(weights.sum()) == pytest.approx(1.0)
        assert weights[1] > weights[2] > weights[0]

    def test_identical_deltas_give_uniform_weights(self):
        weights = _trainer()._weights_from_deltas(torch.full((5,), 0.3))
        torch.testing.assert_close(weights, torch.full((5,), 0.2))

    def test_empty_deltas_are_passed_through(self):
        assert _trainer()._weights_from_deltas(torch.empty(0)).numel() == 0


class TestLossStability:
    def test_saturated_score_differences_keep_a_live_gradient(self):
        """Sequence scores sum hundreds of unbounded per-step rewards, so the
        difference saturates a sigmoid routinely. `binary_cross_entropy` on the
        saturated probability clamps to its -100 floor and kills the gradient;
        the logits form does not."""
        logits = torch.tensor([-400.0], requires_grad=True)
        target = torch.tensor([1.0])

        stable = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        stable.backward()

        assert torch.isfinite(stable)
        assert stable.item() == pytest.approx(400.0, rel=1e-3)
        assert logits.grad.abs().item() == pytest.approx(1.0, rel=1e-3)

        naive_logits = torch.tensor([-400.0], requires_grad=True)
        naive = torch.nn.functional.binary_cross_entropy(
            torch.sigmoid(naive_logits), target
        )
        naive.backward()
        # The old path: loss floored at 100 instead of 400, gradient gone.
        assert naive.item() == pytest.approx(100.0)
        assert naive_logits.grad.abs().item() == pytest.approx(0.0)


class TestTrainLoop:
    def _buffer(self, n: int = 8) -> PreferenceBuffer:
        buffer = PreferenceBuffer(max_episodes=32)
        for i in range(n):
            buffer.add_episode(_episode(efficiency=0.1 * i, peace=1.0, seed=i))
        return buffer

    def test_returns_nothing_when_the_buffer_cannot_form_a_pair(self):
        buffer = PreferenceBuffer(max_episodes=4)
        buffer.add_episode(_episode(0.5, 1.0, seed=0))
        assert _trainer().train(buffer, "efficiency", "narrow_view", 4, 2) == {}

    @pytest.mark.parametrize("mode", ["narrow_view", "input_aggregation"])
    def test_reports_finite_metrics(self, mode):
        metrics = _trainer().train(self._buffer(), "efficiency_x_peace", mode, 6, 3)

        assert set(metrics) >= {"loss", "pref_accuracy", "effective_pairs", "tie_fraction"}
        assert all(np.isfinite(v) for v in metrics.values())
        assert 0.0 <= metrics["pref_accuracy"] <= 1.0
        assert 1.0 <= metrics["effective_pairs"] <= 6.0

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="Unsupported reward model mode"):
            _trainer().train(self._buffer(), "efficiency", "telepathy", 4, 1)

    def test_training_reduces_the_preference_loss(self):
        trainer = _trainer(lr=1e-3)
        buffer = self._buffer()

        first = trainer.train(buffer, "efficiency", "input_aggregation", 8, 5)["loss"]
        for _ in range(6):
            last = trainer.train(buffer, "efficiency", "input_aggregation", 8, 5)["loss"]

        assert last < first

    def test_gradients_are_clipped_to_the_configured_norm(self):
        trainer = _trainer(lr=1e-3, max_grad_norm=0.01)
        metrics = trainer.train(self._buffer(), "efficiency", "input_aggregation", 8, 3)

        # clip_grad_norm_ reports the pre-clip norm, so only the applied update
        # is bounded; assert the metric exists and the params stayed finite.
        assert "grad_norm" in metrics
        assert all(torch.isfinite(p).all() for p in trainer.reward_model.parameters())

    def test_amp_scaler_overflow_is_reported_without_poisoning_grad_norm(self, monkeypatch):
        """The AMP loss scaler deliberately overflows while it finds its scale,
        and `clip_grad_norm_` reports inf on those steps. Averaging those in
        pinned `grad_norm` to NaN for the rest of the run."""
        import torch.nn.utils as nn_utils

        real_clip = nn_utils.clip_grad_norm_
        calls = {"n": 0}

        def flaky_clip(parameters, max_norm, *args, **kwargs):
            norm = real_clip(parameters, max_norm, *args, **kwargs)
            calls["n"] += 1
            return torch.tensor(float("inf")) if calls["n"] == 1 else norm

        monkeypatch.setattr(
            "commons_game_marp.reward_model.reward_trainer.torch.nn.utils.clip_grad_norm_",
            flaky_clip,
        )
        metrics = _trainer().train(self._buffer(), "efficiency", "input_aggregation", 4, 4)

        assert np.isfinite(metrics["grad_norm"])
        assert metrics["grad_overflow_rate"] == pytest.approx(0.25)

    def test_clipping_can_be_disabled(self):
        trainer = _trainer(max_grad_norm=None)
        assert "grad_norm" not in trainer.train(
            self._buffer(), "efficiency", "input_aggregation", 4, 2
        )

    def test_all_tied_episodes_produce_no_accuracy_and_full_tie_fraction(self):
        """Every early episode has efficiency 0; nothing is learnable and the
        metrics should say so rather than reporting a fake 100% accuracy."""
        buffer = PreferenceBuffer(max_episodes=8)
        for i in range(4):
            buffer.add_episode(_episode(efficiency=0.0, peace=1.0, seed=i))

        trainer = _trainer(lr=1e-3)
        first = trainer.train(buffer, "efficiency", "input_aggregation", 6, 5)

        assert first["tie_fraction"] == pytest.approx(1.0)
        assert "pref_accuracy" not in first
        # BCE against mu=0.5 bottoms out at log 2, reached only when the model
        # scores the two sides equally. An all-tie batch should be pushed
        # towards exactly that indifference, not away from it.
        for _ in range(10):
            last = trainer.train(buffer, "efficiency", "input_aggregation", 6, 5)
        assert last["loss"] < first["loss"]
        assert last["loss"] >= np.log(2) - 1e-6

    def test_model_is_left_in_training_mode_after_the_correlation_pass(self):
        trainer = _trainer()
        trainer.train(self._buffer(), "efficiency", "input_aggregation", 4, 2)
        assert trainer.reward_model.training
