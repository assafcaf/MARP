"""Scoring, batching and serialization behaviour of `RewardModel`."""

import numpy as np
import pytest
import torch

from commons_game_marp.reward_model.reward_model import RewardModel


OBS_SHAPE = (7, 7, 3)


@pytest.fixture
def model() -> RewardModel:
    torch.manual_seed(0)
    return RewardModel(obs_shape=OBS_SHAPE, num_actions=8)


def _traj(length: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [
        (rng.integers(0, 256, size=OBS_SHAPE, dtype=np.uint8), int(rng.integers(0, 8)))
        for _ in range(length)
    ]


def test_forward_returns_one_reward_per_row(model):
    obs = torch.zeros(5, *OBS_SHAPE)
    actions = torch.zeros(5, dtype=torch.long)
    assert model(obs, actions).shape == (5,)


def test_obs_scale_is_applied_inside_forward():
    """uint8 frames plus `obs_scale` must equal pre-divided float frames."""
    torch.manual_seed(0)
    scaled = RewardModel(obs_shape=OBS_SHAPE, num_actions=8, obs_scale=1 / 255.0)
    torch.manual_seed(0)
    unscaled = RewardModel(obs_shape=OBS_SHAPE, num_actions=8, obs_scale=1.0)

    frame = np.random.default_rng(1).integers(0, 256, size=(4, *OBS_SHAPE), dtype=np.uint8)
    actions = torch.zeros(4, dtype=torch.long)

    from_uint8 = scaled(torch.from_numpy(frame), actions)
    from_float = unscaled(torch.from_numpy(frame.astype(np.float32) / 255.0), actions)

    torch.testing.assert_close(from_uint8, from_float)


def test_uint8_and_float_inputs_score_identically(model):
    frame = np.random.default_rng(2).integers(0, 256, size=(3, *OBS_SHAPE), dtype=np.uint8)
    actions = torch.arange(3)
    torch.testing.assert_close(
        model(torch.from_numpy(frame), actions),
        model(torch.from_numpy(frame.astype(np.float32)), actions),
    )


def test_batch_sequence_scores_matches_per_sequence_scores(model):
    trajectories = [_traj(6, seed=1), _traj(11, seed=2), _traj(3, seed=3)]

    batched = model.batch_sequence_scores(trajectories)
    individually = torch.stack([model.sequence_score(t) for t in trajectories])

    torch.testing.assert_close(batched, individually, rtol=1e-5, atol=1e-5)


def test_chunking_does_not_change_the_score(model):
    """`chunk_size` is a memory knob; it must not be a numerics knob."""
    trajectories = [_traj(17, seed=4), _traj(9, seed=5)]

    whole = model.batch_sequence_scores(trajectories, chunk_size=1024)
    chunked = model.batch_sequence_scores(trajectories, chunk_size=4)

    torch.testing.assert_close(whole, chunked, rtol=1e-5, atol=1e-5)


def test_empty_trajectories_score_zero_without_shifting_the_others(model):
    trajectories = [[], _traj(5, seed=6), []]

    scores = model.batch_sequence_scores(trajectories)

    assert scores.shape == (3,)
    assert scores[0].item() == 0.0
    assert scores[2].item() == 0.0
    torch.testing.assert_close(scores[1], model.sequence_score(trajectories[1]))


def test_all_empty_and_no_trajectories(model):
    assert model.batch_sequence_scores([]).shape == (0,)
    assert torch.equal(model.batch_sequence_scores([[], []]), torch.zeros(2))


def test_scores_are_differentiable(model):
    score = model.batch_sequence_scores([_traj(5, seed=7)]).sum()
    score.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_gradient_checkpointing_matches_plain_backward():
    torch.manual_seed(3)
    plain = RewardModel(obs_shape=OBS_SHAPE, num_actions=8)
    torch.manual_seed(3)
    checkpointed = RewardModel(obs_shape=OBS_SHAPE, num_actions=8)
    trajectories = [_traj(9, seed=8), _traj(4, seed=9)]

    plain.batch_sequence_scores(trajectories, chunk_size=3).sum().backward()
    checkpointed.batch_sequence_scores(
        trajectories, chunk_size=3, grad_checkpoint=True
    ).sum().backward()

    for a, b in zip(plain.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-5)


def test_predict_batch_matches_predict(model):
    frames = np.random.default_rng(10).integers(0, 256, size=(4, *OBS_SHAPE), dtype=np.uint8)
    actions = [0, 3, 7, 1]

    batched = model.predict_batch(frames, actions)
    one_at_a_time = np.array([model.predict(f, a) for f, a in zip(frames, actions)])

    np.testing.assert_allclose(batched, one_at_a_time, rtol=1e-5, atol=1e-5)


def test_save_load_roundtrip_preserves_scale_and_scores(tmp_path):
    torch.manual_seed(11)
    model = RewardModel(obs_shape=OBS_SHAPE, num_actions=8, obs_scale=1 / 255.0)
    path = tmp_path / "reward_model.pt"
    model.save(str(path))

    restored = RewardModel.load(str(path), device="cpu")

    assert restored.obs_scale == pytest.approx(1 / 255.0)
    assert restored.num_actions == 8
    frames = np.random.default_rng(12).integers(0, 256, size=(3, *OBS_SHAPE), dtype=np.uint8)
    np.testing.assert_allclose(
        model.predict_batch(frames, [0, 1, 2]),
        restored.predict_batch(frames, [0, 1, 2]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_load_defaults_obs_scale_for_older_checkpoints(tmp_path):
    """Checkpoints written before `obs_scale` existed must still load."""
    model = RewardModel(obs_shape=OBS_SHAPE, num_actions=8)
    path = tmp_path / "legacy.pt"
    torch.save(
        {"state_dict": model.state_dict(), "obs_shape": OBS_SHAPE, "num_actions": 8}, path
    )

    restored = RewardModel.load(str(path), device="cpu")

    assert restored.obs_scale == pytest.approx(1.0)
