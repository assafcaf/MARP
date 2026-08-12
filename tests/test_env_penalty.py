"""The FIRE penalty must not leak into the social metrics."""

from src.env.commons_env import HarvestCommonsEnv
from src.env.maps import SMALL_HARVEST_MAP

FIRE_ACTION = 7
STAND_STILL = 4


def _make_env(penalty: bool) -> HarvestCommonsEnv:
    env = HarvestCommonsEnv(
        ascii_map=SMALL_HARVEST_MAP,
        num_agents=2,
        ep_length=50,
        penalty=penalty,
    )
    env.reset()
    return env


def test_penalty_is_returned_to_the_caller():
    env = _make_env(penalty=True)
    _, rewards, _, _ = env.step({"agent-0": FIRE_ACTION, "agent-1": STAND_STILL})
    assert rewards["agent-0"] == -1


def test_penalty_does_not_enter_the_metrics_record():
    env = _make_env(penalty=True)
    env.step({"agent-0": FIRE_ACTION, "agent-1": STAND_STILL})
    # rewards_record feeds efficiency and equality; it must hold the true env
    # reward for the FIRE step, not the -1 penalty.
    assert env.rewards_record["agent-0"] == [0]


def test_metrics_identical_with_and_without_penalty_for_the_same_actions():
    actions = {"agent-0": FIRE_ACTION, "agent-1": STAND_STILL}

    penalised = _make_env(penalty=True)
    plain = _make_env(penalty=False)
    for _ in range(5):
        penalised.step(actions)
        plain.step(actions)

    penalised.compute_social_metrics()
    plain.compute_social_metrics()
    assert penalised.get_social_metrics() == plain.get_social_metrics()
