"""Per-iteration behavioural and reward-model statistics.

What this exists for
--------------------
Episode reward tells you *that* a run improved. It does not tell you how the
agents got there, and in a commons game the how is the whole question: a policy
that strips every patch on step 40 and one that harvests sustainably can post
the same return over a short episode, and a reward model that has learned
nothing looks identical to one that has learned everything if you only watch
its loss.

The reference implementation in `DanfoaTestSOT` logged exactly these breakdowns
-- mean predicted reward per action, on harvest versus non-harvest steps, and
bucketed by how many apples stood nearby -- and they are what made its runs
readable. This module reproduces them and adds the resource-use side.

Design
------
An accumulator, not a logger. `record_step` takes flat `(num_rows,)` arrays --
one entry per agent per environment, matching `VecCommonsEnv`'s row layout --
and appends them; `result` concatenates once and returns a dict of sections
ready for `ResultLogger` to route to TensorBoard tag families.

Empty subsets emit no key at all. A missing point in TensorBoard reads as "this
did not happen this iteration"; a NaN point poisons the axis for the whole run,
and half of these statistics are conditioned on events (a harvest with exactly
five apples nearby) that legitimately do not occur in a given iteration.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Action ids come from `commons_agent.HARVEST_ACTIONS`. TURN_CLOCKWISE and
# TURN_COUNTERCLOCKWISE are merged into one group, matching the reference's
# `on_action/turn`: the two are the same behaviour mirrored, and splitting them
# halves the sample size behind each point for no diagnostic gain.
ACTION_GROUPS: Dict[str, Tuple[int, ...]] = {
    "move_left": (0,),
    "move_right": (1,),
    "move_up": (2,),
    "move_down": (3,),
    "stay": (4,),
    "turn": (5, 6),
    "fire": (7,),
}

# Apple-density buckets for conditioning predicted reward on local abundance.
# `(label, low, high)` with an inclusive `high`; `None` means unbounded.
#
# The reference's buckets overlap -- its "1" is `aip in {1,2}` and its "2" is
# `aip in {2,3}`, so every harvest with exactly two apples nearby is counted in
# both and neither bucket is a real conditional mean. These partition.
NEARBY_BUCKETS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("0", 0, 0),
    ("1-2", 1, 2),
    ("3-4", 3, 4),
    ("5+", 5, None),
)


def _finite_mean(values: np.ndarray) -> Optional[float]:
    """Mean of a subset, or None when the subset is empty."""
    if values.size == 0:
        return None
    mean = float(values.mean())
    return mean if np.isfinite(mean) else None


def _finite_std(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    std = float(values.std())
    return std if np.isfinite(std) else None


def _put(section: Dict[str, float], key: str, value: Optional[float]) -> None:
    """Set `key` only if `value` is a real number."""
    if value is not None and np.isfinite(value):
        section[key] = float(value)


class EpisodeStats:
    """Accumulates one iteration's per-step arrays into flat metric sections.

    Parameters
    ----------
    num_actions : int
        Size of the action space, used for the raw-action entropy denominator.
    num_rows : int
        `num_envs * num_agents` -- the width of every array passed to
        `record_step`. Per-agent-per-episode averages divide by this.
    track_reward_model : bool
        Whether `record_step` will be given predicted rewards. False omits every
        `rm_*` section.
    """

    def __init__(self, num_actions: int, num_rows: int, track_reward_model: bool = False):
        self.num_actions = int(num_actions)
        self.num_rows = int(num_rows)
        self.track_reward_model = bool(track_reward_model)
        self._actions: List[np.ndarray] = []
        self._env_rewards: List[np.ndarray] = []
        self._nearby: List[np.ndarray] = []
        self._last_in_cluster: List[np.ndarray] = []
        self._pred_rewards: List[np.ndarray] = []
        self.steps = 0

    def record_step(
        self,
        actions: Sequence[int],
        env_rewards: Sequence[float],
        nearby_apples: Sequence[int],
        last_in_cluster: Sequence[bool],
        pred_rewards: Optional[Sequence[float]] = None,
    ) -> None:
        """Append one environment step, across every row.

        `env_rewards` must be the *unpenalised* environment reward (the `r` key
        on the info dict), not the value returned by `VecCommonsEnv.step`. With
        `env.penalty` on, the returned reward is -1 for a FIRE action, and a
        harvest mask built from `reward > 0` would still be correct while the
        harvest *count* used as a denominator would not.
        """
        self._actions.append(np.asarray(actions, dtype=np.int64))
        self._env_rewards.append(np.asarray(env_rewards, dtype=np.float64))
        self._nearby.append(np.asarray(nearby_apples, dtype=np.int64))
        self._last_in_cluster.append(np.asarray(last_in_cluster, dtype=bool))
        if self.track_reward_model:
            if pred_rewards is None:
                raise ValueError(
                    "pred_rewards is required when track_reward_model is True"
                )
            self._pred_rewards.append(np.asarray(pred_rewards, dtype=np.float64))
        self.steps += 1

    def result(self) -> Dict[str, Dict[str, float]]:
        """Flat metric sections, keyed by TensorBoard tag prefix."""
        if self.steps == 0:
            return {}

        actions = np.concatenate(self._actions)
        env_rewards = np.concatenate(self._env_rewards)
        nearby = np.concatenate(self._nearby)
        last_in_cluster = np.concatenate(self._last_in_cluster)
        harvested = env_rewards > 0

        sections: Dict[str, Dict[str, float]] = {
            "action": self._action_section(actions),
            "harvest": self._harvest_section(nearby, harvested, last_in_cluster),
        }

        if self.track_reward_model and self._pred_rewards:
            predicted = np.concatenate(self._pred_rewards)
            sections.update(
                self._reward_model_sections(
                    predicted, env_rewards, actions, nearby, harvested
                )
            )
        return {name: values for name, values in sections.items() if values}

    # -- sections ---------------------------------------------------------

    def _action_section(self, actions: np.ndarray) -> Dict[str, float]:
        """How the policy spends its steps.

        Collapse shows here long before it shows in reward: an agent that has
        settled on spinning in place still collects the occasional apple.
        """
        section: Dict[str, float] = {}
        total = actions.size
        for name, ids in ACTION_GROUPS.items():
            if max(ids) >= self.num_actions:
                continue
            count = np.count_nonzero(np.isin(actions, ids))
            section[name] = float(count) / total

        # Entropy over *raw* actions, not the merged groups, so it is directly
        # comparable with `algo/entropy` and with the adaptive entropy
        # controller's `target_entropy` -- both of which are defined over the
        # action space the policy actually samples from.
        counts = np.bincount(actions, minlength=self.num_actions).astype(np.float64)
        probabilities = counts / counts.sum()
        nonzero = probabilities[probabilities > 0]
        section["entropy"] = float(-(nonzero * np.log(nonzero)).sum())
        return section

    def _harvest_section(
        self, nearby: np.ndarray, harvested: np.ndarray, last_in_cluster: np.ndarray
    ) -> Dict[str, float]:
        """How the common pool was used."""
        section: Dict[str, float] = {}
        total_harvests = int(np.count_nonzero(harvested))
        agent_steps = float(self.num_rows * self.steps)

        section["apples_per_agent"] = total_harvests / float(self.num_rows)
        section["harvest_rate"] = total_harvests / agent_steps
        _put(section, "nearby_apples_mean", _finite_mean(nearby.astype(np.float64)))
        _put(
            section,
            "nearby_apples_on_harvest",
            _finite_mean(nearby[harvested].astype(np.float64)),
        )
        # The over-harvesting signal: what fraction of apples taken were the
        # last one standing in their cluster. An apple has no neighbours left to
        # seed regrowth, so a rate near 1 is a policy eating its own future.
        if total_harvests:
            section["last_in_cluster_rate"] = (
                float(np.count_nonzero(last_in_cluster & harvested)) / total_harvests
            )
        return section

    def _reward_model_sections(
        self,
        predicted: np.ndarray,
        env_rewards: np.ndarray,
        actions: np.ndarray,
        nearby: np.ndarray,
        harvested: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Is the learned reward model separating anything?

        Loss and preference accuracy are pair-level and can look healthy while
        the per-step reward the policy actually optimises is flat. These are the
        step-level view: what the model pays for each action, and whether it
        pays more for taking an apple than for not.
        """
        on_action: Dict[str, float] = {}
        for name, ids in ACTION_GROUPS.items():
            if max(ids) >= self.num_actions:
                continue
            _put(on_action, name, _finite_mean(predicted[np.isin(actions, ids)]))

        eaten = predicted[harvested]
        not_eaten = predicted[~harvested]
        outcome_avg: Dict[str, float] = {}
        outcome_std: Dict[str, float] = {}
        outcome: Dict[str, float] = {}

        mean_eaten = _finite_mean(eaten)
        mean_not_eaten = _finite_mean(not_eaten)
        std_eaten = _finite_std(eaten)
        std_not_eaten = _finite_std(not_eaten)
        _put(outcome_avg, "apple_eaten", mean_eaten)
        _put(outcome_avg, "no_apple_eaten", mean_not_eaten)
        _put(outcome_std, "apple_eaten", std_eaten)
        _put(outcome_std, "no_apple_eaten", std_not_eaten)
        if mean_eaten is not None and mean_not_eaten is not None:
            _put(outcome_avg, "delta", mean_eaten - mean_not_eaten)
        if std_eaten is not None and std_not_eaten is not None:
            _put(outcome_std, "delta", std_eaten - std_not_eaten)
        # A d-prime. The raw `delta` the reference logs moves with the model's
        # arbitrary output scale, so it cannot be compared across runs or even
        # across a run whose scale drifts; dividing by the pooled spread gives a
        # separation that can.
        if (
            mean_eaten is not None
            and mean_not_eaten is not None
            and std_eaten is not None
            and std_not_eaten is not None
        ):
            pooled = float(np.sqrt((std_eaten**2 + std_not_eaten**2) / 2.0))
            if pooled > 0:
                _put(outcome, "separation", (mean_eaten - mean_not_eaten) / pooled)

        # Conditioned on a harvest, so this asks: does the model pay more for an
        # apple taken from a dense patch than from the last one in a thin patch?
        by_nearby: Dict[str, float] = {}
        harvest_nearby = nearby[harvested]
        harvest_pred = predicted[harvested]
        for label, low, high in NEARBY_BUCKETS:
            mask = harvest_nearby >= low
            if high is not None:
                mask &= harvest_nearby <= high
            _put(by_nearby, label, _finite_mean(harvest_pred[mask]))

        pred_stats: Dict[str, float] = {}
        _put(pred_stats, "mean", _finite_mean(predicted))
        _put(pred_stats, "std", _finite_std(predicted))
        if predicted.size:
            _put(pred_stats, "min", float(predicted.min()))
            _put(pred_stats, "max", float(predicted.max()))
        _put(pred_stats, "step_corr", _pearson(predicted, env_rewards))

        return {
            "rm_on_action": on_action,
            "rm_outcome_avg": outcome_avg,
            "rm_outcome_std": outcome_std,
            "rm_outcome": outcome,
            "rm_by_nearby_apples": by_nearby,
            "rm_pred": pred_stats,
        }


def _pearson(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    """Pearson correlation, or None when it is undefined.

    A constant series -- an untrained model whose output has collapsed, or an
    episode where nothing was harvested -- has zero variance and no correlation,
    which numpy reports as NaN with a warning.
    """
    if left.size < 2 or right.size != left.size:
        return None
    if left.std() == 0 or right.std() == 0:
        return None
    corr = float(np.corrcoef(left, right)[0, 1])
    return corr if np.isfinite(corr) else None
