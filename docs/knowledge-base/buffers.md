# `src/buffers/` Knowledge Base

## 1. Class map

| Class | File | Inherits from | Instantiated by (call site) |
|---|---|---|---|
| `PRMShardReplayBuffer` | `replay_buffers.py` | SB3 `ReplayBuffer` | `BaseRunner.__init__` (dqn path, `runners.py:48`) as `replay_buffer_class`; `BaseRunner.__init__` (ppo path, `runners.py:66`) as `rollout_buffer_class` when `RP_PARAMETERS.episodial` is `False`; `PRMRunner.__init__` re-sets `self.learner_kwargs['replay_buffer_class']` the same way (`runners.py:290`) |
| `PRMShardReplayBufferEpisodial` | `replay_buffers.py` | `PRMShardReplayBuffer` | `runners.py:66` and `runners.py:290`, same call sites as above, selected when `RP_PARAMETERS.episodial` is `True` |
| `CRMShardReplayBuffer` | `replay_buffers.py` | `PRMShardReplayBuffer` | `CRMRunner.setup_agent` (`runners.py:419`), passed directly as `replay_buffer_class=CRMShardReplayBuffer` into the agent constructor, bypassing `self.learner_kwargs` |
| `PRMShardRolloutBuffer` | `rollout_buffers.py` | SB3 `RolloutBuffer` | **Nowhere.** Imported in `runners.py:11` and re-exported from `src/buffers/__init__.py`, but never assigned to any `*_buffer_class` kwarg or otherwise instantiated. |
| `PRMShardReplayBuffer` (duplicate) | `buffers.py` | SB3 `ReplayBuffer` | **Nowhere.** `buffers.py` is never imported by `__init__.py` or by anything else in the repo. |
| `CRMShardReplayBuffer` (duplicate) | `buffers.py` | local `PRMShardReplayBuffer` (the `buffers.py` copy) | **Nowhere**, same reason. |
| `PRMEpisodeData` / `CRMEpisodeData` | `episode_data.py` | `typing.NamedTuple` | Imported and returned by `replay_buffers.py`'s `get_episodes()` methods; not exported from `__init__.py` but reachable via `replay_buffers` module |

`NRPRunner.init_experiment()` (`runners.py:282-283`) `del`s `replay_buffer_class`/`replay_buffer_kwargs` from `self.learner_kwargs` for the dqn path — its DQN agent falls back to SB3's stock `ReplayBuffer`. For the ppo path it does **not** delete `rollout_buffer_class`/`rollout_buffer_kwargs`, so NRP's non-independent `PPO` agent (`rl_agents/ppo/single_agent.py`) is still constructed with `rollout_buffer_class=PRMShardReplayBuffer` (or the Episodial variant) even though it never calls `.get_episodes()` and has no reward predictor. This is dead weight, not a correctness bug on its own — plain SB3 `PPO.collect_rollouts` only calls `rollout_buffer.add/reset/compute_returns_and_advantage`, all of which `PRMShardReplayBuffer` does **not** implement (it inherits `ReplayBuffer`, not `RolloutBuffer`). See §5 for why this matters.

## 2. Why "shard"

The buffers store more than SB3's plain `(obs, next_obs, action, reward, done)` — each `add()` call also persists:

```python
self.experiment_rewards[self.pos] = np.array(experiment_rewards)   # raw env reward (pre-predictor)
self.true_rewards[self.pos]       = np.array([info['true_reward'] for info in infos])
self.aip[self.pos]                = np.array([info['aip'] for info in infos])
```

while `self.rewards` (SB3's native slot, exposed via the `predicted_rewards` property) holds the reward-predictor's *predicted* reward that was actually fed to the RL algorithm. So one buffer slot carries three parallel reward signals — predicted, raw/experiment, and ground-truth — plus an `aip` (agent-impact/prosociality-adjacent, exact semantics external to this file) metric pulled from `infos`.

`episode_indices` tracks episode start offsets (`add()`: `if len(self.episode_indices)==0 or self.episode_indices[-1]+episode_length<=self.pos: self.episode_indices.append(self.pos)`), and `get_episodes(batch_size)` reconstructs full contiguous episodes (observations, actions, all three reward streams, `aip`) rather than i.i.d. transitions:

```python
def get_episodes(self, batch_size=1):
    ep_indices = np.random.choice(len(self.episode_indices), size=batch_size, replace=True)
    env_indices = np.random.choice(self.n_envs, size=batch_size, replace=True)
    ...
    return PRMEpisodeData(observations=..., actions=..., experiment_rewards=..., true_rewards=..., aip=...)
```

This is the reward-predictor's training interface: the RP (`src/reward_predictor/`) needs raw observation/action *segments* (not single transitions) to re-score them under trajectory-comparison/preference-style objectives, and needs the buffer size to be constrained to a multiple of `episode_length` (`assert buffer_size % episode_length == 0`) so segments never straddle episode boundaries. "Shard" reflects that the buffer partitions storage into fixed-length episode shards addressable by `episode_indices`, on top of being a normal circular SB3 replay buffer for the RL learner's own `sample()` calls.

## 3. Episodial variant

`PRMShardReplayBufferEpisodial(PRMShardReplayBuffer)` overrides only `get_episodes()`. The base class samples one env per episode and returns shape `(batch_size, episode_length, ...)`. The Episodial variant is aware that `self.n_envs` packs multiple agents per logical environment (`n_invs = self.n_envs // self.num_agents`) and, for each sampled logical env, stacks **all agents' transitions for that episode together**:

```python
n_invs = self.n_envs // self.num_agents
...
observations[i] = np.vstack(self.observations[start_idx:end_idx, env_idx*self.num_agents:(env_idx+1)*self.num_agents])
actions[i]      = np.vstack(self.actions[start_idx:end_idx, env_idx*self.num_agents:(env_idx+1)*self.num_agents]).squeeze()
```

Output shape becomes `(batch_size, episode_length * num_agents, ...)` — one flattened multi-agent episode segment per sample, versus the base class's single-agent `(batch_size, episode_length, ...)`. This is selected via `RP_PARAMETERS.episodial` and is presumably used when the reward predictor is trained on joint/aggregate multi-agent episode segments rather than per-agent ones.

## 4. PRM vs CRM buffer diff

`CRMShardReplayBuffer(PRMShardReplayBuffer)` adds:

```python
def __init__(self, ..., state_space=None, n_agents=0, *args, **kwargs):
    ...
    self.n_agents = n_agents
    a, b, c, d, e = self.observations.shape
    self.state_space = state_space
    self.states = np.zeros((a, b // n_agents, *state_space.shape), dtype=np.float32)
```

- **`states` array**: a per-timestep *global environment state* tensor, shaped `(buffer_size, n_envs // n_agents, *state_space.shape)` — one state per logical environment (shared across its `n_agents` agents), separate from the per-agent `observations` array inherited from the base class. `add()` is overridden to take an extra `states` argument and write it before delegating to `super().add(...)`.
- **`n_agents`**: needed to compute the states array's second dimension (`b // n_agents`) and, in `sample()`/`get_episodes()`, to map a "logical env index" to the slice of `n_agents` consecutive per-agent env columns in `observations`/`actions`/reward arrays (`start_env = env_idx*n_agents`).

Rationale: CRM's reward predictor (`CRMRewardPredictor`, confirmed in `runners.py` — constructed with `observation_space=env.state_space`) predicts reward from the **global/centralized state**, not each agent's local observation, so the buffer must separately persist that centralized state alongside the per-agent transition data that PRM's predictor (which scores from per-agent `obs`) doesn't need. `CRMShardReplayBuffer` also overrides `sample()` (PRM's base class does not) to select one agent's slice (`agent_id` param) out of the `n_agents`-wide env block for standard RL sampling, since `n_envs` here packs `n_agents` per logical env.

## 5. Dead code determination

**`PRMShardRolloutBuffer` (`rollout_buffers.py`) is unused in the sense that no runner ever assigns it as `replay_buffer_class`/`rollout_buffer_class`** — confirmed by repo-wide grep; its only references are its own definition, the `__init__.py` re-export, and the unused import in `runners.py:11`.

However, it is not simply orphaned dead code — its shape strongly suggests it was the *originally intended* class for the PPO+PRM rollout-buffer slot and the wiring was later swapped without updating the call site:

- `PRMShardRolloutBuffer.__init__` takes `replay_kwargs={}` ("episode_length is needed from outside") and internally builds a `PRMShardReplayBuffer` from it, exposing `replay_store()` (delegates to the internal replay buffer's `add()`) alongside the inherited `RolloutBuffer` methods (`add`, `reset`, `compute_returns_and_advantage`).
- `runners.py:66-69` builds `"rollout_buffer_kwargs": {"replay_kwargs": {"episode_length": ..., "buffer_size": ..., "num_agemts": ...}}` — a nested `replay_kwargs` dict that matches **only** `PRMShardRolloutBuffer`'s constructor signature. `PRMShardReplayBuffer`/`PRMShardReplayBufferEpisodial` (the classes actually assigned to `rollout_buffer_class`, line 66) take `episode_length`/`num_agemts` as *top-level* kwargs, not nested under `replay_kwargs`; SB3's `OnPolicyAlgorithm._setup_model` would call `buffer_cls(n_steps, obs_space, action_space, device=..., gamma=..., gae_lambda=..., n_envs=..., **rollout_buffer_kwargs)`, i.e. pass `gae_lambda`/`gamma`/`replay_kwargs={...}` straight into `PRMShardReplayBuffer.__init__`, which forwards unrecognized kwargs to SB3's plain `ReplayBuffer.__init__` — that raises `TypeError` (unexpected keyword arguments).
- Independently of that, `rl_agents/ppo/rp_agents.py`'s `PPOPRM.collect_rollouts` (the only consumer of the ppo-path `rollout_buffer_class`) calls `rollout_buffer.replay_store(self._last_obs, new_obs, clipped_actions, pred_rewards, expiriment_rewards, done, infos)` in addition to `rollout_buffer.add/reset/compute_returns_and_advantage`. `replay_store` exists **only** on `PRMShardRolloutBuffer`; `PRMShardReplayBuffer`/`PRMShardReplayBufferEpisodial` have no such method (nor `reset`/`compute_returns_and_advantage`, being `ReplayBuffer` subclasses, not `RolloutBuffer` subclasses).

**Conclusion (high confidence): the PRM+PPO path (`PPOPRM`, selected whenever `RL_PARAMETERS.learner == "ppo"` under `PRMRunner`) is currently broken as wired** — it would raise `TypeError` at buffer construction, or failing that, `AttributeError` on `.replay_store()`/`.reset()` the first time `collect_rollouts` runs. `PRMShardRolloutBuffer` is the buffer this code path actually needs; it is unused only because `runners.py:66` assigns the wrong class family. This is consistent with the stated project context that PRM's live/primary training uses DQN (`DQNPRM`/`IndependentDQNRP`, whose `replay_buffer.add()` call signature in `rl_agents/dqn/rp_agents.py` matches `PRMShardReplayBuffer.add()` exactly) — the PPO+PRM path is plausibly untested/never actually run.

**`src/buffers/buffers.py` is dead code, high confidence**, and appears to be an abandoned earlier draft rather than a base class:
- Not imported anywhere (`__init__.py` imports only from `replay_buffers.py` and `rollout_buffers.py`; repo-wide grep for `buffers.buffers` / `from .buffers import` finds nothing outside the file itself).
- It is independently broken if it were ever instantiated: `get_episodes()` references `PRMEpisodeData` (line 107) and the `add()` signature type-hints `list[dict[str, Any]]` (line 50), but the file's imports (`deque`, `numpy`, `ReplayBuffer`, `Enum`, `spaces`, `ReplayBufferSamples`, `RolloutBuffer`) include neither `PRMEpisodeData`/`CRMEpisodeData` nor `typing.Any` — calling `get_episodes()` or evaluating the annotated `add()` would raise `NameError`.
- Its `PRMShardReplayBuffer.__init__` lacks the `num_agemts`/`num_agents` parameter entirely (compare §6) — an earlier state of the API before per-agent env-slicing was added — reinforcing that `replay_buffers.py` superseded it.

`episode_data.py` is **not dead**: `replay_buffers.py` imports `PRMEpisodeData, CRMEpisodeData` from it (`replay_buffers.py:6`) and both are returned from `get_episodes()`/`CRMShardReplayBuffer.get_episodes()`.

## 6. The `num_agemts` typo

Confirmed present in the **class signature itself**, not just the call site — `replay_buffers.py:20`:

```python
def __init__(self,
            buffer_size,
            observation_space,
            action_space,
            device="auto",
            n_envs=1,
            optimize_memory_usage=False,
            episode_length=100,
            num_agemts=0,
            *args,
            **kwargs):
    ...
    self.num_agents = num_agemts   # correctly-spelled attribute, misspelled param
```

`runners.py` passes the kwarg as `"num_agemts": self.config.ENV_PARAMETERS.num_agent` at all three call sites (`runners.py:50`, `:68`, and implicitly via the PPO `rollout_buffer_kwargs` nesting at `:69`) — matching the misspelling on both ends, so this is **not** a runtime-breaking bug (no `TypeError`/silent-drop-into-`**kwargs`). It's a cosmetic/consistency wart: internally the attribute is correctly `self.num_agents` (used correctly by `PRMShardReplayBufferEpisodial.get_episodes()` as `self.num_agents`), only the constructor parameter name is misspelled, and callers had to match the typo to work. `CRMShardReplayBuffer` avoids the issue entirely by using a differently-named, correctly-spelled parameter, `n_agents`, for the same purpose — so the codebase has two different spellings/names (`num_agemts`→`num_agents`, and `n_agents`) for "number of agents" across sibling classes, which is worth normalizing. The dead `buffers.py` copy of `PRMShardReplayBuffer` has no such parameter at all (see §5), so the typo was introduced only in the `replay_buffers.py` rewrite.
