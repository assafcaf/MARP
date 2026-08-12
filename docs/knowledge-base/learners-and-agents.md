# Learners and Agents

Source scope: `src/rl_agents/**`, `src/learners/**`, `src/callbacks/callbacks.py`. Cross-referenced against `src/experiment_runner/runners.py` for wiring (params, which class is picked per experiment).

## 1. Class hierarchy

All classes ultimately subclass an SB3 algorithm (`sb3_DQN`/`sb3_PPO`). `rl_agents/__init__.py` re-exports the leaf classes used by the runners.

| Class | File | Extends | Adds |
|---|---|---|---|
| `DQN` | `dqn/commons_agent.py` | `stable_baselines3.DQN` | commons-game episode-metrics logging (`efficiency`, `equality`, `sustainability`, `peace`, `fire_*`) via overridden `_update_info_buffer`/`_dump_logs`; multi-logger-safe `set_logger` |
| `IndependentDQN` | `dqn/independent_agent.py` | `DQN` (commons_agent) | Wraps `num_agents` independent `DQN` instances (`self.agents`), each with own replay buffer/policy; overrides `learn`/`train`/`collect_rollouts`/`predict`/`save`/`load` to fan out per-agent |
| `DQNPRM` | `dqn/rp_agents.py` | `DQN` (commons_agent) | Single shared agent; injects `predictor` and overrides `collect_rollouts`/`_store_transition` to store predicted reward instead of env reward |
| `DQNCRM` | `dqn/rp_agents.py` | `DQN` (commons_agent) | Multi-agent (`self.agents`, one per agent, `buffer_size=0` each — see §5), collective reward predictor keyed on joint state; overrides `collect_rollouts`/`_store_transition`/adds `train_multiagent` |
| `IndependentDQNRP` | `dqn/multiagent_rp_dqn.py` | `DQN` (commons_agent) | Wraps `num_agents` `DQNPRM` instances (`predictor=None` each — predictor lives only on the outer wrapper); overrides `collect_rollouts` to predict once per joint step and split rewards per agent |
| `PPO` | `ppo/single_agent.py` | `stable_baselines3.PPO` | Same commons-metrics logging pattern as `DQN` above (`_update_info_buffer`, `_dump_logs(iteration)`, `set_logger`) |
| `IndependentPPO` | `ppo/independent_agent.py` | `PPO` (single_agent) | Wraps `num_agents` independent `PPO` instances; overrides `learn`/`collect_rollouts`/`feedforward`/`train`/`predict` |
| `PPOPRM` | `ppo/rp_agents.py` | `PPO` (single_agent) | Single shared agent; injects `predictor`, overrides `collect_rollouts` to store predicted reward in the rollout buffer and additionally calls `rollout_buffer.replay_store(...)` with both predicted and experiment reward |

No `PPOCRM` / `IndependentPPORP` exist — matches runner wiring (CRM is DQN-only; PRM's PPO path is always the single shared `PPOPRM`, never split into independent PPO+RP agents).

`CnnFeatureExtractor` / `CustomCNN` (`feature_extractors.py`) are SB3 `BaseFeaturesExtractor` subclasses, unrelated to the agent hierarchy — plugged in via `policy_kwargs["features_extractor_class"]` in `runners.py`.

`utils.py` provides `DummyGymEnv` (bare gym.Env used to build placeholder `DummyVecEnv`s for per-agent sub-models) and `average_nested_dicts` (used by `DQN._update_info_buffer` to average episode-metrics dicts across parallel envs).

## 2. Independent vs collective multi-agent training

"Independent" = **separate policy network + separate replay/rollout buffer per agent**, no parameter sharing, no shared critic. Mechanically:

- `IndependentDQN.__init__` builds `self.agents = [DQN(env=dummy_env, ...) for _ in range(num_agents)]` — each a full SB3 `DQN` with its own `policy`, `replay_buffer`, `q_net`, optimizer. The outer `IndependentDQN` object itself is also a `DQN` (superclass init) but its own policy/buffer are unused scaffolding; real state lives in `self.agents`.
- `collect_rollouts` samples actions from every `agent._sample_action(...)` independently, steps the *shared* vec env once with concatenated actions, then routes each agent's slice of `(obs, reward, done, info)` into that agent's own replay buffer (`store_transitions`).
- `train(gradient_steps, batch_size)` just calls `agent.train(...)` for each agent — independent gradient updates, independent losses/optimizers.
- `IndependentPPO` is structurally identical for on-policy: `self.agents = [PPO(...) for _ in range(num_agents)]`, `feedforward` runs each agent's policy forward pass separately, `collect_rollouts` writes each agent's slice into its own `rollout_buffer`, `train()` calls `agent.train()` per agent.
- `IndependentDQNRP` is the same independent-per-agent pattern but each `self.agents[i]` is a `DQNPRM` (so each agent has its own replay buffer storing *predicted* reward) — however there is only **one shared reward predictor** at the outer-wrapper level (`self.predictor`), called once per joint step and its output sliced per agent (`pred_rewards[i::num_agents]`). So policies/buffers are independent, but the reward model is collective/shared even in the "independent" DQNRP path.

Collective/joint path (`DQNPRM`, used for `independent=False`):
- A single `DQN`-derived agent with one policy and one replay buffer sees the concatenated multi-agent transition; the predictor scores it and everything funnels through one shared network — genuine parameter sharing (one policy for all agents).

`DQNCRM` sits in between: it keeps `self.agents` (one policy per agent, so *not* parameter-shared) but reward comes from one collective predictor conditioned on full joint state (`env.get_full_states()`), and per the runner it's driven by `IndependentRLRPLearner` (not `CollectiveRLRPLearner`) — see §5 for why the naming is inconsistent with the CRM concept.

`CollectiveRLRPLearner` vs `IndependentRLRPLearner` (in `learners.py`) mirror this at the orchestration layer: `Collective*.learn_dqn` uses `self.rl_agent.replay_buffer` (singular — fits `DQNPRM`), `Independent*.learn_dqn` uses `self.rl_agent.agents` / `rl_agent.replay_buffers` (plural — fits `IndependentDQNRP` and `DQNCRM`).

## 3. Reward-predictor injection point

Injection always happens inside the overridden `collect_rollouts`, right after `env.step()`, replacing the environment reward with the predictor's output before it is written into the buffer. The env's native reward is still captured separately (as `expiriment_rewards`) and stored alongside for reference/eval, but the value used for RL targets is the predicted one.

**`DQNPRM.collect_rollouts`** (`dqn/rp_agents.py`):
```python
new_obs, expiriment_rewards, dones, infos = env.step(actions)
# reward predictor
pred_rewards = self.predictor.predict(obs_as_tensor(self._last_obs, self.policy.device),
                        th.tensor(actions).to(self.policy.device)).squeeze()
...
self._store_transition(replay_buffer, buffer_actions, new_obs, pred_rewards, expiriment_rewards, dones, infos)
```
`_store_transition` is also overridden to accept both `reward` (predicted) and `expiriment_rewards`, and calls `replay_buffer.add(..., reward_, expiriment_rewards, dones, infos)` — the custom `PRMShardReplayBuffer` stores both.

**`DQNCRM.collect_rollouts`**: same pattern but predictor is queried on the joint state (`env.get_full_states()`), not just observation:
```python
states = np.array(env.get_full_states())
pred_rewards = self.predictor.predict(obs_as_tensor(states, self.policy.device),
                                       th.tensor(actions).to(self.policy.device)).flatten()
```

**`IndependentDQNRP.collect_rollouts`**: predictor queried once per joint step on `self._last_obs` (all agents' obs concatenated), then sliced per agent when storing into each agent's own buffer (`pred_rewards[i::self.num_agents]`).

**`PPOPRM.collect_rollouts`** (`ppo/rp_agents.py`): same idea, but stored into the on-policy `rollout_buffer.add(...)` using `pred_rewards` (not `expiriment_rewards`) as the reward SB3's GAE/return computation uses, plus a side-channel `rollout_buffer.replay_store(self._last_obs, new_obs, clipped_actions, pred_rewards, expiriment_rewards, done, infos)` that persists both reward streams for reward-predictor training later.

In all cases the override point is `collect_rollouts` (and its paired `_store_transition`), not `train()` — the substitution happens at data-collection time, before any gradient step, so the RL algorithm's own `train()`/loss code is untouched and simply consumes whatever reward ended up in the buffer.

## 4. `CollectiveRLRPLearner` / `IndependentRLRPLearner` orchestration

Both live in `src/learners/learners.py` and share `BaseLearner.__init__(rl_agent, reward_predictor, train_rp_freq=1000, rp_learning_starts=10, batch_size=4)`. `train_rp_freq` is accepted but **not actually used to gate anything** in `learn_dqn`/`learn_ppo` (see below) — the real DQN-path gate is episode count vs `rp_learning_starts`; for PPO the reward predictor trains every iteration unconditionally.

DQN-path sequencing (`learn_dqn`, both subclasses look the same shape, differ only in which buffer(s) they pass):

```
_setup_learn()
 └─ loop while total_steps < total_timesteps:
     1. rl_agent.collect_rollouts(...)        # policy acts, predictor scores reward, stored in buffer(s)
        └─ if not continue_training: break
     2. total_steps += rollout.episode_timesteps
        total_episodes += rollout.n_episodes // num_envs
     3. if total_episodes > rp_learning_starts and total_episodes > ep_cnt:
            train_reward_predictor(buffer=...)   # RP gradient step(s) on replay data
            log RP metrics
     4. ep_cnt = total_episodes
        if gradient_steps > 0: rl_agent.train(batch_size, gradient_steps)   # RL gradient step(s)
 └─ callback.on_training_end()
```

So per outer loop iteration: **collect → (maybe) train reward predictor → train RL policy**, gated so the predictor only starts training once `rp_learning_starts` episodes have elapsed, and then at most once per newly-completed episode (`total_episodes > ep_cnt` prevents retraining mid-episode on every rollout call).

`IndependentRLRPLearner.learn_dqn` additionally calls `rl_agent.update_agents_last_obs()` before the loop, and trains the predictor once per sub-agent's buffer: `for rl_agent in self.rl_agent.agents: train_reward_predictor(buffer=rl_agent.replay_buffer)`. `CollectiveRLRPLearner.learn_dqn` trains it once on the single shared `self.rl_agent.replay_buffer`.

PPO-path sequencing (`learn_ppo`, identical body in both subclasses — no independent/collective branching at all despite the class split):
```
_setup_learn()
 └─ loop while total_steps < total_timesteps:
     1. rl_agent.collect_rollouts(env, callback, rl_agent.rollout_buffer, n_rollout_steps=n_steps)
     2. rl_agent._update_current_progress_remaining(...)
     3. train_reward_predictor()          # unconditional, every iteration
     4. if iteration % log_interval == 0: rl_agent._dump_logs(iteration)
     5. rl_agent.train()                  # PPO epochs on rollout_buffer (which already holds predicted reward)
```
Note `learn_ppo` never updates `self.total_steps`, so the `while self.total_steps < total_timesteps` loop only terminates via `collect_rollouts` returning `continue_training=False` — otherwise it runs forever. This looks like a real bug, though PRM's live PPO path is exercised via `PPOPRM` under this learner per `runners.py` (`PRMRunner.init_experiment`, ppo branch), so it's worth flagging if PPO+PRM runs are ever revisited.

## 5. Dead-code determination: `independent_rlrp_learner.py` / `collective_rlrp_learner.py`

**Confirmed dead.** `grep -rn "independent_rlrp_learner\|collective_rlrp_learner"` across the repo returns zero hits outside the files' own definitions — nothing imports them by path, and `src/learners/__init__.py` only imports `CollectiveRLRPLearner`/`IndependentRLRPLearner` from `learners.py`. `runners.py` imports exclusively `from learners import CollectiveRLRPLearner, IndependentRLRPLearner`, i.e. the package-level (`learners.py`) versions.

Comparing the two standalone files against `learners.py`:
- Same class names, same constructor signature, near-identical `learn`/`learn_dqn`/`learn_ppo`/`save`/`load`/`evaluate` bodies — they are clearly an **earlier, un-refactored draft**: `learners.py` factors the shared code into `BaseLearner` (both `CollectiveRLRPLearner`/`IndependentRLRPLearner` there just subclass it), whereas the standalone files duplicate every method independently with no shared base class.
- The standalone `independent_rlrp_learner.py.learn_dqn` references `self.loggers` (`for agent, logger in zip(self.rl_agent.agents, self.loggers)`) — but `self.loggers` is never assigned anywhere in `__init__` or elsewhere in that file, an `AttributeError` waiting to happen. This bug alone indicates the file predates a working version and was abandoned mid-edit.
- The standalone version's `learn_dqn` also calls `self.rl_agent.train_multiagent(...)` and pulls batches via `self.rl_agent.replay_buffer.get_episodes(batch_size=...)`, a different (and more `DQNCRM`-specific) code path than the package version's generic `rl_agent.train(...)` / per-agent-buffer `train_reward_predictor(buffer=...)`.

Conclusion: these two files are superseded duplicates of the classes now living in `learners.py`, left in the tree but never wired into any import path. Safe to treat as dead code / candidates for deletion.

Bonus finding while tracing CRM wiring: `CRMRunner.setup_agent` (`runners.py`) instantiates `IndependentRLRPLearner(... async_rp_training=False, parallel_agents=False, ...)`, but `BaseLearner.__init__` (the live `learners.py` version actually imported) has no `async_rp_training`/`parallel_agents` parameters — this call would raise `TypeError` if ever executed. Consistent with the "CRM never worked" status: the entrypoint is broken at the wiring level, independent of the zero result-directory evidence already gathered.

## 6. `SingleAgentCallback`

`src/callbacks/callbacks.py`, subclasses SB3 `BaseCallback`, used by all three runners for eval/logging during `agent.learn()`.

- `_on_training_start`: dumps run config (`self.args`) to `parameters.json` in the logger dir; creates a `checkpoints/` subdir.
- `_on_rollout_end`: increments an internal iteration counter `self.iterations_`; every `render_frequency` iterations (and not on iteration 0) calls `self._play(render=True)` — runs one deterministic-or-stochastic episode in `eval_env`, renders RGB frames via the underlying SSD env's `.render(mode="RGB")`, resizes to `video_resolution` (derived from the env's `world_map` shape × `tile_size`), and writes an `.mp4` via OpenCV (`save_video`) named `iteration_{n}_score_{score}.mp4`.
- `_play`: rolls out `eval_env` to completion, summing `info['true_reward']` per step (falls back to raw `reward` if that key is absent) as the reported score.
- Model checkpointing (`self.model.save(...)`) is present but commented out — not currently active.

`render_frequency` and `log_interval` are computed differently per learner in `BaseRunner.init_experiment` (`runners.py`):
```python
render_frequency = self.config.RL_PARAMETERS.render_frequency * self.config.ENV_PARAMETERS.ep_length
if self.config.RL_PARAMETERS.learner == "ppo":
    log_interval = 1
    render_frequency = self.config.RL_PARAMETERS.render_frequency          # NOT multiplied by ep_length
else:  # dqn
    log_interval = self.config.ENV_PARAMETERS.num_agent * self.config.ENV_PARAMETERS.num_envs
    render_frequency = self.config.RL_PARAMETERS.render_frequency * self.config.ENV_PARAMETERS.ep_length
```
- For **PPO**, `_on_rollout_end` fires once per `n_steps`-sized rollout/iteration, so `render_frequency` is used directly in iteration units and `log_interval` is fixed at 1 (dump logs every iteration).
- For **DQN**, `_on_rollout_end` fires once per `collect_rollouts` call (roughly once per gradient-update cycle, not per episode), so `render_frequency` is pre-multiplied by `ep_length` to land on a comparable wall-clock/step cadence, and `log_interval` (passed separately into SB3's own episode-count-based logging, e.g. `_dump_logs` triggers in `collect_rollouts` when `self._episode_num % log_interval == 0`) is set to `num_agent * num_envs` so that logs are dumped roughly once per "true" env-episode across all parallel agents/envs.

## Summary

`DQN`/`PPO` (in `commons_agent.py`/`single_agent.py`) add commons-game metric logging on top of raw SB3. `Independent*` variants replace the single policy with a list of fully separate per-agent SB3 models (own policy, buffer, optimizer) driven through overridden `collect_rollouts`/`train`/`predict`; the plain (non-independent) `*PRM` variants keep one shared policy/buffer. Reward-predictor substitution happens uniformly inside each `*PRM`/`*CRM`/`*RP` class's overridden `collect_rollouts`, right after `env.step()`, replacing env reward with `predictor.predict(...)` before storing the transition — the env reward is kept alongside for eval/RP-training but not used as the RL target. `CollectiveRLRPLearner`/`IndependentRLRPLearner` (`learners.py`) orchestrate collect → conditionally train reward predictor (episode-count gated for DQN, every iteration for PPO) → train RL policy, each loop iteration.

`src/learners/independent_rlrp_learner.py` and `collective_rlrp_learner.py` are dead code: nothing imports them (grep-confirmed), they duplicate the classes in `learners.py` without the shared `BaseLearner` refactor, and the independent-learner file has a live bug (`self.loggers` referenced but never set). Separately, `CRMRunner` passes unsupported kwargs (`async_rp_training`, `parallel_agents`) to `IndependentRLRPLearner`'s actual (imported) constructor, which would raise `TypeError` — corroborating that CRM was never made to run end-to-end.
