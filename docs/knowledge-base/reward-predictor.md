# `src/reward_predictor/` Knowledge Base

Scope: `prm/` and `crm/` subpackages, `segment_sampling.py`, `label_schedules.py`, `summaries.py`,
`clip_manager.py`, `comparison_collectors.py`, `utils.py`, `__init__.py`. Cross-references
`docs/knowledge-base/paper-and-configs.md` (paper↔code terminology) and `buffers.md` (buffer
internals) rather than repeating them.

`__init__.py` exports only: `RPMRewardPredictor` (= `prm/reward_model.py`'s `ComparisonRewardPredictor`),
`CRMRewardPredictor` (= `crm/reward_model.py`'s `ComparisonRewardPredictor`), `AgentLoggerSb3`,
`parallel_collect_segments`, `LabelAnnealer`, `function_wrapper`. **Not exported**, and — see §2 —
**not called anywhere in the repo**: `comparison_collectors.py`'s `SyntheticComparisonCollector` /
`HumanComparisonCollector`, and `clip_manager.py`'s `ClipManager` / `SynthClipManager`.

## 1. Pipeline overview — as actually implemented (not as the file layout suggests)

The package looks, from its file names, like the classic OpenAI `rl-teacher` pipeline (segment
sampling → comparison collector → label schedule → reward model). **It isn't wired that way.**
`segment_sampling.py`, `comparison_collectors.py`, `label_schedules.py`, and `clip_manager.py` are
leftover scaffolding from that lineage; grepping the whole repo (`grep -rn` for each class/function
outside its own defining file) turns up **zero call sites**. The real pipeline lives entirely inside
`reward_model.py`'s `ComparisonRewardPredictor.train_predictor()` / `transform_batch()` and the
replay buffer's `get_episodes()` (see `buffers.md`):

```
env.step()                                    (src/env/pettingzoo_env.py)
  └─ per-step reward = 0, EXCEPT last step of episode:
     reward = social-metric value (e.g. Efficiency*Peace), same scalar broadcast to every agent
       ↓
ReplayBuffer.add(experiment_rewards=...)      (PRMShardReplayBuffer / CRMShardReplayBuffer)
  └─ stores per-episode-shard: obs/state, actions, experiment_rewards, true_rewards, aip
       ↓
ReplayBuffer.get_episodes(batch_size)         → PRMEpisodeData / CRMEpisodeData
  └─ samples `batch_size` whole episodes at random (no explicit segment-pairing step)
       ↓
ComparisonRewardPredictor.transform_batch(batch)
  └─ SPLITS the sampled batch in half: first half = "left", second half = "right"
  └─ label = which half has the larger summed experiment_reward   ← THE PREFERENCE LABEL, synthetic
       ↓
ComparisonRewardPredictor._train_step()
  └─ model scores left/right segments → Bradley-Terry-style softmax/cross-entropy loss vs. label
       ↓
predictor.predict(obs, act)                   (rp_agents.py, called every env.step)
  └─ predicted reward REPLACES the env reward the RL algorithm (DQN/PPO) actually trains on
```

Key functions/classes per stage: `PettingZooEnv.step` (env/pettingzoo_env.py, shapes the oracle
signal) → `PRMShardReplayBuffer.add`/`CRMShardReplayBuffer.add` → `get_episodes` →
`ComparisonRewardPredictor.transform_batch` → `ComparisonRewardPredictor._train_step` →
`ComparisonRewardPredictor.predict` → `DQNPRM`/`DQNCRM`/`PPOPRM` `collect_rollouts` (stores
`pred_rewards` as the transition reward, not `experiment_rewards`).

## 2. Segment sampling & comparison collection — where labels actually come from

**`segment_sampling.py` (`parallel_collect_segments`, `collect_segments`, `do_rollout`,
`sample_segment`) is dead code.** It implements a plausible-looking pipeline — roll out a
random-action policy, slice fixed-length windows, collect `n_desired_segments` via a
multiprocessing pool — but nothing in the repo imports it except `__init__.py`, which nothing
downstream imports it *from* either. Same verdict for **`comparison_collectors.py`**:
`SyntheticComparisonCollector._add_synthetic_label` (compares `sum(left_seg["expiriment_rewards"])`
vs `sum(right_seg["expiriment_rewards"])`) and `HumanComparisonCollector` (imports a nonexistent
`human_feedback_api` Django app, builds GCS video URLs, polls a `Comparison` DB model) are never
instantiated anywhere.

**The label mechanism that is actually live** is inline inside
`ComparisonRewardPredictor.transform_batch` in both `prm/reward_model.py` and `crm/reward_model.py`.
It is structurally the same idea as `SyntheticComparisonCollector._add_synthetic_label` (compare
summed reward of two segments → binary label), just reimplemented directly against the replay
buffer instead of through a `Comparison`/collector object:

```python
# prm/reward_model.py — transform_batch()
l = batch.observations.shape[0] // 2
left_obs,  left_acts,  left_r  = batch.observations[:l],  batch.actions[:l],  batch.experiment_rewards[:l].sum(axis=1)
right_obs, right_acts, right_r = batch.observations[l:],  batch.actions[l:],  batch.experiment_rewards[l:].sum(axis=1)
l1 = (left_r >= right_r).float(); l2 = (left_r <= right_r).float()
labels = torch.stack([l1, l2], dim=1)          # soft 2-way label, ties → [1,1]
delta  = (left_r - right_r).abs()               # preference-magnitude weight
```

So: **pairs are not sampled by any dedicated "form a comparison" function — a batch of
`batch_size` whole episodes is drawn from the replay buffer and arbitrarily split down the middle**;
the first half plays "left", the second half plays "right" of a comparison. There is no guarantee
the two halves are behaviorally related — they're just two disjoint random episode samples from the
same buffer window. **The label is 100% synthetic**, generated by literally comparing
`sum(experiment_rewards)` between the two halves — no human ever labels anything, and the dead
`HumanComparisonCollector` path (Django/GCS-backed) confirms a human-in-the-loop mode was designed
for but is not exercised.

Critically, `experiment_rewards` is **not** raw environment reward. Per `env/pettingzoo_env.py`
(`PettingZooEnv.step`, lines ~48-70): if `EXPERIMENT_PARAMETERS.metric != "Efficiency"`, every
per-step reward is zeroed, and only at the terminal step is `experiment_rewards` set to the
configured **social-metric composite** (e.g. `efficiency * peace`, `efficiency * peace * equality`,
`efficiency * sustainability*2`), broadcast identically to every agent's `info` dict for that step.
This is exactly the paper's episode-level social-metric oracle φ(ω) (§4.1, Definition 3) —
`experiment_rewards` *is* the oracle signal, not ground truth game score. The separate
`true_rewards` field (`info['true_reward']`, raw apple-eating reward) is stored purely for logging
(`summaries.py`/`log_step`'s `outcome_avg/apple_eaten` etc.), never used in the loss.

**PRM vs CRM diverge in what gets summed for the label** (see §5) — PRM sums per-agent (or, under
`episodial: True`, per-team) reward from each agent's own trajectory; CRM sums from global state.
See §5 for why this is *not* the paper's Local-Trajectory/Joint-Episode split, despite looking like
a natural candidate for it.

## 3. Label scheduling

`label_schedules.py` defines `LabelAnnealer` and `ConstantLabelSchedule` — **both unused** (only
`LabelAnnealer` is even re-exported from `__init__.py`; grep finds no downstream import). Per the
docstring/implementation, had it been wired in, `LabelAnnealer.n_desired_labels` would anneal the
*number of labels requested* from `pretrain_labels` up to `final_labels` over `final_timesteps`,
via an exponential-decay schedule:

```python
exp_decay_frac = 0.01 ** (timesteps_elapsed / final_timesteps)      # 1 → 0
pretrain_frac  = pretrain_labels / final_labels
desired_frac   = pretrain_frac + (1 - pretrain_frac) * (1 - exp_decay_frac)
return desired_frac * final_labels
```

i.e. a label-request **budget schedule** (how many labels to have collected by now), not a
noise/confidence schedule. `ConstantLabelSchedule` is a simpler wall-clock-rate variant
(`pretrain_labels + elapsed_seconds / seconds_between_labels`). Neither has any effect on training in
the current codebase — the actual "how much to train the reward model" control is
`RP_PARAMETERS.train_freq` / `learning_starts` / `predictor_epochs` / `batch_size`, consumed directly
by `runners.py`/`learners.py`, bypassing this file entirely.

## 4. Reward model architecture

Both `prm/reward_model.py` and `crm/reward_model.py` define a class literally named
`ComparisonRewardPredictor(RewardModel)` (aliased on import as `RPMRewardPredictor` /
`CRMRewardPredictor` — see repo owner's confirmed RPM/PRM naming swap, flagged in
`paper-and-configs.md` too). Shared shape:

| | PRM (`prm/reward_model.py`) | CRM (`crm/reward_model.py`) |
|---|---|---|
| Constructor extra arg | — | `num_outputs` (= `ENV_PARAMETERS.num_agent`) |
| Backbone options | `EmbedCnnNetwork` \| `OneHotCnnNetwork` (`prm/nn.py`), selected via `network=` kwarg | `MlpNetwork` \| `CnnNetwork` (`crm/nn.py`), hardcoded to `CnnNetwork` in `_initialize_buffers_and_model` |
| Loss object | `nn.CrossEntropyLoss(reduction='none')` (per-sample, so it can be weighted) | `nn.CrossEntropyLoss()` (default `reduction='mean'`, **unweighted**) |
| Optimizer | `Adam(lr, weight_decay=1e-4)` | `Adam(lr, weight_decay=1e-4)` |
| Training entry points | `train_predictor` (fresh batch/epoch) **and** `train_concat_batch_predictor` (accumulates batches across epochs into one growing `batch`) — both exist, only `train_predictor` has a confirmed call site (`learners.py`'s `BaseLearner.train_reward_predictor`) | `train_predictor` only |

Config-driven params (from `configs/prm.yaml` / `configs/crm.yaml`, passed via
`RPMRunner`/`CRMRunner.set_up_rp` per the CONFIRMED WIRING):

| Param | PRM value | CRM value | Consumed by |
|---|---|---|---|
| `predictor_epochs` (→ `self.epochs`) | 3 | 4 | inner loop count in `train_predictor` |
| `lr` | 0.0005 | 0.0001 | `Adam` |
| `RP_PARAMETERS.batch_size` (×2 in `runners.py`, PRM only — `batch_size=RP_PARAMETERS.batch_size*2`) | 32 → 64 episodes/call | 1 (CRM doesn't get the `*2`; effectively 1 episode split, i.e. degenerate 1-vs-nothing... see below) | `buffer.get_episodes(batch_size)` inside `transform_batch`'s halving |
| `network` | `"OneHotCnnNetwork"` | n/a (`CnnNetwork` hardcoded) | backbone class selection |
| `emb_dim` | 32 | not set in `crm.yaml` → falls back to class default `emb_dim=8` | action-embedding width |
| `fcnet_hiddens` | `[64, 32, 16]` | not set → `CnnNetwork` default `features_dim=32` (name mismatch: PRM calls it `fcnet_hiddens`, CRM's equivalent knob is `features_dim`) | MLP head widths |
| `train_freq` × `ep_length` | 5 × 400 = 2000 env steps between RP training calls | 1 × 600 = 600 | `learners.py` `train_rp_freq` |
| `learning_starts` (RP) | 32 episodes | 1 episode | gate before RP training starts (`total_episodes > rp_learning_starts`) |

Note on CRM's `RP_PARAMETERS.batch_size: 1`: `CRMRunner.setup_agent` still applies
`batch_size=self.config.RP_PARAMETERS.batch_size*2`, so it's actually 2 episodes/call (1 left vs 1
right) — extremely small compared to PRM's 64; flagged as a plausible contributor to CRM's reported
instability (see §5), not a confirmed root cause.

Loss, PRM (delta-weighted Bradley-Terry cross-entropy — matches the paper's Eq. 6–8 `δ'_xy`-weighted
BCE/softmax formulation):

```python
logits = torch.stack([rewards_left.sum(axis=1), rewards_right.sum(axis=1)], dim=1)
loss = (self.loss(logits, labels)
        * F.softmax(delta / (delta.std() + 1), dim=0).to(device)).sum()
```

Loss, CRM (**plain unweighted** cross-entropy, hard class-index label — does *not* implement the
paper's δ-weighting despite both classes sharing the name `ComparisonRewardPredictor`):

```python
labels = (left_r < right_r).long()                                   # hard class index, not soft/tied-aware
logits = torch.stack([rewards_left.mean(axis=2).sum(1), rewards_right.mean(axis=2).sum(1)], dim=1)
loss = self.loss(logits, labels)                                     # nn.CrossEntropyLoss(), unweighted
```

This resolves the open question flagged in `paper-and-configs.md` §4 ("whether the code implements
the exact δ'-weighted BCE of Eq. 8, or a simpler unweighted BCE, left to the reward_predictor-focused
review to confirm"): **PRM does, CRM does not.**

## 5. Correction: Local-Trajectory vs Joint-Episode are BOTH inside PRM — CRM is a third, separate thing

**Earlier draft of this doc mapped PRM ≈ Local-Trajectory and CRM ≈ Joint-Episode. That mapping was
wrong.** Re-reading the paper's Method section (p.6-8) against the actual buffer code shows both of
the paper's published MARP variants are implemented *inside* `reward_predictor/prm/` — selected
entirely by the `RP_PARAMETERS.episodial` flag in `prm.yaml` — and CRM is not one of the paper's two
variants at all.

The paper is explicit that both variants use the **same single-agent-input reward model**
(§4.2, p.6): *"the reward model architecture is based on single-agent input, with one
observation-action pair as input and a single reward scalar as output. However, the reward model is
simultaneously trained by all agents."* Joint-Episode differs only in how comparison inputs are
formed: *"the reward model is trained on **concatenated inputs that combine all agent trajectories**
from a given episode into a single sequence"* (p.6-7) — i.e. still per-agent `(o,a)` pairs scored by
the same model, just many agents' pairs pooled into one sequence for the purpose of summing/comparing.
Neither variant, per the paper, ever conditions the model on a global/full-map state.

That is exactly what the two PRM buffer classes already do (`buffers.md` §1/§3), both feeding the
identical `ComparisonRewardPredictor` in `reward_predictor/prm/reward_model.py`:

| Axis | PRM, `episodial: False` (Local-Trajectory) | PRM, `episodial: True` (Joint-Episode) |
|---|---|---|
| Buffer class | `PRMShardReplayBuffer.get_episodes()` | `PRMShardReplayBufferEpisodial.get_episodes()` |
| What one "episode" sample contains | one agent's own `(episode_length, obs)` trajectory | **all agents'** per-step observations for that episode `np.vstack`'d into one `(episode_length*num_agents, obs)` sequence |
| `transform_batch`'s `experiment_rewards[:l].sum(axis=1)` | sums over time only, for that one agent | sums over time **and agents together** (both packed into axis 1 by the vstack) — a team-episode-level sum |
| Reward model, loss, δ-weighting | identical `ComparisonRewardPredictor`, identical δ-weighted Bradley-Terry cross-entropy | identical |

This matches Fig. 2's toy example precisely: in both variants "the same episode-level preference
label drives both variants" and the model is always scored per `(o,a)` pair — Joint-Episode just
changes which pairs get pooled into one side of the comparison. **`results/prm-dqn-episodial-*`
directories confirm this variant has real, working results** (e.g.
`prm-dqn-episodial-Efficiency*Peace-fast-4_agents`), consistent with the paper reporting Joint-Episode
as working.

**CRM is architecturally a third thing the paper doesn't describe**: its `ComparisonRewardPredictor`
(`reward_predictor/crm/reward_model.py`) is conditioned on `env.state_space` — the **global full-map
state** (`env.get_full_state()`), identical for every agent at a timestep — plus the **joint action
vector** of all agents concatenated and embedded together, producing a `num_agent`-length vector of
rewards from one centralized forward pass. This is a centralized/global-state design, not an
"aggregated local trajectories" design; nothing in the paper's Local-Trajectory/Joint-Episode
definitions (or Fig. 3's architecture diagram, which always routes through per-agent `(o_i, a_i)`
inputs) describes conditioning the reward model on global state. Concretely:

| Axis | PRM (either variant) | CRM |
|---|---|---|
| Observation input | per-agent egocentric partial view (`env.observation_space`) | global full-state (`env.state_space`) |
| Action input to model | one agent's action | **all agents' joint action vector**, concatenated/embedded together (`crm/nn.py` `CnnNetwork.forward`) |
| Model output shape | scalar — one reward per (obs, action) | vector of length `num_agent` — one reward per agent, from a single centralized forward pass |
| What the comparison label sums | `experiment_rewards[:l].sum(axis=1)` (time, and agents too if episodial) | `experiment_rewards[:l].sum(axis=2).sum(axis=1)` — always time **and** all agents |
| What the loss supervises | full predicted-reward sum | `rewards_left.mean(axis=2).sum(1)` — averages across the `num_agent` output heads before comparing to the team-summed label; individual heads are under-constrained by this loss |
| Buffer class | `PRMShardReplayBuffer`/`PRMShardReplayBufferEpisodial` | `CRMShardReplayBuffer` (adds a `states` array from `state_space`) |
| Backbone network | `OneHotCnnNetwork`/`EmbedCnnNetwork` (`prm/nn.py`) | `CnnNetwork` (`crm/nn.py`) |
| Loss weighting | δ-weighted (magnitude-aware) cross-entropy, matches paper Eq. 8 | plain unweighted cross-entropy — does not match Eq. 8 (see §4) |

**Practical implication for the refactor**: this resolves the "paper reports Joint-Episode working,
code's CRM never worked" conflict flagged in `paper-and-configs.md` — they were never the same thing.
The paper's Joint-Episode results were almost certainly produced by `prm.yaml`/`prm_ppo.yaml` with
`episodial: True`, not by `teach_crm.py`/`CRMRunner`. CRM appears to be a separate, likely earlier or
abandoned, centralized-state reward-model experiment that predates or diverged from what shipped in
the paper — worth confirming with the owner, but the code/paper evidence now points there rather than
at "CRM is Joint-Episode but buggy."

Minor unrelated bug spotted while reading `crm/nn.py`: the base class `MlpHead.forward` does
`emb = self.embed(act.long()); emb = torch.cat(emb, axis=1)` — `torch.cat` on a single tensor (not a
list) is a misuse of the API and would error if ever called. It isn't: both `MlpNetwork` and
`CnnNetwork` override `forward()`, so `MlpHead.forward` is dead code within a dead-ish class (only
`CnnNetwork` is actually instantiated by `crm/reward_model.py`). Also dead: `prm/nn2.py` (an older
`MlpHead`/`MlpNetwork`/`CnnNetwork`/`CustomCNN` set) — `prm/reward_model.py` imports from `.nn`, not
`.nn2`; nothing imports `nn2.py`.

## 6. Logging — `AgentLoggerSb3` / `clip_manager.py`

`AgentLoggerSb3` (`summaries.py`) **is live** — constructed in `runners.py` (`PRMRunner.setup_loggers`,
`CRMRunner.setup_loggers`) and passed into both `RPMRewardPredictor`/`CRMRewardPredictor` as
`agent_logger`. Per training step it logs to the SB3 logger (stdout/tensorboard/csv, not a
human-review tool): rolling `true_reward_per_episode`, `total_steps`, and — via
`analyze_actions`/`log_step` (the latter defined on `ComparisonRewardPredictor` itself, not this
class, but following the same `agent_logger.log_simple` pattern) — predicted-reward breakdowns by
action, by apple-eaten/not, and by `aip` ("apples in proximity", a 3×3-neighborhood apple count from
`env/agent.py`, purely descriptive). `log_plot` additionally renders a matplotlib figure of
before/after observation frames with real-vs-predicted reward captions every 1000 summary steps,
logged as an SB3 `Figure` — a debugging visualization for the researcher, not a human-preference
collection UI.

`clip_manager.py`'s `ClipManager` and `SynthClipManager` are **vestigial, confirmed by grep: zero
call sites anywhere outside their own file**, and not even re-exported from `__init__.py`. They
implement a full human-in-the-loop clip pipeline lifted from `rl-teacher`-style codebases: uploading
MP4 renders to a GCS bucket (`RL_TEACHER_GCS_BUCKET` env var), syncing a red-black `SortTree` against
a Django `human_feedback_api` app (`Clip`/`Comparison`/`SortTree` models that don't exist in this
repo), and interactively prompting the user (`input("Do you want to erase this clip? (y/n)")`) on
load errors. None of this is reachable from any runner/agent/learner in the current codebase — it is
earlier-design residue for a human-preference mode that predates (or was abandoned in favor of) the
fully-synthetic label mechanism actually in use (§2).

## 7. Paper cross-check (`docs/paper-tmlr-anonymized-2026-07-26.pdf`, read directly — full text +
appendix, pp. 1-17; Method section re-verified against source PDF at §5 above)

The paper never uses "PRM", "CRM", "RPM", "personalized", or "collective reward model" — it names the
whole framework **MARP (Multi-Agent Reward Prediction)** with two variants, **Local-Trajectory
Inference** and **Joint-Episode**. **Both variants are implemented inside `reward_predictor/prm/`,
selected by the `episodial` config flag — see §5 for the corrected mapping.** `reward_predictor/crm/`
does not correspond to either published variant; it's a separate centralized-global-state design.

- **Loss** (Eq. 6-8): $s_x=\sum_{(o,a)\in x}\hat r_\theta(o,a)$, $P(x\succ y)=\frac{\exp(s_x)}{\exp(s_x)+\exp(s_y)}$,
  $\mathcal L(\theta)=\sum \delta'_{xy}\cdot\text{BCE}(\mu, P(x\succ y))$ — a magnitude-weighted
  Bradley-Terry loss, citing Christiano et al. (2017). **PRM implements this** (softmax-over-δ
  weighting, §5 above) for both Local-Trajectory and Joint-Episode, since they share the same reward
  model/loss code. **CRM does not** (plain unweighted `CrossEntropyLoss`) — consistent with CRM being
  outside the paper's described method rather than a buggy copy of it.
- Fig. 8 (reward-model architecture) and Appendix A Table 2 (lr 1e-4, batch pairs 64, train steps 50,
  update freq 1000 env steps, warmup 50 episodes, buffer 5000 episodes, FC layers `[128+|A|,128,1]`)
  describe one canonical architecture/hparam set for "the shared reward model" — this is the single
  model shared by Local-Trajectory and Joint-Episode (both live in `prm.yaml`'s `RP_PARAMETERS`);
  `crm.yaml`'s separate, differently-shaped hyperparameters are for the unrelated CRM architecture.
- The paper reports **both of its variants working** (Figs. 4/6/7/9/10/11) — this is **no longer a
  conflict** once Joint-Episode is correctly identified as `prm.yaml`/`episodial: True` (which has
  real results under `results/prm-dqn-episodial-*`) rather than as CRM (which has none). The
  previously-flagged "paper says Joint-Episode works, CRM never worked" discrepancy in
  `paper-and-configs.md` §1 is resolved by this correction, not merely explained away.
- Nothing in the paper describes a human-labeling / clip-video pipeline anywhere in the read pages —
  consistent with §2/§6's finding that `comparison_collectors.py`'s `HumanComparisonCollector` and
  `clip_manager.py` are unused: the paper's method, as published, is the fully-synthetic
  social-metric-oracle pipeline this repo actually runs.
