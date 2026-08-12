# Paper ↔ Code Mapping and Config Reference

Source paper: `docs/paper-tmlr-anonymized-2026-07-26.pdf` — **"Why Study Emergent Behavior When You Can
Regulate It? Aligning Multi-Agent Systems with Reward Prediction"** (anonymous TMLR submission, 19
pages, read in full). Paper's own framework name: **MARP (Multi-Agent Reward Prediction)**.

**Important finding up front: the strings "PRM", "NRP", and "CRM" do not appear anywhere in the paper
text.** These are code-only abbreviations. The mapping below is therefore inferred from matching each
code concept to the closest paper concept, not read directly off the page — flagged wherever inference
was needed.

---

## 1. Terminology conflicts (surface these to the repo owner — not silently resolved)

**Correction (post-review, confirmed against paper p.6-8 and the buffer code directly — see
`reward-predictor.md` §5 for full derivation): the row below originally mapped CRM ≈ Joint-Episode.
That was wrong.** The paper states both its variants share "single-agent input, with one
observation-action pair as input" for the reward model, and Joint-Episode differs only by training
on "concatenated inputs that combine all agent trajectories from a given episode into a single
sequence" — never a global/full-map state. That description is implemented entirely inside
`reward_predictor/prm/`, toggled by `RP_PARAMETERS.episodial` (`prm*.yaml`): `episodial: False` =
Local-Trajectory, `episodial: True` = Joint-Episode (row 31 of §2 below already flagged this flag's
role but the mapping wasn't updated to match — fixed now). CRM's `env.state_space`-conditioned,
global-state design is not one of the paper's two published variants at all.

| Term | README.md says | Repo owner says (given directly) | Paper says | Verdict |
|---|---|---|---|---|
| PRM | "Preference-based Reward Model" | "personalized reward model" | Not used by name; **PRM implements *both* paper variants** — `episodial: False` = **Local-Trajectory Inference** ("Narrow View Inference"), `episodial: True` = **Joint-Episode** ("Input Aggregation") — both share the same `ComparisonRewardPredictor`/reward model/loss code in `reward_predictor/prm/`, differing only in which buffer class aggregates the comparison inputs | Three different definitions for the same acronym, and the code-level truth is that "PRM" isn't one paper concept but a code module that houses *both* of the paper's variants. Owner's "personalized" fits Local-Trajectory specifically, not Joint-Episode (which is still per-agent-input but aggregated) — worth asking whether "personalized" was meant to describe just the non-episodial mode. README's "Preference-based" is technically true of the whole framework, not PRM specifically — likely stale/generic. |
| NRP | "Neural Reward Predictor" | "no reward predictor" (baseline trained on ground-truth env reward) | Not used; closest concept is the **Baseline-Original / Baseline-Penalty** curves in Fig. 4 — plain PPO/DQN trained directly on the environment-defined reward (reproducing Perolat et al. 2017), no learned reward model at all | README's "Neural Reward Predictor" is essentially the **opposite** of what the code does — NRP has NO neural reward predictor, it's the no-reward-model control condition. This is a direct, confirmed contradiction between README and actual runner behavior (`teach_nrp.py` / `NRPRunner` trains straight on env reward — see `RL_PARAMETERS`/env `penalty` flag, no `RP_PARAMETERS`-driven predictor in the loop). Trust the owner's definition. |
| CRM | "Counterfactual Reward Model" | "collective reward model" ("could not make it work") | **Not used, and does not correspond to either published MARP variant.** CRM conditions its reward model on `env.state_space` (global full-map state) plus the joint action vector of all agents — a centralized-critic-style design the paper never describes; both of the paper's variants (Local-Trajectory *and* Joint-Episode) always score individual per-agent `(o,a)` pairs, never global state | README's "Counterfactual" doesn't match anything in the paper (COMA is cited only as related work, not implemented). Owner's "collective" is directionally accurate for what the *code* does (global/team-level state and reward), but this is **not** the paper's Joint-Episode concept — that's actually `prm.yaml`'s `episodial: True` mode (see correction note above), which does have working results (`results/prm-dqn-episodial-*`). CRM is best understood as a separate, likely earlier or abandoned, centralized-state reward-model experiment outside the paper's validated scope — consistent with it never having produced usable results. **This resolves what was previously flagged as the single biggest open question** (paper reports Joint-Episode working vs. CRM never working) — they were never the same method. |

## 2. Paper term → code term mapping

| Paper term | Paper meaning | Code term / file / class |
|---|---|---|
| MARP (Multi-Agent Reward Prediction) | The overall framework: shared learned reward model trained on episode-level preferences, replacing hand-crafted step-wise reward | Repo-wide framework spanning `src/reward_predictor/`, `src/experiment_runner/runners.py` |
| Joint-Episode (MARP-JE) / "Input Aggregation" | Reward model trained on concatenated all-agent-trajectory episode representations; preference oracle is episode-level (𝕆_ω) | **`src/reward_predictor/prm/reward_model.py`, with `RP_PARAMETERS.episodial: True`** — same `ComparisonRewardPredictor`/`RPMRewardPredictor` class as Local-Trajectory, fed by `PRMShardReplayBufferEpisodial` instead of `PRMShardReplayBuffer`. Confirmed, not the earlier CRM guess — see §1 correction note. |
| Local-Trajectory (MARP-LT) / "Narrow View Inference" | Reward model trained on individual per-agent egocentric trajectories; preference label inherited from the episode-level oracle but applied at trajectory level (𝕆_τ) | `src/reward_predictor/prm/reward_model.py`, with `RP_PARAMETERS.episodial: False` (the default) — `ComparisonRewardPredictor`, `RPMRewardPredictor`, `DQNPRM`/`PPOPRM`, `PRMShardReplayBuffer`, `prm*.yaml`, `teach_prm.py` |
| *(not in paper)* Centralized/global-state reward model | No paper equivalent — neither MARP variant conditions the reward model on global state | `src/reward_predictor/crm/reward_model.py` `ComparisonRewardPredictor`, `CRMRewardPredictor`, `DQNCRM`, `CRMShardReplayBuffer`, `crm.yaml`, `teach_crm.py` — a separate, non-paper experimental design (never produced results) |
| Baseline-Original / Baseline-Penalty | PPO/DQN trained directly on environment ground-truth reward (reproducing Perolat et al. 2017); "Penalty" variant presumably adds a tagging/conflict penalty term | `NRPRunner`, `teach_nrp.py`, `nrp*.yaml` — `EXPERIMENT_PARAMETERS.penalty` flag toggles the two variants (see `nrp.yaml`: `penalty: True`; `nrp_ppo.yaml`: `penalty: False`) |
| Social Metric φ(ω) / Preference Oracle 𝕆 | Function scoring an episode's collective outcome; oracle ranks episode pairs by that score to generate preference labels | `EXPERIMENT_PARAMETERS.metric` config field, e.g. `"Efficiency"`, `"Efficiency*Peace"`, `"Efficiency*Sustainability"` |
| Episode-level vs. trajectory-level oracle (𝕆_ω vs 𝕆_τ) | Distinguishes Joint-Episode's episode-pair comparisons from Local-Trajectory's per-agent-trajectory-pair comparisons | `RP_PARAMETERS.episodial` flag in `prm*.yaml` — `PRMShardReplayBufferEpisodial` vs `PRMShardReplayBuffer`; results dirs get an `-episodial` suffix (e.g. `prm-dqn-episodial-Efficiency-fast-4_agents`). **This is the flag that actually selects Joint-Episode vs Local-Trajectory** — see corrected §1/row above. |
| Independent PPO (IPPO) | The actual RL algorithm used for all policy optimization (custom implementation, Appendix A Table 1) | `RL_PARAMETERS.learner: "ppo"` + `EXPERIMENT_PARAMETERS.independent`; `IndependentPPO`, `IndependentDQN`, `IndependentRLRPLearner` in `src/rl_agents`/`src/learners` |
| Harvest Game / Commons Game | The SSD (sequential social dilemma) environment: apples regrow density-dependently, agents can tag each other | `src/env/` (PettingZoo-style commons/harvest env), `ENV_PARAMETERS.map` (`SMALL_HARVEST_MAP`, `MEDIUM2_HARVEST_MAP`, `HARVEST_MAP_LARGER`) |
| $\hat r_\theta(o,a)$ reward model architecture | CNN feature extractor over pixel obs + one-hot action encoding → concat → MLP → linear scalar output (Fig. 8) | `network: "OneHotCnnNetwork"` / `"EmbedCnnNetwork"` in `RP_PARAMETERS`, `fcnet_hiddens` |

## 3. Metrics — per the paper (Section 2, equations 1–5, following Perolat et al. 2017)

Notation: $N$ = num agents, $T$ = episode length, $R^i$ = agent $i$'s cumulative episode reward, $r_t^i$ = agent $i$'s reward at time $t$, $o_t^i$ = agent $i$'s observation at time $t$.

| Metric | Paper definition (equation) | Meaning | Code appearance |
|---|---|---|---|
| **Efficiency** ($U$) | $U = \mathbb{E}\left[\frac{1}{T}\sum_{i=1}^N R^i\right]$ | Mean total reward across all agents — raw collective productivity/throughput | `metric: "Efficiency"` in every config; `Efficiency` / `Efficiency2` / `Efficiency*...` in `results/` dir names |
| **Equality** ($E$) | $E = 1 - \dfrac{\sum_{i=1}^N\sum_{j=1}^N \lvert R^i-R^j\rvert}{2N\sum_{i=1}^N R^i}$ | Gini-coefficient-based fairness of reward distribution across agents | `Efficiency*Equality` composite objective (paper Figs. 6/7/10); not seen as a standalone `results/` dir in this repo's current results, but supported by the metric machinery |
| **Sustainability** ($S$) | $S = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^N t^i\right]$, $t^i = \mathbb{E}[t \mid r_t^i>0]$ | Average time step at which agents receive positive reward — later = more sustainable/patient harvesting, avoids early over-exploitation | `metric: "Efficiency*Sustainability"` in `prm_single.yaml`; `results/prm-single-dqn-Efficiency*Sustainability-fast-1_agents` |
| **Peace** ($P$) | $P = \dfrac{\mathbb{E}\left[NT - \sum_{i=1}^N\sum_{t=1}^T I(o_t^i)\right]}{NT}$, $I(o)=1$ iff $o$ is the "time-out" observation (i.e., agent was tagged and removed) | Proportion of agent-timesteps NOT spent in tagged/removed timeout state — measures absence of inter-agent conflict/tagging | `metric: "Efficiency*Peace"` / `"Efficiency*Peace2"` in `prm.yaml`, `prm_ppo.yaml`; `results/prm-dqn-Efficiency*Peace-fast-4_agents`, `results/prm-dqn-episodial-Efficiency*Peace2-fast-4_agents` |
| **penalty** (config flag, not a paper equation) | Not separately equation-defined; used in the "Baseline-Penalty" curve of Fig. 4 as a ground-truth-reward baseline variant. Text does not spell out the penalty's exact reward-shaping formula (not found in the read pages) | Presumably an environment-level tagging/conflict penalty subtracted from ground-truth reward for the no-reward-model baseline | `EXPERIMENT_PARAMETERS.penalty: True/False`; passed straight into `parallel_env(..., penalty=...)` in `runners.py` line ~113; appends `-penalty` to experiment name |
| **spawn_speed** (config field, not a paper term) | Not named/discussed as an experimental variable anywhere in the read paper text — likely an environment/apple-regrowth-rate hyperparameter not surfaced in the manuscript (possibly related to the "density-dependent" regrowth mentioned in Section 2, but the paper gives no `fast`/`slow` terminology) | `RL_PARAMETERS.spawn_speed: "fast"|"slow"` — controls apple respawn rate in `src/env/`; appears in every `results/` dir name (`...-fast-...`, `...-slow-...`) | 

**Not found in paper**: no equation or explicit discussion of the `penalty` reward-shaping term's formula, and no mention of "spawn_speed" as a named experimental axis — both appear to be engineering-level knobs in the codebase's env that aren't spelled out in the manuscript text (may be in an omitted appendix subsection, or simply undocumented in the paper). Flagging rather than guessing the formula.

## 4. Reward-model loss — consistency check vs. `ComparisonRewardPredictor`

The paper's loss (Section 4.3, Eqs. 6–8) is a **Bradley-Terry preference model**, explicitly cited as following Christiano et al. (2017):

- Latent per-sequence score: $s_x = \sum_{(o,a)\in x}\hat r_\theta(o,a)$ (sum of predicted per-step rewards over a trajectory/episode)
- Preference probability: $P(x\succ y) = \dfrac{\exp(s_x)}{\exp(s_x)+\exp(s_y)}$ (Eq. 7, softmax/Bradley-Terry over summed latent rewards)
- Training loss: $\mathcal{L}(\theta) = \sum_{(x,y,\mu,\delta'_{xy})\in\mathcal{B}} \delta'_{xy}\cdot \text{BCE}(\mu, P(x\succ y))$ (Eq. 8) — weighted binary cross-entropy over sampled pairwise comparisons, where $\delta'_{xy}$ is a softmax-normalized weight scaling each comparison's loss contribution by the magnitude of the social-metric performance gap between the two sequences.

This is **fully consistent** with a "reward predictor trained on pairwise preference comparisons over trajectory segments" — i.e., matches the general shape expected of `ComparisonRewardPredictor` in both `src/reward_predictor/prm/reward_model.py` and `src/reward_predictor/crm/reward_model.py` (class named `ComparisonRewardPredictor` in both — name itself signals pairwise-comparison training). Reward-predictor architecture (Fig. 8, Appendix A) — CNN pixel feature extractor + flatten, one-hot action encoding, concatenate, MLP, linear output — matches `fcnet_hiddens` / `network: "OneHotCnnNetwork"` config fields in `prm*.yaml`. Appendix A Table 2 gives concrete reward-model hyperparameters (lr 1e-4, batch pairs 64, train steps/update 50, update frequency 1000 env steps, warmup 50 episodes, buffer capacity 5000 episodes, FC layers `[128+|A|, 128, 1]`) — these are architecture-level defaults, not necessarily identical to any single repo config's `RP_PARAMETERS` (which vary run-to-run: e.g. `prm.yaml` uses `lr: 0.0005`, `fcnet_hiddens: [64,32,16]`).

Not deeply verified: whether the code's `ComparisonRewardPredictor` implements the *exact* softmax-weighted-by-δ' BCE loss of Eq. 8, or a simpler unweighted BCE — a commented-out line in `prm/reward_model.py:202` (`# loss = (self.loss(logits, labels)*torch.nn.functional.softmax(...)` ) suggests the weighted version was implemented/tried and possibly disabled at some point; left to the reward_predictor-focused review to confirm.

## 5. Naming convention — paper vs. `results/` folder pattern

The paper's manuscript text does not discuss file-naming/ablation-naming conventions (that's purely an engineering artifact of this repo, not paper content). From `src/experiment_runner/runners.py`:

```
self.experiment = f"{EXPERIMENT_PARAMETERS.experiment}-{RL_PARAMETERS.learner}"
self.experiment += "-independent" if EXPERIMENT_PARAMETERS.independent else ""
self.experiment += "-penalty"     if EXPERIMENT_PARAMETERS.penalty     else ""
self.experiment += "-episodial"   if RP_PARAMETERS.episodial (PRM only) else ""

dir_name = run_name_template.format(
    experiment=self.experiment,
    metric=EXPERIMENT_PARAMETERS.metric,
    spawn_speed=RL_PARAMETERS.spawn_speed,
    num_agents=ENV_PARAMETERS.num_agent,
)
# => "{experiment}-{learner}[-independent][-penalty][-episodial]-{metric}-{spawn_speed}-{num_agents}_agents"
```

This exactly reproduces the observed `results/` / `old_results/` directory names, e.g.:
- `nrp-dqn-Efficiency-fast-3_agents` — `experiment=nrp`, `learner=dqn`, no independent/penalty flags
- `nrp-dqn-independent-Efficiency-fast-3_agents` — `independent: True`
- `nrp-dqn-penalty-Efficiency-fast-4_agents` — `penalty: True`
- `prm-dqn-episodial-Efficiency*Peace-fast-4_agents` — `RP_PARAMETERS.episodial: True`, composite metric
- `prm-single-dqn-Efficiency*Sustainability-fast-1_agents` — `experiment: prm-single` (single-agent variant), composite metric
- `old_results/nrp_independent_penalty-Efficiency-fast` — older naming used `_` instead of `-` for the same flags, and predates `{num_agents}_agents` suffix

## 6. Config reference table

All seven files live in `src/configs/`. `config.py` is a trivial recursive dict→attribute wrapper (`Config` class) — not re-analyzed here per task scope.

| Config file | Runner (`experiment`) | Learner | independent | single-agent? | metric | penalty | spawn_speed | Other notable params | Matching `results/`/`old_results/` pattern |
|---|---|---|---|---|---|---|---|---|---|
| `crm.yaml` | `CRMRunner` (`experiment: "crm"`) | dqn | False | No (`num_agent: 7`) | `Efficiency` | False | slow | `RP_PARAMETERS` present (predictor_epochs 4, lr 1e-4); no `network`/`fcnet_hiddens` field (unlike prm*.yaml) — architecture likely hardcoded in `crm/nn.py` | **None** — zero `crm*` dirs exist under `results/` or `old_results/`, consistent with owner's "could not make it work" |
| `nrp.yaml` | `NRPRunner` (`experiment: "nrp"`) | dqn | False | No (`num_agent: 4`) | `Efficiency` | **True** | fast | `MEDIUM2_HARVEST_MAP`, MlpPolicy | `results/nrp-dqn-penalty-Efficiency-fast-4_agents` |
| `nrp_ppo.yaml` | `NRPRunner` | ppo | **True** | No (`num_agent: 12`) | `Efficiency` | False | fast | `HARVEST_MAP_LARGER`, CnnPolicy, IPPO-style hyperparams | `results/nrp-dqn-independent-Efficiency-fast-3_agents` is the DQN analog (note: no `nrp-ppo-*` dir currently present in `results/`, only DQN variants) |
| `prm.yaml` | `PRMRunner` (`experiment: "prm"`) | dqn | False | No (`num_agent: 4`) | `Efficiency*Peace` (also has commented-out `Efficiency2`, `Efficiency*Sustainability`, `Efficiency*Peace2` alternatives) | False | fast | `RP_PARAMETERS.episodial: False`, `network: "OneHotCnnNetwork"` | `results/prm-dqn-Efficiency*Peace-fast-4_agents` (or `-episodial-` variant if `episodial: True` toggled) |
| `prm_ppo.yaml` | `PRMRunner` | ppo | False | No (`num_agent: 12`) | `Efficiency*Peace` | False | slow | `HARVEST_MAP_LARGER`, larger `RP_PARAMETERS.fcnet_hiddens: [256,128,16]` | No corresponding `prm-ppo-*` dir currently present in `results/` (only `prm-dqn-*`) — this config may not have produced archived results yet, or ppo results live elsewhere |
| `prm_single.yaml` | `PRMRunner`/single-agent path (`experiment: "prm-single"`) | dqn | False | **Yes** (`num_agent: 1`, `SMALL_HARVEST_MAP`) | `Efficiency*Sustainability` (commented-out alt: `Efficiency`) | False | fast | `RP_PARAMETERS.predictor_epochs: 8` (higher than multi-agent configs) | `results/prm-single-dqn-Efficiency*Sustainability-fast-1_agents` |
| `prm_single_ppo.yaml` | `PRMRunner`/single-agent path (`experiment: "prm-single"`) | ppo | False | **Yes** (`num_agent: 1`, `SMALL_HARVEST_MAP`) | `Efficiency` | False | fast | Same `RP_PARAMETERS` as `prm_single.yaml` | `results/prm-single-ppo-Efficiency-fast-1_agents` |

Additional `results/`/`old_results/` dirs not directly matched to a current config file (likely produced by earlier/modified config versions no longer checked into `src/configs/`, e.g. metric variants `Efficiency2`, `Efficiency*Peace2` that appear only as commented-out options in `prm.yaml`): `prm-dqn-Efficiency*Peace2-fast-4_agents`, `prm-dqn-Efficiency2-fast-4_agents`, `prm-dqn-episodial-*` variants, and all `old_results/*` entries (which use a pre-refactor underscore-based naming scheme, e.g. `nrp_independent_penalty-Efficiency-fast`, predating the current `-` separator convention and `{num_agents}_agents` suffix).

---

*Compiled by reading the full 19-page paper PDF and all 7 YAML configs plus the relevant slice of `src/experiment_runner/runners.py` (experiment-name construction, lines ~71–120). Reward-predictor internals (`src/reward_predictor/`) were only skimmed for class/loss consistency, not deeply reviewed — that is covered by a separate pass per the task brief.*
