# `src/env/` Knowledge Base

Scope: `src/env/` is the **live** environment package. It is imported as
`from env import parallel_env` in `src/experiment_runner/runners.py`
(`src/experiment_runner/runners.py:12`), which every training entrypoint
goes through. `src/env2/` is a near-duplicate package; its status is
analyzed in section 6.

All line numbers below refer to files under `src/env/` unless stated
otherwise.

## 1. Game mechanics

This is a variant of the DeepMind/SSD "Commons Harvest" social-dilemma
game (Perolat et al. / Jaques et al. lineage — same map format, action
set and beam mechanic as `deepmind/ai-safety-gridworlds`-style
`sequential_social_dilemma_games`), adapted to PettingZoo + gymnasium.

- **Grid world.** The map is a fixed ASCII layout (`maps.py`) parsed into
  a 2D char array (`map_env.py:80` `ascii_to_numpy`). Cell types: `@`
  wall, `A` apple, `P` spawn point (converted to blank at reset), `' '`
  empty, `F` transient fire-beam cell, `S` agent's-own-position marker
  (env2 also adds `C` for a "scope" marker, see §6).
- **Movement.** Each step every agent chooses one of 8 discrete actions
  (`agent.py:15-21`, `commons_agent.py:9-10`): `MOVE_LEFT/RIGHT/UP/DOWN`,
  `STAY`, `TURN_CLOCKWISE/COUNTERCLOCKWISE`, `FIRE`. Move vectors are
  egocentric — they get rotated into world-frame based on the agent's
  current orientation (`map_env.py:333-518`, `rotate_action`). Conflict
  resolution for multiple agents wanting the same cell is handled by
  `update_moves` (`map_env.py:333`): ties are broken by random shuffle +
  iterative re-resolution (an agent whose target cell frees up on a later
  pass gets to move); walls (`@`) always block entry
  (`agent.py:116-124`, `return_valid_pos`).
- **Apples (resource).** Apple cells are the harvestable resource. An
  agent standing on `'A'` "consumes" it on that step: `consume()` in
  `commons_agent.py:63-69` grants `+1` reward and clears the cell to
  `' '`. Apples regrow stochastically and *density-dependently*:
  `spawn_apples()` (`commons_env.py:146-175`) counts existing apples
  within `APPLE_RADIUS=2` (`commons_env.py:7`, circular window `j²+k²
  ≤ APPLE_RADIUS`) of each original apple point and looks up a spawn
  probability from a 4-bucket table indexed by that count
  (`min(num_apples, 3)`):
  ```
  SPAWN_PROB_SLOW = [0, 0.005, 0.02, 0.05]   # commons_env.py:11
  SPAWN_PROB_FAST = [0, 0.01,  0.05, 0.1 ]   # commons_env.py:12  ("paper" values)
  ```
  Zero neighboring apples ⇒ zero regrowth probability ⇒ that patch is
  permanently depleted once fully harvested (classic tragedy-of-the-commons
  collapse dynamic). Selected via the `spawn_speed: "slow"|"fast"` config
  key, consumed in `HarvestCommonsEnv.__init__` (`commons_env.py:32`).
  Apples cannot spawn on a cell an agent currently occupies
  (`commons_env.py:159`).
- **Tagging/zapping ("FIRE").** Action 7 fires a "penalty beam" instead of
  moving. `custom_action` (`commons_env.py:117-122`) calls
  `agent.fire_beam('F')` (no self-cost — `fire_beam` is a no-op that
  subtracts 0, `commons_agent.py:56-58`) then
  `update_map_fire`/`update_map_fire` in `map_env.py:541-627`. The beam
  fires forward from the agent for `ACTIONS['FIRE']` cells — which is set
  to `agent_view_range` at env construction (`commons_env.py:29`, i.e.
  firing range == observation range) — along 3 parallel lanes (center +
  left/right offset by one column, `map_env.py:582-587`). It is blocked by
  walls and by agents (agents absorb the beam and stop it,
  `map_env.py:599-607`). A hit agent's `hit('F')` sets
  `remaining_timeout = TIMEOUT_TIME` (`commons_agent.py:49-54`).
- **Timeout ("outcast").** While `remaining_timeout > 0` an agent: cannot
  move (`return_valid_pos` short-circuits to current pos,
  `commons_agent.py:81-84`), is rendered as a 3×3 wall block `"@"` in its
  own view plus a center `"S"` marker (`commons_agent.py:71-79`), and is
  physically removed from the map to position `(-99, -99)`
  (`OUTCAST_POSITION`, `commons_env.py:13,124-144`) once its timeout
  countdown reaches this state in `custom_map_update`. When the countdown
  hits 0 it is respawned at a random free spawn point with random
  orientation. `TIMEOUT_TIME` defaults to 25 steps but
  `HarvestCommonsEnv.setup_agents` overrides it per-agent to
  `int(ep_length * 0.025)` (`commons_env.py:94,100-101`).
- **Episode.** Fixed length `ep_length` (config `ENV_PARAMETERS.ep_length`,
  default step budget, e.g. 600). Individual agents never terminate
  (`get_done()` always `False`, `commons_agent.py:60-61`); episode end is
  enforced externally in the PettingZoo wrapper (§5), not inside
  `commons_env.py`/`map_env.py`.

## 2. Observation & action space

```
observation (per agent, from env/ raw)  = {"curr_obs": Box(0,255, shape=(2*R+1, 2*R+1, 3), uint8)}
action (per agent)                      = Discrete(8)   # 0..6 base moves/turns/stay, 7 = FIRE
```
where `R = agent_view_range` (config `ENV_PARAMETERS.agent_view_range`,
constructor default `HARVEST_DEFAULT_VIEW_SIZE = 10`,
`commons_agent.py:12`). `HarvestCommonsAgent` uses the same `R` for both
`lateral_view_range` and `frontal_view_range`
(`commons_env.py:100-101` passes `agent_view_range` to both), so the view
is a square, not asymmetric front/back.

Observation pipeline per step (`map_env.py:140-159`, `223-245`,
`284-306`, `647-667`):
```
world_map (chars, whole map)
  -> get_map_with_agents()          # stamp 'P'->agent-id digit char + beam cells
  -> agent.get_state()              # utility_funcs.return_view: crop (2R+1)x(2R+1)
                                     # window centered on agent, zero-padded at map edges
  -> single_agent_map_to_colors()   # char -> RGB via color_map (constants.py)
  -> rotate_view(orientation, .)    # np.rot90 so 'forward' is always 'up' in the obs
  -> {"curr_obs": rgb_arr}          # uint8 HxWx3
```
Color mapping (`constants.py`): `SAME_COLORMAP` (default, all other
agents rendered identically as color `[0,0,255]`, self marked `S` =
`[255,0,0]`) vs `DIFFERENT_COLORMAP` (each of the 9 supported agent ids
gets a distinct color) — selected by `MapEnv(..., same_color=True/False)`;
`HarvestCommonsEnv`/`parallel_env` never expose `same_color` as a kwarg,
so it stays at the `MapEnv` default `True` (SAME_COLORMAP) in env/ (env2
flips this default to `False`, see §6).

`state_space` (whole-map) is a separate, non-per-agent property:
```
state = np.transpose(map_to_colors(get_map_with_agents()).astype(uint8), (2,1,0))  # map_env.py:308-310
state_space = Box(0, 255, shape=state.shape, uint8)                                 # commons_env.py:74-81
```

**Frame stacking is external to this package.** `agent_view_range` and the
raw per-step `{"curr_obs": ...}` dict are all `env/` produces.
`num_frames` (config `ENV_PARAMETERS.num_frames`) is consumed entirely in
`src/experiment_runner/runners.py:119-120`:
```python
env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])  # unwrap dict -> raw Box
env = ss.frame_stack_v1(env, self.config.ENV_PARAMETERS.num_frames)                        # supersuit stacks frames
```
`src/env/` itself has no concept of frame history; it only ever emits a
single current-step RGB crop.

## 3. Reward structure

**Native per-step reward** (`commons_agent.py:63-69`, via
`agent.compute_reward()` in `agent.py:89-94`): `+1` if the agent's new
cell was an apple this step, else `0`. No shaping, no cost for
moving/turning/firing (`fire_beam`/`hit` both do `reward_this_turn -= 0`,
i.e. no-ops — this looks like vestigial code for a once-nonzero fire
penalty). This raw reward is also copied into `infos[agent_id]['true_reward']`
(`map_env.py:150-154`) and `infos[agent_id]['r']`
(`commons_env.py:83-90`).

**`penalty=` config param** — consumed in `pettingzoo_env.py:48-49`
(`ssd_parallel_env.step`), *outside* `commons_env.py`/`map_env.py`
entirely:
```python
if self.penalty:  # simple punishment for fire action
    rews = {agent_id: -1 if infos[agent_id]['fire'] else r for agent_id, r in rews.items()}
```
`infos[...]['fire']` is set in `commons_env.py:87`
(`action[agent_id] == 7`). So `penalty=True` overrides that agent's whole
step reward to `-1` on any step it chose FIRE (regardless of whether the
beam hit anyone), replacing (not adding to) the apple reward for that
step.

**`metric=` config param** — also consumed entirely in
`pettingzoo_env.py` (`ssd_parallel_env.step`, lines 51-77), layered *after*
the `penalty` adjustment:
```python
if self.ssd_env.metric != 'Efficiency':
    rews = {k: 0 for k in rews}          # zero out per-step reward every step
...
# only on the terminal step (num_cycles >= ep_length):
self.ssd_env.compute_social_metrics()     # commons_env.py:199-247
infos[k]['metrics'] = self.ssd_env.get_social_metrics()
if metric == 'Efficiency2':                     rews[k] = metrics['efficiency']
elif metric == 'Efficiency*Peace':              rews[k] = metrics['efficiency'] * metrics['peace']
elif metric == 'Efficiency*Peace2':             rews[k] = metrics['efficiency'] * metrics['peace']**4
elif metric == 'Efficiency*Peace*Equality':     rews[k] = metrics['efficiency'] * metrics['peace'] * metrics['equality']
elif metric == 'Efficiency*Sustainability':     rews[k] = metrics['efficiency'] * (metrics['sustainability']*2)
```
So `metric != "Efficiency"` turns the task from a dense per-step apple
reward into a **sparse, episode-terminal reward** equal to a chosen
combination of the social metrics computed in
`compute_social_metrics()` (`commons_env.py:199-247`):
- `efficiency` = total reward summed over all agents / num_agents
- `equality` = `1 - (sum of pairwise |Δreward|) / (2*N*total_reward)` (Gini-style)
- `sustainability` = mean timestep-of-positive-reward, normalized by `N*ep_length`
- `peace` = `1 - (total timeout-steps across agents) / (N*ep_length)`
- `fire_attempts` / `fire_sucsses`: raw counters (`fire_counter`,
  `fire_sucsses` in `commons_env.py`, note the counters use the
  misspelling `fire_sucsses` throughout the codebase, not `success`).

If `metric == "Efficiency"` (the default), rewards stay as native
per-step apple rewards (`+penalty` adjustment) and metrics are still
computed and attached at episode end for logging, but never substituted
into `rews`.

Note `metric` values `"Efficiency2"` and `"Efficiency*Peace2"` exist only
in `src/env/pettingzoo_env.py` — env2's copy of this file lacks both
branches (see §6).

## 4. Map system

`maps.py` defines each map as a `list[str]`, one string per row, fixed
width per map (rows padded with spaces to equal length in the source).
Chars: `@` wall/border, `A` apple point, `P` agent spawn point, `' '`
empty floor. Maps defined in env/: `SMALL_HARVEST_MAP`,
`MEDIUM_HARVEST_MAP`, `MEDIUM2_HARVEST_MAP`, `HARVEST_MAP`,
`HARVEST_MAP_LARGER`, `ARIGINALHARVEST_MAP_LARGER` [sic, typo in source].

`commons_env.py:18-20` builds a `MAP` dict for 3 of these
(`{"small", "medium", "medium2"}`), but this dict is otherwise unused by
the live path — `HarvestCommonsEnv.__init__` never references it.

Map selection actually happens through `ascii_map=config.ENV_PARAMETERS.map`
threaded from `runners.py:116` → `pettingzoo_env.py:97`
(`HarvestCommonsEnv(ep_length=ep_length, **ssd_args)`) →
`commons_env.py:35`:
```python
super().__init__(eval(ascii_map), num_agents, render, color_map=color_map)
```
i.e. `ascii_map` is a **string naming the module-level constant**, e.g.
`'HARVEST_MAP_LARGER'` (see `src/configs/nrp_ppo.yaml`,
`map: 'HARVEST_MAP_LARGER'`), and it is resolved via Python `eval()`
against `commons_env.py`'s import namespace (which imports all 6 map
names from `maps.py` at `commons_env.py:6`). Passing any other string
(including a real ASCII map, contra the `ascii_map: list of strings`
docstring in `map_env.py:18`) will raise `NameError`/`SyntaxError` from
`eval`. This `eval()`-based map lookup is a code smell worth flagging for
the refactor (arbitrary code execution surface, and it silently diverges
from `MapEnv`'s own docstring that says `ascii_map` should already be the
list of strings).

## 5. PettingZoo API surface

`pettingzoo_env.py` builds a `ParallelEnv` (not AEC) around
`HarvestCommonsEnv`:
```
parallel_env(**kwargs) -> _parallel_env(**kwargs)
_parallel_env(ssd_parallel_env, EzPickle)
    .__init__: builds HarvestCommonsEnv(ep_length, **ssd_args), delegates to ssd_parallel_env
ssd_parallel_env(ParallelEnv)
    possible_agents / observation_spaces / action_spaces / state_space   set in __init__ from HarvestCommonsEnv
    reset(seed, **kwargs)   -> ssd_env.reset()
    step(actions)           -> ssd_env.step(); applies penalty/metric logic (§3); returns
                               (obs, rews, dones, dones, infos)  [5-tuple: dones duplicated
                               for the (terminated, truncated) gymnasium convention]
    get_full_state()        -> self.ssd_env.state          (map_env.py `state` property)
    get_images()             -> self.ssd_env.full_map_to_colors()   -- BROKEN, see below
    get_social_metrics()     -> self.ssd_env.get_social_metrics()
```
- `get_full_state()` (`pettingzoo_env.py:80-81`) **does exist** and works:
  it returns `HarvestCommonsEnv.state` (inherited from `MapEnv.state`,
  `map_env.py:308-310`), a `(3, W, H)` uint8 RGB array of the full map
  including agents.
- `get_images()` (`pettingzoo_env.py:83-84`) calls
  `self.ssd_env.full_map_to_colors()`, a method that **does not exist**
  anywhere in `map_env.py`/`commons_env.py` (the actual method is named
  `map_to_colors`). This method appears unused by `runners.py` (grep found
  no callers), so it's dead/broken code, not on the critical path.
- `state_space` is set once in `__init__`
  (`pettingzoo_env.py:24`, `self.state_space = self.ssd_env.state_space`,
  a `gymnasium.spaces.Box` **instance**, not a method) and inherited from
  `HarvestCommonsEnv.state_space` (`commons_env.py:74-81`, itself a
  `@property`).

**`runners.py` "crm" plumbing** (`src/experiment_runner/runners.py:132-137`,
only when `config.EXPERIMENT_PARAMETERS.experiment == "crm"`):
```python
env.get_full_state = env.env_method("get_full_state")
env.state_space     = env.env_method("state_space")
```
`get_full_state` is plumbed correctly — it's a real bound method on
`ssd_parallel_env`, so `VecEnv.env_method("get_full_state")` can call it
per sub-env. `state_space`, however, is a **plain attribute holding a
`Box` object**, not a callable method, on both `ssd_parallel_env`
(instance attr, `pettingzoo_env.py:24`) and `HarvestCommonsEnv` (a
`@property`). SB3's `VecEnv.env_method(name, *args)` does
`getattr(sub_env, name)(*args)` — i.e. it expects `name` to resolve to a
*callable*; calling the resulting `Box` object like a function
(`Box(...)()`) raises `TypeError: 'Box' object is not callable`. This
looks like a real bug in the `experiment == "crm"` path (untested by the
task's confirmed-live surface, since `crm` is only exercised via
`CRMRunner`, which does then use `vec_env.state_space` as an attribute in
`runners.py:391,421` — meaning it actually relies on the *first* line,
`env.get_full_state = env.env_method("get_full_state")`, executing then
`env.env_method("state_space")` line 137 executing and raising before
`vec_env.state_space` in `CRMRunner.set_up_rp`/`setup_agent` is ever
reached). This should be verified by actually running the `crm` config
before relying on it in a refactor — static reading strongly suggests
it's broken as written.

## 6. env/ vs env2/ diff

| file | status | nature of diff |
|---|---|---|
| `__init__.py` | identical | — |
| `utils/train_utils.py` | identical | both copies are dead/broken: import `DEFAULT_COLORMAP`/`DEFAULT_COLOURS` from `commons_env.py`/`map_env.py`, symbols that don't exist in either package (current names are `DIFFERENT_COLORMAP`/`SAME_COLORMAP`); also imports `DDQNAgent`/`tensorflow`, a training loop no runner touches |
| `utils/utility_funcs.py` | modified | env2's `return_view` re-slices the cropped view (`view[m:, m//2:-m//2]`) after the normal centered crop — shrinks/shifts the observation window; looks experimental/half-finished (no matching change to `observation_space` shape) |
| `agent.py` | modified | env2 doubles the crop passed to `return_view` (`row_size*2, col_size*2`) in `get_state()`; env2 also **deletes** the legacy, unused `HarvestAgent`/`HARVEST_ACTIONS`/`HARVEST_VIEW_SIZE` class block (~65 lines) that env/ still carries as dead code (superseded by `HarvestCommonsAgent` in `commons_agent.py`) |
| `commons_agent.py` | modified | env2 hardcodes `frontal_view_range=HARVEST_DEFAULT_VIEW_SIZE*2` as the default and **drops the `timeout_time` constructor param** (always uses class-level `TIMEOUT_TIME=25`), whereas env/ added a `timeout_time` kwarg so `commons_env.py` can pass a computed `int(ep_length*0.025)` — i.e. env/ has a later feature (per-episode-length-scaled timeout) that env2 never received |
| `commons_env.py` | modified | env2 lost the `timeout_time=` wiring (matches above), lost `MEDIUM2_HARVEST_MAP` from its `MAP` dict/import, and moved one line (`ACTIONS['FIRE'] = ...`) — cosmetic/behavior-preserving otherwise |
| `constants.py` | modified | env2 adds a `'C'` (gray "scope" marker) color entry to both colormaps, in support of a "scope" visualization feature (see `map_env.py` below) not present in env/ |
| `map_env.py` | modified | env2 adds an entire `scope=` feature: a constructor flag, `adjust_single_agent_obs()`, and per-agent/global "gray square in front of the agent" rendering (`map_env.py` diff, +~35 lines); also changes `same_color` default from `True`→`False`; also widens the FIRE beam from 3 lanes to 5 lanes (`start_pos ± right_shift`, `± right_shift*2`) — a real gameplay-mechanics change (wider zap beam) not present in env/ |
| `maps.py` | modified | env2 **lacks `MEDIUM2_HARVEST_MAP`** entirely (env/ added it after the env2 fork point) |
| `pettingzoo_env.py` | modified | env2's `step()` is missing the `'Efficiency2'` and `'Efficiency*Peace2'` metric branches that env/ has — env/ is strictly newer/more-featured here |

**Verdict: env2 is a frozen experimental fork, not the newer version, and
not functionally identical.** Evidence:
- `git log --follow -- src/env2` shows **exactly one commit ever touched
  it**: `8652c81 "stabe prm and baseline"` (2025-03-23). It was added
  already-diverged and never modified again.
- `src/env/pettingzoo_env.py`/`commons_agent.py`/`maps.py` continued to
  receive changes in later commits/mtimes (up to `843b881`, mtimes as
  late as 2025-05-09) — i.e. env/ is the actively maintained line, and it
  both gained features env2 never got (`Efficiency2`/`Efficiency*Peace2`
  metrics, `MEDIUM2_HARVEST_MAP`, per-episode `timeout_time` scaling,
  removal of dead `HarvestAgent` class) and never merged env2's
  experimental additions (the `scope` gray-square overlay, the 5-lane
  wide FIRE beam, the `return_view` re-crop, doubled `frontal_view_range`
  default).
- The **only reference to `env2` anywhere in the repo** is the exploratory
  notebook `src/stam.ipynb:9` (`from env2 import parallel_env`); no
  runner, script, or config touches it.
- env2's changes read as a single exploratory session (rendering a visual
  "attention scope" indicator + a wider zap beam + asymmetric view crop
  experiments) that was abandoned in favor of continuing env/'s line of
  development, rather than env2 being a stepping-stone that was later
  merged back — none of its distinguishing features (`scope`, 5-lane
  beam, `return_view` re-crop) appear in env/ at any point after the
  fork.

**Refactor recommendation:** env2 is dead code and safe to delete, *after*
either (a) confirming with whoever owns `src/stam.ipynb` that the notebook
is throwaway/scratch (its own docstring context already calls it a
"scratch notebook"), or (b) porting the notebook's one import to `env/`
first if it's still in active use. No other consumer was found.
