"""Where a run's files live, and what identifies it.

One run == one directory. Hydra's own output (`.hydra/`, the job log) and the
training artifacts (config.yaml, metrics.jsonl, tensorboard, videos,
extended_info, checkpoints) land in the *same* place, because
`hydra.run.dir`/`hydra.sweep.subdir` are built from the helpers here and the
Trainer logs into `hydra.runtime.output_dir`.

Runs are grouped by configuration, not by launch:

    logs/<config-dir>/<timestamp>[-seed=N]/

`<config-dir>` encodes what makes the configuration what it is -- algorithm,
map, agent count, reward-model mode -- plus any other CLI override that is not
already spelled out there. Every repeat (seed, rerun) of one configuration
therefore sits side by side under a single parent, which is what analysis wants:

    uv run commons-game plot runs logs/<config-dir>/*

A sweep over a value that appears in `<config-dir>` (`-m algorithm=ippo,mappo`)
splits into separate configuration directories; a sweep over seeds fills one
directory with sibling runs.
"""

import os
import time
from typing import Any, Iterable, List, Optional, Sequence, Tuple

# One timestamp for the whole process, so every job of a `-m` sweep shares it
# (Hydra would otherwise re-resolve `${now:...}` per job and drift by seconds).
_LAUNCH_TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

# Overrides the configuration directory name already spells out. Note these are
# exact keys, not prefixes: `algorithm=ippo` is encoded, but
# `algorithm.learning_rate=1e-3` is not, and a sweep over it must still land in
# distinct directories.
_ENCODED_OVERRIDES = frozenset(
    {
        "algorithm",
        "env",
        "reward_model",
        "seed",
        "env.map_type",
        "env.num_agents",
        "reward_model.enabled",
        "reward_model.mode",
        "reward_model.phi",
    }
)

# Groups that change how a run is recorded, not what it runs. A configuration
# is not a different configuration for having had its videos turned off.
_IGNORED_OVERRIDE_PREFIXES: Tuple[str, ...] = ("logging", "hydra")

_MAX_EXTRA_LEN = 40
_MAX_EXTRAS_LEN = 80


def launch_timestamp() -> str:
    """Process-wide launch time, shared by every job of a multirun sweep."""
    return _LAUNCH_TIMESTAMP


def _sanitize(value: str) -> str:
    """Make a config value safe to use as one path segment."""
    cleaned = str(value).strip().replace(os.sep, "_").replace("/", "_")
    cleaned = "_".join(cleaned.split())
    return cleaned[:_MAX_EXTRA_LEN]


def config_dir_name(
    *,
    algorithm: str,
    map_type: str,
    num_agents: int,
    reward_model_enabled: bool,
    reward_model_mode: str,
    reward_model_phi: Optional[str] = None,
    extras: Iterable[str] = (),
) -> str:
    """Name of the directory that groups every run of one configuration."""
    parts = [
        _sanitize(algorithm),
        f"map={_sanitize(map_type)}",
        f"agents={num_agents}",
    ]
    if reward_model_enabled:
        parts.append(f"rm={_sanitize(reward_model_mode)}")
        if reward_model_phi:
            parts.append(f"phi={_sanitize(reward_model_phi)}")
    else:
        parts.append("rm=off")
    parts.extend(extras)
    return "-".join(parts)


def run_leaf_name(seed: Optional[int], timestamp: Optional[str] = None) -> str:
    """Name of a single run inside its configuration directory.

    A null seed is drawn at training time, after this name is fixed, so it is
    left out here; the drawn value is recorded in config.yaml and run_info.json.
    """
    name = timestamp or _LAUNCH_TIMESTAMP
    return f"{name}-seed={seed}" if seed is not None else name


def extras_from_overrides(overrides: Sequence[str]) -> List[str]:
    """CLI overrides that the configuration name does not already capture.

    Without these, `-m algorithm.learning_rate=1e-3,1e-4` would point both jobs
    at one directory. With them the two runs are told apart, and the name says
    how they differ.
    """
    extras: List[str] = []
    budget = _MAX_EXTRAS_LEN
    for override in overrides:
        item = str(override).lstrip("+~")
        key, sep, value = item.partition("=")
        if not sep:
            continue
        key = key.strip()
        if key in _ENCODED_OVERRIDES:
            continue
        if any(key == p or key.startswith(p + ".") for p in _IGNORED_OVERRIDE_PREFIXES):
            continue
        extra = f"{_sanitize(key)}={_sanitize(value)}"
        if len(extra) > budget:
            break
        budget -= len(extra) + 1
        extras.append(extra)
    return extras


def run_path(config: Any, extras: Iterable[str] = ()) -> str:
    """Path of a run relative to `logging.log_dir`.

    Accepts anything with the config's attribute shape -- the composed
    `DictConfig` (from the Hydra resolver) or a `TrainerConfig` (from a
    programmatic `Trainer(...)` call) -- so both agree on the layout.

    `logging.run_name`, when set, replaces the derived configuration directory:
    repeats still nest inside it.
    """
    reward_model = config.reward_model
    group = config.logging.run_name or config_dir_name(
        algorithm=config.algorithm.name,
        map_type=config.env.map_type,
        num_agents=config.env.num_agents,
        reward_model_enabled=reward_model.enabled,
        reward_model_mode=reward_model.mode,
        reward_model_phi=getattr(reward_model, "phi", None),
        extras=extras,
    )
    return os.path.join(group, run_leaf_name(config.seed))


def _resolve_run_path(*_args: Any, _root_: Any) -> str:
    """`${commons.run_path:}` -- used by `hydra.run.dir` and `hydra.sweep.subdir`.

    Reads the composed config directly rather than taking a dozen interpolated
    arguments, and picks up the job's own overrides so sweep jobs stay distinct.
    """
    from omegaconf import OmegaConf

    overrides = OmegaConf.select(_root_, "hydra.overrides.task", default=None) or []
    return run_path(_root_, extras=extras_from_overrides(list(overrides)))


def register_resolvers() -> None:
    """Register the OmegaConf resolvers the run-dir config refers to.

    Safe to call more than once (`replace=True`).
    """
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver(
        "commons.run_path", _resolve_run_path, use_cache=False, replace=True
    )
    OmegaConf.register_new_resolver(
        "commons.launch_time", lambda: _LAUNCH_TIMESTAMP, use_cache=True, replace=True
    )
