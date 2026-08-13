"""`run_info.json` -- provenance for a run directory.

config.yaml says what was configured; this says what actually ran, and where
from. It is written once, at startup, so a run directory found months later
still answers "which commit, which command, which machine, which seed".
"""

import json
import os
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

RUN_INFO_FILENAME = "run_info.json"

_GIT_TIMEOUT_SEC = 5


def _git(args: List[str], cwd: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_info() -> Dict[str, Any]:
    """Commit, branch and dirty flag of the checkout the code was loaded from.

    Never the process CWD: a run launched from elsewhere must still record the
    commit that produced the code it ran.
    """
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = _git(["rev-parse", "HEAD"], repo_dir)
    if commit is None:
        return {"git_commit": None}
    status = _git(["status", "--porcelain"], repo_dir)
    return {
        "git_commit": commit,
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir),
        "git_dirty": bool(status) if status is not None else None,
    }


def _torch_info() -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch": None}
    info: Dict[str, Any] = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if info["cuda_available"]:
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info


def _hydra_overrides() -> Optional[List[str]]:
    """The job's CLI overrides, when running under Hydra."""
    try:
        from hydra.core.hydra_config import HydraConfig
        from omegaconf import OmegaConf

        if not HydraConfig.initialized():
            return None
        overrides = OmegaConf.select(HydraConfig.get(), "overrides.task", default=None)
    except Exception:
        return None
    return list(overrides) if overrides is not None else None


def collect_run_info(config: Any, run_dir: str) -> Dict[str, Any]:
    """Assemble the provenance record for `run_dir`."""
    info: Dict[str, Any] = {
        "run_name": os.path.basename(os.path.normpath(run_dir)),
        # The directory grouping every run of this configuration.
        "config_group": os.path.basename(os.path.dirname(os.path.normpath(run_dir))),
        "run_dir": os.path.abspath(run_dir),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]),
        "hydra_overrides": _hydra_overrides(),
        "algorithm": config.algorithm.name,
        "map_type": config.env.map_type,
        "num_agents": config.env.num_agents,
        "episodes": config.episodes,
        "seed": config.seed,
        "reward_model": (
            {"mode": config.reward_model.mode, "phi": config.reward_model.phi}
            if config.reward_model.enabled
            else None
        ),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    info.update(_git_info())
    info.update(_torch_info())
    return info


def write_run_info(config: Any, run_dir: str) -> str:
    """Write `run_info.json` into `run_dir` and return its path."""
    path = os.path.join(run_dir, RUN_INFO_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collect_run_info(config, run_dir), f, indent=2, sort_keys=True)
    return path
