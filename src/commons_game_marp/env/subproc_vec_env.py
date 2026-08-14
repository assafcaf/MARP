"""Environment copies stepped in parallel worker processes.

Why this exists
---------------
`VecCommonsEnv` steps its copies in a plain Python loop, so `num_envs` bought
sample decorrelation and nothing else: measured throughput was flat at ~1400
agent-steps/s from `num_envs=1` to `num_envs=8`, on one core of twenty. This
runs the same copies in worker processes so the cores get used.

The design follows SuperSuit's `ProcConcatVec` -- workers own a contiguous slice
of the environments, commands and results move over a pipe, and the flat row
layout is the same env-major/agent-minor one `vec_env` already documents:

    row = env_idx * num_agents + agent_idx

Because workers own contiguous env ranges and results are reassembled in worker
order, that layout is preserved exactly.

Deliberately *not* copied from SuperSuit: auto-reset. Episodes here are
fixed-length and lockstep, and both the social metrics and the preference
buffer's episode records depend on exact episode boundaries.

What crosses the pipe
---------------------
Only what the main process actually consumes. Two changes made that small
enough to be worth doing: the full-map `state` render is no longer on the info
dict (it was 1938 bytes per agent per step, ~15 MB per iteration), and the
per-step agent metrics are computed env-side and returned as two scalars rather
than read off live agent objects.

Interface parity
----------------
Identical to `VecCommonsEnv`, so the trainer picks one and never branches again:
`reset`, `step`, `compute_social_metrics`, `render_frame`, `close`, and the
`num_envs` / `num_agents` / `agent_ids` / `num_rows` / `observation_space` /
`action_space` attributes.
"""

import multiprocessing as mp
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .env_spec import EnvSpec

# Commands understood by `_worker`.
_RESET = "reset"
_STEP = "step"
_SOCIAL_METRICS = "social_metrics"
_RENDER = "render"
_SPACES = "spaces"
_CLOSE = "close"


class _WorkerError:
    """A failure travelling back from a worker.

    A dedicated type rather than a `("__error__", ...)` tuple: results carry
    numpy arrays, and comparing `result[0] == "__error__"` against an array
    raises "truth value is ambiguous" instead of being False.
    """

    __slots__ = ("traceback",)

    def __init__(self, traceback: str) -> None:
        self.traceback = traceback


def _worker(remote, spec: EnvSpec, num_envs: int) -> None:
    """Own `num_envs` environments and serve commands until told to close.

    Exceptions are sent back rather than raised into the void: a worker that
    dies silently leaves the parent blocked on a `recv` forever.
    """
    envs = [spec.build() for _ in range(num_envs)]
    agent_ids = list(envs[0].agents.keys())
    try:
        while True:
            command, payload = remote.recv()
            if command == _STEP:
                actions = payload.reshape(num_envs, len(agent_ids))
                obs, rewards, dones, infos = [], [], [], []
                for env_idx, env in enumerate(envs):
                    action_dict = {
                        agent_id: int(actions[env_idx, i])
                        for i, agent_id in enumerate(agent_ids)
                    }
                    o, r, d, i = env.step(action_dict)
                    all_done = bool(d.get("__all__", False))
                    for agent_id in agent_ids:
                        obs.append(o[agent_id]["curr_obs"])
                        rewards.append(float(r[agent_id]))
                        dones.append(bool(d.get(agent_id, False)) or all_done)
                        infos.append(i.get(agent_id, {}))
                remote.send((np.stack(obs, axis=0), rewards, dones, infos))
            elif command == _RESET:
                seeds = payload
                obs, infos = [], []
                for env_idx, env in enumerate(envs):
                    o, i = env.reset(seed=None if seeds is None else seeds[env_idx])
                    for agent_id in agent_ids:
                        obs.append(o[agent_id]["curr_obs"])
                        infos.append(i.get(agent_id, {}))
                remote.send((np.stack(obs, axis=0), infos))
            elif command == _SOCIAL_METRICS:
                metrics = []
                for env in envs:
                    env.compute_social_metrics()
                    metrics.append(dict(env.get_social_metrics()))
                remote.send(metrics)
            elif command == _RENDER:
                remote.send(envs[payload].render())
            elif command == _SPACES:
                remote.send((envs[0].observation_space, envs[0].action_space, agent_ids))
            elif command == _CLOSE:
                remote.send(None)
                break
            else:
                raise RuntimeError(f"unknown command {command!r}")
    except Exception as error:  # noqa: BLE001 - forwarded to the parent
        import traceback

        remote.send(_WorkerError(f"{error}\n{traceback.format_exc()}"))
    finally:
        remote.close()


def split_envs(num_envs: int, num_workers: int) -> List[int]:
    """How many environments each worker owns.

    Remainders are spread one per worker rather than piled onto the last one:
    the vector env steps in lockstep, so every step costs whatever the *slowest*
    worker costs, and an unbalanced split makes that worker the ceiling.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    num_workers = min(num_workers, num_envs)
    base, extra = divmod(num_envs, num_workers)
    return [base + (1 if i < extra else 0) for i in range(num_workers)]


class SubprocVecCommonsEnv:
    """`num_envs` environments spread across `num_workers` processes."""

    def __init__(self, spec: EnvSpec, num_envs: int, num_workers: int) -> None:
        self.num_envs = int(num_envs)
        self._counts = split_envs(self.num_envs, num_workers)
        self.num_workers = len(self._counts)
        self._closed = False

        # `spawn` because the parent has usually initialised CUDA for the
        # policies by now, and forking a live CUDA context is undefined.
        context = mp.get_context("spawn")
        self._remotes = []
        self._processes = []
        for count in self._counts:
            parent_end, child_end = context.Pipe()
            process = context.Process(
                target=_worker, args=(child_end, spec, count), daemon=True
            )
            process.start()
            child_end.close()
            self._remotes.append(parent_end)
            self._processes.append(process)

        self._remotes[0].send((_SPACES, None))
        self.observation_space, self.action_space, self.agent_ids = self._recv(0)
        self.num_agents = len(self.agent_ids)
        # Read off the spec rather than a worker: it is the same value, and a
        # null `algorithm.n_steps` resolves against it during algorithm
        # construction, before any command has been sent.
        self.ep_length = int(spec.ep_length)
        # Offset of each worker's first environment, for render addressing.
        self._offsets = np.cumsum([0] + self._counts[:-1]).tolist()

    # -- plumbing ---------------------------------------------------------

    def _recv(self, index: int):
        result = self._remotes[index].recv()
        if isinstance(result, _WorkerError):
            self.close()
            raise RuntimeError(
                f"environment worker {index} failed:\n{result.traceback}"
            )
        return result

    def _broadcast(self, command: str, payloads=None):
        """Send to every worker first, then collect -- that is the parallelism.

        Sending and receiving one worker at a time would serialise the whole
        thing and reproduce exactly the behaviour this class exists to fix.
        """
        for index, remote in enumerate(self._remotes):
            remote.send((command, None if payloads is None else payloads[index]))
        return [self._recv(index) for index in range(self.num_workers)]

    @property
    def num_rows(self) -> int:
        return self.num_envs * self.num_agents

    def rows_to_agents(self, rows: np.ndarray) -> Dict[str, np.ndarray]:
        from .vec_env import rows_to_agents

        return rows_to_agents(rows, self.num_envs, self.agent_ids)

    def agents_to_rows(self, per_agent: Dict[str, np.ndarray]) -> np.ndarray:
        from .vec_env import agents_to_rows

        return agents_to_rows(per_agent, self.num_envs, self.agent_ids)

    # -- the vector env interface ----------------------------------------

    def reset(self, seeds: Optional[Sequence[Optional[int]]] = None
              ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if seeds is not None and len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        chunks = self._chunk(seeds)
        results = self._broadcast(_RESET, chunks)
        obs = np.concatenate([r[0] for r in results], axis=0)
        infos: List[Dict[str, Any]] = []
        for _, worker_infos in results:
            infos.extend(worker_infos)
        return obs, infos

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_rows,):
            raise ValueError(
                f"expected actions of shape ({self.num_rows},), got {actions.shape}"
            )
        # Split along rows, which is why the env-major layout matters: each
        # worker's environments are contiguous, so its rows are too.
        row_counts = [count * self.num_agents for count in self._counts]
        bounds = np.cumsum(row_counts)[:-1]
        chunks = np.split(actions, bounds)

        results = self._broadcast(_STEP, chunks)
        obs = np.concatenate([r[0] for r in results], axis=0)
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []
        for _, worker_rewards, worker_dones, worker_infos in results:
            rewards.extend(worker_rewards)
            dones.extend(worker_dones)
            infos.extend(worker_infos)
        return (
            obs,
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def compute_social_metrics(self) -> List[Dict[str, float]]:
        metrics: List[Dict[str, float]] = []
        for worker_metrics in self._broadcast(_SOCIAL_METRICS):
            metrics.extend(worker_metrics)
        return metrics

    def render_frame(self, env_idx: int = 0) -> np.ndarray:
        worker = next(
            i for i in reversed(range(self.num_workers)) if self._offsets[i] <= env_idx
        )
        self._remotes[worker].send((_RENDER, env_idx - self._offsets[worker]))
        return self._recv(worker)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            try:
                remote.send((_CLOSE, None))
                remote.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

    def _chunk(self, values):
        """Split a per-env sequence into per-worker lists."""
        if values is None:
            return [None] * self.num_workers
        chunks, start = [], 0
        for count in self._counts:
            chunks.append(list(values[start:start + count]))
            start += count
        return chunks

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
