"""Entropy-coefficient control for the PPO-family algorithms.

Three modes share one interface so IPPO and MAPPO carry no schedule logic of
their own.

`adaptive` exists because the coefficient turned out to be the wrong variable
to control. In run 20260813-125003-seed=0 the policy entropy had already
collapsed to 0.64 nats while `ent_coef` was still 0.068 -- roughly thirty times
the policy-loss magnitude in the total loss, and still losing. No reachable
floor on a fixed schedule would have helped, because the schedule never got
low enough to be the binding constraint. Targeting the entropy itself does.

`ent_coef_lr` and the rollout shape are coupled. `observe_entropy` fires once
per minibatch, not once per env step, so the controller's effective speed is
`ent_coef_lr` scaled by however many minibatches an episode works out to:
roughly `(n_steps / batch_size) * update_epochs * (ep_length / n_steps)`, which
collapses to `update_epochs * ep_length / batch_size` calls per episode. IPPO's
defaults (`n_steps=512`, `batch_size=128`, `update_epochs=2`, `ep_length=600`)
give about 10 calls an episode per agent; MAPPO's defaults give about 12.
Halving `batch_size` doubles the controller's speed at a fixed `ent_coef_lr`,
and the two algorithms already run at different effective rates from the same
nominal `ent_coef_lr` for the same reason. This module does not normalise for
it -- `ent_coef_lr` is a per-minibatch gain, not a per-episode one, and tuning
it means accounting for the rollout settings alongside it.
"""

import math
from typing import Any, Optional

import torch


class EntropyController:
    """Supplies the entropy coefficient for one policy's PPO update.

    IPPO builds one per agent (its networks are per-agent, so its exploration
    pressure should be too, and a diverging agent then shows up directly as a
    diverging coefficient). MAPPO builds one, matching its shared actor.
    """

    MODES = ("fixed", "anneal", "adaptive")

    def __init__(self, config: Any, num_actions: int, device: torch.device) -> None:
        self.mode = str(getattr(config, "ent_coef_mode", "adaptive"))
        if self.mode not in self.MODES:
            raise ValueError(
                f"Unknown ent_coef_mode '{self.mode}'. Available: {list(self.MODES)}"
            )

        self.start = float(getattr(config, "ent_coef", 0.01))
        self.end = float(getattr(config, "ent_coef_end", self.start))
        self.minimum = float(getattr(config, "ent_coef_min", 1e-3))
        self.maximum = float(getattr(config, "ent_coef_max", 0.5))
        frac = float(getattr(config, "target_entropy_frac", 0.6))
        self.target_entropy = frac * math.log(num_actions)

        self._total_episodes = 0
        self._current_episode = 0
        self._log_ent_coef: Optional[torch.Tensor] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._log_min = math.log(self.minimum)
        self._log_max = math.log(self.maximum)

        if self.mode == "adaptive":
            initial = min(max(self.start, self.minimum), self.maximum)
            self._log_ent_coef = torch.tensor(
                math.log(initial), dtype=torch.float32, device=device, requires_grad=True
            )
            self._optimizer = torch.optim.Adam(
                [self._log_ent_coef], lr=float(getattr(config, "ent_coef_lr", 3e-4))
            )

    def set_total_episodes(self, total: int) -> None:
        self._total_episodes = int(total)

    def set_episode(self, episode: int) -> None:
        self._current_episode = int(episode)

    def coefficient(self) -> float:
        """The coefficient to multiply the entropy bonus by, as a plain float.

        Returned detached in every mode: the controller trains its own
        parameter through `observe_entropy`, and letting the policy loss
        backpropagate into it as well would make the two objectives fight.
        """
        if self.mode == "fixed":
            return self.start
        if self.mode == "anneal":
            if self._total_episodes <= 0 or self.start == self.end:
                return self.start
            progress = min(1.0, self._current_episode / self._total_episodes)
            return self.start + (self.end - self.start) * progress
        assert self._log_ent_coef is not None
        return float(self._log_ent_coef.detach().exp().clamp(self.minimum, self.maximum))

    def is_saturated(self) -> bool:
        """True when the coefficient sits at a clamp and can no longer respond.

        A saturated controller is an open loop: entropy can keep moving and the
        coefficient will not. Reported so that degraded state is visible rather
        than looking like a healthy steady value.

        The comparison uses a small relative tolerance rather than an exact
        `<=`/`>=`: `_log_ent_coef` is a float32 tensor clamped in log space, and
        the exp/log round trip through `coefficient()` lands a few ULPs on
        either side of the true bound (observed: 0.0010000000475 against a
        0.001 floor). An exact comparison would report an open loop as still
        responsive purely from that rounding.
        """
        if self.mode != "adaptive":
            return False
        coef = self.coefficient()
        tol = 1e-6
        return coef <= self.minimum * (1 + tol) or coef >= self.maximum * (1 - tol)

    def observe_entropy(self, entropy: float) -> None:
        """One dual-ascent step from an observed policy entropy. Adaptive only.

        `loss = log_ent_coef * (entropy - target)` has gradient `(entropy -
        target)` with respect to the parameter, so entropy below target gives a
        negative gradient and the descent step raises the coefficient. The
        parameter is clamped after the step rather than only on read: clamping
        the read value alone lets the parameter integrate far past the bound
        and take just as long to unwind.
        """
        if self.mode != "adaptive":
            return
        assert self._log_ent_coef is not None and self._optimizer is not None
        loss = self._log_ent_coef * (float(entropy) - self.target_entropy)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        with torch.no_grad():
            self._log_ent_coef.clamp_(self._log_min, self._log_max)
