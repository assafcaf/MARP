"""Human-facing training output.

Everything the training loop says to a person goes through here, so the loop
itself stays free of `print` and the presentation has one place to live.

Two presentations, chosen by `mode`:

- a live `tqdm` bar, for an interactive terminal
- a periodic status line, for a redirected stream -- a bar redrawing itself
  into a log file writes thousands of unreadable lines

`auto` picks between them by asking the stream whether it is a terminal.
"""

import sys
import time
from typing import Any, Mapping, Optional, TextIO

from tqdm import tqdm

MODES = ("auto", "bar", "plain", "quiet")


class TrainingConsole:
    def __init__(
        self,
        mode: str = "auto",
        status_every: int = 10,
        stream: Optional[TextIO] = None,
    ):
        if mode not in MODES:
            raise ValueError(f"unknown console mode {mode!r}; expected one of {MODES}")
        self.stream = stream if stream is not None else sys.stdout
        self.mode = mode
        self.status_every = max(1, status_every)
        self.enabled = mode != "quiet"
        self.uses_bar = self._resolve_uses_bar(mode)
        self._bar: Optional[tqdm] = None
        self._total_episodes: Optional[int] = None
        self._start_time = time.time()
        self._said: set = set()

    @classmethod
    def from_config(cls, logging_config: Any, stream: Optional[TextIO] = None) -> "TrainingConsole":
        return cls(
            mode=getattr(logging_config, "console", "auto"),
            status_every=getattr(logging_config, "status_every", 10),
            stream=stream,
        )

    def _resolve_uses_bar(self, mode: str) -> bool:
        if mode == "bar":
            return True
        if mode in ("plain", "quiet"):
            return False
        # `auto`: a bar only makes sense where it can redraw in place.
        isatty = getattr(self.stream, "isatty", None)
        try:
            return bool(isatty()) if callable(isatty) else False
        except ValueError:
            # A closed stream under pytest's capture.
            return False

    # -- messages ---------------------------------------------------------

    def section(self, title: str) -> None:
        self._write(f"\n=== {title} ===")

    def info(self, message: str) -> None:
        self._write(f"  {message}")

    def warn(self, message: str) -> None:
        self._write(f"  ! {message}")

    def info_once(self, key: str, message: str) -> None:
        """Say something the first time only.

        Phase transitions are noticed inside the episode loop, which asks on
        every episode; keeping the de-duplication here spares each caller its
        own "have I said this yet" flag.
        """
        if key in self._said:
            return
        self._said.add(key)
        self.info(message)

    def _write(self, text: str) -> None:
        if not self.enabled:
            return
        # `tqdm.write` keeps the bar from being torn apart by the message.
        if self._bar is not None:
            self._bar.write(text, file=self.stream)
        else:
            self.stream.write(text + "\n")
        self.stream.flush()

    # -- episode reporting ------------------------------------------------

    def start_episodes(self, total: int) -> None:
        self._total_episodes = total
        self._start_time = time.time()
        if self.uses_bar:
            self._bar = tqdm(
                total=total,
                desc="episodes",
                unit="ep",
                file=self.stream,
                dynamic_ncols=True,
                leave=True,
            )

    def episode_end(
        self,
        episode: int,
        stats: Optional[Mapping[str, Any]] = None,
        advance: int = 1,
    ) -> None:
        """Report finished episodes.

        `episode` is the 0-based index of the last one completed; `advance` is
        how many completed at once, which is `num_envs` when environments run
        in parallel.
        """
        if not self.enabled:
            return
        summary = format_metrics(stats)
        if self._bar is not None:
            if summary:
                self._bar.set_postfix_str(summary, refresh=False)
            self._bar.update(advance)
            return
        if self._should_report(episode):
            done = episode + 1
            total = self._total_episodes if self._total_episodes is not None else done
            elapsed = time.time() - self._start_time
            line = f"[ep {done}/{total}]"
            if summary:
                line += f" {summary}"
            line += f" ({_format_duration(elapsed)} elapsed)"
            self._write(line)

    def _should_report(self, episode: int) -> bool:
        done = episode + 1
        if self._total_episodes is not None and done >= self._total_episodes:
            # The final numbers of a run whose length is not a multiple of the
            # cadence would otherwise never be printed.
            return True
        return done % self.status_every == 0

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def format_metrics(stats: Optional[Mapping[str, Any]]) -> str:
    """`{"reward": 12.3}` -> `reward=12.30`, for bar postfixes and status lines."""
    if not stats:
        return ""
    parts = []
    for name, value in stats.items():
        if isinstance(value, float):
            parts.append(f"{name}={value:.2f}")
        else:
            parts.append(f"{name}={value}")
    return " ".join(parts)


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
