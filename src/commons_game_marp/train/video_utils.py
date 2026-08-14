import os
from typing import Optional

from ..env.utils import utility_funcs


class VideoRecorder:
    def __init__(
        self,
        base_dir: str,
        enabled: bool,
        every_n_episodes: int,
        max_steps: int,
        fps: int,
        keep_frames: bool,
        total_episodes: Optional[int] = None,
        stride: int = 1,
    ):
        self.base_dir = base_dir
        self.enabled = enabled
        self.every_n_episodes = max(1, every_n_episodes)
        self.max_steps = max_steps
        self.fps = fps
        self.keep_frames = keep_frames
        self.total_episodes = total_episodes
        # Episodes the trainer's counter advances between two calls, i.e.
        # `num_envs`. See `should_record`.
        self.stride = max(1, stride)
        self._current_episode: Optional[int] = None
        self._frame_dir: Optional[str] = None

    def should_record(self, episode: int) -> bool:
        """Whether the episode ending this iteration should be captured.

        The counter this receives advances by `stride` at a time -- the trainer
        steps `num_envs` episodes per iteration and reports the last of them,
        `(iteration + 1) * num_envs - 1` -- so it steps *over* round multiples
        rather than landing on them. Testing `episode % every_n == 0` therefore
        silently recorded nothing at num_envs=4: every index it can take is odd
        and every 100-multiple is even, so a 10000-episode run emitted one video
        (the `total_episodes - 1` case below) instead of a hundred.

        So the question is whether the span this iteration covers -- `(episode -
        stride, episode]` -- crosses an interval boundary, not whether it lands
        on one. At stride 1 that is exactly the old modulo test, which keeps the
        serial path unchanged rather than merely equivalent.

        `previous` is floored at 0 so the boundary at 0 is not a crossing, which
        is what keeps the opening iteration unrecorded without a special case
        for `episode == 0`.
        """
        if not self.enabled:
            return False
        # Always record if it's the last episode
        if self.total_episodes is not None and episode == self.total_episodes - 1:
            return True
        previous = max(0, episode - self.stride)
        return episode // self.every_n_episodes > previous // self.every_n_episodes

    def start(self, episode: int) -> None:
        if not self.should_record(episode):
            self._current_episode = None
            self._frame_dir = None
            return
        self._current_episode = episode
        self._frame_dir = os.path.join(self.base_dir, f"episode={episode:04d}")
        os.makedirs(self._frame_dir, exist_ok=True)

    def record(self, env, step: int) -> None:
        if self._current_episode is None or self._frame_dir is None:
            return
        if self.max_steps is not None and step >= self.max_steps:
            return
        img_path = os.path.join(self._frame_dir, f"t={step:04d}.png")
        env.render(img_path, mod="human")

    def finish(self) -> Optional[str]:
        if self._current_episode is None or self._frame_dir is None:
            return None
        video_dir = os.path.dirname(self._frame_dir)
        video_name = f"episode={self._current_episode:04d}"
        utility_funcs.make_video_from_image_dir(
            vid_path=video_dir,
            img_folder=self._frame_dir,
            video_name=video_name,
            fps=self.fps,
        )
        if not self.keep_frames:
            for fname in os.listdir(self._frame_dir):
                if fname.endswith(".png"):
                    os.remove(os.path.join(self._frame_dir, fname))
            os.rmdir(self._frame_dir)
        return os.path.join(video_dir, f"{video_name}.mp4")

    def finalize(self) -> None:
        if self.keep_frames:
            return
        if not os.path.isdir(self.base_dir):
            return
        for entry in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, entry)
            if os.path.isdir(path):
                for fname in os.listdir(path):
                    if fname.endswith(".png"):
                        os.remove(os.path.join(path, fname))
                os.rmdir(path)
