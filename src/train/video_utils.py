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
    ):
        self.base_dir = base_dir
        self.enabled = enabled
        self.every_n_episodes = max(1, every_n_episodes)
        self.max_steps = max_steps
        self.fps = fps
        self.keep_frames = keep_frames
        self.total_episodes = total_episodes
        self._current_episode: Optional[int] = None
        self._frame_dir: Optional[str] = None

    def should_record(self, episode: int) -> bool:
        if not self.enabled:
            return False
        # Always record if it's the last episode
        if self.total_episodes is not None and episode == self.total_episodes - 1:
            return True
        # Skip episode 0, then record based on every_n_episodes
        if episode == 0:
            return False
        return episode % self.every_n_episodes == 0

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
