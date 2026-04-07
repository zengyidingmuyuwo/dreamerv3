import csv
import datetime as dt
import os
import time
from typing import Optional


class EpisodeCSVLogger:
    """Unified episode logger for cross-algorithm comparison."""

    def __init__(
        self,
        algorithm: str,
        scenario: str,
        output_dir: str,
        run_id: Optional[str] = None,
    ):
        self.algorithm = str(algorithm).upper()
        self.scenario = str(scenario)
        self.run_id = run_id or dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        self._start_time = time.time()
        self._path = self._make_path(output_dir)
        self._init_file()

    def _make_path(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        safe_scenario = self.scenario.lower().replace(' ', '')
        filename = (
            f'{self.algorithm.lower()}_{safe_scenario}_{self.run_id}_pid{os.getpid()}.csv'
        )
        return os.path.join(output_dir, filename)

    def _init_file(self) -> None:
        headers = [
            'algorithm',
            'scenario',
            'run_id',
            'episode',
            'timesteps',
            'wall_time_sec',
            'episode_reward',
            'coverage_pct',
            'collision',
        ]
        with open(self._path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    @property
    def path(self) -> str:
        return self._path

    def log_episode(
        self,
        episode: int,
        timesteps: int,
        episode_reward: float,
        coverage_pct: float,
        collision: bool = False,
        wall_time_sec: Optional[float] = None,
    ) -> None:
        if wall_time_sec is None:
            wall_time_sec = time.time() - self._start_time
        row = {
            'algorithm': self.algorithm,
            'scenario': self.scenario,
            'run_id': self.run_id,
            'episode': int(episode),
            'timesteps': int(timesteps),
            'wall_time_sec': float(wall_time_sec),
            'episode_reward': float(episode_reward),
            'coverage_pct': float(coverage_pct),
            'collision': bool(collision),
        }
        with open(self._path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)
