"""
uav_fire.py — embodied.Env wrapper for UAV fire-coverage tasks,
compatible with the DreamerV3 world-model training loop.

Two task strings are supported:
  'circle1'  — fire-point coverage, 3 UAVs (MultiUAVFireEnv)
  'circle8'  — fire-point coverage + obstacle avoidance (UAVFireObstacleEnv)

The environment is located via the following precedence:
  1. Keyword arguments passed directly (fire_points, radius_m, …)
  2. Environment variables:
       UAV_CENTER_CSV, UAV_POINTS_FILE, UAV_ELEVATION_TIF (optional)
  3. Built-in synthetic sample data (works out of the box)

The observation space exposes a single key 'vector' (float32) which is
consumed by DreamerV3's MLP encoder path.
"""

import os
import sys

import numpy as np

import elements
import embodied

# ── locate UAV_Fire_Coverage directory ──────────────────────────────────────
_UAV_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # embodied/envs/
    '..', '..',                                   # repo root
    'UAV_Fire_Coverage',
)
_UAV_DIR = os.path.normpath(_UAV_DIR)
if _UAV_DIR not in sys.path:
    sys.path.insert(0, _UAV_DIR)


def _default_csv(name):
    return os.path.join(_UAV_DIR, 'sample_data', name)


class UAVFireEnvEmbodied(embodied.Env):
    """
    embodied.Env adapter for UAV fire-coverage tasks.

    Parameters
    ----------
    task        : str   'circle1' or 'circle8'
    center_csv  : str   Path to circle-centre CSV (optional; falls back to env
                        var UAV_CENTER_CSV, then sample data).
    points_file : str   Path to fire-points CSV or SHP (optional; similar
                        fallback chain).
    elevation_tif : str Path to GeoTIFF elevation map (circle8 only; optional).
    max_steps   : int   Episode length limit (default 2000).
    """

    def __init__(self, task, center_csv=None, points_file=None,
                 elevation_tif=None, max_steps=2000, **kwargs):
        from data_utils import (load_circle_center, load_fire_points,
                                load_obstacle_map, make_synthetic_obstacle_map)
        from uav_fire_env import MultiUAVFireEnv
        from uav_fire_obstacle_env import UAVFireObstacleEnv

        assert task in ('circle1', 'circle8'), \
            f"UAV task must be 'circle1' or 'circle8', got '{task}'"

        # ── resolve data paths ───────────────────────────────────────────────
        if task == 'circle1':
            center_csv = (
                center_csv
                or os.environ.get('UAV_CENTER_CSV')
                or _default_csv('circle_1_center.csv'))
            points_file = (
                points_file
                or os.environ.get('UAV_POINTS_FILE')
                or _default_csv('circle_1_points.csv'))
        else:
            center_csv = (
                center_csv
                or os.environ.get('UAV_CENTER_CSV')
                or _default_csv('circle_8_center.csv'))
            points_file = (
                points_file
                or os.environ.get('UAV_POINTS_FILE')
                or _default_csv('circle_8_points.csv'))
            elevation_tif = (
                elevation_tif
                or os.environ.get('UAV_ELEVATION_TIF'))

        lat0, lon0, radius = load_circle_center(center_csv)
        fire_pts = load_fire_points(points_file, lat0, lon0)

        if task == 'circle1':
            self._env = MultiUAVFireEnv(fire_pts, radius,
                                        n_uav=3, max_steps=max_steps)
            self._state_dim = self._env.observation_space.shape[0]
        else:
            if elevation_tif:
                obs_grid, cell_size = load_obstacle_map(
                    elevation_tif, lat0, lon0, radius)
            else:
                obs_grid, cell_size = make_synthetic_obstacle_map(radius)
            self._env = UAVFireObstacleEnv(
                fire_pts, radius, obs_grid, cell_size, max_steps=max_steps)
            self._state_dim = self._env.observation_space.shape[0]

        self._done = True

    # ------------------------------------------------------------------
    @property
    def obs_space(self):
        return {
            'vector':      elements.Space(np.float32, (self._state_dim,)),
            'reward':      elements.Space(np.float32),
            'is_first':    elements.Space(bool),
            'is_last':     elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    @property
    def act_space(self):
        return {
            'reset':  elements.Space(bool),
            'action': elements.Space(np.float32, (1,), -1.0, 1.0),
        }

    # ------------------------------------------------------------------
    def step(self, action):
        if action['reset'] or self._done:
            self._done = False
            obs = self._env.reset()
            return self._pack(obs, 0.0, is_first=True)

        raw_action = np.asarray(action['action'], dtype=np.float32)
        obs, reward, done, info = self._env.step(raw_action)
        self._done = done
        return self._pack(obs, reward,
                          is_last=done,
                          is_terminal=done)

    def _pack(self, obs_arr, reward,
              is_first=False, is_last=False, is_terminal=False):
        return {
            'vector':      np.asarray(obs_arr, dtype=np.float32),
            'reward':      np.float32(reward),
            'is_first':    bool(is_first),
            'is_last':     bool(is_last),
            'is_terminal': bool(is_terminal),
        }

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
