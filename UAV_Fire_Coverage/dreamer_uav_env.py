import os

import elements
import embodied
import numpy as np

from UAV_Fire_Coverage.data_utils import (
    generate_sample_circle1_data,
    generate_sample_circle8_data,
    load_circle_data,
    load_elevation_obstacle_map,
)
from UAV_Fire_Coverage.uav_fire_env import UAVFireEnv
from UAV_Fire_Coverage.uav_fire_obstacle_env import UAVFireObstacleEnv


class UAVFire(embodied.Env):

  def __init__(
      self,
      task='circle1',
      num_nearest=6,
      circle1_num_uavs=3,
      circle1_center_csv='',
      circle1_points_file='',
      circle1_circle_id=None,
      circle8_center_csv='',
      circle8_points_file='',
      circle8_circle_id=None,
      elevation_tif='',
      elev_threshold=2000.0,
      obstacle_resolution_m=50.0,
  ):
    self._task = task
    self._done = True
    self._episode_info = None
    self._episode_env_index = -1
    self._envs = []
    self._env = None
    self._is_obstacle_task = (task == 'circle8')

    if task == 'circle1':
      if circle1_center_csv and circle1_points_file and os.path.exists(circle1_center_csv) and os.path.exists(circle1_points_file):
        _, _, radius, fire_points = load_circle_data(
            circle1_center_csv, circle1_points_file, circle_id=circle1_circle_id)
      else:
        (_, _, radius), fire_points = generate_sample_circle1_data()
      clusters = _cluster_fire_points(fire_points, int(circle1_num_uavs))
      self._envs = [
          UAVFireEnv(fire_points=cluster, radius=radius, num_nearest=num_nearest)
          for cluster in clusters
      ]

    elif task == 'circle8':
      obstacle_map = None
      resolution_m = float(obstacle_resolution_m)
      if circle8_center_csv and circle8_points_file and os.path.exists(circle8_center_csv) and os.path.exists(circle8_points_file):
        lat_c, lon_c, radius, fire_points = load_circle_data(
            circle8_center_csv, circle8_points_file, circle_id=circle8_circle_id)
        if elevation_tif and os.path.exists(elevation_tif):
          obstacle_map, resolution_m = load_elevation_obstacle_map(
              elevation_tif,
              lat_center=lat_c,
              lon_center=lon_c,
              region_radius_m=radius,
              elevation_threshold=elev_threshold,
              target_resolution_m=resolution_m,
          )
      else:
        (_, _, radius), fire_points, obstacle_map, resolution_m = generate_sample_circle8_data()
      self._envs = [UAVFireObstacleEnv(
          fire_points=fire_points,
          radius=radius,
          obstacle_map=obstacle_map,
          resolution_m=resolution_m,
          num_nearest=num_nearest,
      )]
    else:
      raise ValueError(f'Unsupported uavfire task: {task}')

    self._switch_env()

  @property
  def obs_space(self):
    state_dim = self._env.state_dim
    return {
        'observation': elements.Space(np.float32, (state_dim,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/coverage_rate': elements.Space(np.float32),
        'log/visited_count': elements.Space(np.float32),
        'log/total_fire_points': elements.Space(np.float32),
        'log/collision': elements.Space(np.float32),
        'log/env_index': elements.Space(np.float32),
    }

  @property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        'action': elements.Space(np.float32, (1,), -1.0, 1.0),
    }

  def step(self, action):
    if action['reset'] or self._done:
      if self._done:
        self._switch_env()
      obs, _ = _env_reset(self._env)
      self._done = False
      self._episode_info = {
          'coverage_rate': 0.0,
          'visited_count': 0.0,
          'total_fire_points': float(self._env.n_fire),
          'collision': 0.0,
      }
      return self._make_obs(obs, 0.0, is_first=True)

    obs, reward, done, info = _env_step(self._env, action['action'])
    self._done = bool(done)
    self._episode_info = info
    is_terminal = bool(info.get('collision', False)) if self._is_obstacle_task else bool(done)
    return self._make_obs(
        obs=obs,
        reward=reward,
        is_last=done,
        is_terminal=is_terminal,
    )

  def close(self):
    for env in self._envs:
      try:
        env.close()
      except Exception:
        pass

  def _switch_env(self):
    self._episode_env_index = (self._episode_env_index + 1) % len(self._envs)
    self._env = self._envs[self._episode_env_index]

  def _make_obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    info = self._episode_info or {}
    return {
        'observation': np.asarray(obs, dtype=np.float32),
        'reward': np.float32(reward),
        'is_first': bool(is_first),
        'is_last': bool(is_last),
        'is_terminal': bool(is_terminal),
        'log/coverage_rate': np.float32(info.get('coverage_rate', 0.0)),
        'log/visited_count': np.float32(info.get('visited_count', 0.0)),
        'log/total_fire_points': np.float32(info.get('total_fire_points', 0.0)),
        'log/collision': np.float32(float(bool(info.get('collision', False)))),
        'log/env_index': np.float32(self._episode_env_index),
    }


def _cluster_fire_points(fire_points, n_clusters):
  n_clusters = max(1, int(n_clusters))
  if len(fire_points) <= n_clusters:
    return [fire_points]
  try:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels = km.fit_predict(fire_points)
  except Exception:
    labels = np.arange(len(fire_points)) % n_clusters
  clusters = [fire_points[labels == k] for k in range(n_clusters)]
  return [x for x in clusters if len(x) > 0]


def _env_reset(env):
  result = env.reset()
  if isinstance(result, tuple):
    return result
  return result, {}


def _env_step(env, action):
  result = env.step(action)
  if len(result) == 5:
    obs, rew, terminated, truncated, info = result
    return obs, rew, bool(terminated or truncated), info
  return result
