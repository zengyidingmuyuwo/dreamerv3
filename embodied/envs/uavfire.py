import os
import sys
import numpy as np
import elements
import embodied


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UAV_DIR = os.path.join(ROOT, 'UAV_Fire_Coverage')
if UAV_DIR not in sys.path:
  sys.path.insert(0, UAV_DIR)

from data_utils import load_circle_data, load_elevation_obstacle_map, generate_sample_circle1_data, generate_sample_circle8_data
from uav_fire_env import UAVFireEnv
from uav_fire_obstacle_env import UAVFireObstacleEnv


class UAVFire(embodied.Env):

  def __init__(
      self, task,
      center_csv='',
      points_file='',
      elevation_tif='',
      circle_id=-1,
      elev_threshold=2000.0,
      num_nearest=6,
      resolution_m=50.0,
      seed=None,
  ):
    assert task in ('circle1', 'circle8'), task
    self._done = True
    self._info = {}
    use_real = bool(center_csv) and bool(points_file) and os.path.exists(center_csv) and os.path.exists(points_file)
    if use_real:
      kwargs = {} if circle_id == -1 else {'circle_id': int(circle_id)}
      lat_c, lon_c, radius, fire_points = load_circle_data(center_csv, points_file, **kwargs)
      obstacle_map = None
      if task == 'circle8' and elevation_tif and os.path.exists(elevation_tif):
        obstacle_map, resolution_m = load_elevation_obstacle_map(
            elevation_tif, lat_c, lon_c, region_radius_m=radius,
            elevation_threshold=elev_threshold, target_resolution_m=resolution_m)
    else:
      if task == 'circle8':
        (_, _, radius), fire_points, obstacle_map, resolution_m = generate_sample_circle8_data()
      else:
        (_, _, radius), fire_points = generate_sample_circle1_data()
        obstacle_map = None
    if task == 'circle8':
      self._env = UAVFireObstacleEnv(
          fire_points=fire_points, radius=radius, obstacle_map=obstacle_map,
          resolution_m=resolution_m, num_nearest=num_nearest, return_dict_obs=True,
          algorithm_name='DREAMER', env_name='Circle8')
    else:
      self._env = UAVFireEnv(
          fire_points=fire_points, radius=radius, num_nearest=num_nearest, return_dict_obs=True,
          algorithm_name='DREAMER', env_name='Circle1')

  @property
  def obs_space(self):
    image_shape = self._env.observation_space['image'].shape
    vector_shape = self._env.observation_space['vector'].shape
    return {
        'image': elements.Space(np.float32, image_shape, -1.0, 1.0),
        'vector': elements.Space(np.float32, vector_shape, -1.0, 1.0),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.float32, self._env.action_space.shape, -1.0, 1.0),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      out = self._env.reset()
      if isinstance(out, tuple):
        obs, _ = out
      else:
        obs = out
      return self._obs(obs, 0.0, is_first=True)
    out = self._env.step(action['action'])
    if len(out) == 5:
      obs, reward, terminated, truncated, self._info = out
      done = bool(terminated or truncated)
    else:
      obs, reward, done, self._info = out
    self._done = done
    return self._obs(obs, reward, is_last=done, is_terminal=done)

  def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    return {
        'image': np.asarray(obs['image'], dtype=np.float32),
        'vector': np.asarray(obs['vector'], dtype=np.float32),
        'reward': np.float32(reward),
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal,
    }

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass
