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
from comparison_logging import EpisodeCSVLogger


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
      num_uavs=3,
      seed=None,
  ):
    assert task in ('circle1', 'circle8'), task
    self._task = task
    self._done = True
    self._info = {}
    self._episode = 0
    self._episode_reward = 0.0
    self._episode_steps = 0
    self._total_steps = 0
    use_real = bool(center_csv) and bool(points_file) and os.path.exists(center_csv) and os.path.exists(points_file)
    lat_c = None
    lon_c = None
    dem_query_metadata = None
    if use_real:
      kwargs = {} if circle_id == -1 else {'circle_id': int(circle_id)}
      lat_c, lon_c, radius, fire_points = load_circle_data(center_csv, points_file, **kwargs)
      obstacle_map = None
      if task == 'circle8' and elevation_tif and os.path.exists(elevation_tif):
        obstacle_map, resolution_m, dem_query_metadata = load_elevation_obstacle_map(
            elevation_tif, lat_c, lon_c, region_radius_m=radius,
            elevation_threshold=elev_threshold, target_resolution_m=resolution_m,
            return_metadata=True)
      elif task == 'circle8':
        elev_dir = os.path.join(ROOT, 'prepare', 'elevation')
        if os.path.isdir(elev_dir):
          tif_candidates = sorted(f for f in os.listdir(elev_dir) if f.lower().endswith(('.tif', '.tiff', '.zip')))
          if tif_candidates:
            tif_path = os.path.join(elev_dir, tif_candidates[0])
            obstacle_map, resolution_m, dem_query_metadata = load_elevation_obstacle_map(
                tif_path, lat_c, lon_c, region_radius_m=radius,
                elevation_threshold=elev_threshold, target_resolution_m=resolution_m,
                return_metadata=True)
    else:
      if task == 'circle8':
        (_, _, radius), fire_points, obstacle_map, resolution_m = generate_sample_circle8_data()
      else:
        (_, _, radius), fire_points = generate_sample_circle1_data()
        obstacle_map = None
    self._num_uavs = max(1, int(num_uavs))
    clusters = self._cluster_fire_points(fire_points, self._num_uavs)
    self._envs = []
    for cluster in clusters:
      if task == 'circle8':
        env = UAVFireObstacleEnv(
            fire_points=cluster, radius=radius, obstacle_map=obstacle_map,
            resolution_m=resolution_m, num_nearest=num_nearest, return_dict_obs=True,
            algorithm_name='DREAMER', env_name='Circle8', lat_center=lat_c, lon_center=lon_c,
            elevation_threshold=elev_threshold, dem_query_metadata=dem_query_metadata)
      else:
        env = UAVFireEnv(
            fire_points=cluster, radius=radius, num_nearest=num_nearest, return_dict_obs=True,
            algorithm_name='DREAMER', env_name='Circle1')
      self._envs.append(env)
    self._num_uavs = len(self._envs)
    self._env = self._envs[0]
    self._single_action_dim = int(self._env.action_space.shape[0])
    self._single_image_dim = int(self._env.observation_space['image'].shape[0])
    self._single_vector_dim = int(self._env.observation_space['vector'].shape[0])
    self._last_infos = [{} for _ in range(self._num_uavs)]
    print(
        f'[Dreamer UAVFire] Centralized control enabled: '
        f'{self._num_uavs} UAV(s), action_dim={self._single_action_dim * self._num_uavs}'
    )
    log_dir = os.path.join(UAV_DIR, 'logs')
    scenario_name = 'Circle8' if task == 'circle8' else 'Circle1'
    self._episode_logger = EpisodeCSVLogger('DREAMER', scenario_name, log_dir)
    print(f'[Dreamer UAVFire] Standard CSV log: {os.path.abspath(self._episode_logger.path)}')

  @property
  def obs_space(self):
    image_shape = (self._single_image_dim * self._num_uavs,)
    vector_shape = (self._single_vector_dim * self._num_uavs,)
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
        'action': elements.Space(np.float32, (self._single_action_dim * self._num_uavs,), -1.0, 1.0),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      self._episode_reward = 0.0
      self._episode_steps = 0
      obs = self._reset_all()
      return self._obs(obs, 0.0, is_first=True)
    actions = self._split_actions(action['action'])
    obs_list, rewards, dones, infos = [], [], [], []
    for env, sub_action in zip(self._envs, actions):
      out = env.step(sub_action)
      if len(out) == 5:
        obs_i, reward_i, terminated_i, truncated_i, info_i = out
        done_i = bool(terminated_i or truncated_i)
      else:
        obs_i, reward_i, done_i, info_i = out
      obs_list.append(obs_i)
      rewards.append(float(reward_i))
      dones.append(bool(done_i))
      infos.append(info_i)
    reward = float(np.sum(rewards))
    self._last_infos = infos
    self._info = self._merge_infos(infos)
    done = bool(np.all(dones))
    self._episode_reward += float(reward)
    self._episode_steps += 1
    self._total_steps += 1
    if done:
      self._episode += 1
      self._episode_logger.log_episode(
          episode=self._episode,
          timesteps=self._total_steps,
          episode_reward=self._episode_reward,
          coverage_pct=float(self._info.get('coverage_rate', 0.0)) * 100.0,
          collision=bool(self._info.get('collision', False)),
      )
    self._done = done
    return self._obs(obs_list, reward, is_last=done, is_terminal=done)

  def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    if isinstance(obs, list):
      image = np.concatenate(
          [np.asarray(item['image'], dtype=np.float32) for item in obs], axis=0)
      vector = np.concatenate(
          [np.asarray(item['vector'], dtype=np.float32) for item in obs], axis=0)
    else:
      image = np.asarray(obs['image'], dtype=np.float32)
      vector = np.asarray(obs['vector'], dtype=np.float32)
    image = np.clip(image, -1.0, 1.0).astype(np.float32)
    vector = np.clip(vector, -1.0, 1.0).astype(np.float32)
    return {
        'image': image,
        'vector': vector,
        'reward': np.float32(reward),
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal,
    }

  def _reset_all(self):
    obs_list = []
    for env in self._envs:
      out = env.reset()
      if isinstance(out, tuple):
        obs, _ = out
      else:
        obs = out
      obs_list.append(obs)
    self._last_infos = [{} for _ in range(self._num_uavs)]
    self._info = self._merge_infos(self._last_infos)
    return obs_list

  def _split_actions(self, action):
    act = np.asarray(action, dtype=np.float32).reshape(-1)
    expected = self._single_action_dim * self._num_uavs
    if act.size == self._single_action_dim:
      act = np.tile(act, self._num_uavs)
    elif act.size < expected:
      act = np.pad(act, (0, expected - act.size), mode='constant')
    elif act.size > expected:
      act = act[:expected]
    act = np.clip(act, -1.0, 1.0).astype(np.float32)
    return [act[i * self._single_action_dim:(i + 1) * self._single_action_dim]
            for i in range(self._num_uavs)]

  def _merge_infos(self, infos):
    if not infos:
      return {}
    total_fire = int(sum(int(info.get('total_fire_points', 0)) for info in infos))
    total_visited = int(sum(int(info.get('visited_count', 0)) for info in infos))
    coverage = (float(total_visited) / float(total_fire)) if total_fire else 0.0
    return {
        'visited_count': total_visited,
        'total_fire_points': total_fire,
        'step': max((int(info.get('step', 0)) for info in infos), default=0),
        'coverage_rate': coverage,
        'collision': any(bool(info.get('collision', False)) for info in infos),
        'mountain_collision': any(bool(info.get('mountain_collision', False)) for info in infos),
        'bird_collision': any(bool(info.get('bird_collision', False)) for info in infos),
        'off_path': any(bool(info.get('off_path', False)) for info in infos),
        'per_uav_info': infos,
    }

  @staticmethod
  def _cluster_fire_points(fire_points, num_uavs):
    points = np.asarray(fire_points, dtype=np.float32)
    if len(points) == 0 or num_uavs <= 1:
      return [points]
    n_clusters = min(int(num_uavs), len(points))
    try:
      from sklearn.cluster import KMeans
      labels = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit_predict(points)
    except Exception:
      labels = np.arange(len(points)) % n_clusters
    clusters = [points[labels == idx] for idx in range(n_clusters)]
    clusters = [cluster for cluster in clusters if len(cluster) > 0]
    return clusters if clusters else [points]

  def close(self):
    for env in self._envs:
      try:
        env.close()
      except Exception:
        pass
