import os
import sys
import numpy as np
import elements
import embodied


_COVERAGE_COMPLETION_EPS = 1e-6


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UAV_DIR = os.path.join(ROOT, 'UAV_Fire_Coverage')
if UAV_DIR not in sys.path:
  sys.path.insert(0, UAV_DIR)

from data_utils import load_circle_data, load_elevation_obstacle_map, generate_sample_circle1_data, generate_sample_circle8_data
from comparison_logging import EpisodeCSVLogger


def _load_env_classes():
  from uav_fire_env import UAVFireEnv
  from uav_fire_obstacle_env import UAVFireObstacleEnv
  return UAVFireEnv, UAVFireObstacleEnv


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
      num_uavs=-1,
      single_round_robin=True,
      seed=None,
  ):
    assert task in ('circle1', 'circle1_single', 'circle8'), task
    self._task = task
    self._done = True
    self._info = {}
    self._episode = 0
    self._episode_reward = 0.0
    self._episode_steps = 0
    self._total_steps = 0
    default_center_csv = os.path.join(
        UAV_DIR, 'prepare', 'circle_8_center.csv' if task == 'circle8' else 'circle_1_center.csv')
    default_points_file = os.path.join(
        UAV_DIR, 'prepare', 'circle_8_points.shp' if task == 'circle8' else 'circle_1_points.shp')
    effective_center_csv = center_csv if center_csv else default_center_csv
    effective_points_file = points_file if points_file else default_points_file
    use_real = (
        bool(effective_center_csv) and bool(effective_points_file) and
        os.path.exists(effective_center_csv) and os.path.exists(effective_points_file)
    )
    radius = 1.0
    fire_points = np.zeros((0, 2), dtype=np.float32)
    lat_c = None
    lon_c = None
    dem_query_metadata = None
    if use_real:
      kwargs = {} if circle_id == -1 else {'circle_id': int(circle_id)}
      lat_c, lon_c, radius, fire_points = load_circle_data(
          effective_center_csv, effective_points_file, **kwargs)
      obstacle_map = None
      if task == 'circle8':
        effective_elevation_tif = elevation_tif if (elevation_tif and os.path.exists(elevation_tif)) else ''
        if not effective_elevation_tif:
          candidates = []
          elev_dir = os.path.join(UAV_DIR, 'prepare', 'elevation')
          if os.path.isdir(elev_dir):
            candidates.extend(
                os.path.join(elev_dir, f)
                for f in sorted(os.listdir(elev_dir))
                if f.lower().endswith(('.tif', '.tiff', '.zip')))
          prepare_dir = os.path.join(UAV_DIR, 'prepare')
          if os.path.isdir(prepare_dir):
            candidates.extend(
                os.path.join(prepare_dir, f)
                for f in sorted(os.listdir(prepare_dir))
                if f.lower().endswith(('.tif', '.tiff', '.zip')))
          candidates = [p for p in candidates if os.path.exists(p)]
          if candidates:
            effective_elevation_tif = candidates[0]
        if effective_elevation_tif:
          obstacle_map, resolution_m, dem_query_metadata = load_elevation_obstacle_map(
              effective_elevation_tif, lat_c, lon_c, region_radius_m=radius,
              elevation_threshold=elev_threshold, target_resolution_m=resolution_m,
              return_metadata=True)
    else:
      if task == 'circle8':
        (_, _, radius), fire_points, obstacle_map, resolution_m = generate_sample_circle8_data()
      else:
        (_, _, radius), fire_points = generate_sample_circle1_data()
        obstacle_map = None
    # Circle1 defaults to 3 UAVs; Circle8 remains single-UAV.
    strict_num_uavs = 1 if task == 'circle8' else 3
    requested_num_uavs = int(num_uavs)
    if requested_num_uavs not in (-1, strict_num_uavs):
      print(
          f'[Dreamer UAVFire] Override num_uavs={requested_num_uavs} -> {strict_num_uavs} '
          f'for strict {task} baseline.'
      )
    self._num_uavs = strict_num_uavs
    UAVFireEnv, UAVFireObstacleEnv = _load_env_classes()
    cluster_count = 3 if task in ('circle1', 'circle1_single') else self._num_uavs
    clusters = self._cluster_fire_points(fire_points, cluster_count)
    self._envs = []
    for idx, cluster in enumerate(clusters):
      if task == 'circle8':
        env = UAVFireObstacleEnv(
            fire_points=cluster, radius=radius, obstacle_map=obstacle_map,
            resolution_m=resolution_m, num_nearest=num_nearest, return_dict_obs=True,
            algorithm_name='DREAMER', env_name='Circle8', lat_center=lat_c, lon_center=lon_c,
            elevation_threshold=elev_threshold, dem_query_metadata=dem_query_metadata,
            dict_image_obs=False, max_steps=1000)
      else:
        env = UAVFireEnv(
            fire_points=cluster, radius=radius, num_nearest=num_nearest, return_dict_obs=True,
            algorithm_name='DREAMER', env_name='Circle1',
            dict_image_obs=False, max_steps=1000)
      self._envs.append(env)
    self._num_uavs = len(self._envs)
    self._cluster_count = self._num_uavs
    self._active_env_idx = -1
    self._single_round_robin = bool(single_round_robin)
    self._active_env = self._envs[0]
    self._env = self._envs[0]
    self._single_action_dim = int(self._env.action_space.shape[0])
    self._single_vector_dim = int(self._env.observation_space['vector'].shape[0])
    # Force joint control for all tasks to keep transitions consistent.
    if task == 'circle1_single':
      print('[Dreamer UAVFire] Override circle1_single -> centralized joint control.')
    self._control_mode = 'centralized'
    self._single_round_robin = False
    self._global_radius = float(radius)
    self._global_fire_capacity = int(np.asarray(fire_points).shape[0])
    self._global_bird_capacity = int(
        sum(int(getattr(env, 'num_birds', 0)) for env in self._envs))
    self._use_joint_global_features = (
        self._control_mode == 'centralized' and self._num_uavs > 1)
    self._global_feature_dim = (
        (2 * self._num_uavs) +
        (3 * self._global_fire_capacity) +
        (3 * self._global_bird_capacity)
    ) if self._use_joint_global_features else 0
    self.num_agents = 1 if self._control_mode == 'single' else self._num_uavs
    self._last_infos = [{} for _ in range(self.num_agents)]
    if self._control_mode == 'single':
      cycle_desc = 'round-robin' if self._single_round_robin else 'fixed-cluster'
      print(
          f'[Dreamer UAVFire] Non-cooperative clustered control enabled: '
          f'{self._cluster_count} single-UAV clusters ({cycle_desc}), '
          f'action_dim={self._single_action_dim}'
      )
    else:
      print(
          f'[Dreamer UAVFire] Centralized control enabled: '
          f'{self._num_uavs} UAVs, action_dim={self._single_action_dim * self._num_uavs}'
      )
    log_dir = os.path.join(UAV_DIR, 'logs')
    scenario_name = 'Circle8' if task == 'circle8' else 'Circle1'
    self._episode_logger = EpisodeCSVLogger('DREAMER', scenario_name, log_dir)
    print(f'[Dreamer UAVFire] Standard CSV log: {os.path.abspath(self._episode_logger.path)}')

  @property
  def obs_space(self):
    if self._control_mode == 'single':
      vector_shape = (self._single_vector_dim,)
    else:
      vector_shape = (self._single_vector_dim * self._num_uavs + self._global_feature_dim,)
    return {
        'vector': elements.Space(np.float32, vector_shape, -1.0, 1.0),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    action_dim = self._single_action_dim if self._control_mode == 'single' else self._single_action_dim * self._num_uavs
    return {
        'action': elements.Space(np.float32, (action_dim,), -1.0, 1.0),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      self._episode_reward = 0.0
      self._episode_steps = 0
      obs = self._reset_single() if self._control_mode == 'single' else self._reset_all()
      return self._obs(obs, 0.0, is_first=True)
    if self._control_mode == 'single':
      sub_action = self._split_actions(action['action'])
      out = self._active_env.step(sub_action)
      if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
      else:
        obs, reward, done, info = out
      reward = float(reward)
      done = bool(done)
      self._last_infos = [info]
      self._info = self._merge_infos(self._last_infos)
      self._info['active_cluster_index'] = int(self._active_env_idx)
      self._info['num_clusters'] = int(self._cluster_count)
    else:
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
      coverage_done = bool(self._info.get('coverage_rate', 0.0) >= 1.0 - _COVERAGE_COMPLETION_EPS)
      collision_done = bool(self._info.get('collision', False))
      timeout_done = bool(np.all(dones))
      done = bool(coverage_done or collision_done or timeout_done)
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
    packed_obs = obs if self._control_mode == 'single' else obs_list
    return self._obs(packed_obs, reward, is_last=done, is_terminal=done)

  def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    if isinstance(obs, list):
      vector = np.concatenate(
          [np.asarray(item['vector'], dtype=np.float32) for item in obs], axis=0)
      if self._use_joint_global_features:
        vector = np.concatenate([vector, self._build_joint_global_features()], axis=0)
    else:
      vector = np.asarray(obs['vector'], dtype=np.float32)
    vector = np.clip(vector, -1.0, 1.0).astype(np.float32)
    return {
        'vector': vector,
        'reward': np.float32(reward),
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal,
    }

  def _build_joint_global_features(self):
    if not self._use_joint_global_features:
      return np.zeros((0,), dtype=np.float32)
    radius = max(self._global_radius, 1.0)
    # 1) all UAV positions
    uav_positions = []
    for env in self._envs:
      pos = np.asarray(getattr(env, 'pos', np.zeros(2, dtype=np.float32)), dtype=np.float32)
      uav_positions.append(np.clip(pos / radius, -1.0, 1.0))
    uav_positions = np.concatenate(uav_positions, axis=0) if uav_positions else np.zeros((0,), np.float32)
    # 2) all remaining fire points (padded) + validity mask
    fire_xy = np.zeros((self._global_fire_capacity, 2), dtype=np.float32)
    fire_mask = np.zeros((self._global_fire_capacity,), dtype=np.float32)
    cursor = 0
    for env in self._envs:
      pts = np.asarray(getattr(env, 'fire_points', np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
      vis = np.asarray(getattr(env, 'visited', np.zeros((len(pts),), dtype=bool)), dtype=bool)
      remain = pts[~vis] if len(pts) else np.zeros((0, 2), dtype=np.float32)
      for point in remain:
        if cursor >= self._global_fire_capacity:
          break
        fire_xy[cursor] = np.clip(point / radius, -1.0, 1.0)
        fire_mask[cursor] = 1.0
        cursor += 1
      if cursor >= self._global_fire_capacity:
        break
    # 3) all dynamic bird positions (padded) + validity mask
    bird_xy = np.zeros((self._global_bird_capacity, 2), dtype=np.float32)
    bird_mask = np.zeros((self._global_bird_capacity,), dtype=np.float32)
    bcursor = 0
    for env in self._envs:
      birds = np.asarray(getattr(env, '_birds_pos', np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
      for bird in birds:
        if bcursor >= self._global_bird_capacity:
          break
        bird_xy[bcursor] = np.clip(bird / radius, -1.0, 1.0)
        bird_mask[bcursor] = 1.0
        bcursor += 1
      if bcursor >= self._global_bird_capacity:
        break
    return np.concatenate([
        uav_positions.astype(np.float32),
        fire_xy.reshape(-1).astype(np.float32),
        fire_mask.astype(np.float32),
        bird_xy.reshape(-1).astype(np.float32),
        bird_mask.astype(np.float32),
    ], axis=0)

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

  def _reset_single(self):
    if not self._envs:
      raise RuntimeError('No UAV environments available to reset.')
    if self._single_round_robin:
      self._active_env_idx = (self._active_env_idx + 1) % self._cluster_count
    else:
      self._active_env_idx = 0
    self._active_env = self._envs[self._active_env_idx]
    out = self._active_env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    self._last_infos = [{}]
    self._info = self._merge_infos(self._last_infos)
    self._info['active_cluster_index'] = int(self._active_env_idx)
    self._info['num_clusters'] = int(self._cluster_count)
    return obs

  def _split_actions(self, action):
    act = np.asarray(action, dtype=np.float32).reshape(-1)
    if self._control_mode == 'single':
      expected = self._single_action_dim
      if act.size != expected:
        raise ValueError(
            f'Expected single-UAV action size {expected} in non-cooperative mode, '
            f'got {act.size}.')
      return np.clip(act, -1.0, 1.0).astype(np.float32)
    expected = self._single_action_dim * self._num_uavs
    if act.size != expected:
      raise ValueError(
          f'Expected centralized action size {expected} for {self._num_uavs} UAV(s), '
          f'got {act.size}.')
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
    points = np.asarray(fire_points)
    if len(points) == 0:
      return [points]
    n_clusters = int(num_uavs)
    try:
      from sklearn.cluster import KMeans
      try:
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        labels = km.fit_predict(points)
      except TypeError:
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = km.fit_predict(points)
    except ImportError:
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
