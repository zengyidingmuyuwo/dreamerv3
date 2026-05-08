import os
import numpy as np

from uav_fire_env import UAVFireEnv
from uav_fire_obstacle_env import UAVFireObstacleEnv


def test_step_adds_wind_displacement_with_time_pattern():
  env = UAVFireEnv(
      fire_points=np.array([[1000.0, 0.0]], dtype=np.float32),
      radius=5000.0,
  )
  env.reset(seed=0)
  env.pos = np.array([0.0, 0.0], dtype=np.float32)
  env.heading = 0.0
  env.step_count = 0

  old_normal = np.random.normal
  np.random.normal = lambda loc=0.0, scale=1.0, size=None: (
      np.zeros(size, dtype=np.float32) if size is not None else 0.0
  )
  try:
    env.step(np.array([0.0], dtype=np.float32))
  finally:
    np.random.normal = old_normal

  expected = np.array([
      env.STEP_SIZE + env.WIND_BASE_M_S,
      env.WIND_GUST_AMPLITUDE_M_S + env.WIND_SPATIAL_AMPLITUDE_M_S,
  ], dtype=np.float32)
  np.testing.assert_allclose(env.pos, expected, atol=1e-4)


def test_best_trajectory_saved_on_new_high_score():
  out_dir = os.path.abspath('trajectory_results')
  os.makedirs(out_dir, exist_ok=True)
  existing = set(os.listdir(out_dir))

  env = UAVFireEnv(
      fire_points=np.array([[0.0, 0.0]], dtype=np.float32),
      radius=5000.0,
      algorithm_name='PPO',
      env_name='Circle1',
  )
  env.reset(seed=0)
  env.pos = np.array([0.0, 0.0], dtype=np.float32)
  env.heading = 0.0

  old_normal = np.random.normal
  np.random.normal = lambda loc=0.0, scale=1.0, size=None: (
      np.zeros(size, dtype=np.float32) if size is not None else 0.0
  )
  try:
    _, _, done, *_ = env.step(np.array([0.0], dtype=np.float32))
  finally:
    np.random.normal = old_normal

  assert done
  assert env._best_ep_score > -float('inf')
  created = [name for name in os.listdir(out_dir) if name not in existing]
  best_files = [name for name in created if name == 'best_PPO_Circle1.png']
  assert best_files
  for name in created:
    os.remove(os.path.join(out_dir, name))


def test_max_steps_and_hard_boundary_relaxed():
  env = UAVFireEnv(
      fire_points=np.array([[3000.0, 0.0]], dtype=np.float32),
      radius=5000.0,
  )
  assert env.MAX_STEPS == 5000
  assert env.HARD_BOUNDARY_FACTOR == 3.0


def test_reset_starts_exactly_at_center_without_random_offset():
  env = UAVFireEnv(
      fire_points=np.array([[1000.0, 200.0], [1100.0, -150.0]], dtype=np.float32),
      radius=5000.0,
  )
  out1 = env.reset(seed=1)
  obs1 = out1[0] if isinstance(out1, tuple) else out1
  np.testing.assert_allclose(env.pos, np.array([0.0, 0.0], dtype=np.float32), atol=1e-8)
  out2 = env.reset(seed=999)
  obs2 = out2[0] if isinstance(out2, tuple) else out2
  np.testing.assert_allclose(env.pos, np.array([0.0, 0.0], dtype=np.float32), atol=1e-8)
  assert obs1 is not None and obs2 is not None


def test_obs_is_clipped_to_space_bounds_for_uavfire_env():
  env = UAVFireEnv(
      fire_points=np.array([[1000.0, 200.0], [1100.0, -150.0]], dtype=np.float32),
      radius=5000.0,
      return_dict_obs=True,
  )
  env.reset(seed=0)
  env.pos = np.array([env.radius * 1.2, -env.radius * 1.3], dtype=np.float32)
  obs = env._get_obs()
  assert np.min(obs['image']) >= -1.0
  assert np.max(obs['image']) <= 1.0
  assert np.min(obs['vector']) >= -1.0
  assert np.max(obs['vector']) <= 1.0


def test_obs_is_clipped_to_space_bounds_for_obstacle_env():
  env = UAVFireObstacleEnv(
      fire_points=np.array([[100.0, 100.0], [200.0, -50.0]], dtype=np.float32),
      radius=2000.0,
      obstacle_map=np.zeros((64, 64), dtype=bool),
      resolution_m=50.0,
      return_dict_obs=True,
  )
  env.reset(seed=0)
  env.pos = np.array([env.radius * 1.4, -env.radius * 1.5], dtype=np.float32)
  obs = env._get_obs()
  assert np.min(obs['image']) >= -1.0
  assert np.max(obs['image']) <= 1.0
  assert np.min(obs['vector']) >= -1.0
  assert np.max(obs['vector']) <= 1.0


def test_render_does_not_draw_red_bird_trail_lines():
  import matplotlib.axes
  env = UAVFireEnv(
      fire_points=np.array([[1000.0, 0.0]], dtype=np.float32),
      radius=5000.0,
      num_birds=2,
  )
  env.reset(seed=0)
  for _ in range(3):
    env.step(np.array([0.0], dtype=np.float32))
  old_plot = matplotlib.axes.Axes.plot
  red_trail_calls = {'count': 0}

  def wrapped_plot(self, *args, **kwargs):
    if kwargs.get('color') == 'red' and float(kwargs.get('alpha', 1.0)) <= 0.25:
      red_trail_calls['count'] += 1
    return old_plot(self, *args, **kwargs)

  matplotlib.axes.Axes.plot = wrapped_plot
  try:
    env.render()
  finally:
    matplotlib.axes.Axes.plot = old_plot
    env.close()
  assert red_trail_calls['count'] == 0


def test_render_does_not_draw_radar_dashed_circle():
  import matplotlib.axes
  import matplotlib.patches
  env = UAVFireEnv(
      fire_points=np.array([[1000.0, 0.0]], dtype=np.float32),
      radius=5000.0,
      num_birds=1,
  )
  env.reset(seed=0)
  old_add_patch = matplotlib.axes.Axes.add_patch
  dashed_radar_calls = {'count': 0}

  def wrapped_add_patch(self, patch):
    if isinstance(patch, matplotlib.patches.Circle) and patch.get_linestyle() == '--':
      dashed_radar_calls['count'] += 1
    return old_add_patch(self, patch)

  matplotlib.axes.Axes.add_patch = wrapped_add_patch
  try:
    env.render()
  finally:
    matplotlib.axes.Axes.add_patch = old_add_patch
    env.close()
  assert dashed_radar_calls['count'] == 0


def test_circle8_background_loads_preprocessed_mask_and_sets_clip_path():
  import tempfile
  import matplotlib.pyplot as plt

  env = UAVFireEnv(
      fire_points=np.array([[0.0, 0.0]], dtype=np.float32),
      radius=200.0,
      env_name='Circle8',
      num_birds=0,
  )
  mask = np.zeros((32, 32), dtype=np.uint8)
  mask[10:22, 10:22] = 1
  tmp = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
  tmp.close()
  np.save(tmp.name, mask)
  old_candidates = env._circle8_obstacle_mask_candidates
  env._circle8_obstacle_mask_candidates = lambda: [tmp.name]
  env._circle8_obstacle_mask = None
  env._circle8_obstacle_mask_loaded = False

  fig, ax = plt.subplots(figsize=(4, 4))
  try:
    env._draw_background_layer(ax)
    assert len(ax.images) == 1
    assert ax.images[0].get_clip_path() is not None
  finally:
    env._circle8_obstacle_mask_candidates = old_candidates
    plt.close(fig)
    os.remove(tmp.name)
