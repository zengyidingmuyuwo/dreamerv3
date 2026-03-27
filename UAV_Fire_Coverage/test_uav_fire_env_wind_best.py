import os
import numpy as np

from uav_fire_env import UAVFireEnv


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
  np.random.normal = lambda loc=0.0, scale=1.0: 0.0
  try:
    env.step(np.array([0.0], dtype=np.float32))
  finally:
    np.random.normal = old_normal

  expected = np.array([env.STEP_SIZE, 5.0], dtype=np.float32)
  np.testing.assert_allclose(env.pos, expected, atol=1e-4)


def test_best_trajectory_saved_on_new_high_score():
  out = os.path.abspath('best_trajectory_record.png')
  if os.path.exists(out):
    os.remove(out)

  env = UAVFireEnv(
      fire_points=np.array([[0.0, 0.0]], dtype=np.float32),
      radius=5000.0,
  )
  env.reset(seed=0)
  env.pos = np.array([0.0, 0.0], dtype=np.float32)
  env.heading = 0.0

  old_normal = np.random.normal
  np.random.normal = lambda loc=0.0, scale=1.0: 0.0
  try:
    _, _, done, *_ = env.step(np.array([0.0], dtype=np.float32))
  finally:
    np.random.normal = old_normal

  assert done
  assert env._best_ep_score > -float('inf')
  assert os.path.exists(out)
  os.remove(out)

