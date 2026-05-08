import numpy as np
import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from uav_fire_env import UAVFireEnv


def test_waypoint_potential_reward_is_zero_sum_on_roundtrip():
  env = UAVFireEnv(
      fire_points=np.array([[1000.0, 0.0]], dtype=np.float32),
      radius=5000.0,
  )
  env.reset(seed=0)
  env.current_waypoint = np.array([500.0, 0.0], dtype=np.float32)

  env.pos = np.array([10.0, 0.0], dtype=np.float32)
  env._prev_wp_dist = 500.0
  r1 = env._waypoint_reward()

  env.pos = np.array([0.0, 0.0], dtype=np.float32)
  env._prev_wp_dist = 490.0
  r2 = env._waypoint_reward()

  assert r1 > 0.0
  assert r2 < 0.0
  assert abs((r1 + r2)) < 1e-6


def test_waypoint_near_diverge_triggers_forced_skip():
  env = UAVFireEnv(
      fire_points=np.array([[1000.0, 0.0], [2000.0, 0.0]], dtype=np.float32),
      radius=5000.0,
  )
  env.reset(seed=0)
  env.waypoints = np.array([[150.0, 0.0], [300.0, 0.0]], dtype=np.float32)
  env.current_waypoint_idx = 0
  env.current_waypoint = env.waypoints[0].copy()
  env.pos = np.array([0.0, 0.0], dtype=np.float32)
  env._prev_wp_dist = 140.0
  env.steps_since_last_waypoint = 7

  reward = env._waypoint_reward()

  assert env.current_waypoint_idx == 1
  assert env.steps_since_last_waypoint == 0
  assert reward >= (env.REWARD_WAYPOINT_REACHED - 2.1)
