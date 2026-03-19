"""
uav_fire_obstacle_env.py — Gym environment for UAV fire-point coverage
with obstacle avoidance (Circle 8).

Extends UAVFireEnv with:
  - 8 directional obstacle-distance sensors added to state (total 31 dims)
  - Collision penalty (terminal) and proximity penalty

Obstacles derive from elevation ≥ 2000 m in the GeoTIFF, or from a
synthetic grid when using sample data.
"""

import numpy as np
import gym
from gym import spaces

from uav_fire_env import (
    K_NEAREST, SPEED, MAX_TURN_RATE, DT, STEP_SIZE, VISIT_RADIUS,
    STEP_PENALTY, VISIT_REWARD, ALL_VISITED_BONUS, SHAPING_SCALE,
)

N_SENSORS = 8            # directional obstacle sensors
SENSOR_RANGE = 800.0     # metres — max sensing distance
COLLISION_PENALTY = -50.0
PROX_SCALE = -2.0
PROX_DECAY = 80.0        # metres


class UAVFireObstacleEnv(gym.Env):
    """Single-UAV fire-point coverage environment with obstacle avoidance."""

    metadata = {'render.modes': []}

    def __init__(self, fire_points, radius_m, obstacle_grid, cell_size,
                 max_steps=2000):
        """
        Parameters
        ----------
        fire_points   : np.ndarray  shape (N, 2)  local (x, y) metres
        radius_m      : float
        obstacle_grid : np.ndarray  shape (G, G)  dtype bool
                        True where terrain is impassable (elev ≥ 2000 m)
        cell_size     : float  metres per grid cell
        max_steps     : int
        """
        super().__init__()
        self.fire_points = np.array(fire_points, dtype=np.float32)
        self.radius_m = float(radius_m)
        self.obstacle_grid = obstacle_grid
        self.cell_size = float(cell_size)
        self.max_steps = max_steps
        self.n_points = len(self.fire_points)
        self.grid_size = obstacle_grid.shape[0]

        self.state_dim = 2 + 2 + 1 + K_NEAREST + 2 * K_NEAREST + N_SENSORS  # 31
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Pre-compute obstacle cell centres for fast distance queries
        iy_obs, ix_obs = np.where(obstacle_grid)
        if len(ix_obs) > 0:
            self._obs_cx = (ix_obs + 0.5) * cell_size - radius_m
            self._obs_cy = (iy_obs + 0.5) * cell_size - radius_m
            self._obs_centres = np.stack(
                [self._obs_cx, self._obs_cy], axis=1).astype(np.float32)
        else:
            self._obs_centres = np.zeros((0, 2), dtype=np.float32)

        self._rng = np.random.default_rng()
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self._visited = np.zeros(self.n_points, dtype=bool)
        self._steps = 0
        # Start at a random position inside the circle that is NOT an obstacle
        while True:
            angle = self._rng.uniform(0, 2 * np.pi)
            r = self._rng.uniform(0, self.radius_m * 0.3)
            pos = np.array([r * np.cos(angle), r * np.sin(angle)], np.float32)
            if not self._is_obstacle(pos):
                break
        self._pos = pos
        self._heading = self._rng.uniform(0, 2 * np.pi)
        self._prev_dist = self._nearest_dist()
        return self._get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).flatten()
        dtheta = float(np.clip(action[0], -1.0, 1.0)) * MAX_TURN_RATE
        self._heading += dtheta

        dx = STEP_SIZE * np.cos(self._heading)
        dy = STEP_SIZE * np.sin(self._heading)
        new_pos = self._pos + np.array([dx, dy], np.float32)

        # Soft boundary: project back inside circle
        dist_from_centre = np.linalg.norm(new_pos)
        if dist_from_centre > self.radius_m:
            new_pos = new_pos / dist_from_centre * self.radius_m

        reward = STEP_PENALTY
        terminated = False

        # Obstacle collision check
        if self._is_obstacle(new_pos):
            reward += COLLISION_PENALTY
            terminated = True
        else:
            self._pos = new_pos

        # Proximity penalty (continuous)
        min_obs_dist = self._min_obstacle_dist()
        if min_obs_dist < SENSOR_RANGE:
            reward += PROX_SCALE * np.exp(-min_obs_dist / PROX_DECAY)

        self._steps += 1

        # Check visits
        for i in np.where(~self._visited)[0]:
            if np.linalg.norm(self.fire_points[i] - self._pos) <= VISIT_RADIUS:
                self._visited[i] = True
                reward += VISIT_REWARD

        all_done = self._visited.all()
        if all_done:
            reward += ALL_VISITED_BONUS

        # Distance shaping
        new_nearest = self._nearest_dist()
        delta = self._prev_dist - new_nearest
        reward += SHAPING_SCALE * delta / STEP_SIZE
        self._prev_dist = new_nearest

        done = all_done or terminated or (self._steps >= self.max_steps)
        info = {
            'visited': int(self._visited.sum()),
            'total': self.n_points,
            'collision': terminated,
        }
        return self._get_obs(), float(reward), done, info

    # ------------------------------------------------------------------
    def _pos_to_grid(self, pos):
        """Convert local (x, y) metres to integer grid indices (ix, iy)."""
        ix = int((pos[0] + self.radius_m) / self.cell_size)
        iy = int((pos[1] + self.radius_m) / self.cell_size)
        ix = np.clip(ix, 0, self.grid_size - 1)
        iy = np.clip(iy, 0, self.grid_size - 1)
        return ix, iy

    def _is_obstacle(self, pos):
        ix, iy = self._pos_to_grid(pos)
        return bool(self.obstacle_grid[iy, ix])

    def _min_obstacle_dist(self):
        """Minimum distance to any obstacle cell (vectorised)."""
        if len(self._obs_centres) == 0:
            return SENSOR_RANGE
        diffs = self._obs_centres - self._pos
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        return float(np.min(dists))

    def _sensor_readings(self):
        """Return N_SENSORS distance readings, one per evenly-spaced direction."""
        readings = np.ones(N_SENSORS, dtype=np.float32) * SENSOR_RANGE
        for s in range(N_SENSORS):
            angle = self._heading + s * (2 * np.pi / N_SENSORS)
            for step in range(1, int(SENSOR_RANGE / STEP_SIZE) + 1):
                probe = self._pos + step * STEP_SIZE * np.array(
                    [np.cos(angle), np.sin(angle)], np.float32)
                if np.linalg.norm(probe) > self.radius_m or self._is_obstacle(probe):
                    readings[s] = step * STEP_SIZE
                    break
        return readings / SENSOR_RANGE   # normalised to [0, 1]

    def _nearest_dist(self):
        unvisited = self.fire_points[~self._visited]
        if len(unvisited) == 0:
            return 0.0
        return float(np.min(np.linalg.norm(unvisited - self._pos, axis=1)))

    def _get_obs(self):
        obs = np.zeros(self.state_dim, dtype=np.float32)
        obs[0] = self._pos[0] / self.radius_m
        obs[1] = self._pos[1] / self.radius_m
        obs[2] = np.sin(self._heading)
        obs[3] = np.cos(self._heading)
        obs[4] = 1.0 - self._visited.mean()

        unvisited_idx = np.where(~self._visited)[0]
        base = 5
        if len(unvisited_idx) > 0:
            unvisited_pts = self.fire_points[unvisited_idx]
            diffs = unvisited_pts - self._pos
            dists = np.linalg.norm(diffs, axis=1)
            k = min(K_NEAREST, len(unvisited_idx))
            nearest_k = np.argsort(dists)[:k]
            for i, ni in enumerate(nearest_k):
                d = dists[ni] / self.radius_m
                phi = np.arctan2(diffs[ni, 1], diffs[ni, 0])
                obs[base + i] = d
                obs[base + K_NEAREST + 2 * i] = np.sin(phi)
                obs[base + K_NEAREST + 2 * i + 1] = np.cos(phi)

        # Obstacle sensor readings (normalised)
        sensor_base = base + K_NEAREST + 2 * K_NEAREST  # = 23
        obs[sensor_base: sensor_base + N_SENSORS] = self._sensor_readings()
        return obs

    def render(self, mode='human'):
        pass

    def close(self):
        pass
