"""
uav_fire_env.py — Gym environment for UAV fire-point coverage (Circle 1).

No obstacle avoidance.  Three UAVs are simulated by cycling through
K-means sub-cluster environments during training.

State space (23 dims):
  pos_x, pos_y        2   normalised position (÷ radius)
  sin(θ), cos(θ)      2   heading direction
  remaining_ratio     1   fraction of unvisited fire points
  dist_k  (k=1..6)    6   distance to k-th nearest unvisited fire point (norm.)
  sin(φ_k), cos(φ_k)  12  angle to k-th nearest unvisited fire point

Action space (1 dim):
  heading-rate ∈ [-1, 1]  scaled by MAX_TURN_RATE (0.25 rad/step)
"""

import numpy as np
import gym
from gym import spaces

K_NEAREST = 6          # number of nearest fire points tracked in state
SPEED = 20.0           # m/s
MAX_TURN_RATE = 0.25   # rad / step
DT = 1.0               # seconds per step
STEP_SIZE = SPEED * DT # 20 m
VISIT_RADIUS = 200.0   # m  — point is considered visited within this range
STEP_PENALTY = -0.05
VISIT_REWARD = 20.0
ALL_VISITED_BONUS = 200.0
SHAPING_SCALE = 0.1


class UAVFireEnv(gym.Env):
    """Single-UAV fire-point coverage environment (no obstacles)."""

    metadata = {'render.modes': []}

    def __init__(self, fire_points, radius_m, max_steps=2000):
        """
        Parameters
        ----------
        fire_points : np.ndarray  shape (N, 2)  local (x, y) metres
        radius_m    : float       circle radius used for normalisation
        max_steps   : int         episode length limit
        """
        super().__init__()
        self.fire_points = np.array(fire_points, dtype=np.float32)
        self.radius_m = float(radius_m)
        self.max_steps = max_steps
        self.n_points = len(self.fire_points)
        self.state_dim = 2 + 2 + 1 + K_NEAREST + 2 * K_NEAREST  # = 23

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self._visited = np.zeros(self.n_points, dtype=bool)
        self._steps = 0
        # Start at circle centre with random heading
        self._pos = np.zeros(2, dtype=np.float32)
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

        self._pos = new_pos
        self._steps += 1

        # Check visits
        reward = STEP_PENALTY
        for i in np.where(~self._visited)[0]:
            if np.linalg.norm(self.fire_points[i] - self._pos) <= VISIT_RADIUS:
                self._visited[i] = True
                reward += VISIT_REWARD

        all_done = self._visited.all()
        if all_done:
            reward += ALL_VISITED_BONUS

        # Distance-shaping reward
        new_nearest = self._nearest_dist()
        delta = self._prev_dist - new_nearest
        reward += SHAPING_SCALE * delta / STEP_SIZE
        self._prev_dist = new_nearest

        done = all_done or (self._steps >= self.max_steps)
        info = {'visited': int(self._visited.sum()), 'total': self.n_points}
        return self._get_obs(), float(reward), done, info

    # ------------------------------------------------------------------
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
        obs[4] = 1.0 - self._visited.mean()   # remaining ratio

        unvisited_idx = np.where(~self._visited)[0]
        if len(unvisited_idx) == 0:
            return obs  # all zeros for distances / angles

        unvisited_pts = self.fire_points[unvisited_idx]
        diffs = unvisited_pts - self._pos
        dists = np.linalg.norm(diffs, axis=1)
        k = min(K_NEAREST, len(unvisited_idx))
        nearest_k = np.argsort(dists)[:k]

        for i, ni in enumerate(nearest_k):
            d = dists[ni] / self.radius_m
            phi = np.arctan2(diffs[ni, 1], diffs[ni, 0])
            obs[5 + i] = d
            obs[5 + K_NEAREST + 2 * i] = np.sin(phi)
            obs[5 + K_NEAREST + 2 * i + 1] = np.cos(phi)

        return obs

    def render(self, mode='human'):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Multi-UAV wrapper
# ---------------------------------------------------------------------------

class MultiUAVFireEnv:
    """
    Simulates 3 UAVs on Circle 1 by cycling through 3 K-means sub-clusters.

    Usage:
        env = MultiUAVFireEnv(fire_points, radius_m)
        obs = env.reset()
        while True:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()   # automatically cycles to next sub-env
    """

    def __init__(self, fire_points, radius_m, n_uav=3, max_steps=2000):
        from sklearn.cluster import KMeans
        self.n_uav = n_uav
        km = KMeans(n_clusters=n_uav, random_state=0, n_init=10)
        labels = km.fit_predict(fire_points)
        self._envs = [
            UAVFireEnv(fire_points[labels == k], radius_m, max_steps)
            for k in range(n_uav)
        ]
        self._current = 0
        # expose spaces from the first sub-env
        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space

    @property
    def env(self):
        return self._envs[self._current]

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            self._current = (self._current + 1) % self.n_uav
        return obs, reward, done, info

    def close(self):
        for e in self._envs:
            e.close()
