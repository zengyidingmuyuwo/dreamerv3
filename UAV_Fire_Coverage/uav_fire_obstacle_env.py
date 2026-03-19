"""
UAV fire-coverage environment with elevation-based obstacle avoidance.

Extends :class:`UAVFireEnv` by adding an obstacle layer derived from a
digital elevation model (DEM).  Pixels with elevation ≥ 2000 m are treated
as obstacles.

Additional state features (appended to the base state vector)
--------------------------------------------------------------
  8 × obstacle-distance sensors — one per cardinal / inter-cardinal direction,
  each normalised to [0, 1] by MAX_SENSOR_RANGE.
  Sensor angle offsets (relative to East):  0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

Additional rewards / penalties
-------------------------------
  PENALTY_COLLISION  (terminal)  — UAV enters an obstacle pixel
  PENALTY_PROXIMITY  — proportional to closeness to nearest obstacle
"""

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_TUPLE_5 = True
except ImportError:
    import gym
    from gym import spaces
    _GYM_TUPLE_5 = False

from uav_fire_env import UAVFireEnv


class UAVFireObstacleEnv(UAVFireEnv):
    """Single fixed-wing UAV fire-coverage + obstacle-avoidance environment."""

    NUM_SENSORS       = 8        # directional distance sensors
    MAX_SENSOR_RANGE  = 600.0    # metres
    RAY_STEP_M        = 20.0     # metres per ray-casting step

    PENALTY_COLLISION = -50.0
    PENALTY_PROXIMITY = -2.0     # multiplied by exp(-dist/scale)
    PROXIMITY_SCALE   = 80.0     # metres

    def __init__(self, fire_points, radius,
                 obstacle_map=None, resolution_m=50.0,
                 num_nearest=6):
        """
        Parameters
        ----------
        fire_points : array-like, shape (N, 2)
        radius : float
        obstacle_map : np.ndarray (H, W) bool, optional
            Binary obstacle grid.  ``True`` = obstacle (elevation ≥ 2000 m).
            If *None*, the environment has no obstacles (same as UAVFireEnv).
        resolution_m : float
            Side length of each obstacle-map pixel in metres (default 50).
        num_nearest : int
        """
        super(UAVFireObstacleEnv, self).__init__(
            fire_points=fire_points,
            radius=radius,
            num_nearest=num_nearest,
        )

        self.obstacle_map  = obstacle_map   # (H, W) bool or None
        self.resolution_m  = float(resolution_m)

        # Extend state dimension with NUM_SENSORS obstacle distances
        extra = self.NUM_SENSORS
        self.state_dim += extra

        self.observation_space = spaces.Box(
            low=np.concatenate([
                np.full(self.state_dim - extra, -1.0),
                np.zeros(extra),             # sensor distances in [0, 1]
            ]).astype(np.float32),
            high=np.ones(self.state_dim, dtype=np.float32),
        )

    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset; if the centroid-based start is inside an obstacle, fall back
        to the circle centre (0, 0) to avoid an immediate collision."""
        result = super().reset(seed=seed, options=options)
        if self._at_obstacle(self.pos):
            self.pos = np.zeros(2, dtype=np.float32)
            self._prev_min_dist = self._min_dist_to_nearest()
        return result

    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action):
        if self._done:
            if _GYM_TUPLE_5:
                return self._get_obs(), 0.0, True, False, {}
            return self._get_obs(), 0.0, True, {}

        # ── Physics (same as base) ───────────────────────────────────────────
        delta = float(np.asarray(action).flat[0])
        delta = np.clip(delta, -1.0, 1.0) * self.MAX_TURN_RATE
        self.heading = (self.heading + delta) % (2.0 * np.pi)
        self.pos = self.pos + self.STEP_SIZE * np.array(
            [np.cos(self.heading), np.sin(self.heading)], dtype=np.float32
        )
        self._trajectory.append(self.pos.copy())
        self.step_count += 1

        # ── Collision check ──────────────────────────────────────────────────
        reward = self.REWARD_STEP
        if self._at_obstacle(self.pos):
            reward += self.PENALTY_COLLISION
            self._done = True
            info = {
                'visited_count': int(np.sum(self.visited)),
                'total_fire_points': self.n_fire,
                'step': self.step_count,
                'coverage_rate': float(np.sum(self.visited)) / self.n_fire,
                'collision': True,
            }
            if _GYM_TUPLE_5:
                return self._get_obs(), float(reward), True, False, info
            return self._get_obs(), float(reward), True, info

        # ── Proximity penalty ────────────────────────────────────────────────
        min_dist = self._min_obstacle_dist()
        if min_dist < self.PROXIMITY_SCALE * 3:
            reward += self.PENALTY_PROXIMITY * np.exp(-min_dist / self.PROXIMITY_SCALE)

        # ── Visit, shaping & boundary (same as base) ─────────────────────────
        n_before = int(np.sum(self.visited))
        reward  += self._check_visits()
        reward  += self._shaping_reward(n_before)
        reward  += self._boundary_penalty()

        done = bool(np.all(self.visited)) or (self.step_count >= self.MAX_STEPS)
        if np.all(self.visited):
            reward += self.REWARD_COMPLETE
        self._done = done

        info = {
            'visited_count': int(np.sum(self.visited)),
            'total_fire_points': self.n_fire,
            'step': self.step_count,
            'coverage_rate': float(np.sum(self.visited)) / self.n_fire,
            'collision': False,
        }
        if _GYM_TUPLE_5:
            return self._get_obs(), float(reward), done, False, info
        return self._get_obs(), float(reward), done, info

    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        base_obs = super()._get_obs()
        sensors  = self._obstacle_sensors()
        return np.concatenate([base_obs, sensors]).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────

    def _obstacle_sensors(self):
        """Compute NUM_SENSORS normalised obstacle-distance readings."""
        sensors = np.ones(self.NUM_SENSORS, dtype=np.float32)
        if self.obstacle_map is None:
            return sensors
        for i in range(self.NUM_SENSORS):
            angle = i * (2.0 * np.pi / self.NUM_SENSORS)  # absolute angle from East
            d = self._ray_cast(self.pos, angle)
            sensors[i] = float(np.clip(d / self.MAX_SENSOR_RANGE, 0.0, 1.0))
        return sensors

    def _ray_cast(self, start, angle):
        """Return distance (metres) to the nearest obstacle along *angle*."""
        if self.obstacle_map is None:
            return self.MAX_SENSOR_RANGE
        dx = np.cos(angle) * self.RAY_STEP_M
        dy = np.sin(angle) * self.RAY_STEP_M
        pos = start.astype(float).copy()
        for step in range(int(self.MAX_SENSOR_RANGE / self.RAY_STEP_M)):
            pos[0] += dx
            pos[1] += dy
            if self._at_obstacle(pos):
                return step * self.RAY_STEP_M
        return self.MAX_SENSOR_RANGE

    def _at_obstacle(self, pos):
        """Return True if *pos* (local metric) falls inside an obstacle cell."""
        if self.obstacle_map is None:
            return False
        H, W = self.obstacle_map.shape
        cx, cy = W // 2, H // 2
        j = int(cx + pos[0] / self.resolution_m)
        i = int(cy - pos[1] / self.resolution_m)   # y-axis is north (up)
        if 0 <= i < H and 0 <= j < W:
            return bool(self.obstacle_map[i, j])
        return False

    def _min_obstacle_dist(self):
        """Minimum distance to any obstacle in the 8 sensor directions."""
        if self.obstacle_map is None:
            return self.MAX_SENSOR_RANGE
        sensors = self._obstacle_sensors()
        return float(np.min(sensors) * self.MAX_SENSOR_RANGE)

    # ─────────────────────────────────────────────────────────────────────────

    def render(self, mode='human'):
        """Visualise environment including obstacle map."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            return

        if not hasattr(self, '_fig') or self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(7, 7))
            plt.ion()

        ax = self._ax
        ax.clear()

        # Obstacle map background
        if self.obstacle_map is not None:
            H, W = self.obstacle_map.shape
            ext = [-W // 2 * self.resolution_m, W // 2 * self.resolution_m,
                   -H // 2 * self.resolution_m, H // 2 * self.resolution_m]
            ax.imshow(self.obstacle_map, cmap='Reds', alpha=0.35,
                      extent=ext, origin='upper', zorder=0)

        # Boundary circle
        ax.add_patch(mpatches.Circle((0, 0), self.radius,
                                     fill=False, color='steelblue', lw=2))

        # Fire points
        unv = self.fire_points[~self.visited]
        vis = self.fire_points[self.visited]
        if len(unv): ax.scatter(unv[:, 0], unv[:, 1], c='red',      s=40, zorder=3, label='Unvisited')
        if len(vis): ax.scatter(vis[:, 0], vis[:, 1], c='limegreen', s=40, zorder=3, label='Visited')

        # Trajectory
        if len(self._trajectory) > 1:
            traj = np.array(self._trajectory)
            ax.plot(traj[:, 0], traj[:, 1], 'b-', lw=0.5, alpha=0.5)

        # UAV
        ax.scatter(*self.pos, c='blue', s=120, marker='^', zorder=5)
        aln = self.radius * 0.06
        ax.annotate('', xy=(self.pos[0] + aln * np.cos(self.heading),
                             self.pos[1] + aln * np.sin(self.heading)),
                    xytext=self.pos,
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))

        lim = self.radius * 1.15
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'UAV Fire+Obstacle  step={self.step_count}  '
                     f'visited={np.sum(self.visited)}/{self.n_fire}')
        self._fig.canvas.draw()
        plt.pause(0.001)