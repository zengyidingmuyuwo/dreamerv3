"""
UAV fire-coverage environment (no obstacles).

A single fixed-wing UAV must visit every fire point inside a circular region.
The environment follows the OpenAI Gym interface and is compatible with both
PPO and SAC implementations in this repository.

Fixed-wing UAV model
--------------------
- Constant forward speed ``UAV_SPEED`` m/s
- Heading updated by action: Δθ = action × MAX_TURN_RATE  (rad/step)
- Position updated by: Δpos = STEP_SIZE × [cos θ, sin θ]

State vector (23-dimensional by default with num_nearest=6)
-----------------------------------------------------------
 [0]  pos_x        normalised by region radius, range ~[-1, 1]
 [1]  pos_y        normalised by region radius, range ~[-1, 1]
 [2]  sin(heading)
 [3]  cos(heading)
 [4]  remaining_ratio   fraction of unvisited fire points, [0, 1]
 [5..5+3k)  for each of the k nearest unvisited fire points:
            dist_k  (normalised, [0, 1])
            sin(angle_k)
            cos(angle_k)

Guidance vector (4-dimensional)
-------------------------------
Relative position of current and lookahead waypoint:
  [cur_dx, cur_dy, next_dx, next_dy], each normalised by region radius.

Action space
------------
Scalar continuous: Δθ ∈ [-1, 1]  (scaled by MAX_TURN_RATE inside step())
"""

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_TUPLE_5 = True   # gymnasium step() returns (obs, rew, terminated, truncated, info)
except ImportError:
    import gym
    from gym import spaces
    _GYM_TUPLE_5 = False  # classic gym step() returns (obs, rew, done, info)

from global_planner import DronePlanner


class UAVFireEnv(gym.Env):
    """Single fixed-wing UAV fire-point coverage environment."""

    metadata = {'render.modes': ['human']}

    # ── UAV physics ──────────────────────────────────────────────────────────
    UAV_SPEED    = 20.0    # m/s  (typical small fixed-wing)
    MAX_TURN_RATE = 0.25   # rad/step  → min-turn-radius ≈ 80 m at 20 m/s
    DT           = 1.0     # s per step
    STEP_SIZE    = UAV_SPEED * DT  # metres per step

    # ── Task parameters ───────────────────────────────────────────────────────
    VISIT_RADIUS  = 200.0  # m  — fire point considered "visited" within this range
    MAX_STEPS     = 3000   # maximum steps per episode

    # ── Rewards ───────────────────────────────────────────────────────────────
    REWARD_STEP       = -1.0    # per-step energy/time penalty to discourage orbiting
    REWARD_VISIT      = 100.0   # per fire point visited
    REWARD_COMPLETE   = 200.0   # bonus for visiting all fire points
    PENALTY_BOUNDARY  = 0.0     # no per-step penalty; UAV is projected back into
                                # the circle which is sufficient boundary enforcement
    REWARD_APPROACH   = 0.1     # potential-based shaping coefficient: reward
                                # proportional to reduction in distance to the
                                # nearest unvisited fire point (normalised by
                                # STEP_SIZE so one straight-line approach step
                                # yields exactly REWARD_APPROACH)
    REWARD_WAYPOINT_POTENTIAL = 0.2
    REWARD_WAYPOINT_REACHED  = 10.0
    PENALTY_OFF_PATH         = 0.0
    OFF_PATH_DIST_M          = 600.0
    WAYPOINT_REACH_M         = 80.0
    WAYPOINT_TIMEOUT_NEAR_M  = 200.0
    WAYPOINT_TIMEOUT_STEPS   = 300
    WAYPOINT_DIVERGE_EPS     = 1e-6
    HARD_BOUNDARY_FACTOR     = 2.0
    MAX_ALLOWED_RADIUS_M     = 2_000_000.0

    def __init__(self, fire_points, radius, num_nearest=6, return_dict_obs=False):
        """
        Parameters
        ----------
        fire_points : array-like, shape (N, 2)
            Fire point locations in local metric coordinates (metres from
            circle centre).  The UAV starts at (0, 0).
        radius : float
            Radius of the circular operational region (metres).
        num_nearest : int
            Number of nearest unvisited fire points included in the state.
        """
        super(UAVFireEnv, self).__init__()

        self.fire_points  = np.asarray(fire_points, dtype=np.float32)
        self.radius       = float(radius)
        self.num_nearest  = int(num_nearest)
        self.return_dict_obs = bool(return_dict_obs)
        self.n_fire       = len(self.fire_points)
        self._validate_coordinate_scale()

        # State dimension: 5 base + 3 per nearest fire point
        self.state_dim = 5 + self.num_nearest * 3
        self.vector_dim = 4

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        if self.return_dict_obs:
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32),
                'vector': spaces.Box(low=-1.0, high=1.0, shape=(self.vector_dim,), dtype=np.float32),
            })
        else:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.state_dim + self.vector_dim,),
                dtype=np.float32
            )

        # Internal state (initialised in reset)
        self.pos     = np.zeros(2, dtype=np.float32)
        self.heading = 0.0
        self.visited = np.zeros(self.n_fire, dtype=bool)
        self.step_count = 0
        self._done = False
        self._trajectory = []   # for rendering
        self._planner = DronePlanner(obstacle_map=None, resolution_m=50.0)
        self.waypoints = np.zeros((0, 2), dtype=np.float32)
        self.current_waypoint_idx = 0
        self.current_waypoint = None
        self._global_plan_path = np.zeros((0, 2), dtype=np.float32)
        self.steps_since_last_waypoint = 0

    def _validate_coordinate_scale(self):
        if self.n_fire == 0:
            return
        norms = np.linalg.norm(self.fire_points.astype(np.float64), axis=1)
        p95 = float(np.percentile(norms, 95))
        if (self.radius <= 0) or (self.radius > self.MAX_ALLOWED_RADIUS_M):
            raise ValueError(
                f"radius={self.radius} 异常，疑似坐标系不一致（应为米尺度局部坐标）。"
            )
        if p95 > self.MAX_ALLOWED_RADIUS_M:
            raise ValueError(
                f"fire_points 尺度异常（p95={p95:.1f}），疑似把经纬度/UTM混用到了环境输入。"
            )

    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # ── Start near the centroid of fire points (much faster than always
        #    starting at the origin, which can be far from all fire points) ───
        centroid = (self.fire_points.mean(axis=0).astype(np.float32)
                    if self.n_fire > 0 else np.zeros(2, dtype=np.float32))
        jitter_r = min(self.VISIT_RADIUS * 2.0, self.radius * 0.25)
        angle    = np.random.uniform(0.0, 2.0 * np.pi)
        dist     = np.random.uniform(0.0, jitter_r)
        self.pos = centroid + dist * np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float32)
        # Clamp into the circular region
        d = float(np.linalg.norm(self.pos))
        if d > self.radius * 0.9:
            self.pos = self.pos * (self.radius * 0.9 / d)

        self.heading = np.random.uniform(0.0, 2.0 * np.pi)
        self.visited    = np.zeros(self.n_fire, dtype=bool)
        self.step_count = 0
        self._done      = False
        self._trajectory = [self.pos.copy()]
        self.steps_since_last_waypoint = 0
        self._prev_min_dist = self._min_dist_to_nearest()  # for shaping
        self._plan_waypoints()
        obs = self._get_obs()
        if _GYM_TUPLE_5:
            return obs, {}
        return obs

    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action):
        if self._done:
            if _GYM_TUPLE_5:
                return self._get_obs(), 0.0, True, False, {}
            return self._get_obs(), 0.0, True, {}

        # ── Update heading and position ──────────────────────────────────────
        delta = float(np.asarray(action).flat[0])
        delta = np.clip(delta, -1.0, 1.0) * self.MAX_TURN_RATE
        self.heading = (self.heading + delta) % (2.0 * np.pi)
        self.pos = self.pos + self.STEP_SIZE * np.array(
            [np.cos(self.heading), np.sin(self.heading)], dtype=np.float32
        )
        hard_out = float(np.linalg.norm(self.pos)) > self.radius * self.HARD_BOUNDARY_FACTOR
        self._trajectory.append(self.pos.copy())
        self.step_count += 1

        # ── Reward bookkeeping ───────────────────────────────────────────────
        reward  = self.REWARD_STEP
        n_before = int(np.sum(self.visited))
        reward  += self._check_visits()
        reward  += self._shaping_reward(n_before)
        reward  += self._boundary_penalty()
        reward  += self._waypoint_reward()
        off_path = self._is_off_path()

        # ── Termination ──────────────────────────────────────────────────────
        done = self._done or hard_out or bool(np.all(self.visited)) or (self.step_count >= self.MAX_STEPS)
        if np.all(self.visited):
            reward += self.REWARD_COMPLETE
        self._done = done

        info = {
            'visited_count': int(np.sum(self.visited)),
            'total_fire_points': self.n_fire,
            'step': self.step_count,
            'coverage_rate': float(np.sum(self.visited)) / self.n_fire,
            'off_path': bool(off_path),
        }
        if _GYM_TUPLE_5:
            return self._get_obs(), float(reward), done, False, info
        return self._get_obs(), float(reward), done, info

    # ─────────────────────────────────────────────────────────────────────────

    def _check_visits(self):
        """Check newly visited fire points and return visit rewards."""
        reward = 0.0
        dists = np.linalg.norm(self.fire_points - self.pos, axis=1)
        newly_visited = (~self.visited) & (dists <= self.VISIT_RADIUS)
        self.visited |= newly_visited
        reward += self.REWARD_VISIT * float(np.sum(newly_visited))
        return reward

    def _boundary_penalty(self):
        """Project UAV back inside the circle (no per-step penalty)."""
        dist_from_centre = float(np.linalg.norm(self.pos))
        if dist_from_centre > self.radius:
            self.pos = self.pos * (self.radius / dist_from_centre)
            return self.PENALTY_BOUNDARY   # 0.0 by default
        return 0.0

    def _min_dist_to_nearest(self):
        """Distance (metres) to the nearest unvisited fire point, or 0 if all visited."""
        unvisited = np.where(~self.visited)[0]
        if len(unvisited) == 0:
            return 0.0
        return float(np.min(np.linalg.norm(self.fire_points[unvisited] - self.pos, axis=1)))

    def _shaping_reward(self, n_visited_before):
        """Potential-based shaping: reward for reducing distance to nearest fire point.

        Skips the step immediately after a visit (the "nearest fire point" jumps
        to a farther one, which is expected and should not be penalised).
        Updates ``self._prev_min_dist`` for the next step.
        """
        n_now   = int(np.sum(self.visited))
        new_min = self._min_dist_to_nearest()
        if n_now == n_visited_before:
            # No visit this step: apply approach shaping
            shaping = self.REWARD_APPROACH * (self._prev_min_dist - new_min) / self.STEP_SIZE
        else:
            shaping = 0.0  # reset baseline without penalising the jump
        self._prev_min_dist = new_min
        return float(shaping)

    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        """Build the state observation vector."""
        pos_norm = self.pos / self.radius   # normalised to ~[-1, 1]

        heading_feat = np.array([np.sin(self.heading), np.cos(self.heading)],
                                dtype=np.float32)

        remaining_ratio = float(np.sum(~self.visited)) / self.n_fire

        nearest_feat = self._nearest_fire_features()

        obs = np.concatenate([
            pos_norm,
            heading_feat,
            [remaining_ratio],
            nearest_feat,
        ]).astype(np.float32)
        vec = self._waypoint_vector()
        if self.return_dict_obs:
            return {'image': obs, 'vector': vec}
        return np.concatenate([obs, vec]).astype(np.float32)

    def _plan_waypoints(self):
        result = self._planner.plan(self.pos, self.fire_points)
        self.waypoints = result.waypoints
        self._global_plan_path = result.waypoints
        self.current_waypoint_idx = 0
        self.current_waypoint = (
            self.waypoints[0].copy() if len(self.waypoints) else self.pos.copy()
        )
        self._prev_wp_dist = self._dist_to_waypoint()
        self.steps_since_last_waypoint = 0

    def _dist_to_waypoint(self):
        if self.current_waypoint is None:
            return 0.0
        return float(np.linalg.norm(self.current_waypoint - self.pos))

    def _waypoint_vector(self):
        if self.current_waypoint is None:
            return np.zeros(4, dtype=np.float32)
        cur_rel = (self.current_waypoint - self.pos) / max(self.radius, 1.0)
        if self.current_waypoint_idx + 1 < len(self.waypoints):
            next_wp = self.waypoints[self.current_waypoint_idx + 1]
            next_rel = (next_wp - self.pos) / max(self.radius, 1.0)
        else:
            next_rel = np.zeros(2, dtype=np.float32)
        rel = np.concatenate([cur_rel, next_rel], axis=0)
        return np.clip(rel.astype(np.float32), -1.0, 1.0)

    def _advance_waypoint(self):
        if self.current_waypoint_idx + 1 < len(self.waypoints):
            self.current_waypoint_idx += 1
            self.current_waypoint = self.waypoints[self.current_waypoint_idx].copy()
            return True
        return False

    def _waypoint_reward(self):
        if self.current_waypoint is None:
            return 0.0
        old_dist = float(self._prev_wp_dist)
        cur = self._dist_to_waypoint()
        self.steps_since_last_waypoint += 1
        shaped = float(self.REWARD_WAYPOINT_POTENTIAL * (old_dist - cur))
        near_and_diverging = (
            (cur <= self.WAYPOINT_TIMEOUT_NEAR_M) and (cur > old_dist + self.WAYPOINT_DIVERGE_EPS)
        )
        timed_out = self.steps_since_last_waypoint > self.WAYPOINT_TIMEOUT_STEPS
        while (cur <= self.WAYPOINT_REACH_M) or near_and_diverging or timed_out:
            shaped += self.REWARD_WAYPOINT_REACHED
            moved = self._advance_waypoint()
            self.steps_since_last_waypoint = 0
            if not moved:
                self.current_waypoint = None
                cur = 0.0
                break
            cur = self._dist_to_waypoint()
            near_and_diverging = False
            timed_out = False
        if cur > self.OFF_PATH_DIST_M:
            shaped += self.PENALTY_OFF_PATH
        self._prev_wp_dist = cur
        return shaped

    def _is_off_path(self):
        return self.current_waypoint is not None and self._dist_to_waypoint() > self.OFF_PATH_DIST_M

    def _nearest_fire_features(self):
        """Return (dist, sin_angle, cos_angle) for the K nearest unvisited pts."""
        feat = np.zeros(self.num_nearest * 3, dtype=np.float32)
        unvisited_idx = np.where(~self.visited)[0]
        if len(unvisited_idx) == 0:
            return feat

        pts   = self.fire_points[unvisited_idx]
        diffs = pts - self.pos
        dists = np.linalg.norm(diffs, axis=1)

        k     = min(self.num_nearest, len(unvisited_idx))
        order = np.argsort(dists)[:k]

        for i, j in enumerate(order):
            d     = float(np.clip(dists[j] / self.radius, 0.0, 1.0))
            angle = float(np.arctan2(diffs[j, 1], diffs[j, 0]))
            feat[i * 3]     = d
            feat[i * 3 + 1] = np.sin(angle)
            feat[i * 3 + 2] = np.cos(angle)

        return feat

    # ─────────────────────────────────────────────────────────────────────────

    def render(self, mode='human'):
        """Matplotlib visualisation (non-blocking)."""
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

        # Boundary circle
        ax.add_patch(mpatches.Circle((0, 0), self.radius,
                                     fill=False, color='steelblue', lw=2))

        # Fire points
        unv = self.fire_points[~self.visited]
        vis = self.fire_points[self.visited]
        if len(unv): ax.scatter(unv[:, 0], unv[:, 1], c='red',   s=40, zorder=3, label='Unvisited')
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
        ax.set_title(f'UAV Fire Coverage  step={self.step_count}  '
                     f'visited={np.sum(self.visited)}/{self.n_fire}')
        self._fig.canvas.draw()
        plt.pause(0.001)

    def close(self):
        if hasattr(self, '_fig') and self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
