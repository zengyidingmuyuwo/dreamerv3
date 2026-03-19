# UAV Fire Coverage — PPO, SAC & DreamerV3 Path Planning

This module implements reinforcement-learning-based path planning for
fixed-wing UAVs performing fire-point coverage missions inside the
DreamerV3 repository.  **Three algorithms** are compared:

| Algorithm | Type | Script prefix |
|-----------|------|---------------|
| PPO | Model-free on-policy | `ppo_uav_*.py` |
| SAC | Model-free off-policy | `sac_uav_*.py` |
| DreamerV3 | Model-based (world model) | `dreamer_uav_*.py` |

Two tasks are supported:

| Task | Circle | UAVs | Obstacles |
|------|--------|------|-----------|
| Fire-point coverage | Circle 1 | 3 (one per cluster) | None |
| Fire-point coverage + obstacle avoidance | Circle 8 | 1 | Elevation ≥ 2000 m |

---

## File structure

```
UAV_Fire_Coverage/
├── data_utils.py                # Data loading & coordinate conversion
├── uav_fire_env.py              # Gym environment — fire coverage (no obstacles)
├── uav_fire_obstacle_env.py     # Gym environment — fire coverage + obstacle avoidance
├── ppo_uav_circle1.py           # PPO training — Circle 1 (3 UAVs)
├── ppo_uav_circle8.py           # PPO training — Circle 8 (obstacles)
├── sac_uav_circle1.py           # SAC training — Circle 1 (3 UAVs)
├── sac_uav_circle8.py           # SAC training — Circle 8 (obstacles)
├── dreamer_uav_circle1.py       # DreamerV3 training — Circle 1 (3 UAVs)
├── dreamer_uav_circle8.py       # DreamerV3 training — Circle 8 (obstacles)
└── sample_data/
    ├── circle_1_center.csv      # Sample circle-1 centre coordinates
    ├── circle_1_points.csv      # Sample circle-1 fire points (60 points)
    ├── circle_8_center.csv      # Sample circle-8 centre coordinates
    └── circle_8_points.csv      # Sample circle-8 fire points (30 points)
```

The `embodied.Env` adapter (for DreamerV3) lives at:

```
embodied/envs/uav_fire.py        # UAVFireEnvEmbodied — DreamerV3 interface
```

Config profiles for the two UAV tasks are in:

```
dreamerv3/configs.yaml           # uav_circle1 / uav_circle8 sections
```

---

## Requirements

```bash
# Core (PPO & SAC)
pip install numpy torch gym scikit-learn

# DreamerV3 (already in this repository)
pip install -r requirements.txt

# Optional — only for real data files
pip install pyshp      # read .shp fire-point files
pip install rasterio   # read GeoTIFF elevation maps
```

---

## Quick start (sample data)

All scripts work out-of-the-box with synthetic sample data — **no real data
files are needed**.

```bash
cd UAV_Fire_Coverage

# PPO — Circle 1 (3-UAV fire coverage)
python ppo_uav_circle1.py

# SAC — Circle 1 (3-UAV fire coverage)
python sac_uav_circle1.py

# PPO — Circle 8 (fire coverage + obstacles)
python ppo_uav_circle8.py

# SAC — Circle 8 (fire coverage + obstacles)
python sac_uav_circle8.py

# DreamerV3 — Circle 1
python dreamer_uav_circle1.py

# DreamerV3 — Circle 8
python dreamer_uav_circle8.py
```

---

## Using your real data files

### Circle 1 — fire coverage

```bash
python ppo_uav_circle1.py \
    --center_csv  "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_1_center.csv" \
    --points_file "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_1_points.shp"

python sac_uav_circle1.py \
    --center_csv  "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_1_center.csv" \
    --points_file "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_1_points.shp"

python dreamer_uav_circle1.py \
    --center_csv  "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_1_center.csv" \
    --points_file "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_1_points.shp"
```

### Circle 8 — fire coverage + obstacle avoidance

```bash
python ppo_uav_circle8.py \
    --center_csv    "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_8_center.csv" \
    --points_file   "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_8_points.shp" \
    --elevation_tif "E:/lzd/fire data/各种图/数据完整的区域高程图.tif"

python sac_uav_circle8.py \
    --center_csv    "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_8_center.csv" \
    --points_file   "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_8_points.shp" \
    --elevation_tif "E:/lzd/fire data/各种图/数据完整的区域高程图.tif"

python dreamer_uav_circle8.py \
    --center_csv    "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_8_center.csv" \
    --points_file   "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_8_points.shp" \
    --elevation_tif "E:/lzd/fire data/各种图/数据完整的区域高程图.tif"
```

---

## Fire-point CSV format

If `pyshp` is not available you can supply a CSV instead of a `.shp` file.
The file must have a `latitude` and `longitude` column (header is
case-insensitive):

```csv
latitude,longitude
25.4812,101.1934
25.5023,101.2145
```

## Circle-centre CSV format

```csv
latitude,longitude,radius_m
25.5,101.2,5000.0
```

---

## Environment design

### UAV physics model

| Parameter | Value |
|-----------|-------|
| Speed | 20 m/s (constant) |
| Max turn rate | 0.25 rad/step (~14°/s) |
| Time step | 1 s |
| Step size | 20 m |
| Visit radius | 200 m |

### State space

**Circle 1 (23 dimensions):**

| Feature | Dim | Description |
|---------|-----|-------------|
| `pos_x`, `pos_y` | 2 | Normalised position (÷ radius) |
| `sin(θ)`, `cos(θ)` | 2 | Heading direction |
| `remaining_ratio` | 1 | Fraction of unvisited fire points |
| `dist_k` | 6 | Distance to k-th nearest unvisited fire point (normalised) |
| `sin(φ_k)`, `cos(φ_k)` | 12 | Angle to k-th nearest unvisited fire point |

**Circle 8** adds 8 directional obstacle-distance sensors (23 + 8 = **31 dimensions**).

### Action space

Single continuous action: heading-change rate ∈ [−1, 1], scaled by `MAX_TURN_RATE`.

### Reward function

| Event | Reward |
|-------|--------|
| Each step | −0.05 |
| Fire point visited | +20 |
| All fire points visited | +200 |
| Boundary violation (soft) | 0 (UAV projected back; no penalty) |
| Obstacle collision (Circle 8, terminal) | −50 |
| Proximity to obstacle (Circle 8) | −2 × exp(−d / 80) |
| Approaching nearest fire point | +0.1 × Δdist / step_size (shaping) |

---

## Multi-UAV strategy (Circle 1)

The 60 fire points in Circle 1 are divided into 3 sub-clusters using
K-means (one cluster per UAV).  A single shared policy is trained across
all three sub-cluster environments by cycling through them one episode at a
time.  During final evaluation the policy is applied independently to each
cluster, simulating three simultaneous UAVs.

---

## DreamerV3 world-model integration

The DreamerV3 agent learns a compact **world model** (RSSM) of the UAV
environment entirely from experience.  During training it:

1. **Encodes** vector observations into latent tokens via an MLP encoder.
2. **Updates** the RSSM recurrent state using the latent tokens and previous
   actions.
3. **Imagines** future trajectories of length `imag_length` steps by rolling
   out the RSSM under the current policy.
4. **Trains** actor and critic purely in imagination space using λ-returns.

This allows the agent to plan ahead without executing every action in the
real environment, making it significantly more **sample-efficient** than
PPO or SAC.

### Running DreamerV3 via the main CLI

You can also invoke training directly through the DreamerV3 CLI from the
repo root:

```bash
# Circle 1 (sample data)
python -m dreamerv3.main --configs defaults uav_circle1

# Circle 8 (real data via env vars)
UAV_CENTER_CSV=<path> UAV_POINTS_FILE=<path> UAV_ELEVATION_TIF=<path> \
    python -m dreamerv3.main --configs defaults uav_circle8
```

### Saved models

DreamerV3 saves checkpoints automatically to the `logdir` directory
(default: `./logdir/uav_circle1_<timestamp>/` or `./logdir/uav_circle8_<timestamp>/`).

PPO / SAC models are saved every `--save_interval` episodes (default: 200)
and at the end of training:

| Script | Default save directory |
|--------|----------------------|
| `ppo_uav_circle1.py` | `./ppo_circle1_model/` |
| `ppo_uav_circle8.py` | `./ppo_circle8_model/` |
| `sac_uav_circle1.py` | `./sac_circle1_model/` |
| `sac_uav_circle8.py` | `./sac_circle8_model/` |

To resume PPO/SAC training from a saved model, add `--load`.
To change the save directory, use `--save_dir`.

To resume DreamerV3 from a checkpoint, use `--load_ckpt <path>`.

---

## Comparing algorithms

After training, compare the episode reward curves across algorithms:

- **PPO** logs per-episode rewards to the terminal.
- **SAC** logs per-episode rewards to the terminal.
- **DreamerV3** writes `metrics.jsonl` and `scores.jsonl` to the logdir,
  viewable with:

```bash
python scores/view.py ./logdir/uav_circle1_*/scores.jsonl
```
