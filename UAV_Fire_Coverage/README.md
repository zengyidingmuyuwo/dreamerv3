# UAV Fire Coverage — PPO & SAC Path Planning

This module implements reinforcement-learning-based path planning for fixed-wing UAVs performing fire-point coverage missions.  Two tasks are supported:

| Task | Circle | UAVs | Obstacles |
|------|--------|------|-----------|
| Fire-point coverage | Circle 1 | 3 (one per cluster) | None |
| Fire-point coverage + obstacle avoidance | Circle 8 | 1 | Elevation ≥ 2000 m |

Both PPO and SAC algorithms are implemented for each task.

---

## File structure

```
UAV_Fire_Coverage/
├── data_utils.py               # Data loading & coordinate conversion
├── uav_fire_env.py             # Gym environment — fire coverage (no obstacles)
├── uav_fire_obstacle_env.py    # Gym environment — fire coverage + obstacle avoidance
├── ppo_uav_circle1.py          # PPO training — Circle 1 (3 UAVs)
├── ppo_uav_circle8.py          # PPO training — Circle 8 (obstacles)
├── sac_uav_circle1.py          # SAC training — Circle 1 (3 UAVs)
├── sac_uav_circle8.py          # SAC training — Circle 8 (obstacles)
└── sample_data/
    ├── circle_1_center.csv     # Sample circle-1 centre coordinates
    ├── circle_1_points.csv     # Sample circle-1 fire points
    ├── circle_8_center.csv     # Sample circle-8 centre coordinates
    └── circle_8_points.csv     # Sample circle-8 fire points
```

---

## Requirements

```bash
pip install numpy torch gym scikit-learn
# Optional — needed only for real data files:
pip install pyshp      # read .shp fire-point files
pip install rasterio   # read GeoTIFF elevation maps
```

---

## Quick start (sample data)

All four training scripts work out-of-the-box with **synthetic sample data** — no real data files are needed.

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
```

### Circle 8 — fire coverage + obstacle avoidance

```bash
python ppo_uav_circle8.py \
    --center_csv   "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_8_center.csv" \
    --points_file  "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_8_points.shp" \
    --elevation_tif "E:/lzd/fire data/各种图/数据完整的区域高程图.tif"

python sac_uav_circle8.py \
    --center_csv   "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_8_center.csv" \
    --points_file  "E:/lzd/python/贪心圆/111-copilot-process-fire-data-and-cluster/output/circle_8_points.shp" \
    --elevation_tif "E:/lzd/fire data/各种图/数据完整的区域高程图.tif"
```

### Fire-point CSV format

If `pyshp` is not available you can supply a CSV instead of a `.shp` file.
The file must have a `latitude` and `longitude` column (header is case-insensitive):

```csv
latitude,longitude
25.4812,101.1934
25.5023,101.2145
...
```

### Circle-centre CSV format

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
| Visit radius | 120 m |

### State space

**Circle 1** (23 dimensions):

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
| Each step | −0.01 |
| Fire point visited | +10 |
| All fire points visited | +100 |
| Boundary violation (soft) | −1 |
| Obstacle collision (Circle 8, terminal) | −50 |
| Proximity to obstacle (Circle 8) | −2 × exp(−d / 80) |

---

## Multi-UAV strategy (Circle 1)

The 60 fire points in Circle 1 are divided into **3 sub-clusters** using
K-means (one cluster per UAV).  A **single shared policy** is trained
across all three sub-cluster environments by cycling through them one
episode at a time.  During final evaluation the policy is applied
independently to each cluster, simulating three simultaneous UAVs.

---

## Saved models

Models are saved every `--save_interval` episodes (default: 200) and at
the end of training:

| Script | Default save directory |
|--------|----------------------|
| `ppo_uav_circle1.py` | `./ppo_circle1_model/` |
| `ppo_uav_circle8.py` | `./ppo_circle8_model/` |
| `sac_uav_circle1.py` | `./sac_circle1_model/` |
| `sac_uav_circle8.py` | `./sac_circle8_model/` |

To resume training from a saved model, add `--load`.
To change the save directory, use `--save_dir <path>`.

---

## Visualisation

Add `--render` to any training command to enable real-time matplotlib visualisation.
This is disabled by default for faster training.

---

## Key command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_episodes` | 2000 | Total training episodes |
| `--gamma` | 0.99 | Discount factor |
| `--lr` / `--lr_actor` | 3e-4 | Learning rate |
| `--batch_size` | 256 (SAC) / 64 (PPO) | Mini-batch size |
| `--num_uavs` | 3 | Number of UAV clusters (Circle 1 only) |
| `--elev_threshold` | 2000.0 | Elevation obstacle threshold in metres |
| `--render` | off | Enable live visualisation |
| `--load` | off | Resume from saved model |
