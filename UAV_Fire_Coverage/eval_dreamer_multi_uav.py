#!/usr/bin/env python3
"""
Evaluate one single-UAV DreamerV3 policy over Circle1 3-way clusters and
overlay trajectories as pseudo multi-UAV execution.
"""

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_utils import load_circle_data
from uav_fire_env import UAVFireEnv


@dataclass
class EvalResult:
    fire_points: np.ndarray
    visited: np.ndarray
    trajectory: np.ndarray
    coverage: float


def cluster_fire_points(fire_points, n_clusters=3):
    points = np.asarray(fire_points, dtype=np.float32)
    if len(points) == 0:
        return [points]
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        labels = km.fit_predict(points)
    except ImportError:
        labels = np.arange(len(points)) % n_clusters
    clusters = [points[labels == idx] for idx in range(n_clusters)]
    clusters = [c for c in clusters if len(c) > 0]
    return clusters if clusters else [points]


def build_spaces(env):
    import elements
    obs_space = {
        "image": elements.Space(np.float32, env.observation_space["image"].shape, -1.0, 1.0),
        "vector": elements.Space(np.float32, env.observation_space["vector"].shape, -1.0, 1.0),
        "reward": elements.Space(np.float32),
        "is_first": elements.Space(bool),
        "is_last": elements.Space(bool),
        "is_terminal": elements.Space(bool),
    }
    act_space = {
        "action": elements.Space(np.float32, env.action_space.shape, -1.0, 1.0),
    }
    return obs_space, act_space


def load_agent(from_checkpoint, config_names, obs_space, act_space, task_name):
    import elements
    import ruamel.yaml as yaml
    from dreamerv3.agent import Agent

    config_path = os.path.join(ROOT, "dreamerv3", "configs.yaml")
    parsed = yaml.YAML(typ="safe").load(elements.Path(config_path).read())
    config = elements.Config(parsed["defaults"])
    for name in config_names:
        if not name:
            continue
        if name not in parsed:
            raise KeyError(f"Unknown config name: {name}")
        if name != "defaults":
            config = config.update(parsed[name])
    config = config.update(task=task_name)
    agent_cfg = elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    )
    agent = Agent(obs_space, act_space, agent_cfg)
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(from_checkpoint, keys=["agent"])
    return agent


def reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, rew, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, rew, done, info = out
        done = bool(done)
    return obs, float(rew), done, info


def run_single_cluster(agent, env):
    carry = agent.init_policy(1)
    obs = reset_env(env)
    is_first = True
    reward = 0.0
    for _ in range(env.MAX_STEPS):
        model_obs = {
            "image": np.asarray(obs["image"], np.float32)[None, ...],
            "vector": np.asarray(obs["vector"], np.float32)[None, ...],
            "reward": np.asarray([reward], np.float32),
            "is_first": np.asarray([is_first], bool),
            "is_last": np.asarray([False], bool),
            "is_terminal": np.asarray([False], bool),
        }
        carry, action, _ = agent.policy(carry, model_obs, mode="eval")
        act = np.asarray(action["action"], dtype=np.float32)[0]
        obs, reward, done, _ = step_env(env, act)
        is_first = False
        if done:
            break
    visited = env.visited.copy()
    coverage = float(np.sum(visited)) / max(len(visited), 1)
    traj = np.asarray(env._trajectory, dtype=np.float32)
    return EvalResult(
        fire_points=env.fire_points.copy(),
        visited=visited,
        trajectory=traj,
        coverage=coverage,
    )


def plot_results(results, radius, save_path, title):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(mpatches.Circle((0, 0), radius, fill=False, color="steelblue", lw=2))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    shown_unvisited = False
    shown_start = False
    total_visited = 0
    total_points = 0

    for i, result in enumerate(results):
        fp = result.fire_points
        vm = result.visited
        traj = result.trajectory
        total_visited += int(np.sum(vm))
        total_points += int(len(vm))

        unv = fp[~vm]
        vis = fp[vm]
        color = colors[i % len(colors)]
        if len(unv):
            ax.scatter(
                unv[:, 0], unv[:, 1], c="lightcoral", s=20, alpha=0.35, zorder=2,
                label="Unvisited" if not shown_unvisited else None,
            )
            shown_unvisited = True
        if len(vis):
            ax.scatter(
                vis[:, 0], vis[:, 1], c=color, s=28, marker="o", zorder=4,
                label=f"UAV{i + 1} Visited",
            )
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], color=color, lw=2.0, zorder=3, label=f"UAV{i + 1} Path")
        if len(traj):
            ax.scatter(
                traj[0, 0], traj[0, 1], c="black", marker="x", s=60, zorder=5,
                label="Start" if not shown_start else None,
            )
            shown_start = True

    coverage = float(total_visited) / max(float(total_points), 1.0)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title} | Coverage={coverage * 100:.2f}% ({total_visited}/{total_points})")
    ax.legend(loc="best")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return coverage, total_visited, total_points


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prepare_dir = os.path.join(script_dir, "prepare")
    parser = argparse.ArgumentParser(description="Evaluate Dreamer single-UAV policy on Circle1 clustered sectors.")
    parser.add_argument("--from_checkpoint", required=True, type=str, help="Dreamer checkpoint path to load.")
    parser.add_argument("--center_csv", default=os.path.join(prepare_dir, "circle_1_center.csv"), type=str)
    parser.add_argument("--points_file", default=os.path.join(prepare_dir, "circle_1_points.shp"), type=str)
    parser.add_argument("--circle_id", default=-1, type=int)
    parser.add_argument("--num_clusters", default=3, type=int)
    parser.add_argument("--num_nearest", default=6, type=int)
    parser.add_argument("--configs", default="defaults,uavfire", type=str,
                        help="Comma-separated dreamerv3 config names.")
    parser.add_argument("--task_name", default="uavfire_circle1", type=str,
                        help="Agent task tag used to build Dreamer config.")
    parser.add_argument("--save_path", default=os.path.join(script_dir, "trajectory_results", "eval_dreamer_multi_uav_circle1.png"), type=str)
    args = parser.parse_args()

    kwargs = {} if args.circle_id == -1 else {"circle_id": int(args.circle_id)}
    lat_c, lon_c, radius, fire_points = load_circle_data(args.center_csv, args.points_file, **kwargs)
    clusters = cluster_fire_points(fire_points, n_clusters=args.num_clusters)
    if len(clusters) != args.num_clusters:
        print(
            f"[Eval][Warn] Requested {args.num_clusters} clusters but got {len(clusters)} "
            f"(empty clusters were removed)."
        )
    print(f"[Eval] Circle1 center=({lat_c:.6f},{lon_c:.6f}) radius={radius:.1f}m")
    print("[Eval] Cluster sizes:", ", ".join(f"UAV{i+1}={len(c)}" for i, c in enumerate(clusters)))

    probe_env = UAVFireEnv(
        fire_points=clusters[0],
        radius=radius,
        num_nearest=args.num_nearest,
        return_dict_obs=True,
        algorithm_name="DREAMER",
        env_name="Circle1",
    )
    obs_space, act_space = build_spaces(probe_env)
    probe_env.close()

    config_names = [x.strip() for x in args.configs.split(",") if x.strip()]
    agent = load_agent(args.from_checkpoint, config_names, obs_space, act_space, args.task_name)
    print(f"[Eval] Loaded checkpoint: {args.from_checkpoint}")

    results = []
    for idx, cluster in enumerate(clusters):
        env = UAVFireEnv(
            fire_points=cluster,
            radius=radius,
            num_nearest=args.num_nearest,
            return_dict_obs=True,
            algorithm_name="DREAMER",
            env_name="Circle1",
        )
        result = run_single_cluster(agent, env)
        env.close()
        results.append(result)
        print(
            f"[Eval] UAV{idx+1}: coverage={result.coverage * 100:.2f}% "
            f"visited={int(np.sum(result.visited))}/{len(result.visited)}"
        )

    coverage, total_visited, total_points = plot_results(
        results, radius, args.save_path, "Dreamer Single-Brain Multi-Sector Evaluation (Circle1)"
    )
    print(f"[Eval] Total coverage={coverage * 100:.2f}% ({total_visited}/{total_points})")
    print(f"[Eval] Figure saved to: {os.path.abspath(args.save_path)}")


if __name__ == "__main__":
    main()
