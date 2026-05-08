#!/usr/bin/env python3
"""
Distributed Circle1 evaluation:
- Load a single-UAV Dreamer policy.
- Run full 3-UAV Circle1 environment.
- Feed each UAV local observation to the same single policy independently.
- Concatenate 3 actions and step centrally.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import elements
from dreamerv3.agent import Agent
from embodied.envs.uavfire import UAVFire


def _build_agent(from_checkpoint, config_names, task_name):
    import ruamel.yaml as yaml

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

    probe_env = UAVFire("circle1_single")
    obs_space = probe_env.obs_space
    act_space = {"action": probe_env.act_space["action"]}
    probe_env.close()

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


def _split_local_obs(obs, num_uavs):
    img = np.asarray(obs["image"], np.float32)
    vec = np.asarray(obs["vector"], np.float32)
    img_dim = int(img.shape[0] // num_uavs)
    vec_dim = int(vec.shape[0] // num_uavs)
    out = []
    for i in range(num_uavs):
        out.append({
            "image": img[i * img_dim:(i + 1) * img_dim],
            "vector": vec[i * vec_dim:(i + 1) * vec_dim],
        })
    return out


def _plot_trajectories(env, save_path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, ax = plt.subplots(figsize=(9, 9))
    radius = float(env._envs[0].radius)
    ax.add_patch(mpatches.Circle((0, 0), radius, fill=False, color="steelblue", lw=2))

    shown_unvisited = False
    shown_start = False
    total_visited, total_points = 0, 0
    for i, sub_env in enumerate(env._envs):
        fp = np.asarray(sub_env.fire_points, dtype=np.float32)
        vm = np.asarray(sub_env.visited, dtype=bool)
        tr = np.asarray(sub_env._trajectory, dtype=np.float32)
        total_visited += int(np.sum(vm))
        total_points += int(len(vm))
        color = colors[i % len(colors)]

        unv = fp[~vm]
        vis = fp[vm]
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
        if len(tr) > 1:
            ax.plot(tr[:, 0], tr[:, 1], color=color, lw=2.0, zorder=3, label=f"UAV{i + 1} Path")
        if len(tr):
            ax.scatter(
                tr[0, 0], tr[0, 1], c="black", marker="x", s=60, zorder=5,
                label="Start" if not shown_start else None,
            )
            shown_start = True

    coverage = float(total_visited) / max(float(total_points), 1.0)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Dreamer Distributed Circle1 | Coverage={coverage * 100:.2f}% ({total_visited}/{total_points})")
    ax.legend(loc="best")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close(fig)
    return coverage, total_visited, total_points


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Distributed 3-UAV Circle1 evaluation with single-UAV Dreamer brain.")
    parser.add_argument("--from_checkpoint", required=True, type=str, help="Dreamer checkpoint path.")
    parser.add_argument("--configs", default="defaults,uavfire_single", type=str,
                        help="Comma-separated config names.")
    parser.add_argument("--task_name", default="uavfire_circle1_single", type=str,
                        help="Task tag used to build the single-agent Dreamer config.")
    parser.add_argument("--save_path", default=os.path.join(script_dir, "trajectory_results", "eval_dreamer_distributed_circle1.png"), type=str)
    args = parser.parse_args()

    config_names = [x.strip() for x in args.configs.split(",") if x.strip()]
    agent = _build_agent(args.from_checkpoint, config_names, args.task_name)
    print(f"[Eval] Loaded single-UAV Dreamer checkpoint: {args.from_checkpoint}")

    env = UAVFire("circle1")
    num_uavs = int(env._num_uavs)
    action_dim = int(env.act_space["action"].shape[0])
    single_action_dim = int(action_dim // num_uavs)
    carries = [agent.init_policy(1) for _ in range(num_uavs)]

    obs = env.step({
        "reset": True,
        "action": np.zeros((action_dim,), dtype=np.float32),
    })
    reward = 0.0
    is_first = True
    steps = 0

    while True:
        local_obs = _split_local_obs(obs, num_uavs)
        actions = []
        for i in range(num_uavs):
            model_obs = {
                "image": np.asarray(local_obs[i]["image"], np.float32)[None, ...],
                "vector": np.asarray(local_obs[i]["vector"], np.float32)[None, ...],
                "reward": np.asarray([reward], np.float32),
                "is_first": np.asarray([is_first], bool),
                "is_last": np.asarray([False], bool),
                "is_terminal": np.asarray([False], bool),
            }
            carries[i], action, _ = agent.policy(carries[i], model_obs, mode="eval")
            act_i = np.asarray(action["action"], dtype=np.float32)[0].reshape(single_action_dim)
            actions.append(act_i)

        joint_action = np.concatenate(actions, axis=0).astype(np.float32)
        obs = env.step({"reset": False, "action": joint_action})
        reward = float(obs["reward"])
        is_first = False
        steps += 1
        if bool(obs["is_last"]):
            break
        if steps >= int(env._envs[0].MAX_STEPS):
            break

    coverage, visited, total = _plot_trajectories(env, args.save_path)
    env.close()
    print(f"[Eval] Steps={steps}")
    print(f"[Eval] Coverage={coverage * 100:.2f}% ({visited}/{total})")
    print(f"[Eval] Figure saved to: {os.path.abspath(args.save_path)}")


if __name__ == "__main__":
    main()
