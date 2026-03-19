"""
dreamer_uav_circle1.py — DreamerV3 (world-model) training for Circle 1.

Trains the DreamerV3 agent on the 3-UAV fire-point coverage task using
the RSSM world model.  The agent simultaneously learns:
  - A world model (encoder + RSSM dynamics + decoder)
  - An actor-critic policy in imagined latent rollouts

Quick start (sample data — no GPU needed):
    python dreamer_uav_circle1.py

Real data:
    python dreamer_uav_circle1.py \\
        --center_csv  <path>/circle_1_center.csv \\
        --points_file <path>/circle_1_points.shp \\
        --logdir      ./logdir/uav_circle1

Advanced — pass additional DreamerV3 flags after a double-dash:
    python dreamer_uav_circle1.py -- --run.steps 5e5 --batch_size 8

Data-path environment variables (alternative to CLI flags):
    UAV_CENTER_CSV=<path>  UAV_POINTS_FILE=<path>  python dreamer_uav_circle1.py
"""

import argparse
import os
import sys

# ── locate the repo root & dreamerv3 package ────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main():
    parser = argparse.ArgumentParser(
        description='DreamerV3 training — UAV Circle 1 fire coverage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Pass additional DreamerV3 config overrides after a double-dash:\n'
            '  python dreamer_uav_circle1.py -- --run.steps 5e5'
        ),
    )
    parser.add_argument(
        '--center_csv', default=None,
        help='Path to circle_1_center.csv  '
             '(default: UAV_CENTER_CSV env var or built-in sample data)',
    )
    parser.add_argument(
        '--points_file', default=None,
        help='Path to circle_1_points.shp / .csv  '
             '(default: UAV_POINTS_FILE env var or built-in sample data)',
    )
    parser.add_argument(
        '--logdir', default='./logdir/uav_circle1_{timestamp}',
        help='Directory for logs, checkpoints and metrics',
    )
    parser.add_argument(
        '--load_ckpt', default='',
        help='Path to a previous checkpoint to resume from',
    )
    parser.add_argument(
        '--steps', type=float, default=1e6,
        help='Total environment steps (default: 1 000 000)',
    )
    parser.add_argument(
        '--train_ratio', type=float, default=512.0,
        help='Train updates per environment step (default: 512)',
    )
    # Capture any remaining DreamerV3 flags
    args, extra = parser.parse_known_args()

    # ── propagate data-path choices via environment variables ───────────────
    if args.center_csv:
        os.environ.setdefault('UAV_CENTER_CSV', args.center_csv)
    if args.points_file:
        os.environ.setdefault('UAV_POINTS_FILE', args.points_file)

    # ── build the DreamerV3 argv list ────────────────────────────────────────
    dreamer_argv = [
        '--configs', 'defaults', 'uav_circle1',
        '--task', 'uav_circle1',
        '--logdir', args.logdir,
        '--run.steps', str(int(args.steps)),
        '--run.train_ratio', str(args.train_ratio),
    ]
    if args.load_ckpt:
        dreamer_argv += ['--run.from_checkpoint', args.load_ckpt]

    # append any extra overrides the user passed after '--'
    dreamer_argv += extra

    print('=' * 60)
    print('DreamerV3  ×  UAV Fire Coverage  —  Circle 1 (3 UAVs)')
    print('=' * 60)
    print('Config overrides:', dreamer_argv)
    print()

    from dreamerv3.main import main as dreamer_main
    dreamer_main(dreamer_argv)


if __name__ == '__main__':
    main()
