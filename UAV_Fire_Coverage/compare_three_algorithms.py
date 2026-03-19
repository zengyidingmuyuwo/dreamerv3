"""
Compare PPO / SAC / Dreamer training outcomes from log files.

Usage examples
--------------
python compare_three_algorithms.py \
  --ppo_log ./ppo_circle1.log \
  --sac_log ./sac_circle1.log \
  --dreamer_scores ~/logdir/dreamer/uavfire_circle1/scores.jsonl
"""

import argparse
import json
import re
from pathlib import Path


PAT_COVERAGE = re.compile(r'coverage=([0-9]+(?:\.[0-9]+)?)%')
PAT_TOTAL_REWARD = re.compile(r'total_reward=([-+]?[0-9]+(?:\.[0-9]+)?)')


def parse_policy_log(path):
  coverages = []
  rewards = []
  for line in Path(path).read_text(encoding='utf-8', errors='ignore').splitlines():
    cov = PAT_COVERAGE.search(line)
    rew = PAT_TOTAL_REWARD.search(line)
    if cov:
      coverages.append(float(cov.group(1)))
    if rew:
      rewards.append(float(rew.group(1)))
  return {
      'coverage_mean_pct': _mean(coverages),
      'coverage_last_pct': coverages[-1] if coverages else None,
      'reward_mean': _mean(rewards),
      'reward_last': rewards[-1] if rewards else None,
      'num_coverage_records': len(coverages),
      'num_reward_records': len(rewards),
  }


def parse_dreamer_scores(path):
  scores = []
  for line in Path(path).read_text(encoding='utf-8', errors='ignore').splitlines():
    line = line.strip()
    if not line:
      continue
    try:
      item = json.loads(line)
    except json.JSONDecodeError:
      continue
    if 'episode/score' in item:
      scores.append(float(item['episode/score']))
  return {
      'episode_score_mean': _mean(scores),
      'episode_score_last': scores[-1] if scores else None,
      'num_scores': len(scores),
  }


def _mean(values):
  return sum(values) / len(values) if values else None


def _fmt(value):
  if value is None:
    return 'n/a'
  return f'{value:.3f}'


def main():
  parser = argparse.ArgumentParser(description='Compare PPO/SAC/Dreamer logs.')
  parser.add_argument('--ppo_log', type=str, default='')
  parser.add_argument('--sac_log', type=str, default='')
  parser.add_argument('--dreamer_scores', type=str, default='')
  args = parser.parse_args()

  if not any([args.ppo_log, args.sac_log, args.dreamer_scores]):
    raise ValueError(
        'At least one non-empty path must be provided: '
        '--ppo_log, --sac_log, or --dreamer_scores')

  rows = []
  if args.ppo_log:
    rows.append(('PPO', parse_policy_log(args.ppo_log)))
  if args.sac_log:
    rows.append(('SAC', parse_policy_log(args.sac_log)))
  if args.dreamer_scores:
    rows.append(('Dreamer', parse_dreamer_scores(args.dreamer_scores)))

  print('Algorithm comparison summary')
  print('=' * 80)
  for name, metrics in rows:
    print(name)
    for k, v in metrics.items():
      if isinstance(v, float):
        v = _fmt(v)
      print(f'  - {k}: {v}')
    print('-' * 80)


if __name__ == '__main__':
  main()
