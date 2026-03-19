"""
sac_uav_circle1.py — SAC training for Circle 1 (3-UAV fire coverage).

Soft Actor-Critic with automatic entropy tuning.
Uses the same MultiUAVFireEnv cycling strategy as the PPO script.

Quick start (sample data):
    python sac_uav_circle1.py

Real data:
    python sac_uav_circle1.py \\
        --center_csv  <path>/circle_1_center.csv \\
        --points_file <path>/circle_1_points.shp
"""

import argparse
import os
import sys
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data_utils import load_circle_center, load_fire_points
from uav_fire_env import MultiUAVFireEnv

SAMPLE_CENTER = os.path.join(_HERE, 'sample_data', 'circle_1_center.csv')
SAMPLE_POINTS = os.path.join(_HERE, 'sample_data', 'circle_1_points.csv')

LR = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 256
WARMUP_STEPS = 1000
UPDATE_EVERY = 1
MAX_EPISODES = 2000
SAVE_INTERVAL = 200
LOG_STD_MIN = -5
LOG_STD_MAX = 2


# ── replay buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, n):
        idx = np.random.choice(len(self.buf), n, replace=False)
        batch = [self.buf[i] for i in idx]
        s, a, r, ns, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.FloatTensor(np.array(a)),
                torch.FloatTensor(np.array(r)).unsqueeze(1),
                torch.FloatTensor(np.array(ns)),
                torch.FloatTensor(np.array(d)).unsqueeze(1))

    def __len__(self):
        return len(self.buf)


# ── networks ──────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], -1))


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        return action, log_prob.sum(-1, keepdim=True)

    def get_action(self, state):
        action, _ = self(state)
        return action.detach()


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1 - tau) + s.data * tau)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--center_csv', default=SAMPLE_CENTER)
    parser.add_argument('--points_file', default=SAMPLE_POINTS)
    parser.add_argument('--n_uav', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=MAX_EPISODES)
    parser.add_argument('--save_dir', default='./sac_circle1_model')
    parser.add_argument('--save_interval', type=int, default=SAVE_INTERVAL)
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    lat0, lon0, radius = load_circle_center(args.center_csv)
    fire_pts = load_fire_points(args.points_file, lat0, lon0)
    print(f'Circle 1: centre=({lat0:.4f},{lon0:.4f}), '
          f'radius={radius:.0f}m, fire_pts={len(fire_pts)}')

    env = MultiUAVFireEnv(fire_pts, radius, n_uav=args.n_uav)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = GaussianPolicy(state_dim, action_dim)
    q1 = QNetwork(state_dim, action_dim)
    q2 = QNetwork(state_dim, action_dim)
    q1_target = QNetwork(state_dim, action_dim)
    q2_target = QNetwork(state_dim, action_dim)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    pol_opt = optim.Adam(policy.parameters(), lr=LR)
    q1_opt = optim.Adam(q1.parameters(), lr=LR)
    q2_opt = optim.Adam(q2.parameters(), lr=LR)

    # Automatic entropy tuning
    target_entropy = -float(action_dim)
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_opt = optim.Adam([log_alpha], lr=LR)

    buffer = ReplayBuffer(BUFFER_SIZE)
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, 'model.pt')
    if args.load and os.path.exists(ckpt):
        data = torch.load(ckpt)
        policy.load_state_dict(data['policy'])
        print('Loaded checkpoint:', ckpt)

    total_steps = 0
    episode_rewards = []

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            if total_steps < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                s_t = torch.FloatTensor(state).unsqueeze(0)
                action = policy.get_action(s_t).numpy()[0]

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward
            total_steps += 1

            if len(buffer) >= BATCH_SIZE and total_steps % UPDATE_EVERY == 0:
                s, a, r, ns, d = buffer.sample(BATCH_SIZE)
                with torch.no_grad():
                    na, nlp = policy(ns)
                    alpha_val = log_alpha.exp()
                    q_min = torch.min(q1_target(ns, na), q2_target(ns, na))
                    y = r + GAMMA * (1 - d) * (q_min - alpha_val * nlp)

                # Q losses
                q1_loss = F.mse_loss(q1(s, a), y)
                q1_opt.zero_grad(); q1_loss.backward(); q1_opt.step()
                q2_loss = F.mse_loss(q2(s, a), y)
                q2_opt.zero_grad(); q2_loss.backward(); q2_opt.step()

                # Policy loss
                new_a, new_lp = policy(s)
                alpha_val = log_alpha.exp()
                pol_loss = (alpha_val * new_lp
                            - torch.min(q1(s, new_a), q2(s, new_a))).mean()
                pol_opt.zero_grad(); pol_loss.backward(); pol_opt.step()

                # Alpha loss
                alpha_loss = -(log_alpha * (new_lp + target_entropy).detach()).mean()
                alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

                soft_update(q1_target, q1, TAU)
                soft_update(q2_target, q2, TAU)

        episode_rewards.append(ep_reward)
        if ep % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f'Episode {ep:5d}  avg_reward={avg:.1f}  '
                  f'alpha={log_alpha.exp().item():.4f}')

        if ep % args.save_interval == 0:
            torch.save({'policy': policy.state_dict()}, ckpt)
            print(f'  Saved → {ckpt}')

    torch.save({'policy': policy.state_dict()}, ckpt)
    print('Training complete.  Final model:', ckpt)


if __name__ == '__main__':
    main()
