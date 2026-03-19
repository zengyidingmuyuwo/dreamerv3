"""
ppo_uav_circle1.py — PPO training for Circle 1 (3-UAV fire coverage).

Trains a shared policy with Proximal Policy Optimisation.
The 60 fire points in Circle 1 are split into 3 K-means sub-clusters;
the policy is trained by cycling through sub-cluster environments.

Quick start (sample data):
    python ppo_uav_circle1.py

Real data:
    python ppo_uav_circle1.py \\
        --center_csv  <path>/circle_1_center.csv \\
        --points_file <path>/circle_1_points.shp
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ── locate sibling modules ──────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from data_utils import load_circle_center, load_fire_points
from uav_fire_env import MultiUAVFireEnv

# ── defaults ─────────────────────────────────────────────────────────────────
SAMPLE_CENTER = os.path.join(_HERE, 'sample_data', 'circle_1_center.csv')
SAMPLE_POINTS = os.path.join(_HERE, 'sample_data', 'circle_1_points.csv')

# ── hyper-parameters ─────────────────────────────────────────────────────────
LR = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
UPDATE_STEPS = 2048   # steps per rollout
EPOCHS = 10
MINIBATCH = 256
MAX_EPISODES = 2000
SAVE_INTERVAL = 200


# ── network ──────────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(256, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.shared(x)
        mean = torch.tanh(self.actor_mean(x))
        std = self.actor_log_std.exp().expand_as(mean)
        return mean, std, self.critic(x).squeeze(-1)

    def act(self, state):
        mean, std, value = self(state)
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, states, actions):
        mean, std, values = self(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, values, entropy


# ── PPO update ────────────────────────────────────────────────────────────────
def ppo_update(net, optimizer, states, actions, log_probs_old, returns,
               advantages):
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    log_probs_old = torch.FloatTensor(log_probs_old)
    returns = torch.FloatTensor(returns)
    advantages = torch.FloatTensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(states)
    for _ in range(EPOCHS):
        idxs = np.random.permutation(n)
        for start in range(0, n, MINIBATCH):
            mb = idxs[start: start + MINIBATCH]
            lp, v, ent = net.evaluate(states[mb], actions[mb])
            ratio = torch.exp(lp - log_probs_old[mb])
            adv = advantages[mb]
            clip = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
            pol_loss = -torch.min(ratio * adv, clip).mean()
            val_loss = VALUE_COEF * (v - returns[mb]).pow(2).mean()
            ent_loss = -ENTROPY_COEF * ent.mean()
            loss = pol_loss + val_loss + ent_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()


def compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    returns, advantages = [], []
    gae = 0.0
    next_value = 0.0
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * next_value * (1 - d) - v
        gae = delta + gamma * lam * (1 - d) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + v)
        next_value = v
    return returns, advantages


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--center_csv', default=SAMPLE_CENTER)
    parser.add_argument('--points_file', default=SAMPLE_POINTS)
    parser.add_argument('--n_uav', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=MAX_EPISODES)
    parser.add_argument('--save_dir', default='./ppo_circle1_model')
    parser.add_argument('--save_interval', type=int, default=SAVE_INTERVAL)
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    lat0, lon0, radius = load_circle_center(args.center_csv)
    fire_pts = load_fire_points(args.points_file, lat0, lon0)
    print(f'Circle 1: centre=({lat0:.4f},{lon0:.4f}), '
          f'radius={radius:.0f}m, fire_pts={len(fire_pts)}')

    env = MultiUAVFireEnv(fire_pts, radius, n_uav=args.n_uav)
    state_dim = env.observation_space.shape[0]
    net = ActorCritic(state_dim)
    optimizer = optim.Adam(net.parameters(), lr=LR)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, 'model.pt')
    if args.load and os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt))
        print('Loaded checkpoint:', ckpt)

    episode_rewards = []
    buf_s, buf_a, buf_lp, buf_r, buf_v, buf_d = [], [], [], [], [], []
    state = env.reset()
    ep_reward = 0.0

    for ep in range(1, args.episodes + 1):
        while True:
            s_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                a, lp, v = net.act(s_t)
            action = a.numpy()[0]
            next_state, reward, done, _ = env.step(action)
            buf_s.append(state)
            buf_a.append(action)
            buf_lp.append(lp.item())
            buf_r.append(reward)
            buf_v.append(v.item())
            buf_d.append(float(done))
            ep_reward += reward
            state = next_state
            if done:
                state = env.reset()
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                break

        if len(buf_s) >= UPDATE_STEPS or done:
            returns, advantages = compute_gae(buf_r, buf_v, buf_d)
            ppo_update(net, optimizer,
                       np.array(buf_s), np.array(buf_a),
                       np.array(buf_lp), np.array(returns),
                       np.array(advantages))
            buf_s, buf_a, buf_lp, buf_r, buf_v, buf_d = [], [], [], [], [], []

        if ep % 10 == 0:
            avg = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            print(f'Episode {ep:5d}  avg_reward={avg:.1f}')

        if ep % args.save_interval == 0:
            torch.save(net.state_dict(), ckpt)
            print(f'  Saved → {ckpt}')

    torch.save(net.state_dict(), ckpt)
    print('Training complete.  Final model:', ckpt)


if __name__ == '__main__':
    main()
