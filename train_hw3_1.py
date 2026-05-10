#!/usr/bin/env python3
"""
HW3-1: Naive DQN on Static GridWorld (30%)

Trains two variants:
  1. Online DQN (no replay buffer)
  2. DQN with Experience Replay Buffer

Both use the same network architecture and hyperparameters.
The only difference is the training procedure.
"""

import os
import sys
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gridworld import Gridworld
from agents.naive_dqn import NaiveDQN
from agents.replay_buffer import ExperienceReplayBuffer
from utils.plotting import plot_rewards, plot_win_rate, plot_comparison

# ── Hyperparameters ──────────────────────────────────────────────────
NUM_EPISODES = 1000
MAX_STEPS = 50           # max steps per episode
LR = 1e-3
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
BUFFER_CAPACITY = 1000
BATCH_SIZE = 64
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'hw3_1')


def train_online(seed=SEED):
    """Train naive DQN without experience replay (online single-sample updates)."""
    np.random.seed(seed)
    env = Gridworld(size=4, mode='static')
    agent = NaiveDQN(lr=LR, gamma=GAMMA, epsilon=EPSILON_START,
                     epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY)

    episode_rewards = []
    episode_wins = []

    for ep in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.train_step_online(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_wins.append(env.board.components['Player'].pos == env.board.components['Goal'].pos)

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            wr = np.mean(episode_wins[-100:]) * 100
            print(f"  [Online DQN] Ep {ep+1:4d} | Avg Reward: {avg:7.2f} | Win Rate: {wr:5.1f}% | ε: {agent.epsilon:.3f}")

    return episode_rewards, episode_wins


def train_replay(seed=SEED):
    """Train DQN with experience replay buffer."""
    np.random.seed(seed)
    env = Gridworld(size=4, mode='static')
    agent = NaiveDQN(lr=LR, gamma=GAMMA, epsilon=EPSILON_START,
                     epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY)
    buffer = ExperienceReplayBuffer(capacity=BUFFER_CAPACITY)

    episode_rewards = []
    episode_wins = []

    for ep in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            # Train from replay buffer once we have enough samples
            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                agent.train_step_replay(*batch)

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_wins.append(env.board.components['Player'].pos == env.board.components['Goal'].pos)

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            wr = np.mean(episode_wins[-100:]) * 100
            print(f"  [Replay DQN] Ep {ep+1:4d} | Avg Reward: {avg:7.2f} | Win Rate: {wr:5.1f}% | ε: {agent.epsilon:.3f}")

    return episode_rewards, episode_wins


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("HW3-1: Naive DQN on Static GridWorld")
    print("=" * 60)

    print("\n▶ Training Online DQN (no replay) ...")
    online_rewards, online_wins = train_online()

    print("\n▶ Training DQN with Experience Replay ...")
    replay_rewards, replay_wins = train_replay()

    # ── Plots ────────────────────────────────────────────────────────
    print("\n▶ Generating plots ...")
    rewards_dict = {
        'Online DQN (no replay)': online_rewards,
        'DQN + Experience Replay': replay_rewards,
    }
    plot_rewards(rewards_dict, 'HW3-1: Reward Curves — Static Mode',
                 os.path.join(RESULTS_DIR, 'reward_curves.png'))

    wins_dict = {
        'Online DQN (no replay)': online_wins,
        'DQN + Experience Replay': replay_wins,
    }
    plot_win_rate(wins_dict, 'HW3-1: Win Rate — Static Mode',
                  os.path.join(RESULTS_DIR, 'win_rate.png'))

    # Final stats
    final_stats = {
        'Online DQN': {
            'avg_reward': float(np.mean(online_rewards[-100:])),
            'win_rate': float(np.mean(online_wins[-100:]) * 100),
        },
        'Replay DQN': {
            'avg_reward': float(np.mean(replay_rewards[-100:])),
            'win_rate': float(np.mean(replay_wins[-100:]) * 100),
        },
    }
    plot_comparison(final_stats, 'HW3-1: Online vs Replay DQN',
                    os.path.join(RESULTS_DIR, 'comparison.png'))

    print("\n" + "=" * 60)
    print("Final Results (last 100 episodes):")
    for name, stats in final_stats.items():
        print(f"  {name:15s} | Avg Reward: {stats['avg_reward']:7.2f} | Win Rate: {stats['win_rate']:5.1f}%")
    print("=" * 60)
    print(f"Plots saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
