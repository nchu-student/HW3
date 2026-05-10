#!/usr/bin/env python3
"""
HW3-4 (Bonus): Rainbow DQN on Random-Mode GridWorld

Rainbow-style DQN combining five improvements:
  1. Double DQN
  2. Dueling architecture
  3. Prioritized Experience Replay
  4. Multi-step returns (n=3)
  5. Noisy Networks

Also trains a Double DQN baseline for comparison.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gridworld import Gridworld
from agents.rainbow_dqn import RainbowDQN
from agents.double_dqn import DoubleDQN
from agents.replay_buffer import ExperienceReplayBuffer
from utils.plotting import plot_rewards, plot_win_rate, plot_comparison

# ── Hyperparameters ──────────────────────────────────────────────────
NUM_EPISODES = 3000
MAX_STEPS = 50
LR = 1e-3
GAMMA = 0.9
BATCH_SIZE = 64
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'hw3_4')


def train_rainbow(seed=SEED):
    """Train Rainbow DQN."""
    np.random.seed(seed)
    env = Gridworld(size=4, mode='random')
    agent = RainbowDQN(
        lr=LR, gamma=GAMMA,
        n_step=3,
        buffer_capacity=10000,
        target_update_freq=50,
    )

    episode_rewards = []
    episode_wins = []

    for ep in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.train_step(BATCH_SIZE)

            if done:
                break

        agent.end_episode()
        episode_rewards.append(total_reward)
        episode_wins.append(env.board.components['Player'].pos == env.board.components['Goal'].pos)

        if (ep + 1) % 300 == 0:
            avg = np.mean(episode_rewards[-300:])
            wr = np.mean(episode_wins[-300:]) * 100
            print(f"  [Rainbow DQN ] Ep {ep+1:4d} | Avg Reward: {avg:7.2f} | Win Rate: {wr:5.1f}% | ε: {agent.epsilon:.3f}")

    return episode_rewards, episode_wins


def train_double_baseline(seed=SEED):
    """Train Double DQN baseline for comparison."""
    np.random.seed(seed)
    env = Gridworld(size=4, mode='random')
    agent = DoubleDQN(
        lr=LR, gamma=GAMMA,
        epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.998,
        target_update_freq=50,
    )
    buffer = ExperienceReplayBuffer(capacity=10000)

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

            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                agent.train_step(*batch)

            if done:
                break

        agent.end_episode()
        episode_rewards.append(total_reward)
        episode_wins.append(env.board.components['Player'].pos == env.board.components['Goal'].pos)

        if (ep + 1) % 300 == 0:
            avg = np.mean(episode_rewards[-300:])
            wr = np.mean(episode_wins[-300:]) * 100
            print(f"  [Double DQN  ] Ep {ep+1:4d} | Avg Reward: {avg:7.2f} | Win Rate: {wr:5.1f}% | ε: {agent.epsilon:.3f}")

    return episode_rewards, episode_wins


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("HW3-4 (Bonus): Rainbow DQN on Random-Mode GridWorld")
    print("=" * 60)

    print("\n▶ Training Double DQN (baseline) ...")
    base_rewards, base_wins = train_double_baseline()

    print("\n▶ Training Rainbow DQN ...")
    rain_rewards, rain_wins = train_rainbow()

    # ── Plots ────────────────────────────────────────────────────────
    print("\n▶ Generating plots ...")
    rewards_dict = {
        'Double DQN (baseline)': base_rewards,
        'Rainbow DQN': rain_rewards,
    }
    plot_rewards(rewards_dict, 'HW3-4: Reward Curves — Random Mode',
                 os.path.join(RESULTS_DIR, 'reward_curves.png'), window=100)

    wins_dict = {
        'Double DQN (baseline)': base_wins,
        'Rainbow DQN': rain_wins,
    }
    plot_win_rate(wins_dict, 'HW3-4: Win Rate — Random Mode',
                  os.path.join(RESULTS_DIR, 'win_rate.png'), window=200)

    final_stats = {
        'Double DQN': {
            'avg_reward': float(np.mean(base_rewards[-100:])),
            'win_rate': float(np.mean(base_wins[-100:]) * 100),
        },
        'Rainbow DQN': {
            'avg_reward': float(np.mean(rain_rewards[-100:])),
            'win_rate': float(np.mean(rain_wins[-100:]) * 100),
        },
    }
    plot_comparison(final_stats, 'HW3-4: Double DQN vs Rainbow DQN',
                    os.path.join(RESULTS_DIR, 'comparison.png'))

    print("\n" + "=" * 60)
    print("Final Results (last 100 episodes):")
    for name, stats in final_stats.items():
        print(f"  {name:15s} | Avg Reward: {stats['avg_reward']:7.2f} | Win Rate: {stats['win_rate']:5.1f}%")
    print("=" * 60)
    print(f"Plots saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
