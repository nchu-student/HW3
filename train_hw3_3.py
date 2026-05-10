#!/usr/bin/env python3
"""
HW3-3: Keras DQN on Random-Mode GridWorld (30%)

Trains and compares:
  1. Keras DQN WITHOUT training tips (baseline)
  2. Keras DQN WITH training tips (Huber loss, gradient clipping, LR scheduling, soft update)

Mode: 'random' — all pieces placed randomly each episode.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gridworld import Gridworld
from agents.dqn_keras import KerasDQN
from agents.replay_buffer import ExperienceReplayBuffer
from utils.plotting import plot_rewards, plot_win_rate, plot_comparison

# ── Hyperparameters ──────────────────────────────────────────────────
NUM_EPISODES = 3000
MAX_STEPS = 50
LR = 1e-3
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.998
BUFFER_CAPACITY = 10000
BATCH_SIZE = 128
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'hw3_3')

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_keras_dqn(use_tips, label, seed=SEED):
    """Train a Keras DQN agent."""
    np.random.seed(seed)
    env = Gridworld(size=4, mode='random')
    agent = KerasDQN(
        lr=LR, gamma=GAMMA, epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY,
        use_training_tips=use_tips,
    )
    buffer = ExperienceReplayBuffer(capacity=BUFFER_CAPACITY)

    episode_rewards = []
    episode_wins = []
    hard_update_freq = 100  # for baseline without tips

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

        # Hard update for baseline (no tips)
        if not use_tips and (ep + 1) % hard_update_freq == 0:
            agent.hard_update_target()

        episode_rewards.append(total_reward)
        episode_wins.append(env.board.components['Player'].pos == env.board.components['Goal'].pos)

        if (ep + 1) % 300 == 0:
            avg = np.mean(episode_rewards[-300:])
            wr = np.mean(episode_wins[-300:]) * 100
            print(f"  [{label:25s}] Ep {ep+1:4d} | Avg Reward: {avg:7.2f} | Win Rate: {wr:5.1f}% | ε: {agent.epsilon:.3f}")

    return episode_rewards, episode_wins


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("HW3-3: Keras DQN on Random-Mode GridWorld")
    print("=" * 60)

    print("\n▶ Training Keras DQN (no training tips) ...")
    base_rewards, base_wins = train_keras_dqn(use_tips=False, label='Keras DQN (baseline)')

    print("\n▶ Training Keras DQN (with training tips) ...")
    tips_rewards, tips_wins = train_keras_dqn(use_tips=True, label='Keras DQN + Training Tips')

    # ── Plots ────────────────────────────────────────────────────────
    print("\n▶ Generating plots ...")
    rewards_dict = {
        'Keras DQN (baseline)': base_rewards,
        'Keras DQN + Training Tips': tips_rewards,
    }
    plot_rewards(rewards_dict, 'HW3-3: Reward Curves — Random Mode (Keras)',
                 os.path.join(RESULTS_DIR, 'reward_curves.png'), window=100)

    wins_dict = {
        'Keras DQN (baseline)': base_wins,
        'Keras DQN + Training Tips': tips_wins,
    }
    plot_win_rate(wins_dict, 'HW3-3: Win Rate — Random Mode (Keras)',
                  os.path.join(RESULTS_DIR, 'win_rate.png'), window=200)

    final_stats = {
        'Baseline': {
            'avg_reward': float(np.mean(base_rewards[-100:])),
            'win_rate': float(np.mean(base_wins[-100:]) * 100),
        },
        'With Tips': {
            'avg_reward': float(np.mean(tips_rewards[-100:])),
            'win_rate': float(np.mean(tips_wins[-100:]) * 100),
        },
    }
    plot_comparison(final_stats, 'HW3-3: Baseline vs Training Tips (Keras)',
                    os.path.join(RESULTS_DIR, 'comparison.png'))

    print("\n" + "=" * 60)
    print("Final Results (last 100 episodes):")
    for name, stats in final_stats.items():
        print(f"  {name:15s} | Avg Reward: {stats['avg_reward']:7.2f} | Win Rate: {stats['win_rate']:5.1f}%")
    print("=" * 60)
    print(f"Plots saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
