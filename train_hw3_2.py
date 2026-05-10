#!/usr/bin/env python3
"""
HW3-2: Double DQN & Dueling DQN on Player-Mode GridWorld (40%)

Trains and compares:
  1. Double DQN
  2. Dueling DQN

Both use experience replay and target networks on 'player' mode
(random player start, fixed goal/pit/wall).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gridworld import Gridworld
from agents.double_dqn import DoubleDQN
from agents.dueling_dqn import DuelingDQN
from agents.replay_buffer import ExperienceReplayBuffer
from utils.plotting import plot_rewards, plot_win_rate, plot_comparison

# ── Hyperparameters ──────────────────────────────────────────────────
NUM_EPISODES = 2000
MAX_STEPS = 50
LR = 1e-3
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.997
BUFFER_CAPACITY = 5000
BATCH_SIZE = 64
TARGET_UPDATE = 50
SEED = 42
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'hw3_2')


def train_agent(agent_cls, agent_name, seed=SEED):
    """Generic training loop for any DQN agent with replay buffer."""
    np.random.seed(seed)
    env = Gridworld(size=4, mode='player')
    agent = agent_cls(
        lr=LR, gamma=GAMMA, epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY,
        target_update_freq=TARGET_UPDATE,
    )
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

            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                agent.train_step(*batch)

            if done:
                break

        agent.end_episode()
        episode_rewards.append(total_reward)
        episode_wins.append(env.board.components['Player'].pos == env.board.components['Goal'].pos)

        if (ep + 1) % 200 == 0:
            avg = np.mean(episode_rewards[-200:])
            wr = np.mean(episode_wins[-200:]) * 100
            print(f"  [{agent_name:12s}] Ep {ep+1:4d} | Avg Reward: {avg:7.2f} | Win Rate: {wr:5.1f}% | ε: {agent.epsilon:.3f}")

    return episode_rewards, episode_wins


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("HW3-2: Double DQN & Dueling DQN on Player-Mode GridWorld")
    print("=" * 60)

    print("\n▶ Training Double DQN ...")
    double_rewards, double_wins = train_agent(DoubleDQN, 'Double DQN')

    print("\n▶ Training Dueling DQN ...")
    dueling_rewards, dueling_wins = train_agent(DuelingDQN, 'Dueling DQN')

    # ── Plots ────────────────────────────────────────────────────────
    print("\n▶ Generating plots ...")
    rewards_dict = {
        'Double DQN': double_rewards,
        'Dueling DQN': dueling_rewards,
    }
    plot_rewards(rewards_dict, 'HW3-2: Reward Curves — Player Mode',
                 os.path.join(RESULTS_DIR, 'reward_curves.png'))

    wins_dict = {
        'Double DQN': double_wins,
        'Dueling DQN': dueling_wins,
    }
    plot_win_rate(wins_dict, 'HW3-2: Win Rate — Player Mode',
                  os.path.join(RESULTS_DIR, 'win_rate.png'))

    final_stats = {
        'Double DQN': {
            'avg_reward': float(np.mean(double_rewards[-100:])),
            'win_rate': float(np.mean(double_wins[-100:]) * 100),
        },
        'Dueling DQN': {
            'avg_reward': float(np.mean(dueling_rewards[-100:])),
            'win_rate': float(np.mean(dueling_wins[-100:]) * 100),
        },
    }
    plot_comparison(final_stats, 'HW3-2: Double vs Dueling DQN',
                    os.path.join(RESULTS_DIR, 'comparison.png'))

    print("\n" + "=" * 60)
    print("Final Results (last 100 episodes):")
    for name, stats in final_stats.items():
        print(f"  {name:15s} | Avg Reward: {stats['avg_reward']:7.2f} | Win Rate: {stats['win_rate']:5.1f}%")
    print("=" * 60)
    print(f"Plots saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
