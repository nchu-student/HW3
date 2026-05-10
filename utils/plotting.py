"""
Plotting utilities for DQN training results.

Produces publication-quality matplotlib charts for:
- Reward curves (smoothed with moving average)
- Win rate over episodes
- Side-by-side algorithm comparisons
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

# ── Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'lines.linewidth': 1.8,
})

COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#00BCD4']


def _smooth(values, window=50):
    """Moving average smoothing."""
    if len(values) < window:
        window = max(1, len(values) // 5)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def plot_rewards(rewards_dict, title, save_path, window=50):
    """
    Plot overlaid reward curves.

    Parameters
    ----------
    rewards_dict : dict[str, list[float]]
        Algorithm name → list of episode rewards.
    title : str
    save_path : str
    window : int
        Smoothing window size.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots()

    for i, (name, rewards) in enumerate(rewards_dict.items()):
        color = COLORS[i % len(COLORS)]
        smoothed = _smooth(rewards, window)
        episodes = np.arange(len(smoothed)) + window
        ax.plot(episodes, smoothed, color=color, label=name)
        # faint raw data
        ax.plot(range(len(rewards)), rewards, color=color, alpha=0.12, linewidth=0.5)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  📊 Saved reward plot → {save_path}")


def plot_win_rate(win_rates_dict, title, save_path, window=100):
    """
    Plot win rate (fraction of wins in a sliding window).

    Parameters
    ----------
    win_rates_dict : dict[str, list[bool]]
        Algorithm name → list of episode outcomes (True = win).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots()

    for i, (name, wins) in enumerate(win_rates_dict.items()):
        color = COLORS[i % len(COLORS)]
        win_arr = np.array(wins, dtype=np.float32)
        smoothed = _smooth(win_arr, window) * 100
        episodes = np.arange(len(smoothed)) + window
        ax.plot(episodes, smoothed, color=color, label=name)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate (%)')
    ax.set_ylim(-5, 105)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  📊 Saved win-rate plot → {save_path}")


def plot_comparison(results, title, save_path):
    """
    Bar chart comparing final metrics across algorithms.

    Parameters
    ----------
    results : dict[str, dict]
        Algorithm name → {'avg_reward': float, 'win_rate': float, ...}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    names = list(results.keys())
    avg_rewards = [results[n].get('avg_reward', 0) for n in names]
    win_rates = [results[n].get('win_rate', 0) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(names, avg_rewards, color=COLORS[:len(names)], edgecolor='white')
    ax1.set_ylabel('Avg Reward (last 100 eps)')
    ax1.set_title('Average Reward')
    for bar, val in zip(bars1, avg_rewards):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    bars2 = ax2.bar(names, win_rates, color=COLORS[:len(names)], edgecolor='white')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate (last 100 eps)')
    ax2.set_ylim(0, 110)
    for bar, val in zip(bars2, win_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    fig.suptitle(title, fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  📊 Saved comparison plot → {save_path}")
