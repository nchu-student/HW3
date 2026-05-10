"""
HW3-1: Naive DQN Agent (PyTorch).

A simple 3-layer fully-connected Q-network trained with:
- ε-greedy exploration
- Optional experience replay buffer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Fully-connected Q-network.

    Architecture: Input(64) → 150 → ReLU → 100 → ReLU → 4
    """

    def __init__(self, state_dim=64, action_dim=4, hidden1=150, hidden2=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class NaiveDQN:
    """
    Naive DQN agent.

    Parameters
    ----------
    state_dim : int
    action_dim : int
    lr : float
    gamma : float
        Discount factor.
    epsilon : float
        Initial exploration rate.
    epsilon_min : float
    epsilon_decay : float
        Multiplicative decay per episode.
    """

    def __init__(
        self,
        state_dim=64,
        action_dim=4,
        lr=1e-3,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return q_values.argmax(dim=1).item()

    def train_step_online(self, state, action, reward, next_state, done):
        """
        Single-sample online update (no replay buffer).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # Current Q-value
        q_values = self.q_net(state_t)
        q_value = q_values[0, action]

        # Target Q-value
        with torch.no_grad():
            next_q = self.q_net(next_state_t).max(dim=1)[0]
            target = reward + (1 - done) * self.gamma * next_q

        loss = self.loss_fn(q_value, target.squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_step_replay(self, states, actions, rewards, next_states, dones):
        """
        Mini-batch update using experience replay buffer samples.
        """
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for the taken actions
        q_values = self.q_net(states_t)
        q_value = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.q_net(next_states_t).max(dim=1)[0]
            target = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
