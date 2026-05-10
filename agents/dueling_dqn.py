"""
HW3-2: Dueling DQN Agent (PyTorch).

Decomposes Q(s, a) into:
  Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))

Where:
- V(s)    is the state-value stream  (scalar)
- A(s, a) is the advantage stream    (per-action)

This allows the network to learn which states are valuable
independently of the effect of each action.

Also uses a target network (Double DQN style).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy


class DuelingQNetwork(nn.Module):
    """
    Dueling architecture:
      shared → FC(64→150) → ReLU → FC(150→100) → ReLU
        ├─ Value stream:     FC(100→1)
        └─ Advantage stream: FC(100→4)
    """

    def __init__(self, state_dim=64, action_dim=4, hidden1=150, hidden2=100):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.value_stream = nn.Linear(hidden2, 1)
        self.advantage_stream = nn.Linear(hidden2, action_dim)

    def forward(self, x):
        features = self.shared(x)
        value = self.value_stream(features)          # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)
        # Q = V + A - mean(A)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


class DuelingDQN:
    """
    Dueling DQN agent with target network (Double DQN style).

    Parameters
    ----------
    state_dim : int
    action_dim : int
    lr : float
    gamma : float
    epsilon, epsilon_min, epsilon_decay : float
    target_update_freq : int
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
        target_update_freq=50,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.episode_count = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_t).argmax(dim=1).item()

    def train_step(self, states, actions, rewards, next_states, dones):
        """Mini-batch Dueling + Double DQN update."""
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q
        q_values = self.policy_net(states_t)
        q_value = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target with dueling architecture
        with torch.no_grad():
            best_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def end_episode(self):
        """ε-decay and periodic target network sync."""
        self.episode_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.episode_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
