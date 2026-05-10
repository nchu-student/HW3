"""
HW3-4 (Bonus): Rainbow DQN Agent (PyTorch).

Combines improvements over vanilla DQN:
  1. Double DQN            — reduce overestimation
  2. Dueling architecture  — V(s) + A(s,a)
  3. Prioritized Replay    — sample important transitions more often
  4. Multi-step returns    — n-step TD targets
  5. Noisy Networks        — parametric noise for exploration

Note: We use expected-value Q-learning (not full C51 distributional)
because the 4×4 GridWorld has a very small, discrete reward structure
where C51's 51-atom distribution adds unnecessary complexity without
benefit. The five techniques above already capture the essence of
Rainbow and perform well on this environment.

Reference:
  Hessel et al., "Rainbow: Combining Improvements in Deep RL" (2018)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from collections import deque


# ═══════════════════════════════════════════════════════════════════════
# 1. Noisy Linear Layer
# ═══════════════════════════════════════════════════════════════════════
class NoisyLinear(nn.Module):
    """
    Factorised Gaussian noisy linear layer.
    y = (μ_w + σ_w ⊙ ε_w) · x + (μ_b + σ_b ⊙ ε_b)
    """

    def __init__(self, in_features, out_features, sigma_init=0.17):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ═══════════════════════════════════════════════════════════════════════
# 2. Prioritized Replay Buffer
# ═══════════════════════════════════════════════════════════════════════
class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay."""

    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.size = 0

        self.states = [None] * capacity
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = [None] * capacity
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities[:self.size].max() if self.size > 0 else 1.0

        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        prios = self.priorities[:self.size] ** self.alpha
        probs = prios / prios.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        states = np.array([self.states[i] for i in indices], dtype=np.float32)
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = np.array([self.next_states[i] for i in indices], dtype=np.float32)
        dones = self.dones[indices]

        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        self.priorities[indices] = np.abs(td_errors) + 1e-6

    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════════════════════════
# 3. N-step Reward Buffer
# ═══════════════════════════════════════════════════════════════════════
class NStepBuffer:
    """Accumulates n-step returns before storing in the main replay buffer."""

    def __init__(self, n_step=3, gamma=0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def is_ready(self):
        return len(self.buffer) == self.n_step

    def get(self):
        """Compute n-step return."""
        state, action = self.buffer[0][0], self.buffer[0][1]
        n_reward = 0
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            n_reward += (self.gamma ** i) * r
            if d:
                next_state = self.buffer[i][3]
                done = True
                return state, action, n_reward, next_state, done
        next_state = self.buffer[-1][3]
        done = self.buffer[-1][4]
        return state, action, n_reward, next_state, done

    def reset(self):
        self.buffer.clear()


# ═══════════════════════════════════════════════════════════════════════
# 4. Rainbow Network (Dueling + Noisy — expected-value Q-learning)
# ═══════════════════════════════════════════════════════════════════════
class RainbowNetwork(nn.Module):
    """
    Dueling network with noisy layers.
    Q(s,a) = V(s) + A(s,a) - mean(A)
    """

    def __init__(self, state_dim=64, action_dim=4, hidden1=150, hidden2=100):
        super().__init__()
        # Shared feature extractor (deterministic)
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
        )
        # Value stream (noisy)
        self.value_hidden = NoisyLinear(hidden1, hidden2)
        self.value = NoisyLinear(hidden2, 1)

        # Advantage stream (noisy)
        self.advantage_hidden = NoisyLinear(hidden1, hidden2)
        self.advantage = NoisyLinear(hidden2, action_dim)

    def forward(self, x):
        features = self.feature(x)

        v = F.relu(self.value_hidden(features))
        v = self.value(v)                    # (batch, 1)

        a = F.relu(self.advantage_hidden(features))
        a = self.advantage(a)                # (batch, action_dim)

        q = v + a - a.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ═══════════════════════════════════════════════════════════════════════
# 5. Rainbow DQN Agent
# ═══════════════════════════════════════════════════════════════════════
class RainbowDQN:
    """
    Rainbow-style DQN combining:
    Double DQN + Dueling + PER + N-step + Noisy Nets

    Uses Huber loss and gradient clipping for stability.
    """

    def __init__(
        self,
        state_dim=64,
        action_dim=4,
        lr=1e-3,
        gamma=0.9,
        n_step=3,
        buffer_capacity=10000,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=3000,
        target_update_freq=50,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.n_step = n_step
        self.target_update_freq = target_update_freq
        self.episode_count = 0
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_count = 0
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = RainbowNetwork(state_dim, action_dim).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')  # Huber loss, per-element

        self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha)
        self.n_step_buffer = NStepBuffer(n_step, gamma)

    @property
    def beta(self):
        return min(1.0, self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames)

    def select_action(self, state):
        """ε-greedy + noisy net exploration."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        self.policy_net.reset_noise()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store with n-step processing."""
        self.n_step_buffer.push(state, action, reward, next_state, done)
        if self.n_step_buffer.is_ready():
            s, a, r, ns, d = self.n_step_buffer.get()
            self.replay_buffer.push(s, a, r, ns, float(d))
        if done:
            while len(self.n_step_buffer.buffer) > 0:
                s, a, r, ns, d = self.n_step_buffer.get()
                self.replay_buffer.push(s, a, r, ns, float(d))
                self.n_step_buffer.buffer.popleft()

    def train_step(self, batch_size=64):
        """Double DQN + Dueling + PER + N-step update with Huber loss."""
        if len(self.replay_buffer) < batch_size:
            return 0.0

        self.frame_count += 1

        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(batch_size, self.beta)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        # Reset noise
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        # Current Q-values
        q_values = self.policy_net(states_t)
        q_value = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target: select action with policy, evaluate with target
        with torch.no_grad():
            gamma_n = self.gamma ** self.n_step
            best_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target = rewards_t + (1 - dones_t) * gamma_n * next_q

        # Per-element Huber loss, weighted by PER importance
        td_errors = q_value - target
        elementwise_loss = self.loss_fn(q_value, target)
        loss = (weights_t * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        return loss.item()

    def end_episode(self):
        self.episode_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.n_step_buffer.reset()
        if self.episode_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
