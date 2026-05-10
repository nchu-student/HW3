"""
Experience Replay Buffer for DQN training.

Stores (state, action, reward, next_state, done) transitions
and supports random mini-batch sampling.
"""

import random
from collections import deque, namedtuple
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ExperienceReplayBuffer:
    """
    Fixed-size circular buffer for experience replay.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a random mini-batch.

        Returns
        -------
        states : np.ndarray of shape (batch_size, state_dim)
        actions : np.ndarray of shape (batch_size,)
        rewards : np.ndarray of shape (batch_size,)
        next_states : np.ndarray of shape (batch_size, state_dim)
        dones : np.ndarray of shape (batch_size,)
        """
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
