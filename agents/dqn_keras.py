"""
HW3-3: DQN Agent converted to TensorFlow / Keras.

Implements a Double DQN with training tips:
  - Huber loss instead of MSE for stable gradients
  - Gradient clipping (clipnorm=1.0)
  - Learning rate scheduling (exponential decay)
  - Soft target update (τ = 0.005) instead of hard copy
"""

import numpy as np

# TensorFlow imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress info/warning logs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses


def build_q_network(state_dim=64, action_dim=4, hidden1=150, hidden2=100):
    """Build a Keras Q-network (functional API)."""
    inputs = keras.Input(shape=(state_dim,))
    x = layers.Dense(hidden1, activation='relu')(inputs)
    x = layers.Dense(hidden2, activation='relu')(x)
    outputs = layers.Dense(action_dim)(x)
    return keras.Model(inputs=inputs, outputs=outputs, name='QNetwork')


class KerasDQN:
    """
    Double DQN agent implemented in Keras with training tips.

    Training Tips
    -------------
    1. Huber loss (less sensitive to outliers than MSE)
    2. Gradient clipping (clipnorm=1.0)
    3. Learning rate scheduling (exponential decay)
    4. Soft target update (polyak averaging, τ=0.005)
    """

    def __init__(
        self,
        state_dim=64,
        action_dim=4,
        lr=1e-3,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.997,
        tau=0.005,
        use_training_tips=True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.use_training_tips = use_training_tips

        # Build networks
        self.policy_net = build_q_network(state_dim, action_dim)
        self.target_net = build_q_network(state_dim, action_dim)
        self.target_net.set_weights(self.policy_net.get_weights())

        # Optimizer with training tips
        if use_training_tips:
            lr_schedule = optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=500,
                decay_rate=0.95,
                staircase=True,
            )
            self.optimizer = optimizers.Adam(
                learning_rate=lr_schedule,
                clipnorm=1.0,          # Gradient clipping
            )
            self.loss_fn = losses.Huber(delta=1.0)  # Huber loss
        else:
            self.optimizer = optimizers.Adam(learning_rate=lr)
            self.loss_fn = losses.MeanSquaredError()

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        q_values = self.policy_net(state_t, training=False)
        return int(tf.argmax(q_values, axis=1).numpy()[0])

    @tf.function
    def _train_step_graph(self, states, actions, rewards, next_states, dones):
        """Compiled TF graph for training step."""
        # Double DQN: select action with policy, evaluate with target
        next_q_policy = self.policy_net(next_states, training=False)
        best_actions = tf.argmax(next_q_policy, axis=1)
        next_q_target = self.target_net(next_states, training=False)
        next_q = tf.reduce_sum(
            next_q_target * tf.one_hot(best_actions, self.action_dim),
            axis=1,
        )
        targets = rewards + (1.0 - dones) * self.gamma * next_q

        with tf.GradientTape() as tape:
            q_values = self.policy_net(states, training=True)
            q_action = tf.reduce_sum(
                q_values * tf.one_hot(actions, self.action_dim),
                axis=1,
            )
            loss = self.loss_fn(targets, q_action)

        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
        return loss

    def train_step(self, states, actions, rewards, next_states, dones):
        """Mini-batch Double DQN update (Keras)."""
        states_t = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_t = tf.convert_to_tensor(actions, dtype=tf.int64)
        rewards_t = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_t = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_t = tf.convert_to_tensor(dones, dtype=tf.float32)

        loss = self._train_step_graph(states_t, actions_t, rewards_t, next_states_t, dones_t)
        return float(loss.numpy())

    def soft_update_target(self):
        """Soft (polyak) update: θ_target ← τ·θ_policy + (1-τ)·θ_target"""
        for target_var, policy_var in zip(
            self.target_net.trainable_variables,
            self.policy_net.trainable_variables,
        ):
            target_var.assign(self.tau * policy_var + (1 - self.tau) * target_var)

    def hard_update_target(self):
        """Hard copy: θ_target ← θ_policy"""
        self.target_net.set_weights(self.policy_net.get_weights())

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.use_training_tips:
            self.soft_update_target()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
