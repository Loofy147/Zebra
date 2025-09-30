import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

logging.basicConfig(level=logging.INFO)


class InterventionEnvironment:
    """
    Simulates the environment for reinforcement learning.
    State: Current system metrics
    Action: Proposed intervention type
    Reward: Performance improvement after intervention
    """

    def __init__(self):
        self.current_state = self._generate_initial_state()
        self.intervention_history = []
        self.step_count = 0

    def _generate_initial_state(self) -> np.ndarray:
        """Generate initial system state."""
        return np.array([
            50.0,
            100.0,
            150.0,
            100.0,
            0.01,
            0.3,
            0.5
        ], dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_state = self._generate_initial_state()
        self.intervention_history = []
        self.step_count = 0
        return self.current_state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action and return next state, reward, done, info.

        Actions:
            0: No intervention
            1: Optimize algorithm
            2: Scale resources
            3: Cache optimization
            4: Database indexing

        Returns:
            (next_state, reward, done, info)
        """
        self.step_count += 1

        action_effects = {
            0: np.array([0, 0, 0, 0, 0, 0, 0]),
            1: np.array([-10, -20, -30, 5, -0.005, 0.05, 0]),
            2: np.array([-5, -10, -15, 20, 0, 0.15, 0.1]),
            3: np.array([-15, -25, -35, 10, 0, 0.1, 0.05]),
            4: np.array([-8, -15, -25, 3, -0.002, 0.08, 0])
        }

        noise = np.random.normal(0, 2, size=7)
        effect = action_effects.get(action, np.zeros(7))
        self.current_state = np.clip(self.current_state + effect + noise, 0, 1000)

        reward = self._calculate_reward(action, effect)

        done = self.step_count >= 50

        info = {
            'action': action,
            'step': self.step_count,
            'latency_improvement': -effect[0:3].mean() if action > 0 else 0
        }

        self.intervention_history.append({
            'action': action,
            'reward': reward,
            'state': self.current_state.copy()
        })

        return self.current_state.copy(), reward, done, info

    def _calculate_reward(self, action: int, effect: np.ndarray) -> float:
        """Calculate reward based on performance improvement."""
        latency_improvement = -effect[0:3].mean()

        request_rate_change = effect[3]

        error_rate_change = -effect[4]

        resource_cost = abs(effect[5]) + abs(effect[6])

        reward = (
            latency_improvement * 0.4 +
            request_rate_change * 0.2 +
            error_rate_change * 100 +
            -resource_cost * 10
        )

        if action == 0:
            reward = -1

        return reward


class DQNNetwork(nn.Module):
    """Deep Q-Network for learning optimal intervention policies."""

    def __init__(self, state_dim: int = 7, action_dim: int = 5,
                 hidden_dims: List[int] = [128, 64]):
        super(DQNNetwork, self).__init__()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-values for each action."""
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class ReinforcementLearningAgent:
    """
    DQN-based RL agent for learning optimal intervention strategies.
    Uses double DQN with prioritized experience replay concepts.
    """

    def __init__(self, state_dim: int = 7, action_dim: int = 5,
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer()

        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'average_reward': 0.0,
            'loss_history': []
        }

        logging.info(f"RL Agent initialized on device: {self.device}")

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            explore: Whether to use exploration

        Returns:
            Selected action index
        """
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def train_step(self, batch_size: int = 64) -> Optional[float]:
        """
        Perform one training step on a batch from replay buffer.

        Returns:
            Training loss or None if buffer too small
        """
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_episode(self, env: InterventionEnvironment) -> Dict[str, float]:
        """
        Train for one episode.

        Returns:
            Episode statistics
        """
        state = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0

        done = False
        while not done:
            action = self.select_action(state, explore=True)

            next_state, reward, done, info = env.step(action)

            self.replay_buffer.push(state, action, reward, next_state, done)

            loss = self.train_step()
            if loss is not None:
                episode_loss += loss

            episode_reward += reward
            state = next_state
            steps += 1

        self.decay_epsilon()

        if self.training_stats['episodes'] % 10 == 0:
            self.update_target_network()

        self.training_stats['episodes'] += 1
        self.training_stats['total_steps'] += steps
        self.training_stats['average_reward'] = (
            0.95 * self.training_stats['average_reward'] + 0.05 * episode_reward
        )
        self.training_stats['loss_history'].append(episode_loss / steps if steps > 0 else 0)

        return {
            'episode': self.training_stats['episodes'],
            'reward': episode_reward,
            'steps': steps,
            'epsilon': self.epsilon,
            'avg_reward': self.training_stats['average_reward']
        }

    def recommend_intervention(self, system_state: Dict) -> Dict[str, any]:
        """
        Recommend best intervention based on current system state.

        Args:
            system_state: Current system metrics

        Returns:
            Recommendation with action and confidence
        """
        state_array = np.array([
            system_state.get('latency_p50', 0),
            system_state.get('latency_p95', 0),
            system_state.get('latency_p99', 0),
            system_state.get('request_rate', 0),
            system_state.get('error_rate', 0),
            system_state.get('cpu_usage', 0),
            system_state.get('memory_usage', 0)
        ], dtype=np.float32)

        action = self.select_action(state_array, explore=False)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            confidence = torch.softmax(q_values, dim=1).max().item()

        action_names = [
            'no_intervention',
            'optimize_algorithm',
            'scale_resources',
            'cache_optimization',
            'database_indexing'
        ]

        return {
            'recommended_action': action_names[action],
            'action_index': action,
            'confidence': confidence,
            'q_values': q_values.cpu().numpy().tolist(),
            'exploration_rate': self.epsilon
        }

    def save_checkpoint(self, path: str):
        """Save agent checkpoint."""
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_stats': self.training_stats
        }
        torch.save(checkpoint, path)
        logging.info(f"RL Agent checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_stats = checkpoint['training_stats']
        logging.info(f"RL Agent checkpoint loaded from {path}")


rl_agent = ReinforcementLearningAgent()