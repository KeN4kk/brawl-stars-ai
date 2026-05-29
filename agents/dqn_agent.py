import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from models.networks import DQNNetwork, DuelingDQN
from typing import Tuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    def is_full(self):
        return len(self.memory) == self.memory.maxlen

class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: list, 
                 learning_rate: float = 0.001, gamma: float = 0.99, 
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, buffer_size: int = 100000,
                 device: str = 'cpu', agent_type: str = 'balanced'):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.agent_type = agent_type
        
        # Networks
        self.policy_net = DQNNetwork(state_size, hidden_layers, action_size).to(device)
        self.target_net = DQNNetwork(state_size, hidden_layers, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Statistics
        self.steps = 0
        self.episodes = 0
        self.total_reward = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state: np.ndarray, action: int, next_state: np.ndarray, 
                        reward: float, done: bool):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, next_state, reward, done)
    
    def train_step(self, batch_size: int) -> float:
        """Perform one training step"""
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute Q-values
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes += 1
    
    def get_stats(self) -> dict:
        """Get agent statistics"""
        return {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'buffer_size': len(self.replay_buffer),
            'agent_type': self.agent_type
        }

class DuelingDQNAgent(DQNAgent):
    """Dueling DQN Agent - Separates value and advantage"""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: list, **kwargs):
        # Initialize parent class
        super().__init__(state_size, action_size, hidden_layers, **kwargs)
        
        # Replace with dueling networks
        self.policy_net = DuelingDQN(state_size, hidden_layers, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, hidden_layers, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, *args):
        transition = Transition(*args)
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[list, np.ndarray, np.ndarray]:
        """Sample with priorities"""
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        # Importance weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        transitions = [self.buffer[i] for i in indices]
        
        return transitions, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)
    
    def is_full(self):
        return len(self.buffer) == self.capacity
