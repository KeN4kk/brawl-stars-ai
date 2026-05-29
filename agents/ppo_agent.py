import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from models.networks import PPONetwork
from collections import deque
from typing import Tuple
import random

class RolloutBuffer:
    """Buffer for collecting rollouts in PPO"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def push(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.log_probs),
            np.array(self.dones)
        )
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent"""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: list,
                 learning_rate: float = 0.0003, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_ratio: float = 0.2,
                 value_loss_coef: float = 0.5, entropy_coef: float = 0.01,
                 num_epochs: int = 10, batch_size: int = 64,
                 device: str = 'cpu', agent_type: str = 'balanced'):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.agent_type = agent_type
        
        # Network
        self.network = PPONetwork(state_size, hidden_layers, action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer()
        
        # Statistics
        self.steps = 0
        self.episodes = 0
        self.update_count = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float]:
        """Select action and get log probability"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _, _ = self.network.get_action_and_value(state_tensor)
            
            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().numpy()[0]
        
        return action, log_prob
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        value: float, log_prob: float, done: bool):
        """Store transition in rollout buffer"""
        self.rollout_buffer.push(state, action, reward, value, log_prob, done)
    
    def compute_gae(self, states: np.ndarray, rewards: np.ndarray, 
                   values: np.ndarray, dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def train(self) -> dict:
        """Perform PPO training on collected rollouts"""
        states, actions, rewards, values, log_probs, dones = self.rollout_buffer.get()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(states, rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Training loop
        total_loss = 0
        num_updates = 0
        
        for epoch in range(self.num_epochs):
            # Create mini-batches
            indices = np.random.permutation(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                # Get batch
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Forward pass
                policy_action, new_log_probs, entropy, values = \
                    self.network.get_action_and_value(batch_states, batch_actions)
                
                # Compute ratio and surrogate loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 
                                        1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + \
                       self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
        
        self.rollout_buffer.clear()
        self.update_count += 1
        self.steps += len(states)
        
        return {
            'loss': total_loss / max(num_updates, 1),
            'num_updates': num_updates,
            'mean_advantage': advantages.mean(),
            'mean_return': returns.mean()
        }
    
    def get_state_value(self, state: np.ndarray) -> float:
        """Get value estimate for a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, _, _, value = self.network.get_action_and_value(state_tensor)
            return value.cpu().numpy()[0]
    
    def get_stats(self) -> dict:
        """Get agent statistics"""
        return {
            'steps': self.steps,
            'episodes': self.episodes,
            'update_count': self.update_count,
            'agent_type': self.agent_type
        }

class A3CAgent:
    """Asynchronous Advantage Actor-Critic (A3C) Agent"""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: list,
                 learning_rate: float = 0.0001, gamma: float = 0.99,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
                 device: str = 'cpu', agent_type: str = 'balanced'):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.agent_type = agent_type
        
        # Network
        self.network = PPONetwork(state_size, hidden_layers, action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Statistics
        self.steps = 0
        self.episodes = 0
    
    def compute_loss(self, states: np.ndarray, actions: np.ndarray,
                    rewards: np.ndarray, next_states: np.ndarray,
                    dones: np.ndarray) -> torch.Tensor:
        """Compute A3C loss"""
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Get current policy and value
        policy, value = self.network(states_tensor)
        
        # Get next value
        with torch.no_grad():
            _, next_value = self.network(next_states_tensor)
        
        # Compute TD error
        target = rewards_tensor + self.gamma * next_value.squeeze() * (1 - dones_tensor)
        advantage = target - value.squeeze()
        
        # Policy loss
        std = torch.exp(self.network.log_std)
        dist = torch.distributions.Normal(policy, std)
        log_prob = dist.log_prob(actions_tensor).sum(dim=-1)
        policy_loss = -(log_prob * advantage.detach()).mean()
        
        # Value loss
        value_loss = advantage.pow(2).mean()
        
        # Entropy
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        
        return loss
    
    def train_step(self, states: np.ndarray, actions: np.ndarray,
                  rewards: np.ndarray, next_states: np.ndarray,
                  dones: np.ndarray) -> float:
        """Perform one training step"""
        
        loss = self.compute_loss(states, actions, rewards, next_states, dones)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.steps += len(states)
        
        return loss.item()
