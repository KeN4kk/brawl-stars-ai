import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np

class DQNNetwork(nn.Module):
    """Deep Q-Network for discrete action selection"""
    
    def __init__(self, input_size: int, hidden_layers: List[int], action_size: int, dropout: float = 0.2):
        super(DQNNetwork, self).__init__()
        
        self.input_size = input_size
        self.action_size = action_size
        
        # Build network
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ActorCriticNetwork(nn.Module):
    """Actor-Critic Network for PPO"""
    
    def __init__(self, input_size: int, hidden_layers: List[int], action_size: int, dropout: float = 0.2):
        super(ActorCriticNetwork, self).__init__()
        
        self.input_size = input_size
        self.action_size = action_size
        
        # Shared backbone
        backbone_layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            backbone_layers.append(nn.Linear(prev_size, hidden_size))
            backbone_layers.append(nn.ReLU())
            backbone_layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(prev_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], action_size),
            nn.Tanh()
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(prev_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], 1)
        )
    
    def forward(self, x):
        backbone_out = self.backbone(x)
        actor_out = self.actor(backbone_out)
        critic_out = self.critic(backbone_out)
        return actor_out, critic_out

class PPONetwork(nn.Module):
    """PPO (Proximal Policy Optimization) Network"""
    
    def __init__(self, input_size: int, hidden_layers: List[int], action_size: int, dropout: float = 0.2):
        super(PPONetwork, self).__init__()
        
        self.input_size = input_size
        self.action_size = action_size
        
        # Shared layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.shared = nn.Sequential(*layers)
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(prev_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], action_size),
            nn.Tanh()
        )
        
        # Value head
        self.value = nn.Sequential(
            nn.Linear(prev_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], 1)
        )
        
        # Log standard deviation for continuous control
        self.log_std = nn.Parameter(torch.zeros(action_size))
    
    def forward(self, x):
        shared_out = self.shared(x)
        policy_out = self.policy(shared_out)
        value_out = self.value(shared_out)
        return policy_out, value_out
    
    def get_action_and_value(self, x, action=None):
        """Get action, log probability, and value estimate"""
        policy_out, value = self.forward(x)
        
        # Create distribution
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(policy_out, std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)

class DuelingDQN(nn.Module):
    """Dueling DQN - Separates value and advantage streams"""
    
    def __init__(self, input_size: int, hidden_layers: List[int], action_size: int, dropout: float = 0.2):
        super(DuelingDQN, self).__init__()
        
        self.action_size = action_size
        
        # Shared layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.shared = nn.Sequential(*layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], action_size)
        )
    
    def forward(self, x):
        shared = self.shared(x)
        value = self.value_stream(shared)
        advantages = self.advantage_stream(shared)
        
        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class NoisyLinear(nn.Module):
    """Linear layer with learnable parameter noise for exploration"""
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # Register noisy buffers
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def forward(self, x):
        # Sample noise
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()
        
        # Compute weights and biases
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        
        return F.linear(x, weight, bias)

class RainbowDQN(nn.Module):
    """Rainbow DQN - Combines multiple DQN improvements"""
    
    def __init__(self, input_size: int, hidden_layers: List[int], action_size: int, 
                 num_atoms: int = 51, dropout: float = 0.2):
        super(RainbowDQN, self).__init__()
        
        self.action_size = action_size
        self.num_atoms = num_atoms
        
        # Shared layers with noisy
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Advantage stream (Noisy layers)
        self.advantage = nn.Sequential(
            NoisyLinear(hidden_layers[1], hidden_layers[1]),
            nn.ReLU(),
            NoisyLinear(hidden_layers[1], action_size * num_atoms)
        )
        
        # Value stream (Noisy layers)
        self.value = nn.Sequential(
            NoisyLinear(hidden_layers[1], hidden_layers[1]),
            nn.ReLU(),
            NoisyLinear(hidden_layers[1], num_atoms)
        )
    
    def forward(self, x):
        shared = self.shared(x)
        
        # Value and advantage
        adv = self.advantage(shared).view(-1, self.action_size, self.num_atoms)
        val = self.value(shared).view(-1, 1, self.num_atoms)
        
        # Combine using dueling architecture
        q_atoms = val + adv - adv.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probability distributions
        return F.softmax(q_atoms, dim=2)
