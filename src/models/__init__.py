"""Modern dynamics models for model-based reinforcement learning."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class EnsembleDynamicsModel(nn.Module):
    """Ensemble dynamics model for uncertainty estimation.
    
    This model learns to predict next states and rewards given current states and actions.
    Uses an ensemble of neural networks to estimate prediction uncertainty.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_networks: int = 5,
        num_layers: int = 4,
        activation: str = "relu",
        dropout_rate: float = 0.1,
    ):
        """Initialize ensemble dynamics model.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Dimension of action space.
            hidden_dim: Hidden layer dimension.
            num_networks: Number of networks in ensemble.
            num_layers: Number of hidden layers.
            activation: Activation function name.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_networks = num_networks
        
        # Create ensemble of networks
        self.networks = nn.ModuleList([
            self._create_network(state_dim, action_dim, hidden_dim, num_layers, activation, dropout_rate)
            for _ in range(num_networks)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _create_network(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int, 
        num_layers: int,
        activation: str,
        dropout_rate: float
    ) -> nn.Module:
        """Create a single dynamics network.
        
        Args:
            state_dim: State dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden dimension.
            num_layers: Number of layers.
            activation: Activation function.
            dropout_rate: Dropout rate.
            
        Returns:
            Neural network module.
        """
        layers = []
        input_dim = state_dim + action_dim
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            
        # Output layer (predicts next state and reward)
        layers.append(nn.Linear(hidden_dim, state_dim + 1))  # +1 for reward
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights.
        
        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ensemble.
        
        Args:
            states: Current states.
            actions: Actions taken.
            
        Returns:
            Tuple of (next_states, rewards) predictions.
        """
        inputs = torch.cat([states, actions], dim=-1)
        
        predictions = []
        for network in self.networks:
            pred = network(inputs)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_networks, batch_size, state_dim + 1]
        
        next_states = predictions[:, :, :self.state_dim]
        rewards = predictions[:, :, self.state_dim:]
        
        return next_states, rewards
    
    def predict_with_uncertainty(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next states and rewards with uncertainty estimates.
        
        Args:
            states: Current states.
            actions: Actions taken.
            
        Returns:
            Tuple of (mean_next_states, std_next_states, mean_rewards, std_rewards).
        """
        next_states, rewards = self.forward(states, actions)
        
        mean_next_states = torch.mean(next_states, dim=0)
        std_next_states = torch.std(next_states, dim=0)
        
        mean_rewards = torch.mean(rewards, dim=0)
        std_rewards = torch.std(rewards, dim=0)
        
        return mean_next_states, std_next_states, mean_rewards, std_rewards


class ProbabilisticDynamicsModel(nn.Module):
    """Probabilistic dynamics model using Gaussian distributions.
    
    This model learns to predict the distribution of next states and rewards,
    providing better uncertainty quantification than deterministic models.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        activation: str = "relu",
        min_std: float = 0.01,
    ):
        """Initialize probabilistic dynamics model.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Dimension of action space.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
            activation: Activation function name.
            min_std: Minimum standard deviation for numerical stability.
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.min_std = min_std
        
        # Shared encoder
        layers = []
        input_dim = state_dim + action_dim
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.encoder = nn.Sequential(*layers)
        
        # Output heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, state_dim + 1)  # +1 for reward
        self.log_std_head = nn.Linear(hidden_dim, state_dim + 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights.
        
        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[Normal, Normal]:
        """Forward pass returning distributions.
        
        Args:
            states: Current states.
            actions: Actions taken.
            
        Returns:
            Tuple of (next_state_distribution, reward_distribution).
        """
        inputs = torch.cat([states, actions], dim=-1)
        features = F.relu(self.encoder(inputs))
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        std = torch.exp(log_std) + self.min_std
        
        # Split into next state and reward components
        next_state_mean = mean[:, :self.state_dim]
        next_state_std = std[:, :self.state_dim]
        reward_mean = mean[:, self.state_dim:]
        reward_std = std[:, self.state_dim:]
        
        next_state_dist = Normal(next_state_mean, next_state_std)
        reward_dist = Normal(reward_mean, reward_std)
        
        return next_state_dist, reward_dist
    
    def sample(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the learned distributions.
        
        Args:
            states: Current states.
            actions: Actions taken.
            
        Returns:
            Tuple of (sampled_next_states, sampled_rewards).
        """
        next_state_dist, reward_dist = self.forward(states, actions)
        
        next_states = next_state_dist.sample()
        rewards = reward_dist.sample()
        
        return next_states, rewards
    
    def log_prob(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        next_states: torch.Tensor, 
        rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities of observed transitions.
        
        Args:
            states: Current states.
            actions: Actions taken.
            next_states: Observed next states.
            rewards: Observed rewards.
            
        Returns:
            Tuple of (next_state_log_probs, reward_log_probs).
        """
        next_state_dist, reward_dist = self.forward(states, actions)
        
        next_state_log_probs = next_state_dist.log_prob(next_states)
        reward_log_probs = reward_dist.log_prob(rewards)
        
        return next_state_log_probs, reward_log_probs


class ReplayBuffer:
    """Experience replay buffer for storing transitions."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store.
            state_dim: State dimension.
            action_dim: Action dimension.
            device: Device to store tensors on.
        """
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate tensors for efficiency
        self.states = torch.zeros((capacity, state_dim), device=device)
        self.actions = torch.zeros((capacity, action_dim), device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.next_states = torch.zeros((capacity, state_dim), device=device)
        self.dones = torch.zeros((capacity, 1), device=device)
    
    def add(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """Add a transition to the buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode ended.
        """
        self.states[self.ptr] = torch.from_numpy(state).float()
        self.actions[self.ptr] = torch.from_numpy(action).float()
        self.rewards[self.ptr] = torch.tensor(reward).float()
        self.next_states[self.ptr] = torch.from_numpy(next_state).float()
        self.dones[self.ptr] = torch.tensor(done).float()
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
