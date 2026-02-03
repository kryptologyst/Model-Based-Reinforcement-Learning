"""Advanced model-based reinforcement learning algorithms."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ..models import EnsembleDynamicsModel, ProbabilisticDynamicsModel, ReplayBuffer
from ..utils import get_device


class ModelPredictiveControl:
    """Model Predictive Control (MPC) planner for model-based RL.
    
    Uses a learned dynamics model to plan actions by optimizing over a finite horizon.
    """
    
    def __init__(
        self,
        dynamics_model: Union[EnsembleDynamicsModel, ProbabilisticDynamicsModel],
        horizon: int = 10,
        num_candidates: int = 1000,
        num_iterations: int = 5,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """Initialize MPC planner.
        
        Args:
            dynamics_model: Learned dynamics model.
            horizon: Planning horizon.
            num_candidates: Number of action sequences to sample.
            num_iterations: Number of optimization iterations.
            temperature: Temperature for action sampling.
            device: Device to run computations on.
        """
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.device = device or get_device()
        
        self.action_dim = dynamics_model.action_dim
        self.state_dim = dynamics_model.state_dim
        
    def plan(
        self, 
        state: np.ndarray, 
        action_bounds: Tuple[float, float] = (-1.0, 1.0)
    ) -> np.ndarray:
        """Plan actions using MPC.
        
        Args:
            state: Current state.
            action_bounds: Action bounds (min, max).
            
        Returns:
            Planned action sequence.
        """
        state_tensor = torch.from_numpy(state).float().to(self.device)
        
        # Initialize random action sequences
        actions = torch.randn(
            (self.num_candidates, self.horizon, self.action_dim),
            device=self.device
        ) * self.temperature
        
        # Clip actions to bounds
        actions = torch.clamp(actions, action_bounds[0], action_bounds[1])
        
        best_reward = float('-inf')
        best_actions = None
        
        for iteration in range(self.num_iterations):
            # Evaluate action sequences
            rewards = self._evaluate_sequences(state_tensor, actions)
            
            # Select best sequences
            _, top_indices = torch.topk(rewards, self.num_candidates // 2)
            top_actions = actions[top_indices]
            
            # Refine top sequences with noise
            noise = torch.randn_like(top_actions) * 0.1
            refined_actions = top_actions + noise
            refined_actions = torch.clamp(refined_actions, action_bounds[0], action_bounds[1])
            
            # Combine top and refined actions
            actions = torch.cat([top_actions, refined_actions], dim=0)
            
            # Update best
            if rewards.max() > best_reward:
                best_reward = rewards.max().item()
                best_actions = actions[rewards.argmax()].cpu().numpy()
        
        return best_actions[0] if best_actions is not None else np.zeros(self.action_dim)
    
    def _evaluate_sequences(
        self, 
        initial_state: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate action sequences using the dynamics model.
        
        Args:
            initial_state: Initial state.
            actions: Action sequences to evaluate.
            
        Returns:
            Total rewards for each sequence.
        """
        batch_size = actions.shape[0]
        states = initial_state.unsqueeze(0).repeat(batch_size, 1)
        total_rewards = torch.zeros(batch_size, device=self.device)
        
        for t in range(self.horizon):
            action_t = actions[:, t]
            
            if isinstance(self.dynamics_model, EnsembleDynamicsModel):
                mean_next_states, std_next_states, mean_rewards, std_rewards = self.dynamics_model.predict_with_uncertainty(states, action_t)
                next_states = mean_next_states  # Use mean prediction
                rewards = mean_rewards
            else:
                next_states, rewards = self.dynamics_model.sample(states, action_t)
            
            total_rewards += rewards.squeeze()
            states = next_states
        
        return total_rewards


class PETSAgent:
    """Probabilistic Ensemble Trajectory Sampling (PETS) agent.
    
    Combines ensemble dynamics models with trajectory sampling for planning.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_networks: int = 5,
        horizon: int = 15,
        num_candidates: int = 1000,
        num_iterations: int = 5,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        device: Optional[torch.device] = None,
    ):
        """Initialize PETS agent.
        
        Args:
            state_dim: State dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
            num_networks: Number of networks in ensemble.
            horizon: Planning horizon.
            num_candidates: Number of action candidates.
            num_iterations: Number of optimization iterations.
            learning_rate: Learning rate for dynamics model.
            batch_size: Training batch size.
            device: Device to run computations on.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or get_device()
        
        # Initialize dynamics model
        self.dynamics_model = EnsembleDynamicsModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_networks=num_networks,
        ).to(self.device)
        
        # Initialize MPC planner
        self.planner = ModelPredictiveControl(
            dynamics_model=self.dynamics_model,
            horizon=horizon,
            num_candidates=num_candidates,
            num_iterations=num_iterations,
            device=self.device,
        )
        
        # Training components
        self.optimizer = torch.optim.Adam(self.dynamics_model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=100000,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
        )
        
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using MPC planning.
        
        Args:
            state: Current state.
            
        Returns:
            Selected action.
        """
        return self.planner.plan(state)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode ended.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train_dynamics_model(self, num_epochs: int = 5) -> Dict[str, float]:
        """Train the dynamics model.
        
        Args:
            num_epochs: Number of training epochs.
            
        Returns:
            Training metrics.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}
        
        total_loss = 0.0
        
        for epoch in range(num_epochs):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            
            # Predict next states and rewards
            pred_next_states, pred_rewards = self.dynamics_model(states, actions)
            
            # Compute losses
            state_loss = self.criterion(pred_next_states.mean(dim=0), next_states)
            reward_loss = self.criterion(pred_rewards.mean(dim=0), rewards)
            
            total_loss = state_loss + reward_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        return {"loss": total_loss.item()}


class MBPOAgent:
    """Model-Based Policy Optimization (MBPO) agent.
    
    Combines model-based planning with policy gradient methods.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_networks: int = 7,
        horizon: int = 1,
        rollout_batch_size: int = 1000,
        real_ratio: float = 0.05,
        learning_rate: float = 3e-4,
        device: Optional[torch.device] = None,
    ):
        """Initialize MBPO agent.
        
        Args:
            state_dim: State dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
            num_networks: Number of networks in ensemble.
            horizon: Model rollout horizon.
            rollout_batch_size: Batch size for model rollouts.
            real_ratio: Ratio of real data in training.
            learning_rate: Learning rate.
            device: Device to run computations on.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.rollout_batch_size = rollout_batch_size
        self.real_ratio = real_ratio
        self.device = device or get_device()
        
        # Initialize dynamics model
        self.dynamics_model = EnsembleDynamicsModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_networks=num_networks,
        ).to(self.device)
        
        # Initialize policy (simple Gaussian policy)
        self.policy = self._create_policy(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Training components
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics_model.parameters(), lr=learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Replay buffers
        self.real_buffer = ReplayBuffer(100000, state_dim, action_dim, self.device)
        self.model_buffer = ReplayBuffer(100000, state_dim, action_dim, self.device)
        
    def _create_policy(self, state_dim: int, action_dim: int, hidden_dim: int) -> nn.Module:
        """Create a Gaussian policy network.
        
        Args:
            state_dim: State dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden dimension.
            
        Returns:
            Policy network.
        """
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),  # mean and log_std
        )
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using the policy.
        
        Args:
            state: Current state.
            deterministic: Whether to use deterministic action selection.
            
        Returns:
            Selected action.
        """
        state_tensor = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            output = self.policy(state_tensor)
            mean, log_std = output[:self.action_dim], output[self.action_dim:]
            std = torch.exp(log_std)
            
            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
        
        return action.cpu().numpy()
    
    def store_real_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store real transition in replay buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode ended.
        """
        self.real_buffer.add(state, action, reward, next_state, done)
    
    def train_dynamics_model(self) -> Dict[str, float]:
        """Train the dynamics model.
        
        Returns:
            Training metrics.
        """
        if len(self.real_buffer) < 1000:
            return {"dynamics_loss": 0.0}
        
        # Sample real data
        states, actions, rewards, next_states, dones = self.real_buffer.sample(1000)
        
        # Train dynamics model
        pred_next_states, pred_rewards = self.dynamics_model(states, actions)
        
        state_loss = F.mse_loss(pred_next_states.mean(dim=0), next_states)
        reward_loss = F.mse_loss(pred_rewards.mean(dim=0), rewards)
        dynamics_loss = state_loss + reward_loss
        
        self.dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        self.dynamics_optimizer.step()
        
        return {"dynamics_loss": dynamics_loss.item()}
    
    def generate_model_rollouts(self) -> None:
        """Generate model rollouts and store in model buffer."""
        if len(self.real_buffer) < 1000:
            return
        
        # Sample initial states from real buffer
        states, _, _, _, _ = self.real_buffer.sample(self.rollout_batch_size)
        
        for step in range(self.horizon):
            # Select actions using policy
            with torch.no_grad():
                policy_output = self.policy(states)
                mean = policy_output[:, :self.action_dim]
                log_std = policy_output[:, self.action_dim:]
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                actions = dist.sample()
            
            # Predict next states and rewards
            mean_next_states, std_next_states, mean_rewards, std_rewards = self.dynamics_model.predict_with_uncertainty(states, actions)
            
            # Add noise for exploration
            noise = torch.randn_like(mean_next_states) * 0.01
            next_states = mean_next_states + noise
            
            # Store in model buffer
            for i in range(self.rollout_batch_size):
                self.model_buffer.add(
                    states[i].cpu().numpy(),
                    actions[i].cpu().numpy(),
                    mean_rewards[i].item(),
                    next_states[i].cpu().numpy(),
                    False,  # Model rollouts don't have terminal states
                )
            
            states = next_states
    
    def train_policy(self) -> Dict[str, float]:
        """Train the policy using both real and model data.
        
        Returns:
            Training metrics.
        """
        # Determine batch sizes
        real_batch_size = int(self.batch_size * self.real_ratio)
        model_batch_size = self.batch_size - real_batch_size
        
        # Sample from both buffers
        if len(self.real_buffer) >= real_batch_size:
            real_data = self.real_buffer.sample(real_batch_size)
        else:
            real_data = None
        
        if len(self.model_buffer) >= model_batch_size:
            model_data = self.model_buffer.sample(model_batch_size)
        else:
            model_data = None
        
        # Combine data
        if real_data is not None and model_data is not None:
            states = torch.cat([real_data[0], model_data[0]], dim=0)
            actions = torch.cat([real_data[1], model_data[1]], dim=0)
            rewards = torch.cat([real_data[2], model_data[2]], dim=0)
            next_states = torch.cat([real_data[3], model_data[3]], dim=0)
            dones = torch.cat([real_data[4], model_data[4]], dim=0)
        elif real_data is not None:
            states, actions, rewards, next_states, dones = real_data
        elif model_data is not None:
            states, actions, rewards, next_states, dones = model_data
        else:
            return {"policy_loss": 0.0}
        
        # Compute policy loss (simplified REINFORCE)
        policy_output = self.policy(states)
        mean = policy_output[:, :self.action_dim]
        log_std = policy_output[:, self.action_dim:]
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Use rewards as advantages (simplified)
        policy_loss = -(log_probs * rewards.squeeze()).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {"policy_loss": policy_loss.item()}
    
    @property
    def batch_size(self) -> int:
        """Get batch size for training."""
        return 256
