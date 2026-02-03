"""Unit tests for model-based reinforcement learning components."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.models import EnsembleDynamicsModel, ProbabilisticDynamicsModel, ReplayBuffer
from src.algorithms import PETSAgent, MBPOAgent, ModelPredictiveControl
from src.utils import set_seed, get_device, EarlyStopping


class TestEnsembleDynamicsModel:
    """Test cases for EnsembleDynamicsModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = EnsembleDynamicsModel(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        
        assert model.state_dim == 4
        assert model.action_dim == 1
        assert model.num_networks == 3
        assert len(model.networks) == 3
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = EnsembleDynamicsModel(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        
        states = torch.randn(10, 4)
        actions = torch.randn(10, 1)
        
        next_states, rewards = model(states, actions)
        
        assert next_states.shape == (3, 10, 4)  # [num_networks, batch_size, state_dim]
        assert rewards.shape == (3, 10, 1)  # [num_networks, batch_size, 1]
    
    def test_predict_with_uncertainty(self):
        """Test uncertainty prediction."""
        model = EnsembleDynamicsModel(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        
        states = torch.randn(10, 4)
        actions = torch.randn(10, 1)
        
        mean_next_states, std_next_states, mean_rewards, std_rewards = model.predict_with_uncertainty(states, actions)
        
        assert mean_next_states.shape == (10, 4)
        assert std_next_states.shape == (10, 4)
        assert mean_rewards.shape == (10, 1)
        assert std_rewards.shape == (10, 1)
        
        # Check that std values are non-negative
        assert torch.all(std_next_states >= 0)
        assert torch.all(std_rewards >= 0)


class TestProbabilisticDynamicsModel:
    """Test cases for ProbabilisticDynamicsModel."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = ProbabilisticDynamicsModel(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
        )
        
        assert model.state_dim == 4
        assert model.action_dim == 1
        assert model.min_std > 0
    
    def test_forward_pass(self):
        """Test forward pass returns distributions."""
        model = ProbabilisticDynamicsModel(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
        )
        
        states = torch.randn(10, 4)
        actions = torch.randn(10, 1)
        
        next_state_dist, reward_dist = model(states, actions)
        
        assert hasattr(next_state_dist, 'sample')
        assert hasattr(reward_dist, 'sample')
        assert next_state_dist.mean.shape == (10, 4)
        assert reward_dist.mean.shape == (10, 1)
    
    def test_sample(self):
        """Test sampling from distributions."""
        model = ProbabilisticDynamicsModel(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
        )
        
        states = torch.randn(10, 4)
        actions = torch.randn(10, 1)
        
        next_states, rewards = model.sample(states, actions)
        
        assert next_states.shape == (10, 4)
        assert rewards.shape == (10, 1)


class TestReplayBuffer:
    """Test cases for ReplayBuffer."""
    
    def test_initialization(self):
        """Test buffer initialization."""
        buffer = ReplayBuffer(
            capacity=1000,
            state_dim=4,
            action_dim=1,
            device=torch.device('cpu'),
        )
        
        assert buffer.capacity == 1000
        assert buffer.size == 0
        assert buffer.ptr == 0
    
    def test_add_and_sample(self):
        """Test adding and sampling transitions."""
        buffer = ReplayBuffer(
            capacity=1000,
            state_dim=4,
            action_dim=1,
            device=torch.device('cpu'),
        )
        
        # Add some transitions
        for i in range(10):
            state = np.random.randn(4)
            action = np.random.randn(1)
            reward = float(i)
            next_state = np.random.randn(4)
            done = i % 3 == 0
            
            buffer.add(state, action, reward, next_state, done)
        
        assert buffer.size == 10
        
        # Sample batch
        batch = buffer.sample(5)
        states, actions, rewards, next_states, dones = batch
        
        assert states.shape == (5, 4)
        assert actions.shape == (5, 1)
        assert rewards.shape == (5, 1)
        assert next_states.shape == (5, 4)
        assert dones.shape == (5, 1)


class TestPETSAgent:
    """Test cases for PETSAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = PETSAgent(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        
        assert agent.state_dim == 4
        assert agent.action_dim == 1
        assert agent.dynamics_model.num_networks == 3
    
    def test_select_action(self):
        """Test action selection."""
        agent = PETSAgent(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        
        state = np.random.randn(4)
        action = agent.select_action(state)
        
        assert action.shape == (1,)
        assert isinstance(action, np.ndarray)
    
    def test_store_transition(self):
        """Test storing transitions."""
        agent = PETSAgent(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        
        state = np.random.randn(4)
        action = np.random.randn(1)
        reward = 1.0
        next_state = np.random.randn(4)
        done = False
        
        agent.store_transition(state, action, reward, next_state, done)
        
        assert len(agent.replay_buffer) == 1


class TestMBPOAgent:
    """Test cases for MBPOAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = MBPOAgent(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        
        assert agent.state_dim == 4
        assert agent.action_dim == 1
        assert hasattr(agent, 'policy')
    
    def test_select_action(self):
        """Test action selection."""
        agent = MBPOAgent(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        
        state = np.random.randn(4)
        action = agent.select_action(state)
        
        assert action.shape == (1,)
        assert isinstance(action, np.ndarray)
    
    def test_store_real_transition(self):
        """Test storing real transitions."""
        agent = MBPOAgent(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        
        state = np.random.randn(4)
        action = np.random.randn(1)
        reward = 1.0
        next_state = np.random.randn(4)
        done = False
        
        agent.store_real_transition(state, action, reward, next_state, done)
        
        assert len(agent.real_buffer) == 1


class TestModelPredictiveControl:
    """Test cases for ModelPredictiveControl."""
    
    def test_initialization(self):
        """Test MPC initialization."""
        dynamics_model = EnsembleDynamicsModel(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        
        mpc = ModelPredictiveControl(
            dynamics_model=dynamics_model,
            horizon=5,
            num_candidates=100,
        )
        
        assert mpc.horizon == 5
        assert mpc.num_candidates == 100
    
    def test_plan(self):
        """Test MPC planning."""
        dynamics_model = EnsembleDynamicsModel(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        
        mpc = ModelPredictiveControl(
            dynamics_model=dynamics_model,
            horizon=5,
            num_candidates=100,
        )
        
        state = np.random.randn(4)
        action = mpc.plan(state)
        
        assert action.shape == (1,)
        assert isinstance(action, np.ndarray)


class TestUtilities:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Check that random numbers are reproducible
        np.random.seed(42)
        val1 = np.random.randn()
        
        set_seed(42)
        np.random.seed(42)
        val2 = np.random.randn()
        
        assert val1 == val2
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        
        # Mock model
        model = Mock()
        model.state_dict.return_value = {}
        
        # Test that it doesn't stop early with improving loss
        assert not early_stopping(0.1, model)
        assert not early_stopping(0.05, model)
        assert not early_stopping(0.02, model)
        
        # Test that it stops after patience is exceeded
        assert early_stopping(0.03, model)  # Worse than best (0.02)
        assert early_stopping(0.04, model)  # Still worse
        assert early_stopping(0.05, model)  # Should trigger early stopping


class TestIntegration:
    """Integration tests."""
    
    def test_training_loop(self):
        """Test basic training loop."""
        agent = PETSAgent(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        
        # Add some transitions
        for i in range(100):
            state = np.random.randn(4)
            action = np.random.randn(1)
            reward = float(i)
            next_state = np.random.randn(4)
            done = i % 10 == 0
            
            agent.store_transition(state, action, reward, next_state, done)
        
        # Train dynamics model
        metrics = agent.train_dynamics_model(num_epochs=2)
        
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
    
    @patch('gymnasium.make')
    def test_evaluation(self, mock_gym):
        """Test evaluation with mocked environment."""
        # Mock environment
        mock_env = Mock()
        mock_env.reset.return_value = (np.random.randn(4), {})
        mock_env.step.return_value = (np.random.randn(4), 1.0, False, False, {})
        mock_env.close.return_value = None
        mock_gym.return_value = mock_env
        
        agent = PETSAgent(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        
        # Add some transitions for evaluation
        for i in range(50):
            state = np.random.randn(4)
            action = np.random.randn(1)
            reward = float(i)
            next_state = np.random.randn(4)
            done = i % 10 == 0
            
            agent.store_transition(state, action, reward, next_state, done)
        
        # Train a bit
        agent.train_dynamics_model(num_epochs=1)
        
        # Test action selection
        state = np.random.randn(4)
        action = agent.select_action(state)
        
        assert action.shape == (1,)
        assert isinstance(action, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__])
