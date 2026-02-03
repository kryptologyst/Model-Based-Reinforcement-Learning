"""Core utilities for model-based reinforcement learning."""

import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env, spaces


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_obs(obs: np.ndarray, running_stats: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Normalize observations using running statistics.
    
    Args:
        obs: Observation array.
        running_stats: Dictionary containing running mean and std.
        
    Returns:
        Normalized observation.
    """
    if running_stats is None:
        return obs
    
    mean = running_stats.get("mean", 0.0)
    std = running_stats.get("std", 1.0)
    return (obs - mean) / (std + 1e-8)


def update_running_stats(
    obs: np.ndarray, 
    running_stats: Dict[str, Any], 
    alpha: float = 0.01
) -> Dict[str, Any]:
    """Update running statistics for observation normalization.
    
    Args:
        obs: Observation array.
        running_stats: Current running statistics.
        alpha: Update rate.
        
    Returns:
        Updated running statistics.
    """
    if "mean" not in running_stats:
        running_stats["mean"] = np.zeros_like(obs)
        running_stats["std"] = np.ones_like(obs)
        running_stats["count"] = 0
    
    running_stats["count"] += 1
    running_stats["mean"] = (1 - alpha) * running_stats["mean"] + alpha * obs
    running_stats["std"] = (1 - alpha) * running_stats["std"] + alpha * np.abs(obs - running_stats["mean"])
    
    return running_stats


def create_env(env_name: str, seed: Optional[int] = None) -> Env:
    """Create and configure a Gymnasium environment.
    
    Args:
        env_name: Name of the environment.
        seed: Random seed for the environment.
        
    Returns:
        Configured environment.
    """
    import gymnasium as gym
    
    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=seed)
    return env


def get_space_size(space: spaces.Space) -> int:
    """Get the size of a Gymnasium space.
    
    Args:
        space: Gymnasium space.
        
    Returns:
        Size of the space.
    """
    if isinstance(space, spaces.Box):
        return space.shape[0] if len(space.shape) == 1 else np.prod(space.shape)
    elif isinstance(space, spaces.Discrete):
        return space.n
    elif isinstance(space, spaces.MultiDiscrete):
        return sum(space.nvec)
    else:
        raise ValueError(f"Unsupported space type: {type(space)}")


class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
            restore_best_weights: Whether to restore best weights when stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss.
            model: Model to potentially restore weights for.
            
        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False
