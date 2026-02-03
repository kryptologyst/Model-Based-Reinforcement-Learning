"""Training utilities for model-based reinforcement learning."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..algorithms import PETSAgent, MBPOAgent
from ..utils import EarlyStopping, get_device, set_seed


class Trainer:
    """Base trainer class for model-based RL agents."""
    
    def __init__(
        self,
        agent: Union[PETSAgent, MBPOAgent],
        env_name: str,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 200,
        eval_frequency: int = 50,
        save_frequency: int = 100,
        log_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        """Initialize trainer.
        
        Args:
            agent: Model-based RL agent.
            env_name: Name of the environment.
            max_episodes: Maximum number of training episodes.
            max_steps_per_episode: Maximum steps per episode.
            eval_frequency: Frequency of evaluation.
            save_frequency: Frequency of saving checkpoints.
            log_dir: Directory for logging.
            device: Device to run training on.
            seed: Random seed.
        """
        self.agent = agent
        self.env_name = env_name
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.eval_frequency = eval_frequency
        self.save_frequency = save_frequency
        self.device = device or get_device()
        
        if seed is not None:
            set_seed(seed)
        
        # Setup logging
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Training metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.training_losses: List[Dict[str, float]] = []
        
    def train(self) -> Dict[str, Any]:
        """Train the agent.
        
        Returns:
            Training results and metrics.
        """
        self.logger.info(f"Starting training for {self.max_episodes} episodes")
        
        start_time = time.time()
        
        for episode in tqdm(range(self.max_episodes), desc="Training"):
            episode_reward, episode_length = self._train_episode()
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Train dynamics model
            if hasattr(self.agent, 'train_dynamics_model'):
                dynamics_metrics = self.agent.train_dynamics_model()
                self.training_losses.append(dynamics_metrics)
            
            # Train policy (for MBPO)
            if hasattr(self.agent, 'train_policy'):
                policy_metrics = self.agent.train_policy()
                if "policy_loss" in policy_metrics:
                    self.training_losses[-1].update(policy_metrics)
            
            # Generate model rollouts (for MBPO)
            if hasattr(self.agent, 'generate_model_rollouts'):
                self.agent.generate_model_rollouts()
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                self.logger.info(
                    f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                    f"Avg Length = {avg_length:.2f}"
                )
            
            # Evaluation
            if episode % self.eval_frequency == 0:
                eval_results = self._evaluate()
                self.logger.info(f"Evaluation at episode {episode}: {eval_results}")
            
            # Save checkpoint
            if episode % self.save_frequency == 0:
                self._save_checkpoint(episode)
        
        training_time = time.time() - start_time
        
        results = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_losses": self.training_losses,
            "training_time": training_time,
            "final_evaluation": self._evaluate(),
        }
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        return results
    
    def _train_episode(self) -> Tuple[float, int]:
        """Train for one episode.
        
        Returns:
            Tuple of (episode_reward, episode_length).
        """
        import gymnasium as gym
        
        env = gym.make(self.env_name)
        state, _ = env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        
        for step in range(self.max_steps_per_episode):
            # Select action
            action = self.agent.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            if hasattr(self.agent, 'store_transition'):
                self.agent.store_transition(state, action, reward, next_state, done)
            elif hasattr(self.agent, 'store_real_transition'):
                self.agent.store_real_transition(state, action, reward, next_state, done)
            
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if done:
                break
        
        env.close()
        return episode_reward, episode_length
    
    def _evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the agent.
        
        Args:
            num_episodes: Number of episodes to evaluate.
            
        Returns:
            Evaluation metrics.
        """
        import gymnasium as gym
        
        env = gym.make(self.env_name)
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            for step in range(self.max_steps_per_episode):
                action = self.agent.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        env.close()
        
        return {
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "mean_length": np.mean(eval_lengths),
            "std_length": np.std(eval_lengths),
        }
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint.
        
        Args:
            episode: Current episode number.
        """
        checkpoint = {
            "episode": episode,
            "agent_state_dict": self.agent.dynamics_model.state_dict(),
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_losses": self.training_losses,
        }
        
        if hasattr(self.agent, 'policy'):
            checkpoint["policy_state_dict"] = self.agent.policy.state_dict()
        
        torch.save(checkpoint, self.log_dir / f"checkpoint_episode_{episode}.pt")
        self.logger.info(f"Checkpoint saved at episode {episode}")


class CurriculumTrainer(Trainer):
    """Trainer with curriculum learning capabilities."""
    
    def __init__(
        self,
        agent: Union[PETSAgent, MBPOAgent],
        env_name: str,
        curriculum_schedule: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        """Initialize curriculum trainer.
        
        Args:
            agent: Model-based RL agent.
            env_name: Name of the environment.
            curriculum_schedule: Curriculum learning schedule.
            **kwargs: Additional arguments for base trainer.
        """
        super().__init__(agent, env_name, **kwargs)
        self.curriculum_schedule = curriculum_schedule or []
        self.current_curriculum_level = 0
    
    def _train_episode(self) -> Tuple[float, int]:
        """Train for one episode with curriculum.
        
        Returns:
            Tuple of (episode_reward, episode_length).
        """
        # Update curriculum level
        self._update_curriculum_level()
        
        # Train episode with current curriculum
        return super()._train_episode()
    
    def _update_curriculum_level(self) -> None:
        """Update curriculum level based on schedule."""
        if not self.curriculum_schedule:
            return
        
        current_episode = len(self.episode_rewards)
        
        for i, schedule in enumerate(self.curriculum_schedule):
            if current_episode >= schedule["start_episode"]:
                self.current_curriculum_level = i
        
        # Apply curriculum modifications
        if self.current_curriculum_level < len(self.curriculum_schedule):
            curriculum = self.curriculum_schedule[self.current_curriculum_level]
            
            # Modify environment parameters if specified
            if "env_params" in curriculum:
                self.logger.info(f"Applying curriculum level {self.current_curriculum_level}: {curriculum['env_params']}")


class HyperparameterScheduler:
    """Scheduler for hyperparameters during training."""
    
    def __init__(self, schedule: Dict[str, List[Dict[str, Any]]]):
        """Initialize hyperparameter scheduler.
        
        Args:
            schedule: Dictionary mapping parameter names to schedules.
        """
        self.schedule = schedule
    
    def get_value(self, param_name: str, episode: int) -> Any:
        """Get parameter value for current episode.
        
        Args:
            param_name: Name of the parameter.
            episode: Current episode number.
            
        Returns:
            Parameter value.
        """
        if param_name not in self.schedule:
            return None
        
        param_schedule = self.schedule[param_name]
        
        for entry in reversed(param_schedule):
            if episode >= entry["start_episode"]:
                return entry["value"]
        
        return param_schedule[0]["value"] if param_schedule else None
