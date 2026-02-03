"""Comprehensive evaluation metrics for model-based reinforcement learning."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats

from ..algorithms import PETSAgent, MBPOAgent
from ..utils import get_device


class Evaluator:
    """Comprehensive evaluator for model-based RL agents."""
    
    def __init__(
        self,
        agent: Union[PETSAgent, MBPOAgent],
        env_name: str,
        num_eval_episodes: int = 100,
        confidence_level: float = 0.95,
        device: Optional[torch.device] = None,
    ):
        """Initialize evaluator.
        
        Args:
            agent: Model-based RL agent to evaluate.
            env_name: Name of the environment.
            num_eval_episodes: Number of episodes for evaluation.
            confidence_level: Confidence level for confidence intervals.
            device: Device to run evaluation on.
        """
        self.agent = agent
        self.env_name = env_name
        self.num_eval_episodes = num_eval_episodes
        self.confidence_level = confidence_level
        self.device = device or get_device()
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_performance(self) -> Dict[str, Any]:
        """Evaluate agent performance comprehensively.
        
        Returns:
            Dictionary containing all evaluation metrics.
        """
        self.logger.info(f"Starting evaluation with {self.num_eval_episodes} episodes")
        
        # Basic performance metrics
        performance_metrics = self._evaluate_performance()
        
        # Sample efficiency metrics
        efficiency_metrics = self._evaluate_sample_efficiency()
        
        # Stability and robustness metrics
        stability_metrics = self._evaluate_stability()
        
        # Model quality metrics
        model_metrics = self._evaluate_model_quality()
        
        # Combine all metrics
        all_metrics = {
            "performance": performance_metrics,
            "efficiency": efficiency_metrics,
            "stability": stability_metrics,
            "model_quality": model_metrics,
        }
        
        self.logger.info("Evaluation completed")
        return all_metrics
    
    def _evaluate_performance(self) -> Dict[str, Any]:
        """Evaluate basic performance metrics.
        
        Returns:
            Performance metrics.
        """
        import gymnasium as gym
        
        env = gym.make(self.env_name)
        episode_rewards = []
        episode_lengths = []
        success_rates = []
        
        for episode in range(self.num_eval_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            for step in range(1000):  # Max steps
                action = self.agent.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                state = next_state
                
                if done:
                    # Check if episode was successful
                    success = info.get("is_success", False) if isinstance(info, dict) else False
                    success_rates.append(success)
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        env.close()
        
        # Compute statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        ci_reward = self._compute_confidence_interval(episode_rewards)
        
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        success_rate = np.mean(success_rates) if success_rates else 0.0
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "reward_ci": ci_reward,
            "mean_length": mean_length,
            "std_length": std_length,
            "success_rate": success_rate,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }
    
    def _evaluate_sample_efficiency(self) -> Dict[str, Any]:
        """Evaluate sample efficiency metrics.
        
        Returns:
            Sample efficiency metrics.
        """
        # This would typically require training history
        # For now, return placeholder metrics
        return {
            "samples_to_threshold": None,
            "learning_curve_slope": None,
            "convergence_episode": None,
        }
    
    def _evaluate_stability(self) -> Dict[str, Any]:
        """Evaluate stability and robustness metrics.
        
        Returns:
            Stability metrics.
        """
        # Evaluate across multiple seeds
        seed_rewards = []
        
        for seed in range(5):  # Test with 5 different seeds
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            import gymnasium as gym
            env = gym.make(self.env_name)
            
            episode_rewards = []
            for _ in range(10):  # 10 episodes per seed
                state, _ = env.reset(seed=seed)
                episode_reward = 0.0
                
                for step in range(1000):
                    action = self.agent.select_action(state, deterministic=True)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    state = next_state
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
            
            env.close()
            seed_rewards.append(np.mean(episode_rewards))
        
        # Compute stability metrics
        reward_variance = np.var(seed_rewards)
        reward_range = np.max(seed_rewards) - np.min(seed_rewards)
        
        return {
            "reward_variance": reward_variance,
            "reward_range": reward_range,
            "seed_rewards": seed_rewards,
            "coefficient_of_variation": np.std(seed_rewards) / np.mean(seed_rewards),
        }
    
    def _evaluate_model_quality(self) -> Dict[str, Any]:
        """Evaluate dynamics model quality.
        
        Returns:
            Model quality metrics.
        """
        if not hasattr(self.agent, 'dynamics_model'):
            return {"error": "No dynamics model found"}
        
        # Collect test data
        import gymnasium as gym
        env = gym.make(self.env_name)
        
        test_states = []
        test_actions = []
        test_next_states = []
        test_rewards = []
        
        for _ in range(100):  # Collect 100 transitions
            state, _ = env.reset()
            
            for step in range(10):  # 10 steps per episode
                action = env.action_space.sample()  # Random action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                test_states.append(state)
                test_actions.append(action)
                test_next_states.append(next_state)
                test_rewards.append(reward)
                
                state = next_state
                
                if done:
                    break
        
        env.close()
        
        # Convert to tensors
        test_states = torch.tensor(np.array(test_states), dtype=torch.float32).to(self.device)
        test_actions = torch.tensor(np.array(test_actions), dtype=torch.float32).to(self.device)
        test_next_states = torch.tensor(np.array(test_next_states), dtype=torch.float32).to(self.device)
        test_rewards = torch.tensor(np.array(test_rewards), dtype=torch.float32).to(self.device)
        
        # Evaluate model predictions
        with torch.no_grad():
            if hasattr(self.agent.dynamics_model, 'predict_with_uncertainty'):
                mean_next_states, std_next_states, mean_rewards, std_rewards = self.agent.dynamics_model.predict_with_uncertainty(
                    test_states, test_actions
                )
                pred_next_states = mean_next_states  # Use mean
                pred_rewards = mean_rewards
            else:
                pred_next_states, pred_rewards = self.agent.dynamics_model(test_states, test_actions)
                pred_next_states = pred_next_states.mean(dim=0)
                pred_rewards = pred_rewards.mean(dim=0)
        
        # Compute prediction errors
        state_mse = torch.mean((pred_next_states - test_next_states) ** 2).item()
        reward_mse = torch.mean((pred_rewards.squeeze() - test_rewards) ** 2).item()
        
        # Compute R-squared
        state_r2 = 1 - torch.sum((test_next_states - pred_next_states) ** 2) / torch.sum((test_next_states - test_next_states.mean()) ** 2)
        reward_r2 = 1 - torch.sum((test_rewards - pred_rewards.squeeze()) ** 2) / torch.sum((test_rewards - test_rewards.mean()) ** 2)
        
        return {
            "state_mse": state_mse,
            "reward_mse": reward_mse,
            "state_r2": state_r2.item(),
            "reward_r2": reward_r2.item(),
            "state_mae": torch.mean(torch.abs(pred_next_states - test_next_states)).item(),
            "reward_mae": torch.mean(torch.abs(pred_rewards.squeeze() - test_rewards)).item(),
        }
    
    def _compute_confidence_interval(self, data: List[float]) -> Tuple[float, float]:
        """Compute confidence interval for data.
        
        Args:
            data: List of values.
            
        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        alpha = 1 - self.confidence_level
        n = len(data)
        
        if n < 2:
            return (data[0], data[0]) if data else (0.0, 0.0)
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        # Use t-distribution for small samples
        if n < 30:
            t_val = stats.t.ppf(1 - alpha/2, n - 1)
            margin = t_val * std / np.sqrt(n)
        else:
            # Use normal distribution for large samples
            z_val = stats.norm.ppf(1 - alpha/2)
            margin = z_val * std / np.sqrt(n)
        
        return (mean - margin, mean + margin)
    
    def compare_agents(self, other_agent: Union[PETSAgent, MBPOAgent]) -> Dict[str, Any]:
        """Compare two agents.
        
        Args:
            other_agent: Another agent to compare with.
            
        Returns:
            Comparison metrics.
        """
        # Evaluate both agents
        self_metrics = self.evaluate_performance()
        
        other_evaluator = Evaluator(other_agent, self.env_name, self.num_eval_episodes)
        other_metrics = other_evaluator.evaluate_performance()
        
        # Compute comparison metrics
        reward_diff = (self_metrics["performance"]["mean_reward"] - 
                      other_metrics["performance"]["mean_reward"])
        
        length_diff = (self_metrics["performance"]["mean_length"] - 
                      other_metrics["performance"]["mean_length"])
        
        success_diff = (self_metrics["performance"]["success_rate"] - 
                       other_metrics["performance"]["success_rate"])
        
        return {
            "reward_difference": reward_diff,
            "length_difference": length_diff,
            "success_rate_difference": success_diff,
            "self_metrics": self_metrics,
            "other_metrics": other_metrics,
        }


class AblationStudy:
    """Conduct ablation studies for model-based RL components."""
    
    def __init__(self, base_config: Dict[str, Any], env_name: str):
        """Initialize ablation study.
        
        Args:
            base_config: Base configuration for the agent.
            env_name: Name of the environment.
        """
        self.base_config = base_config
        self.env_name = env_name
    
    def run_ablation(self, ablation_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run ablation study.
        
        Args:
            ablation_configs: Dictionary mapping ablation names to config modifications.
            
        Returns:
            Results of ablation study.
        """
        results = {}
        
        for ablation_name, config_modifications in ablation_configs.items():
            # Create modified config
            config = self.base_config.copy()
            config.update(config_modifications)
            
            # Train and evaluate agent
            # This would require implementing the full training loop
            # For now, return placeholder results
            results[ablation_name] = {
                "config": config,
                "performance": None,  # Would be filled with actual results
            }
        
        return results
