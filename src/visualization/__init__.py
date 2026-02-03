"""Visualization utilities for model-based reinforcement learning."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.animation import FuncAnimation

from ..algorithms import PETSAgent, MBPOAgent
from ..utils import get_device


class Visualizer:
    """Visualization utilities for model-based RL."""
    
    def __init__(self, output_dir: str = "assets"):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.logger = logging.getLogger(__name__)
    
    def plot_learning_curves(
        self, 
        episode_rewards: List[float], 
        episode_lengths: List[int],
        training_losses: List[Dict[str, float]],
        title: str = "Learning Curves"
    ) -> None:
        """Plot learning curves.
        
        Args:
            episode_rewards: List of episode rewards.
            episode_lengths: List of episode lengths.
            training_losses: List of training loss dictionaries.
            title: Plot title.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(episode_rewards, alpha=0.7)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        
        # Moving average of rewards
        window_size = min(50, len(episode_rewards) // 10)
        if window_size > 1:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            axes[0, 0].plot(range(window_size-1, len(episode_rewards)), moving_avg, 
                           color='red', linewidth=2, label=f'Moving Avg ({window_size})')
            axes[0, 0].legend()
        
        # Episode lengths
        axes[0, 1].plot(episode_lengths, alpha=0.7)
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Length")
        
        # Training losses
        if training_losses:
            dynamics_losses = [loss.get("dynamics_loss", 0) for loss in training_losses]
            policy_losses = [loss.get("policy_loss", 0) for loss in training_losses]
            
            axes[1, 0].plot(dynamics_losses, label="Dynamics Loss", alpha=0.7)
            if any(policy_losses):
                axes[1, 0].plot(policy_losses, label="Policy Loss", alpha=0.7)
            axes[1, 0].set_title("Training Losses")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].legend()
            axes[1, 0].set_yscale('log')
        
        # Reward distribution
        axes[1, 1].hist(episode_rewards, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title("Reward Distribution")
        axes[1, 1].set_xlabel("Reward")
        axes[1, 1].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Learning curves saved to {self.output_dir / 'learning_curves.png'}")
    
    def plot_model_predictions(
        self,
        agent: Union[PETSAgent, MBPOAgent],
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
        title: str = "Model Predictions"
    ) -> None:
        """Plot model predictions vs actual values.
        
        Args:
            agent: Trained agent.
            states: Input states.
            actions: Input actions.
            next_states: Actual next states.
            rewards: Actual rewards.
            title: Plot title.
        """
        device = get_device()
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            if hasattr(agent.dynamics_model, 'predict_with_uncertainty'):
                mean_next_states, std_next_states, mean_rewards, std_rewards = agent.dynamics_model.predict_with_uncertainty(
                    states_tensor, actions_tensor
                )
                pred_next_states = mean_next_states.cpu().numpy()
                pred_rewards = mean_rewards.cpu().numpy()
            else:
                pred_next_states, pred_rewards = agent.dynamics_model(states_tensor, actions_tensor)
                pred_next_states = pred_next_states.mean(dim=0).cpu().numpy()
                pred_rewards = pred_rewards.mean(dim=0).cpu().numpy()
        
        # Plot next state predictions
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        state_dim = states.shape[1]
        for i in range(min(4, state_dim)):
            row, col = i // 2, i % 2
            
            axes[row, col].scatter(next_states[:, i], pred_next_states[:, i], alpha=0.6)
            axes[row, col].plot([next_states[:, i].min(), next_states[:, i].max()],
                              [next_states[:, i].min(), next_states[:, i].max()], 
                              'r--', alpha=0.8)
            axes[row, col].set_title(f"State Dimension {i}")
            axes[row, col].set_xlabel("Actual")
            axes[row, col].set_ylabel("Predicted")
            
            # Add R² score
            r2 = np.corrcoef(next_states[:, i], pred_next_states[:, i])[0, 1] ** 2
            axes[row, col].text(0.05, 0.95, f"R² = {r2:.3f}", 
                              transform=axes[row, col].transAxes, 
                              bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot reward predictions
        plt.figure(figsize=(8, 6))
        plt.scatter(rewards, pred_rewards.flatten(), alpha=0.6)
        plt.plot([rewards.min(), rewards.max()], [rewards.min(), rewards.max()], 'r--', alpha=0.8)
        plt.title("Reward Predictions")
        plt.xlabel("Actual Reward")
        plt.ylabel("Predicted Reward")
        
        r2_reward = np.corrcoef(rewards, pred_rewards.flatten())[0, 1] ** 2
        plt.text(0.05, 0.95, f"R² = {r2_reward:.3f}", 
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "reward_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model predictions saved to {self.output_dir}")
    
    def plot_uncertainty_analysis(
        self,
        agent: Union[PETSAgent, MBPOAgent],
        states: np.ndarray,
        actions: np.ndarray,
        title: str = "Uncertainty Analysis"
    ) -> None:
        """Plot uncertainty analysis for model predictions.
        
        Args:
            agent: Trained agent.
            states: Input states.
            actions: Input actions.
            title: Plot title.
        """
        device = get_device()
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            if hasattr(agent.dynamics_model, 'predict_with_uncertainty'):
                mean_next_states, std_next_states, mean_rewards, std_rewards = agent.dynamics_model.predict_with_uncertainty(
                    states_tensor, actions_tensor
                )
                pred_next_states_mean = mean_next_states.cpu().numpy()
                pred_next_states_std = std_next_states.cpu().numpy()
                pred_rewards_mean = mean_rewards.cpu().numpy()
                pred_rewards_std = std_rewards.cpu().numpy()
            else:
                # For ensemble models, compute uncertainty manually
                pred_next_states, pred_rewards = agent.dynamics_model(states_tensor, actions_tensor)
                pred_next_states_mean = pred_next_states.mean(dim=0).cpu().numpy()
                pred_next_states_std = pred_next_states.std(dim=0).cpu().numpy()
                pred_rewards_mean = pred_rewards.mean(dim=0).cpu().numpy()
                pred_rewards_std = pred_rewards.std(dim=0).cpu().numpy()
        
        # Plot uncertainty vs prediction magnitude
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # State uncertainty
        state_magnitude = np.linalg.norm(pred_next_states_mean, axis=1)
        state_uncertainty = np.mean(pred_next_states_std, axis=1)
        
        axes[0, 0].scatter(state_magnitude, state_uncertainty, alpha=0.6)
        axes[0, 0].set_title("State Uncertainty vs Magnitude")
        axes[0, 0].set_xlabel("State Magnitude")
        axes[0, 0].set_ylabel("Average Uncertainty")
        
        # Reward uncertainty
        reward_magnitude = np.abs(pred_rewards_mean.flatten())
        reward_uncertainty = pred_rewards_std.flatten()
        
        axes[0, 1].scatter(reward_magnitude, reward_uncertainty, alpha=0.6)
        axes[0, 1].set_title("Reward Uncertainty vs Magnitude")
        axes[0, 1].set_xlabel("Reward Magnitude")
        axes[0, 1].set_ylabel("Uncertainty")
        
        # Uncertainty distribution
        axes[1, 0].hist(state_uncertainty, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title("State Uncertainty Distribution")
        axes[1, 0].set_xlabel("Uncertainty")
        axes[1, 0].set_ylabel("Frequency")
        
        axes[1, 1].hist(reward_uncertainty, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title("Reward Uncertainty Distribution")
        axes[1, 1].set_xlabel("Uncertainty")
        axes[1, 1].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "uncertainty_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Uncertainty analysis saved to {self.output_dir}")
    
    def create_trajectory_animation(
        self,
        agent: Union[PETSAgent, MBPOAgent],
        env_name: str,
        num_episodes: int = 3,
        max_steps: int = 200,
        title: str = "Agent Trajectory"
    ) -> None:
        """Create animation of agent trajectories.
        
        Args:
            agent: Trained agent.
            env_name: Name of the environment.
            num_episodes: Number of episodes to animate.
            max_steps: Maximum steps per episode.
            title: Animation title.
        """
        import gymnasium as gym
        
        env = gym.make(env_name)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate_episode(episode):
            ax.clear()
            ax.set_title(f"{title} - Episode {episode + 1}")
            
            state, _ = env.reset()
            episode_reward = 0.0
            
            states = [state]
            rewards = [0]
            
            for step in range(max_steps):
                action = agent.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                states.append(next_state)
                rewards.append(reward)
                episode_reward += reward
                
                state = next_state
                
                if done:
                    break
            
            # Plot trajectory
            if len(states[0]) >= 2:  # At least 2D state space
                states_array = np.array(states)
                ax.plot(states_array[:, 0], states_array[:, 1], 'b-', alpha=0.7, linewidth=2)
                ax.scatter(states_array[0, 0], states_array[0, 1], color='green', s=100, label='Start')
                ax.scatter(states_array[-1, 0], states_array[-1, 1], color='red', s=100, label='End')
                ax.legend()
                ax.set_xlabel("State Dimension 0")
                ax.set_ylabel("State Dimension 1")
            else:
                # 1D case
                ax.plot(range(len(states)), [s[0] for s in states], 'b-', alpha=0.7, linewidth=2)
                ax.set_xlabel("Step")
                ax.set_ylabel("State Value")
            
            ax.text(0.02, 0.98, f"Total Reward: {episode_reward:.2f}", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        # Create animation
        anim = FuncAnimation(fig, animate_episode, frames=num_episodes, interval=2000, repeat=True)
        
        # Save animation
        anim.save(self.output_dir / "trajectory_animation.gif", writer='pillow', fps=0.5)
        plt.close()
        
        env.close()
        
        self.logger.info(f"Trajectory animation saved to {self.output_dir / 'trajectory_animation.gif'}")
    
    def plot_comparison(
        self,
        results: Dict[str, Any],
        title: str = "Agent Comparison"
    ) -> None:
        """Plot comparison between different agents or configurations.
        
        Args:
            results: Dictionary containing comparison results.
            title: Plot title.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Extract data
        agent_names = list(results.keys())
        mean_rewards = [results[name]["performance"]["mean_reward"] for name in agent_names]
        std_rewards = [results[name]["performance"]["std_reward"] for name in agent_names]
        mean_lengths = [results[name]["performance"]["mean_length"] for name in agent_names]
        success_rates = [results[name]["performance"]["success_rate"] for name in agent_names]
        
        # Mean rewards comparison
        axes[0, 0].bar(agent_names, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
        axes[0, 0].set_title("Mean Rewards")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Mean lengths comparison
        axes[0, 1].bar(agent_names, mean_lengths, alpha=0.7)
        axes[0, 1].set_title("Mean Episode Lengths")
        axes[0, 1].set_ylabel("Length")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Success rates comparison
        axes[1, 0].bar(agent_names, success_rates, alpha=0.7)
        axes[1, 0].set_title("Success Rates")
        axes[1, 0].set_ylabel("Success Rate")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Model quality comparison (if available)
        if "model_quality" in results[agent_names[0]]:
            model_r2 = [results[name]["model_quality"]["state_r2"] for name in agent_names]
            axes[1, 1].bar(agent_names, model_r2, alpha=0.7)
            axes[1, 1].set_title("Model R² Scores")
            axes[1, 1].set_ylabel("R² Score")
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, "Model Quality\nNot Available", 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Model Quality")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "agent_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Agent comparison saved to {self.output_dir / 'agent_comparison.png'}")
