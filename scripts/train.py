#!/usr/bin/env python3
"""Main training script for model-based reinforcement learning."""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from omegaconf import OmegaConf

from src.algorithms import PETSAgent, MBPOAgent
from src.training import Trainer
from src.evaluation import Evaluator
from src.utils import get_device, set_seed


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_agent(config: Dict[str, Any]) -> Any:
    """Create agent based on configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Initialized agent.
    """
    agent_config = config["agent"]
    env_config = config["env"]
    
    # Get device
    device = get_device() if config["training"]["device"] == "auto" else torch.device(config["training"]["device"])
    
    if agent_config["type"] == "PETS":
        agent = PETSAgent(
            state_dim=agent_config["state_dim"],
            action_dim=agent_config["action_dim"],
            hidden_dim=agent_config["hidden_dim"],
            num_networks=agent_config["num_networks"],
            horizon=agent_config["horizon"],
            num_candidates=agent_config["num_candidates"],
            num_iterations=agent_config["num_iterations"],
            learning_rate=agent_config["learning_rate"],
            batch_size=agent_config["batch_size"],
            device=device,
        )
    elif agent_config["type"] == "MBPO":
        agent = MBPOAgent(
            state_dim=agent_config["state_dim"],
            action_dim=agent_config["action_dim"],
            hidden_dim=agent_config["hidden_dim"],
            num_networks=agent_config["num_networks"],
            horizon=agent_config["horizon"],
            rollout_batch_size=agent_config["rollout_batch_size"],
            real_ratio=agent_config["real_ratio"],
            learning_rate=agent_config["learning_rate"],
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_config['type']}")
    
    return agent


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train model-based RL agent")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file path")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config["training"]["seed"])
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Agent type: {config['agent']['type']}")
    logger.info(f"Environment: {config['env']['name']}")
    
    # Create agent
    agent = create_agent(config)
    logger.info(f"Created {config['agent']['type']} agent")
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        agent.dynamics_model.load_state_dict(checkpoint["agent_state_dict"])
        if "policy_state_dict" in checkpoint:
            agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    
    if not args.eval_only:
        # Create trainer
        trainer = Trainer(
            agent=agent,
            env_name=config["env"]["name"],
            max_episodes=config["training"]["max_episodes"],
            max_steps_per_episode=config["env"]["max_steps_per_episode"],
            eval_frequency=config["training"]["eval_frequency"],
            save_frequency=config["training"]["save_frequency"],
            log_dir=str(output_dir),
            device=get_device(),
            seed=config["training"]["seed"],
        )
        
        # Train agent
        logger.info("Starting training...")
        training_results = trainer.train()
        
        # Save training results
        torch.save(training_results, output_dir / "training_results.pt")
        logger.info(f"Training completed. Results saved to {output_dir}")
    
    # Evaluate agent
    logger.info("Starting evaluation...")
    evaluator = Evaluator(
        agent=agent,
        env_name=config["env"]["name"],
        num_eval_episodes=config["evaluation"]["num_eval_episodes"],
        confidence_level=config["evaluation"]["confidence_level"],
    )
    
    eval_results = evaluator.evaluate_performance()
    
    # Save evaluation results
    torch.save(eval_results, output_dir / "evaluation_results.pt")
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    logger.info(f"Mean Length: {eval_results['mean_length']:.2f} ± {eval_results['std_length']:.2f}")
    logger.info(f"Success Rate: {eval_results['success_rate']:.2f}")
    
    if "model_quality" in eval_results:
        model_metrics = eval_results["model_quality"]
        logger.info("Model Quality:")
        logger.info(f"State MSE: {model_metrics['state_mse']:.4f}")
        logger.info(f"Reward MSE: {model_metrics['reward_mse']:.4f}")
        logger.info(f"State R²: {model_metrics['state_r2']:.4f}")
        logger.info(f"Reward R²: {model_metrics['reward_r2']:.4f}")


if __name__ == "__main__":
    main()
