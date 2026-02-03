"""Streamlit demo application for model-based reinforcement learning."""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
import torch
import yaml
from PIL import Image

from src.algorithms import PETSAgent, MBPOAgent
from src.evaluation import Evaluator
from src.training import Trainer
from src.utils import get_device, set_seed
from src.visualization import Visualizer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Model-Based Reinforcement Learning Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_agent(config: Dict[str, Any]) -> Any:
    """Create agent based on configuration."""
    agent_config = config["agent"]
    device = get_device()
    
    if agent_config["type"] == "PETS":
        return PETSAgent(
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
        return MBPOAgent(
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

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Model-Based Reinforcement Learning Demo</h1>', 
                unsafe_allow_html=True)
    
    # Safety disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Safety Notice:</strong> This is a research and educational demonstration. 
        Do not use for production control of real-world systems without proper safety measures and validation.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Agent selection
    agent_type = st.sidebar.selectbox(
        "Agent Type",
        ["PETS", "MBPO"],
        help="PETS: Probabilistic Ensemble Trajectory Sampling\nMBPO: Model-Based Policy Optimization"
    )
    
    # Environment selection
    env_name = st.sidebar.selectbox(
        "Environment",
        ["CartPole-v1", "Pendulum-v1", "MountainCar-v0"],
        help="Select the environment to train on"
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    max_episodes = st.sidebar.slider("Max Episodes", 10, 1000, 100)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
    batch_size = st.sidebar.slider("Batch Size", 32, 512, 256)
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    hidden_dim = st.sidebar.slider("Hidden Dimension", 64, 512, 256)
    num_networks = st.sidebar.slider("Number of Networks", 3, 10, 5)
    
    # Create configuration
    config = {
        "env": {"name": env_name, "max_steps_per_episode": 200, "seed": 42},
        "agent": {
            "type": agent_type,
            "state_dim": 4 if env_name == "CartPole-v1" else 3,
            "action_dim": 1,
            "hidden_dim": hidden_dim,
            "num_networks": num_networks,
            "horizon": 10,
            "num_candidates": 1000,
            "num_iterations": 5,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "rollout_batch_size": 1000,
            "real_ratio": 0.05,
        },
        "training": {
            "max_episodes": max_episodes,
            "eval_frequency": 10,
            "save_frequency": 50,
            "log_dir": "logs",
            "device": "auto",
            "seed": 42,
        },
        "evaluation": {"num_eval_episodes": 10, "confidence_level": 0.95},
    }
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Training", "Evaluation", "Visualization", "About"])
    
    with tab1:
        st.header("Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Start Training", type="primary"):
                with st.spinner("Training in progress..."):
                    # Set random seed
                    set_seed(config["training"]["seed"])
                    
                    # Create agent
                    agent = create_agent(config)
                    
                    # Create trainer
                    trainer = Trainer(
                        agent=agent,
                        env_name=config["env"]["name"],
                        max_episodes=config["training"]["max_episodes"],
                        max_steps_per_episode=config["env"]["max_steps_per_episode"],
                        eval_frequency=config["training"]["eval_frequency"],
                        save_frequency=config["training"]["save_frequency"],
                        device=get_device(),
                        seed=config["training"]["seed"],
                    )
                    
                    # Train agent
                    training_results = trainer.train()
                    
                    # Store results in session state
                    st.session_state.training_results = training_results
                    st.session_state.agent = agent
                    st.session_state.config = config
                    
                    st.success("Training completed!")
        
        with col2:
            if "training_results" in st.session_state:
                results = st.session_state.training_results
                
                st.subheader("Training Summary")
                
                # Display metrics
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric(
                        "Final Reward",
                        f"{results['episode_rewards'][-1]:.2f}",
                        delta=f"{results['episode_rewards'][-1] - results['episode_rewards'][0]:.2f}"
                    )
                
                with col_b:
                    st.metric(
                        "Final Length",
                        f"{results['episode_lengths'][-1]:.0f}",
                        delta=f"{results['episode_lengths'][-1] - results['episode_lengths'][0]:.0f}"
                    )
                
                # Training time
                st.metric("Training Time", f"{results['training_time']:.1f}s")
                
                # Learning curve
                st.subheader("Learning Curve")
                st.line_chart(results['episode_rewards'])
    
    with tab2:
        st.header("Evaluation")
        
        if "agent" in st.session_state:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("Run Evaluation", type="primary"):
                    with st.spinner("Evaluating agent..."):
                        evaluator = Evaluator(
                            agent=st.session_state.agent,
                            env_name=st.session_state.config["env"]["name"],
                            num_eval_episodes=20,
                        )
                        
                        eval_results = evaluator.evaluate_performance()
                        st.session_state.eval_results = eval_results
                        
                        st.success("Evaluation completed!")
            
            with col2:
                if "eval_results" in st.session_state:
                    results = st.session_state.eval_results
                    
                    st.subheader("Evaluation Results")
                    
                    # Performance metrics
                    st.metric("Mean Reward", f"{results['mean_reward']:.2f}")
                    st.metric("Std Reward", f"{results['std_reward']:.2f}")
                    st.metric("Mean Length", f"{results['mean_length']:.2f}")
                    st.metric("Success Rate", f"{results['success_rate']:.2f}")
                    
                    # Confidence interval
                    ci = results['reward_ci']
                    st.metric("Reward CI", f"[{ci[0]:.2f}, {ci[1]:.2f}]")
        else:
            st.info("Please train an agent first before evaluation.")
    
    with tab3:
        st.header("Visualization")
        
        if "agent" in st.session_state and "training_results" in st.session_state:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Visualization Options")
                
                plot_learning = st.checkbox("Learning Curves", value=True)
                plot_predictions = st.checkbox("Model Predictions", value=True)
                plot_uncertainty = st.checkbox("Uncertainty Analysis", value=True)
                plot_trajectory = st.checkbox("Trajectory Animation", value=True)
                
                if st.button("Generate Visualizations", type="primary"):
                    with st.spinner("Generating visualizations..."):
                        # Create visualizer
                        with tempfile.TemporaryDirectory() as temp_dir:
                            visualizer = Visualizer(output_dir=temp_dir)
                            
                            # Generate plots
                            if plot_learning:
                                visualizer.plot_learning_curves(
                                    st.session_state.training_results['episode_rewards'],
                                    st.session_state.training_results['episode_lengths'],
                                    st.session_state.training_results['training_losses'],
                                )
                            
                            if plot_predictions:
                                # Generate some test data for predictions
                                import gymnasium as gym
                                env = gym.make(st.session_state.config["env"]["name"])
                                
                                states, actions, next_states, rewards = [], [], [], []
                                for _ in range(50):
                                    state, _ = env.reset()
                                    for _ in range(10):
                                        action = env.action_space.sample()
                                        next_state, reward, terminated, truncated, _ = env.step(action)
                                        done = terminated or truncated
                                        
                                        states.append(state)
                                        actions.append(action)
                                        next_states.append(next_state)
                                        rewards.append(reward)
                                        
                                        state = next_state
                                        if done:
                                            break
                                
                                env.close()
                                
                                visualizer.plot_model_predictions(
                                    st.session_state.agent,
                                    np.array(states),
                                    np.array(actions),
                                    np.array(next_states),
                                    np.array(rewards),
                                )
                            
                            if plot_uncertainty:
                                visualizer.plot_uncertainty_analysis(
                                    st.session_state.agent,
                                    np.array(states),
                                    np.array(actions),
                                )
                            
                            if plot_trajectory:
                                visualizer.create_trajectory_animation(
                                    st.session_state.agent,
                                    st.session_state.config["env"]["name"],
                                    num_episodes=3,
                                )
                            
                            # Display generated images
                            st.success("Visualizations generated!")
            
            with col2:
                st.subheader("Generated Visualizations")
                
                # Display images if they exist
                temp_dir = Path(tempfile.gettempdir())
                
                if (temp_dir / "learning_curves.png").exists():
                    st.image(str(temp_dir / "learning_curves.png"), caption="Learning Curves")
                
                if (temp_dir / "model_predictions.png").exists():
                    st.image(str(temp_dir / "model_predictions.png"), caption="Model Predictions")
                
                if (temp_dir / "uncertainty_analysis.png").exists():
                    st.image(str(temp_dir / "uncertainty_analysis.png"), caption="Uncertainty Analysis")
                
                if (temp_dir / "trajectory_animation.gif").exists():
                    st.image(str(temp_dir / "trajectory_animation.gif"), caption="Trajectory Animation")
        else:
            st.info("Please train an agent first before generating visualizations.")
    
    with tab4:
        st.header("About")
        
        st.markdown("""
        ## Model-Based Reinforcement Learning Demo
        
        This application demonstrates modern model-based reinforcement learning algorithms:
        
        ### Algorithms Implemented
        
        - **PETS (Probabilistic Ensemble Trajectory Sampling)**: Uses ensemble dynamics models 
          with trajectory sampling for planning
        - **MBPO (Model-Based Policy Optimization)**: Combines model-based planning with 
          policy gradient methods
        
        ### Features
        
        - **Modern Architecture**: PyTorch 2.x, Gymnasium, type hints
        - **Uncertainty Estimation**: Ensemble and probabilistic models
        - **Comprehensive Evaluation**: Performance, sample efficiency, stability metrics
        - **Visualization**: Learning curves, model predictions, uncertainty analysis
        - **Interactive Demo**: Real-time training and evaluation
        
        ### Technical Details
        
        - **Device Support**: CUDA, MPS (Apple Silicon), CPU
        - **Reproducibility**: Deterministic seeding
        - **Safety**: Research/educational focus with disclaimers
        - **Extensibility**: Modular design for easy customization
        
        ### Usage
        
        1. Select agent type and environment
        2. Configure training parameters
        3. Start training and monitor progress
        4. Evaluate performance
        5. Generate visualizations
        
        ### Safety Notice
        
        This is a research and educational demonstration. Do not use for production control 
        of real-world systems without proper safety measures and validation.
        """)
        
        # Technical specifications
        st.subheader("Technical Specifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Dependencies:**
            - PyTorch 2.x
            - Gymnasium
            - NumPy, Pandas
            - Matplotlib, Seaborn
            - Streamlit
            """)
        
        with col2:
            st.markdown("""
            **Requirements:**
            - Python 3.10+
            - CUDA/MPS support (optional)
            - 4GB+ RAM recommended
            """)

if __name__ == "__main__":
    main()
