# Model-Based Reinforcement Learning

A comprehensive implementation of model-based reinforcement learning algorithms with a focus on research and education.

## Overview

This project provides a clean, reproducible implementation of advanced model-based reinforcement learning algorithms, including:

- **PETS (Probabilistic Ensemble Trajectory Sampling)**: Uses ensemble dynamics models with trajectory sampling for planning
- **MBPO (Model-Based Policy Optimization)**: Combines model-based planning with policy gradient methods
- **Model Predictive Control (MPC)**: Planning-based control using learned dynamics models

## Features

- **Modern Architecture**: PyTorch 2.x, Gymnasium, comprehensive type hints
- **Uncertainty Estimation**: Ensemble and probabilistic dynamics models
- **Comprehensive Evaluation**: Performance, sample efficiency, stability, and model quality metrics
- **Interactive Demo**: Streamlit-based web application for real-time training and evaluation
- **Production Ready**: Proper project structure, CI/CD, testing, and documentation

## Safety Notice

⚠️ **This is a research and educational demonstration. Do not use for production control of real-world systems without proper safety measures and validation.**

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Model-Based-Reinforcement-Learning.git
cd Model-Based-Reinforcement-Learning

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Train a PETS agent on CartPole
python scripts/train.py --config configs/default.yaml

# Train an MBPO agent
python scripts/train.py --config configs/mbpo.yaml

# Evaluate a trained agent
python scripts/train.py --config configs/default.yaml --eval-only --checkpoint logs/checkpoint_episode_100.pt
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── algorithms/         # RL algorithms (PETS, MBPO, MPC)
│   ├── models/            # Dynamics models (ensemble, probabilistic)
│   ├── training/          # Training utilities and trainers
│   ├── evaluation/        # Evaluation metrics and tools
│   ├── visualization/     # Plotting and visualization
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── tests/                 # Unit tests
├── demo/                  # Streamlit demo application
├── assets/                # Generated visualizations and outputs
├── logs/                  # Training logs and checkpoints
└── data/                  # Data storage
```

## Algorithms

### PETS (Probabilistic Ensemble Trajectory Sampling)

PETS learns an ensemble of dynamics models and uses trajectory sampling for planning. Key features:

- Ensemble uncertainty estimation
- Model Predictive Control planning
- Trajectory sampling with optimization

```python
from src.algorithms import PETSAgent

agent = PETSAgent(
    state_dim=4,
    action_dim=1,
    hidden_dim=256,
    num_networks=5,
    horizon=10,
    num_candidates=1000,
)
```

### MBPO (Model-Based Policy Optimization)

MBPO combines model-based planning with policy gradient methods. Key features:

- Ensemble dynamics models
- Model rollouts for policy training
- Real and model data mixing

```python
from src.algorithms import MBPOAgent

agent = MBPOAgent(
    state_dim=4,
    action_dim=1,
    hidden_dim=256,
    num_networks=7,
    horizon=1,
    rollout_batch_size=1000,
    real_ratio=0.05,
)
```

## Configuration

Configuration is managed through YAML files in the `configs/` directory:

```yaml
# configs/default.yaml
env:
  name: "CartPole-v1"
  max_steps_per_episode: 200
  seed: 42

agent:
  type: "PETS"
  state_dim: 4
  action_dim: 1
  hidden_dim: 256
  num_networks: 5
  horizon: 10
  learning_rate: 0.001

training:
  max_episodes: 1000
  eval_frequency: 50
  device: "auto"  # auto, cuda, mps, cpu
```

## Evaluation

The framework provides comprehensive evaluation metrics:

### Performance Metrics
- Mean reward ± 95% confidence interval
- Episode length statistics
- Success rate (when applicable)

### Sample Efficiency
- Steps to reach performance threshold
- Learning curve analysis
- Convergence analysis

### Stability and Robustness
- Reward variance across seeds
- Sensitivity analysis
- Domain randomization tests

### Model Quality
- Prediction accuracy (MSE, MAE, R²)
- Uncertainty calibration
- Model error analysis

```python
from src.evaluation import Evaluator

evaluator = Evaluator(agent, env_name="CartPole-v1")
results = evaluator.evaluate_performance()
print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
```

## Visualization

Generate comprehensive visualizations:

```python
from src.visualization import Visualizer

visualizer = Visualizer(output_dir="assets")

# Learning curves
visualizer.plot_learning_curves(episode_rewards, episode_lengths, training_losses)

# Model predictions
visualizer.plot_model_predictions(agent, states, actions, next_states, rewards)

# Uncertainty analysis
visualizer.plot_uncertainty_analysis(agent, states, actions)

# Trajectory animations
visualizer.create_trajectory_animation(agent, env_name, num_episodes=3)
```

## Development

### Code Quality

The project uses modern Python development practices:

- **Type Hints**: Comprehensive type annotations
- **Code Formatting**: Black for formatting, Ruff for linting
- **Testing**: Pytest with comprehensive test coverage
- **Documentation**: NumPy/Google-style docstrings

```bash
# Format code
black src/ tests/ scripts/

# Lint code
ruff check src/ tests/ scripts/

# Type checking
mypy src/

# Run tests
pytest tests/ -v
```

### CI/CD

GitHub Actions provides automated testing and quality checks:

- Code formatting and linting
- Unit tests across Python 3.10-3.12
- Coverage reporting
- Quick rollout tests

## Environment Support

### Supported Environments

- **CartPole-v1**: Classic control benchmark
- **Pendulum-v1**: Continuous control
- **MountainCar-v0**: Sparse reward environment

### Device Support

Automatic device detection with fallback:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU

## Reproducibility

- Deterministic seeding for all random sources
- Fixed evaluation seeds
- Reproducible configurations
- Version-controlled dependencies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure code quality (black, ruff, mypy)
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{model_based_rl,
  title={Model-Based Reinforcement Learning},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Model-Based-Reinforcement-Learning}
}
```

## Acknowledgments

- PETS algorithm: Chua et al. (2018)
- MBPO algorithm: Janner et al. (2019)
- Gymnasium: Brockman et al. (2016)
- PyTorch: Paszke et al. (2019)

## References

1. Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). Deep reinforcement learning in a handful of trials using probabilistic dynamics models. NeurIPS.
2. Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). When to trust your model: Model-based policy optimization. NeurIPS.
3. Brockman, G., et al. (2016). Openai gym. arXiv preprint arXiv:1606.01540.
4. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. NeurIPS.
# Model-Based-Reinforcement-Learning
