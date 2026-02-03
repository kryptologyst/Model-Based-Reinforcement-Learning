"""Model-based reinforcement learning package."""

from .algorithms import PETSAgent, MBPOAgent, ModelPredictiveControl
from .models import EnsembleDynamicsModel, ProbabilisticDynamicsModel, ReplayBuffer
from .training import Trainer, CurriculumTrainer, HyperparameterScheduler
from .evaluation import Evaluator, AblationStudy
from .visualization import Visualizer
from .utils import (
    set_seed,
    get_device,
    count_parameters,
    normalize_obs,
    update_running_stats,
    create_env,
    get_space_size,
    EarlyStopping,
)

__version__ = "0.1.0"
__author__ = "AI Research Team"

__all__ = [
    "PETSAgent",
    "MBPOAgent", 
    "ModelPredictiveControl",
    "EnsembleDynamicsModel",
    "ProbabilisticDynamicsModel",
    "ReplayBuffer",
    "Trainer",
    "CurriculumTrainer",
    "HyperparameterScheduler",
    "Evaluator",
    "AblationStudy",
    "Visualizer",
    "set_seed",
    "get_device",
    "count_parameters",
    "normalize_obs",
    "update_running_stats",
    "create_env",
    "get_space_size",
    "EarlyStopping",
]
