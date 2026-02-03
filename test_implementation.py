#!/usr/bin/env python3
"""Simple test script to verify the model-based RL implementation."""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.utils import set_seed, get_device
        print("✓ Utils imported successfully")
        
        from src.models import EnsembleDynamicsModel, ReplayBuffer
        print("✓ Models imported successfully")
        
        from src.algorithms import PETSAgent
        print("✓ Algorithms imported successfully")
        
        from src.training import Trainer
        print("✓ Training imported successfully")
        
        from src.evaluation import Evaluator
        print("✓ Evaluation imported successfully")
        
        from src.visualization import Visualizer
        print("✓ Visualization imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")
    
    try:
        import torch
        import numpy as np
        from src.utils import set_seed, get_device
        from src.models import EnsembleDynamicsModel, ReplayBuffer
        from src.algorithms import PETSAgent
        
        # Test device detection
        device = get_device()
        print(f"✓ Device detected: {device}")
        
        # Test seed setting
        set_seed(42)
        print("✓ Seed set successfully")
        
        # Test model creation
        model = EnsembleDynamicsModel(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        print("✓ Ensemble model created")
        
        # Test replay buffer
        buffer = ReplayBuffer(
            capacity=1000,
            state_dim=4,
            action_dim=1,
            device=device,
        )
        print("✓ Replay buffer created")
        
        # Test agent creation
        agent = PETSAgent(
            state_dim=4,
            action_dim=1,
            hidden_dim=64,
            num_networks=3,
        )
        print("✓ PETS agent created")
        
        # Test basic forward pass
        states = torch.randn(5, 4)
        actions = torch.randn(5, 1)
        
        next_states, rewards = model(states, actions)
        print(f"✓ Forward pass successful: {next_states.shape}, {rewards.shape}")
        
        # Test action selection
        state = np.random.randn(4)
        action = agent.select_action(state)
        print(f"✓ Action selection successful: {action.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path("configs/default.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Default config loaded")
            
            # Check required keys
            required_keys = ["env", "agent", "training", "evaluation"]
            for key in required_keys:
                if key in config:
                    print(f"✓ Config has {key} section")
                else:
                    print(f"✗ Config missing {key} section")
                    return False
            
            return True
        else:
            print("✗ Config file not found")
            return False
            
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Model-Based Reinforcement Learning - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_config_loading,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
