#!/usr/bin/env python3
"""Quick start script for the model-based RL project."""

import subprocess
import sys
from pathlib import Path

def main():
    """Main quick start function."""
    print("🤖 Model-Based Reinforcement Learning - Quick Start")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("src").exists() or not Path("configs").exists():
        print("❌ Please run this script from the project root directory")
        return 1
    
    print("📋 Available commands:")
    print()
    print("1. Train PETS agent on CartPole:")
    print("   python scripts/train.py --config configs/default.yaml")
    print()
    print("2. Train MBPO agent:")
    print("   python scripts/train.py --config configs/mbpo.yaml")
    print()
    print("3. Launch interactive demo:")
    print("   streamlit run demo/app.py")
    print()
    print("4. Run unit tests:")
    print("   python -m pytest tests/ -v")
    print()
    print("5. Check code quality:")
    print("   black --check src/ tests/ scripts/")
    print("   ruff check src/ tests/ scripts/")
    print()
    
    # Ask user what they want to do
    choice = input("What would you like to do? (1-5, or 'q' to quit): ").strip()
    
    if choice == '1':
        print("\n🚀 Starting PETS training...")
        subprocess.run([sys.executable, "scripts/train.py", "--config", "configs/default.yaml"])
    elif choice == '2':
        print("\n🚀 Starting MBPO training...")
        subprocess.run([sys.executable, "scripts/train.py", "--config", "configs/mbpo.yaml"])
    elif choice == '3':
        print("\n🌐 Launching Streamlit demo...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "demo/app.py"])
    elif choice == '4':
        print("\n🧪 Running tests...")
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    elif choice == '5':
        print("\n🔍 Checking code quality...")
        subprocess.run([sys.executable, "-m", "black", "--check", "src/", "tests/", "scripts/"])
        subprocess.run([sys.executable, "-m", "ruff", "check", "src/", "tests/", "scripts/"])
    elif choice.lower() == 'q':
        print("\n👋 Goodbye!")
        return 0
    else:
        print("\n❌ Invalid choice. Please run the script again.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
