#!/usr/bin/env python3
"""
SignBridge Pro - Quick Setup
One-command setup for the entire project
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and show progress"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, shell=True, capture_output=False)

    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False

    print(f"✅ Completed: {description}")
    return True

def main():
    print("="*60)
    print("🚀 SignBridge Pro - Quick Setup")
    print("="*60)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required!")
        return

    print(f"\n🐍 Python: {sys.version}")

    # Step 1: Install requirements
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("\n⚠️  Some packages may have failed. Continuing...")

    # Step 2: Download dataset
    if not run_command("python download_dataset.py", "Downloading dataset"):
        print("❌ Dataset download failed!")
        return

    # Step 3: Train model
    print("\n" + "="*60)
    print("🤖 Ready to train model?")
    print("   This will take 10-30 minutes depending on your hardware.")
    print("="*60)

    response = input("\nTrain model now? (y/n): ").lower().strip()

    if response == 'y':
        if not run_command("python train_model.py", "Training model"):
            print("❌ Model training failed!")
            return
    else:
        print("\n⏭️  Skipping training. You can run 'python train_model.py' later.")
        print("   The app will work in demo mode without a trained model.")

    # Step 4: Start server
    print("\n" + "="*60)
    print("🎉 Setup complete!")
    print("="*60)
    print("\nTo start the server:")
    print("   python app.py")
    print("\nThen open: http://localhost:5000")
    print("="*60)

    response = input("\nStart server now? (y/n): ").lower().strip()

    if response == 'y':
        run_command("python app.py", "Starting server")

if __name__ == "__main__":
    main()
