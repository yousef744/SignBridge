#!/usr/bin/env python3
"""
SignBridge Pro - Dataset Downloader
Downloads Sign Language MNIST dataset from Kaggle
"""

import os
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path

def download_dataset():
    """Download Sign Language MNIST dataset"""

    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("📥 SignBridge Pro - Dataset Downloader")
    print("=" * 60)

    # Sign Language MNIST dataset URLs (direct download)
    # Using alternative source since Kaggle needs API
    urls = {
        "train": "https://raw.githubusercontent.com/ardamavi/Sign-Language-Digits-Dataset/master/Dataset/train.csv",
        "test": "https://raw.githubusercontent.com/ardamavi/Sign-Language-Digits-Dataset/master/Dataset/test.csv"
    }

    # Alternative: Create synthetic dataset if download fails
    try:
        print("\n🔄 Downloading dataset...")

        # Try to download from GitHub
        for name, url in urls.items():
            filepath = dataset_dir / f"{name}.csv"
            if not filepath.exists():
                print(f"   Downloading {name}.csv...")
                urllib.request.urlretrieve(url, filepath)
                print(f"   ✅ {name}.csv downloaded!")
            else:
                print(f"   ✅ {name}.csv already exists!")

        print("\n🎉 Dataset downloaded successfully!")

    except Exception as e:
        print(f"\n⚠️  Download failed: {e}")
        print("🔄 Creating synthetic dataset for testing...")
        create_synthetic_dataset(dataset_dir)

    print(f"\n📁 Dataset location: {dataset_dir.absolute()}")
    print("=" * 60)

def create_synthetic_dataset(dataset_dir):
    """Create synthetic ASL dataset for testing"""
    import numpy as np

    print("\n🎨 Creating synthetic ASL dataset...")

    # Create synthetic data (28x28 images flattened = 784 features)
    np.random.seed(42)

    # 26 letters + space = 27 classes
    n_samples_train = 1000
    n_samples_test = 200
    n_features = 784  # 28x28

    # Generate random data
    X_train = np.random.randint(0, 256, size=(n_samples_train, n_features), dtype=np.uint8)
    y_train = np.random.randint(0, 27, size=n_samples_train)

    X_test = np.random.randint(0, 256, size=(n_samples_test, n_features), dtype=np.uint8)
    y_test = np.random.randint(0, 27, size=n_samples_test)

    # Save as CSV
    train_df = pd.DataFrame(X_train)
    train_df.insert(0, 'label', y_train)
    train_df.to_csv(dataset_dir / "train.csv", index=False)

    test_df = pd.DataFrame(X_test)
    test_df.insert(0, 'label', y_test)
    test_df.to_csv(dataset_dir / "test.csv", index=False)

    print("   ✅ Synthetic dataset created!")
    print(f"   📊 Training samples: {n_samples_train}")
    print(f"   📊 Test samples: {n_samples_test}")

if __name__ == "__main__":
    download_dataset()
