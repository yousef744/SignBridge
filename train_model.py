#!/usr/bin/env python3
"""
SignBridge Pro - Model Training
Trains CNN model on Sign Language MNIST dataset
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ASL Alphabet mapping
ASL_LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z', 26: 'space'
}

def load_data():
    """Load and preprocess dataset"""
    print("\n📂 Loading dataset...")

    dataset_dir = Path("dataset")

    # Check if dataset exists
    if not (dataset_dir / "train.csv").exists():
        print("❌ Dataset not found! Run download_dataset.py first.")
        return None, None, None, None

    # Load CSV files
    train_df = pd.read_csv(dataset_dir / "train.csv")
    test_df = pd.read_csv(dataset_dir / "test.csv")

    # Extract labels and features
    y_train = train_df['label'].values
    X_train = train_df.drop('label', axis=1).values

    y_test = test_df['label'].values
    X_test = test_df.drop('label', axis=1).values

    # Normalize pixel values (0-255 -> 0-1)
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Reshape for CNN (28x28x1)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # One-hot encode labels
    num_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print(f"   ✅ Training samples: {X_train.shape[0]}")
    print(f"   ✅ Test samples: {X_test.shape[0]}")
    print(f"   ✅ Image shape: {X_train.shape[1:]}")
    print(f"   ✅ Number of classes: {num_classes}")

    return X_train, y_train, X_test, y_test

def create_model(input_shape=(28, 28, 1), num_classes=27):
    """Create CNN model architecture"""
    print("\n🏗️  Creating model...")

    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    print("   ✅ Model created!")
    model.summary()

    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Train the model"""
    print("\n🚀 Starting training...")
    print("=" * 60)

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False
    )

    # Train
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    print("\n✅ Training completed!")

    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n📊 Evaluating model...")

    loss, accuracy, top5_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"   📈 Test Accuracy: {accuracy:.4f}")
    print(f"   📈 Test Loss: {loss:.4f}")
    print(f"   📈 Top-5 Accuracy: {top5_acc:.4f}")

    return accuracy

def save_model(model, history):
    """Save model and training history"""
    print("\n💾 Saving model...")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Save model in multiple formats
    # 1. Keras format
    model.save('models/asl_model.keras')
    print("   ✅ Saved: models/asl_model.keras")

    # 2. TensorFlow SavedModel format
    model.save('models/asl_model_saved')
    print("   ✅ Saved: models/asl_model_saved/")

    # 3. Save as TFLite for mobile
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('models/asl_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("   ✅ Saved: models/asl_model.tflite")

    # 4. Save training history
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("   ✅ Saved: models/training_history.pkl")

    # 5. Save label mapping
    with open('models/label_mapping.json', 'w') as f:
        json.dump(ASL_LABELS, f)
    print("   ✅ Saved: models/label_mapping.json")

    # 6. Plot training history
    plot_history(history)

    print("\n🎉 Model saved successfully!")

def plot_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: models/training_history.png")

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("🤖 SignBridge Pro - Model Training")
    print("=" * 60)

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    if X_train is None:
        print("\n❌ Failed to load dataset!")
        return

    # Create model
    num_classes = y_train.shape[1]
    model = create_model(num_classes=num_classes)

    # Train
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test)

    # Save
    save_model(model, history)

    print("\n" + "=" * 60)
    print("🎉 Training pipeline completed!")
    print(f"🎯 Final Accuracy: {accuracy:.2%}")
    print("=" * 60)

if __name__ == "__main__":
    main()
