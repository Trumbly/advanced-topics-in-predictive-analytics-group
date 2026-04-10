import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json
import time

# --- Configuration ---
SPECTROGRAM_PATH = "data/processed/spectrograms.npy"
LABEL_PATH = "data/processed/labels.npy"
EXPERIMENT_ID = "CNN_Baseline_V1"
LOG_DIR = "experiments/logs"
os.makedirs(os.path.join(LOG_DIR, EXPERIMENT_ID), exist_ok=True)
RESULTS_FILE = os.path.join(LOG_DIR, EXPERIMENT_ID, "results.json")

# --- Data Loading ---
try:
    X = np.load(SPECTROGRAM_PATH)
    Y = np.load(LABEL_PATH)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

# --- Model Definition (Simple 2D CNN) ---
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Main Training Function ---
def run_experiment():
    print("--- Starting CNN Baseline Experiment ---")
    
    # Assuming the labels are one-hot encoded based on the problem structure (206 classes)
    # We will use the provided Y shape directly for categorical crossentropy.
    
    # Split data: 90% Train, 10% Validation
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.1, random_state=42
    )
    
    input_shape = X.shape[1:] # (128, 256)
    num_classes = Y.shape[1]   # 206

    # Build and train model
    model = build_cnn_model(input_shape, num_classes)
    
    start_time = time.time()
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=32,
        epochs=15, # Keep epochs low to stay under 5 min limit
        verbose=1
    )
    end_time = time.time()
    
    # Evaluate on validation set (final check)
    val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)
    
    # Record results
    results = {
        "val_accuracy": float(val_acc),
        "approach": "Simple 2D CNN",
        "notes": f"Trained for {15} epochs. Used standard CNN architecture on spectrogram images. Validation time: {end_time - start_time:.2f}s."
    }
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"--- Experiment Finished ---")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"Validation Accuracy: {results['val_accuracy']:.4f}")

if __name__ == "__main__":
    run_experiment()