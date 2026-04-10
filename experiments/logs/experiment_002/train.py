import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import time
import json

# --- Configuration ---
SPECTROGRAM_PATH = "data/processed/spectrograms.npy"
LABEL_PATH = "data/processed/labels.npy"
EXPERIMENT_ID = "experiment_002"
LOG_DIR = f"experiments/logs/{EXPERIMENT_ID}"
RESULTS_FILE = os.path.join(LOG_DIR, "results.json")

# Setup directories
os.makedirs(LOG_DIR, exist_ok=True)

# --- Data Loading ---
try:
    X = np.load(SPECTROGRAM_PATH)
    Y_one_hot = np.load(LABEL_PATH)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

# Assuming X shape is (N, F, T) -> (2000, 128, 256)
# Assuming Y_one_hot shape is (N, C) -> (2000, 206)

# --- Strategy: Simple 2D CNN on spectrogram images ---
# Input shape is (128, 256), which is (Frequency, Time)
# We reshape it for CNN compatibility: (Batch, Height, Width, Channels)
# Since the spectrogram is already 2D, we treat it as 1 channel (grayscale image)
X_cnn = X[..., np.newaxis] # Shape becomes (2000, 128, 256, 1)

# Split data (simple train/val split for efficiency)
VAL_SIZE = 0.2
X_train, X_val, Y_train, Y_val = train_test_split(
    X_cnn, Y_one_hot, test_size=VAL_SIZE, random_state=42
)

# --- Model Definition: Simple 2D CNN ---
def build_cnn_model(input_shape):
    model = Sequential([
        # Layer 1: Conv -> Pool
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # Layer 2: Conv -> Pool
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # Layer 3: Conv -> Pool
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        # Classifier head
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(206, activation='softmax') # num_species = 206
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_cnn_model((128, 256, 1))

# --- Training ---
EPOCHS = 10
BATCH_SIZE = 32

print("Starting CNN training...")
start_time = time.time()
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0 # Keep output clean for script execution
)
elapsed_time = time.time() - start_time
print(f"Training finished in {elapsed_time:.2f} seconds.")

# --- Evaluation ---
val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)

# --- Saving Results ---
results = {
    "val_accuracy": float(val_acc),
    "approach": "Simple 2D CNN",
    "notes": f"Trained for {EPOCHS} epochs. Used a standard CNN architecture on the spectrogram images. Time taken: {elapsed_time:.2f}s."
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f)

print(f"Results saved to {RESULTS_FILE}")