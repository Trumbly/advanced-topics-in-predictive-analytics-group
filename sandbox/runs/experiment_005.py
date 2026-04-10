import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json
import time

# --- Configuration ---
SPECTROGRAM_PATH = "data/processed/spectrograms.npy"
LABEL_PATH = "data/processed/labels.npy"
EXPERIMENT_ID = "experiment_005"
LOG_DIR = "experiments/logs"
RESULTS_PATH = os.path.join(LOG_DIR, EXPERIMENT_ID, "results.json")

# Ensure directories exist
os.makedirs(os.path.join(LOG_DIR, EXPERIMENT_ID), exist_ok=True)

# --- Data Loading ---
try:
    X = np.load(SPECTROGRAM_PATH)
    Y = np.load(LABEL_PATH)
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

# Assuming the input shape is (N, 128, 256) and label shape is (N, 206)
# We will reshape/process Y if it's not one-hot encoded correctly, but for simplicity,
# we'll assume Y is suitable for sparse categorical crossentropy or already one-hot.

# Split data (Simulating train/val split if the loaded data is the full set)
# Since the prompt doesn't specify the split, we'll assume the loaded data is the training set
# and we will use a small held-out subset for validation if possible, or just train on all
# and use a small split for validation accuracy reporting, mimicking a proper setup.
NUM_SAMPLES = X.shape[0]
VAL_SIZE = int(0.1 * NUM_SAMPLES)
TRAIN_SIZE = NUM_SAMPLES - VAL_SIZE

X_train, X_val = X[:TRAIN_SIZE], X[TRAIN_SIZE:]
Y_train, Y_val = Y[:TRAIN_SIZE], Y[TRAIN_SIZE:]

# --- Model Definition: Simple CNN with Batch Normalization (Escalation from simple CNN) ---
def build_cnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Classifier Head
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# --- Training Setup ---
INPUT_SHAPE = X.shape[1:] # (128, 256)
NUM_CLASSES = Y.shape[1]   # 206
BATCH_SIZE = 32
EPOCHS = 15 # Keep epochs low to ensure runtime constraint

model = build_cnn_model(INPUT_SHAPE, NUM_CLASSES)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Training ---
start_time = time.time()
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=0 # Keep output clean
)
train_time = time.time() - start_time

# --- Evaluation ---
val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)

# --- Results Saving ---
results = {
    "val_accuracy": float(val_acc),
    "approach": "Deeper CNN with Batch Normalization",
    "notes": f"Increased CNN depth and added Batch Normalization layers compared to previous simple CNN attempts. Trained for {EPOCHS} epochs. (Validation Loss: {val_loss:.4f})"
}

with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Training complete. Validation Accuracy: {val_acc*100:.2f}%")
print(f"Results saved to {RESULTS_PATH}")