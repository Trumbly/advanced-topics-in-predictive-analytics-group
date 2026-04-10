import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json
import time

# --- Configuration ---
SPECTROGRAM_PATH = "data/processed/spectrograms.npy"
LABEL_PATH = "data/processed/labels.npy"
EXPERIMENT_ID = "experiment_004"
MODEL_DIR = f"experiments/logs/{EXPERIMENT_ID}"
RESULTS_FILE = os.path.join(MODEL_DIR, "results.json")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Data Loading ---
try:
    X_all = np.load(SPECTROGRAM_PATH)
    y_all = np.load(LABEL_PATH)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

# For simplicity and speed, we will use the entire dataset for training and validation
# In a real scenario, we would split explicitly, but given the limited time/data context,
# we treat the loaded data as ready for training/validation split if necessary,
# but for this attempt, we will use the whole set and rely on the model's inherent validation capability if possible.

# Since the history suggests simple models failed, we escalate to Transfer Learning: EfficientNet
# This requires reshaping the data for image-based CNNs.
# Input shape is [128, 256] (Frequency, Time) -> Ideal for CNN input (Height, Width, Channels)
# We assume the input shape is (Samples, Height, Width) -> (Samples, 128, 256)
# We need to add a channel dimension: (Samples, 128, 256, 1)

print(f"Data loaded: X shape {X_all.shape}, y shape {y_all.shape}")

# --- Prepare Data for CNN/Transfer Learning ---
# Add channel dimension: (N, H, W) -> (N, H, W, 1)
X_reshaped = np.expand_dims(X_all, axis=-1).astype('float32')

# Split data (simple 80/20 split for validation check)
X_train, X_val, y_train, y_val = train_test_split(
    X_reshaped, y_all, test_size=0.2, random_state=42
)

# --- Model: Transfer Learning with EfficientNetB0 ---
# We use a pre-trained model for robust feature extraction.
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 256, 1)
)

# Freeze base layers for initial quick training to prevent catastrophic forgetting
base_model.trainable = False

# Build the classification head
inputs = tf.keras.layers.Input(shape=(128, 256, 1))
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(206, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Training ---
print("Starting training with EfficientNetB0...")
start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,  # Keep epochs low to respect time limit
    verbose=1
)

end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds.")

# --- Evaluation ---
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

# --- Saving Results ---
results = {
    "val_accuracy": float(val_acc),
    "approach": "Transfer Learning (EfficientNetB0)",
    "notes": f"Used EfficientNetB0 pre-trained on ImageNet, frozen base layers, and trained a classification head. Training time: {end_time - start_time:.1f}s."
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f)

print(f"Results saved to {RESULTS_FILE}")