import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os
import json
import time

# --- Configuration ---
INPUT_SHAPE = (128, 256)
NUM_SPECIES = 206
DATA_PATH = "data/processed/spectrograms.npy"
LABEL_PATH = "data/processed/labels.npy"
EXPERIMENT_ID = "experiment_003"
LOG_DIR = f"experiments/logs/{EXPERIMENT_ID}"
RESULTS_FILE = os.path.join(LOG_DIR, "results.json")

# --- Setup ---
os.makedirs(LOG_DIR, exist_ok=True)

# --- Data Loading ---
try:
    X = np.load(DATA_PATH)
    Y = np.load(LABEL_PATH)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

# Since the data summary suggests 2000 samples, and CNNs usually process images,
# we assume the first dimension of X is the sample dimension (N, H, W).
# We will use the entire dataset for training and validation split later.

# --- Strategy: Simple 2D CNN (Refinement/Optimization) ---
# Previous attempt used a simple 2D CNN. This attempt will use a slightly deeper/more robust
# architecture by adding BatchNormalization and Dropout for regularization, which is a standard
# improvement over a basic CNN structure.

def build_cnn_model(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(32, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D((2, 2))(x) # Halve dimensions
    
    # Block 2
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D((2, 2))(x) # Halve dimensions
    
    # Block 3
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D((2, 2))(x) # Halve dimensions
    
    # Classifier Head
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

# --- Data Splitting ---
# Split into training (80%) and validation (20%)
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# --- Model Compilation and Training ---
model = build_cnn_model(INPUT_SHAPE, NUM_SPECIES)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Starting model training...")
start_time = time.time()

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=30, # Reduced epochs to keep complexity low and time under 5 mins
    batch_size=32,
    verbose=0 # Set to 0 for clean execution
)

end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds.")

# --- Evaluation ---
val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)

# --- Saving Results ---
results = {
    "val_accuracy": float(val_acc),
    "approach": "Deep CNN with BatchNorm & Dropout",
    "notes": f"Improved CNN structure over baseline by adding BatchNormalization, LeakyReLU, and Dropout for better regularization. Trained for 30 epochs. Total time: {end_time - start_time:.2f}s."
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f)

print(f"Results saved to {RESULTS_FILE}")
print(f"Final Validation Accuracy: {val_acc*100:.2f}%")