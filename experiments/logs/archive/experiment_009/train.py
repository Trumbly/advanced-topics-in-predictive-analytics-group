import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, models

# --- 1. Data Loading and Setup (As required) ---
X = np.load("data/spectrograms.npy")
y = np.load("data/labels.npy")
X = X[..., np.newaxis]
X = X[:2000]
y = y[:2000]
X_train, X_val = X[:1600], X[1600:] # Assuming 80% for training, 20% for validation for a full split
y_train = y[:1600]
y_val = y[1600:]

# --- Model Components ---

# 1. Convolutional Block (Mimicking the feature extraction phase)
def conv_block(input_tensor, filters=32, kernel_size=3):
    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

# 2. Self-Attention Mechanism (Simplified implementation for spatial feature weighting)
def attention_block(input_tensor, num_heads=4):
    # This block simulates global contextual feature weighting (Attention)
    # Input: (Batch, H, W, C)
    
    # Step 1: Squeeze (Global Average Pooling across H and W) -> (Batch, 1, 1, C)
    squeezed = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    squeezed = tf.keras.layers.Reshape((1, 1, -1))(squeezed)
    
    # Step 2: Apply Attention/Reconstruction
    # We will use a simple mechanism: feature multiplication weighted by a learned vector
    attention_weights = tf.keras.layers.Conv2D(1, 1, padding='same')(squeezed) # Weight map (Batch, 1, 1, 1)
    reconstructed = tf.keras.layers.Multiply()([input_tensor, attention_weights])
    
    return reconstructed

# 3. Full Model Definition
def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Block 1: Initial Feature Extraction
    x = conv_block(inputs, filters=32)(inputs)
    
    # Block 2: Deeper Feature Extraction
    x = conv_block(x, filters=64)(x)
    
    # Block 3: Global Contextual Refinement (Attention)
    x = attention_block(x)(x)
    
    # Classifier Head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# --- Training and Evaluation ---

import tensorflow as tf
from tensorflow import keras

# Re-define necessary constants and model setup for execution context
INPUT_SHAPE = (128, 128, 1) # Based on typical image dimensions for segmentation tasks
model = build_model(input_shape=INPUT_SHAPE)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training phase
print("\n--- Training Model ---")
history = model.fit(
    X_train=X_train, 
    y_train=y_train, 
    validation_data=(X_val, y_val),
    epochs=10, # Reduced epochs for notebook run time
    batch_size=32,
    verbose=1
)

# Evaluation (Optional: Print final scores)
print("\n--- Final Evaluation ---")
loss, acc = model.evaluate(X_val, y_val, batch_size=32)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}")

# Clean up TensorFlow session resources
tf.keras.backend.clear_session()