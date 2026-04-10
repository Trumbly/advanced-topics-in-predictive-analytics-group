import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense, Input
)
from tensorflow.keras import regularizers
from sklearn.metrics import roc_auc_score

# --- 0. Data Loading and Preprocessing (Mandatory Start) ---
X = np.load("data/spectrograms.npy")
y = np.load("data/labels.npy")
X = X[..., np.newaxis]
X = X[:2000]
y = y[:2000]
X_train, X_val = X[:1600], X[1600:]
y_train, y_val = y[:1600], y[1600:]
print(f"Shapes — X_train: {X_train.shape}, y_train: {y_train.shape}")

# --- 1. Custom Keras Components ---

def squeeze_excitation_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation Block for channel-wise re-weighting."""
    # Global Average Pooling to get channel descriptors
    se = GlobalAveragePooling2D()(input_tensor)
    
    # Reduction and expansion via Dense layers
    se_reduced = Dense(tf.keras.backend.int_shape(se.shape)[-1] // ratio, activation='relu', kernel_initializer='he_normal')(se)
    se_expanded = Dense(tf.keras.backend.int_shape(se.shape)[-1], activation='sigmoid', kernel_initializer='he_normal')(se_reduced)
    
    # Reshape and multiply back to the input tensor's feature map dimensions
    se_reshaped = tf.keras.layers.Reshape((1, 1, tf.keras.backend.int_shape(input_tensor)[-1]))(se_expanded)
    return tf.keras.layers.Multiply()([input_tensor, se_reshaped])

def attention_module(input_tensor):
    """Simulated efficient cross-attention/context module using feature transformation."""
    # In Keras, true cross-attention requires explicit Query, Key, Value mapping.
    # We approximate this by using multiple convolutions to capture different relational views.
    
    # Q, K, V projections (using 1x1 convolutions)
    q = Conv2D(tf.keras.backend.int_shape(input_tensor.shape)[-1] // 2, (1, 1), padding='same', name='query')(input_tensor)
    k = Conv2D(tf.keras.backend.int_shape(input_tensor.shape)[-1] // 2, (1, 1), padding='same', name='key')(input_tensor)
    v = Conv2D(tf.keras.backend.int_shape(input_tensor.shape)[-1] // 2, (1, 1), padding='same', name='value')(input_tensor)
    
    # Simplified attention mechanism: Weighted sum based on compatibility measure
    # (Dot product approximation: element-wise multiplication followed by softmax across spatial dimensions)
    # We stabilize this by using Conv2D to emulate the transformation matrix multiplication structure
    
    # Calculate weight (Attention Score)
    attention_weights = tf.keras.layers.Multiply()([q, k])
    
    # Normalize attention map (Simplified attention scaling)
    attention_map = tf.keras.layers.Activation('softmax')(attention_weights)
    
    # Apply attention map to context features
    output = tf.keras.layers.Multiply()([attention_map, v])
    return output


def build_model():
    # Initial feature extraction blocks (Mimicking initial feature learning)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(input_tensor)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Second level of feature extraction
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Apply advanced attention mechanism
    x = attention_module(x)
    
    # Final classification block
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # Flatten and pass through Dense layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Final classification layer
    output_tensor = tf.keras.layers.Dense(128, activation='relu')(x)
    output_tensor = tf.keras.layers.Dense(num_classes, activation='linear')(output_tensor)

    model = tf.keras.Model(inputs=[input_tensor], outputs=output_tensor)
    return model

# Define placeholders for the model structure
input_tensor = tf.keras.Input(shape=(128, 128, 3), name='input_image') # Assuming 128x128 input
num_classes = 2 # Assuming binary classification for the final layer

# Re-implementing the attention module wrapper for cleaner model definition
def attention_module(input_tensor):
    # This wrapper keeps the logic contained within the Functional API structure
    a = attention_module(input_tensor)
    return a

model = build_model()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model Compiled Successfully.")

# --- Execution Start ---
# The model definition above must be executed before any fitting can occur.
# We re-run the definition flow here for completeness if this were a single script environment.
# Note: In a real environment, defining the functional graph once is sufficient.
# ------------------------
# The provided structure above is the functional equivalent of the model definition.
# For execution robustness, we assume the model definition above is what is trained.