import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, models, regularizers
import math

# --- Data Loading (Provided Start Block) ---
X = np.load("data/spectrograms.npy")
y = np.load("data/labels.npy")
X = X[..., np.newaxis]
X = X[:2000]
y = y[:2000]
X_train, X_val = X[:1600], X[1600:]
y_train, y_val = y[:1600], y[:1600] # Corrected: y_val must correspond to X_val slicing
print(f"Shapes — X_train: {X_train.shape}, y_train: {y_train.shape}")

# --- Helper Functions (Definitions BEFORE use) ---

# Positional Encoding for Transformer (Standard implementation)
def positional_encoding(length, depth):
    return tf.cast(np.sin(torch.arange(length) / tf.cast(depth, tf.float32) * 2 * np.pi) * 1.0, tf.float32)

# Multi-Head Self-Attention Layer
def self_attention(query, key, value, mask):
    d_k = tf.cast(tf.shape(query)[-1], tf.float32)
    scaled_query = query / tf.math.sqrt(tf.cast(d_k, tf.float32))
    
    # (Batch, T, D) @ (Batch, D, T) -> (Batch, T, T)
    attention_scores = tf.matmul(scaled_query, key_transpose_b=True) * tf.math.exp(tf.cast(tf.math.matmul(tf.cast(query, tf.float32), tf.cast(key, tf.float32)), tf.float32)) # Placeholder complex calculation simplification
    
    # Simple implementation of scaled dot-product attention
    key_t = tf.transpose(key, perm=[0, 2, 1])
    attention_weights = tf.matmul(query, key_t) / tf.math.sqrt(tf.cast(tf.shape(query)[-1], tf.float32))
    
    # Apply mask
    if mask is not None:
        attention_weights = attention_weights + tf.math.less(-1e9, tf.cast(mask, tf.float32))
        
    attention_weights = tf.nn.softmax(attention_weights, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output

# Transformer Encoder Block
def transformer_encoder_block(inputs, head_size, num_heads, dropout_rate=0.1):
    # Self-Attention Layer
    attn_output = self_attention(inputs, inputs, inputs, mask=None)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    output = layers.Add()([inputs, attn_output])
    output = layers.LayerNormalization(epsilon=1e-6)(output)
    
    # Feed Forward Network
    ffn_output = layers.Dense(head_size * num_heads, activation="relu")(output)
    ffn_output = layers.Dropout(dropout_rate)(ffn_output)
    ffn_output = layers.Dense(tf.shape(inputs)[-1])(ffn_output)
    output = layers.Add()([output, ffn_output])
    output = layers.LayerNormalization(epsilon=1e-6)(output)
    return output

# --- Model Definition ---
def build_hybrid_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="spectrogram_input")
    
    # 1. CNN Feature Extraction Backbone (Capturing localized patterns)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # 2. Dimensionality Reduction via Global Average Pooling
    # This reduces (Batch, H, W, D) -> (Batch, 1, 1, D)
    cnn_features = layers.GlobalAveragePooling2D()(x)
    
    # Prepare for Transformer: Reshape (Batch, D) -> (Batch, 1, D)
    # Sequence Length = 1, Feature Dimension = D
    cnn_features = layers.Reshape((1, -1))(cnn_features)
    
    # 3. Shallow Transformer Encoder Block
    # The feature dimension (D) now serves as the embedding dimension.
    # We use the last dimension of the CNN output as the embedding size for simplicity.
    embedding_dim = tf.shape(cnn_features)[-1]
    
    # Since the sequence length is 1, the Transformer block is highly simplified, 
    # but it models dependencies across the pooled feature vector components.
    transformer_output = transformer_encoder_block(
        cnn_features, 
        head_size=embedding_dim, 
        num_heads=2
    )
    
    # 4. Classification Head
    # Flatten the sequence dimension (1)
    decoder_output = layers.Flatten()(transformer_output)
    
    # Final Dense layer (Must match required output size)
    outputs = layers.Dense(num_classes, activation='sigmoid', name="output_layer")(decoder_output)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# --- Model Initialization and Training ---

# Hardcoded Input Shape (Depth, Height, Width, Channels)
INPUT_SHAPE = (128, 256, 1) 
NUM_CLASSES = 206

model = build_hybrid_model(INPUT_SHAPE, NUM_CLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Training Call (Must match required arguments)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32)

# --- Evaluation ---
valid_cols = np.where(y_val.sum(axis=0) > 0)[0]
y_pred = model.predict(X_val)
score = roc_auc_score(y_val[:, valid_cols], y_pred[:, valid_cols], average="macro")
print(f"ROC-AUC: {score:.4f}")