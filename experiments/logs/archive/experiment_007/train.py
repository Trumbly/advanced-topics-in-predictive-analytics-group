import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D, Dense, LayerNormalization, MultiHeadAttention, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

# --- Data Loading (Verbatim) ---
X = np.load("data/spectrograms.npy")
y = np.load("data/labels.npy")
X = X[..., np.newaxis]
X = X[:2000]
y = y[:2000]
X_train, X_val = X[:1600], X[1600:]
y_train, y_val = y[:1600], y[1600:]
print(f"Shapes — X_train: {X_train.shape}, y_train: {y_train.shape}")

# --- Model Components Definition ---

def conv_block(x, filters, kernel_size=3):
    """Standard CNN block structure."""
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    return x

def transformer_encoder_block(inputs, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
    """A simplified Transformer Encoder block."""
    # 1. Self-Attention Sublayer
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    # Residual connection + LayerNorm
    out1 = tf.keras.layers.Add()([inputs, attn_output])
    out1 = LayerNormalization(epsilon=1e-6)(out1)

    # 2. Feed Forward Sublayer
    ffn_output = Dense(ff_dim)(out1)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    ffn_output = Dense(embed_dim)(ffn_output)
    # Residual connection + LayerNorm
    outputs = tf.keras.layers.Add()([out1, ffn_output])
    outputs = LayerNormalization(epsilon=1e-6)(outputs)
    return outputs

# --- Model Building ---

def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # 1. CNN Feature Extractor Backbone (Capturing local spectral patterns)
    # We process (128, 256, 1)
    x = conv_block(inputs, 64)  # Output shape: (None, 128, 256, 64)
    x = conv_block(x, 128)     # Output shape: (None, 128, 256, 128)
    
    # Global pooling is usually used, but since we need a sequence for Transformer, 
    # we must retain the temporal (sequence) dimension.
    # We reshape/flatten to get (Batch, Sequence_Length, Features)
    # Assuming the CNN layers preserved the temporal dimension (e.g., 128 time steps)
    # We use the last dimension's size (Channels) as the feature dimension for the transformer input.
    cnn_output = x 
    
    # Reshape: (B, Time_steps, Features) -> (B, 128, 128)
    # We target the shape (Batch, Sequence_Length, Feature_Dimension)
    # Sequence_Length = 128 (Height/Time dimension)
    # Feature_Dimension = 128 (The last channel dimension)
    seq_len = tf.shape(cnn_output)[1] 
    feature_dim = tf.shape(cnn_output)[-1]
    
    cnn_output_flat = Reshape((seq_len, feature_dim))(cnn_output)
    
    # 2. Transformer Encoder (Modeling global temporal dependencies)
    # We use the feature_dim as the embedding dimension for the Transformer
    transformer_output = transformer_encoder_block(
        cnn_output_flat, 
        embed_dim=feature_dim, 
        num_heads=4, 
        ff_dim=feature_dim * 2
    )
    
    # 3. Classification Head
    # Use GlobalAveragePooling over the sequence dimension to get a fixed-size vector
    # (B, Sequence_Length, Features) -> (B, Features)
    pooled_output = GlobalAveragePooling1D()(transformer_output)
    
    # Final Dense Layer (Verbatim requirement)
    output = Dense(206, activation='sigmoid')(pooled_output)
    
    model = Model(inputs=inputs, outputs=output)
    return model

# --- Model Instantiation and Training ---
INPUT_SHAPE = (128, 256, 1) # Hardcoded (Height, Width, Channels)
model = build_hybrid_model(INPUT_SHAPE)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Model Training (Verbatim requirement)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32)

# --- Evaluation (Verbatim) ---
valid_cols = np.where(y_val.sum(axis=0) > 0)[0]
y_pred = model.predict(X_val)
score = roc_auc_score(y_val[:, valid_cols], y_pred[:, valid_cols], average="macro")
print(f"ROC-AUC: {score:.4f}")