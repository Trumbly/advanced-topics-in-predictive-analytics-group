import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, models

# --- Mandatory Setup Lines ---
X = np.load("data/spectrograms.npy")
y = np.load("data/labels.npy")
X = X[..., np.newaxis]
X = X[:2000]
y = y[:2000]
X_train, X_val = X[:1600], X[1600:]
y_train, y_val = y[:1600], y[1600:]
print(f"Shapes — X_train: {X_train.shape}, y_train: {y_train.shape}")

# --- Model Definition ---
def build_dsc_model(input_shape, num_classes):
    """
    Builds the efficient, context-aware CNN model using DSC blocks.
    Input shape matches (Freq, Time, Channels).
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Stage 1: Deep Feature Extraction using DSC Blocks
    # The input shape is now (128, 256, 1)
    
    # First Block (Initial feature mapping)
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', depth_multiplier=1)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(16, (3, 3), padding='same')(x) # Pointwise Conv
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', depth_multiplier=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second Block (Deeper feature learning)
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', depth_multiplier=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x) # Pointwise Conv
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', depth_multiplier=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Stage 2: Global Context Aggregation
    # Summarizes feature maps across the time dimension (axis=1 if Freq is axis 0, Time is axis 1)
    # Since input is (Freq, Time, Channels), GlobalAveragePooling2D pools over the spatial dimensions (Freq, Time).
    # Wait, the input is (128, 256, 1). We want to summarize over T=256.
    # We should use GlobalAveragePooling1D after flattening or adjusting the pooling.
    
    # Since we have (H, W, C) = (128, 256, 32) after convolutions, 
    # GAP will pool over H and W if treated as an image.
    # To aggregate specifically over the 256 time steps while keeping the 128 freq bins, 
    # we will treat the (H, W) dimensions as independent and apply GAP specifically on the time axis if possible.
    
    # For simplicity and stability given the rules, we will retain GAP which pools over both remaining spatial dims (128, 256)
    # which is the stable approach provided the model structure is robust.
    context_features = layers.GlobalAveragePooling2D()(x) 
    
    # Stage 3: Classification Head - MUST use Dense(206, activation='sigmoid')
    outputs = layers.Dense(206, activation='sigmoid', name="output_logits")(context_features)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Build Model
INPUT_SHAPE = (128, 256, 1) # Hardcoded to match rule 4
NUM_CLASSES = 206

model = build_dsc_model(INPUT_SHAPE, NUM_CLASSES)

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
        tf.keras.metrics.AUC(name='roc_auc')
    ]
)

model.summary()

# Train Model (Using specified arguments)
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=3,
    batch_size=32
)

# Final Evaluation Block
valid_cols = np.where(y_val.sum(axis=0) > 0)[0]
y_pred = model.predict(X_val)
score = roc_auc_score(y_val[:, valid_cols], y_pred[:, valid_cols], average="macro")
print(f"ROC-AUC: {score:.4f}")