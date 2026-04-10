import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

X = np.load("data/spectrograms.npy")
y = np.load("data/labels.npy")
X = X[..., np.newaxis]
X = X[:2000]
y = y[:2000]
X_train, X_val = X[:1600], X[1600:]
y_train, y_val = y[:1600], y[1600:]
print(f"Shapes — X_train: {X_train.shape}, y_train: {y_train.shape}")

def build_deep_cnn_model(input_shape):
    """
    Builds a deep CNN backbone for feature extraction.
    Progressively increases filters and depth as requested.
    """
    inputs = Input(shape=input_shape)
    
    # Block 1: Increase filters and depth
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)  # Halves spatial dimensions
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 2: Further increase capacity
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x) # Added regularization for deeper model
    
    # Block 3: Deepest feature extraction layers
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Flatten and dense classification head
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.4)(x)
    
    # Final layer must match constraints
    outputs = Dense(206, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

input_shape = (128, 256, 1)
model = build_deep_cnn_model(input_shape)

# Compile the model for multi-label classification
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Train the model using the specified parameters
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32)

valid_cols = np.where(y_val.sum(axis=0) > 0)[0]
y_pred = model.predict(X_val)
score = roc_auc_score(y_val[:, valid_cols], y_pred[:, valid_cols], average="macro")
print(f"ROC-AUC: {score:.4f}")