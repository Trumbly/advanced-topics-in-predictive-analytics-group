import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- Setup Code Provided By User ---
X = np.load("data/spectrograms.npy")
y = np.load("data/labels.npy")
X = X[..., np.newaxis]
X = X[:2000]
y = y[:2000]
X_train, X_val = X[:1600], X[1600:]
y_train, y_val = y[:1600], y[1600:]
print(f"Shapes — X_train: {X_train.shape}, y_train: {y_train.shape}")

# --- Model Definition Function ---
def build_cnn_model(input_shape, num_classes):
    """
    Builds the 3-layer CNN model tailored for the given input shape and multi-label output.
    Input_shape is expected to be (Mel_Bins, Time_Frames, 1).
    """
    input_tensor = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same', name='conv1')(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Final layer constrained to Dense(206, activation='sigmoid')
    output_tensor = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

# --- Training Function ---
def train_model_keras(model, X_train, y_train, X_val, y_val):
    """
    Compiles and trains the model using the specified parameters.
    """
    # Compile using appropriate loss for multi-label classification
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model as required
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=3, 
        batch_size=32,
        verbose=1
    )
    return history

if __name__ == '__main__':
    # --- Model Initialization ---
    # Hardcoded input shape: (Mel_Bins, Time_Frames, Channels) = (128, 256, 1)
    INPUT_SHAPE = (128, 256, 1)
    N_SPECIES = 206
    
    model = build_cnn_model(INPUT_SHAPE, N_SPECIES)
    model.summary()
    
    # --- Training Execution ---
    print("\n*** Starting Keras Model Training ***")
    train_model_keras(model, X_train, y_train, X_val, y_val)

# --- Validation Code Provided By User ---
valid_cols = np.where(y_val.sum(axis=0) > 0)[0]
y_pred = model.predict(X_val)
score = roc_auc_score(y_val[:, valid_cols], y_pred[:, valid_cols], average="macro")
print(f"ROC-AUC: {score:.4f}")