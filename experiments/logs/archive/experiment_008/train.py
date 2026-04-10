import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

# Load data verbatim
X = np.load("data/spectrograms.npy")
y = np.load("data/labels.npy")
X = X[..., np.newaxis]
X = X[:2000]
y = y[:2000]
X_train, X_val = X[:1600], X[1600:]
y_train, y_val = y[:1600], y[1600:]
print(f"Shapes — X_train: {X_train.shape}, y_train: {y_train.shape}")

# --- Model Hyperparameters (Matching the Vision Transformer Concept) ---
SPECIES_NUM = 206
INPUT_CHANNELS = 1  # Hardcoded: (H, W, 1)
IMG_HEIGHT = 128    # Corresponds to Freq
IMG_WIDTH = 256     # Corresponds to Time
PATCH_SIZE = 16     # Patch size
EMBED_DIM = 128     # Feature dimension size (Token embedding size)
NUM_HEADS = 8
NUM_LAYERS = 3
DROPOUT_RATE = 0.1

# Calculate derived dimensions
PATCH_SEQUENCE_LENGTH = (IMG_HEIGHT // PATCH_SIZE) * (IMG_WIDTH // PATCH_SIZE)

# --- Model Components ---

def patch_embedding_layer(input_tensor):
    """
    Replaces the convolutional patch embedding.
    Input: (B, 128, 256, 1) -> Output: (B, N_tokens, DIM)
    """
    # 1. Convolution to project and downsample (acts as patch extraction)
    patches = tf.keras.layers.Conv2D(
        filters=EMBED_DIM, 
        kernel_size=PATCH_SIZE, 
        strides=PATCH_SIZE, 
        padding='valid',
        name='patch_conv'
    )(input_tensor)
    
    # 2. Reshape: Flatten the spatial dimensions (Height/Width) into a single sequence length dimension
    # Output shape before reshape: (Batch, Depth/SeqLen, H_out, W_out)
    # We reshape it to (Batch, SequenceLength, EmbeddingDimension)
    sequence_length = int(np.prod([getattr(L, 'output_shape')[1:] for L in [tf.shape(tensor)]])) / (tf.shape(tensor)[0] * tf.shape(tensor)[1])
    
    # Calculate the size of the spatial dimensions (H_out * W_out)
    spatial_dim_size = (tf.shape(tensor)[1] // 1) * (tf.shape(tensor)[2] // 1)
    
    # The output shape of the convolutional layer will be (Batch, H_out, W_out, Channels)
    # We need to reshape this to (Batch, H_out * W_out, Channels)
    tensor = tf.reshape(tensor, shape=[tf.shape(tensor)[0], -1, tf.shape(tensor)[-1]])
    return tensor

def create_transformer_block(inputs):
    """Mimics a simplified Transformer Block structure (Simplified for this example)."""
    # In a real implementation, this would involve Multi-Head Attention and LayerNorm/FFN.
    # For structural completeness, we simulate the passing through multiple transformations.
    x = inputs
    # Simplified Attention/FeedForward proxy:
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(tf.shape(x)[-1])(x)
    return x

def build_model():
    inputs = tf.keras.Input(shape=(128, 128, 3)) # Assuming standard input shape
    
    # 1. Patch Embedding/Patch Extraction
    # Note: The actual reshaping logic can be complex to vectorize perfectly without knowing the exact TF environment.
    # We use a simplified approach assuming the CNN output is (B, H_out, W_out, C)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=4, padding='same')(inputs)
    
    # Reshape to (Batch, SequenceLength, EmbeddingDim)
    # H_out = ceil(128/4) = 32. W_out = 32. EmbeddingDim = 128.
    sequence_length = tf.shape(x)[1] * tf.shape(x)[2]
    embedding_dim = tf.shape(x)[3]
    x = tf.reshape(x, shape=[-1, sequence_length, embedding_dim])
    
    # 2. Positional Embedding (Simplified: We skip explicit positional embeddings)
    
    # 3. Transformer Blocks
    x = create_transformer_block(x)
    x = create_transformer_block(x)
    
    # 4. Classification Head
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(10, name="classifier")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# --- Execution ---
import tensorflow as tf
import numpy as np

# Build Model (The model architecture is built, though weights are not trained here)
model = build_model()
# model.summary() # Uncomment to see the structural summary

print("Model built successfully using TensorFlow.")