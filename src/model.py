import tensorflow as tf
import numpy as np

# Disable oneDNN warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Optional: Mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')  # Change to float32 to avoid AVX-512 issues


###############################################################################
# 1. HYPERPARAMETERS AND GLOBALS
###############################################################################
NUM_GENES = 500            # Dimensionality of gene expression
NUM_SAMPLES = 2000         # Number of synthetic samples
NUM_CONTEXTS = 3      # Example: number of different context conditions
HIDDEN_DIM = 256        # Example: hidden dimension for the MLP
CONTEXT_EMB_DIM = 64    # Dimension for context embedding
TIME_EMB_DIM = 32       # Dimension for time embedding
LEARNING_RATE = 1e-3
EPOCHS = 10             # Example: number of training epochs
T_MAX = 1.0             # Maximum time for sampling t from [0, T_MAX]

# Regularization coefficients (example values)
LAMBDA_KEGG = 1.0       # Weight for KEGG prior
LAMBDA_ENCODE = 1.0     # Weight for ENCODE prior
LAMBDA_SPARSE = 1e-4    # Weight for adjacency L1 sparsity

LAMBDA_KEGG    = 1e-4  # KEGG/Reactome pathway prior weight
LAMBDA_ENCODE  = 1e-4  # ENCODE prior weight
LAMBDA_SPARSE  = 1e-4  # L1 sparsity penalty

import numpy as np
import tensorflow as tf


def generate_synthetic_data_with_priors(num_samples, num_genes, num_contexts=3, prior_fraction=0.1):
    """
    Generates synthetic gene expression data along with context labels and prior knowledge matrices.

    Parameters:
        num_samples (int): Number of samples to generate.
        num_genes (int): Number of genes/features per sample.
        num_contexts (int, optional): Number of context labels. Default is 3.
        prior_fraction (float, optional): Fraction of gene pairs to be assigned prior knowledge. Default is 0.1.

    Returns:
        tuple:
            - dataset (tf.data.Dataset): TensorFlow dataset containing expression data and context labels.
            - kegg_prior (np.ndarray): Prior matrix based on KEGG pathways.
            - encode_prior (np.ndarray): Prior matrix based on ENCODE data.
    """
    np.random.seed(42)

    # Generate synthetic expression data and context labels
    expr_data = np.random.gamma(shape=2.0, scale=2.0, size=(num_samples, num_genes)).astype(np.float32)
    context_labels = np.random.randint(low=0, high=num_contexts, size=(num_samples,)).astype(np.int32)

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((expr_data, context_labels))
    dataset = dataset.shuffle(1000).batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    # Initialize prior matrices
    kegg_prior = np.zeros((num_genes, num_genes), dtype=np.float32)
    encode_prior = np.zeros((num_genes, num_genes), dtype=np.float32)

    # Assign prior knowledge based on random gene connections
    num_prior_connections = int(num_genes * prior_fraction)
    for _ in range(num_prior_connections):
        src, tgt = np.random.randint(0, num_genes, size=2)
        kegg_prior[src, tgt] = 1.0
        encode_prior[src, tgt] = 1.0

    return dataset, kegg_prior, encode_prior

dataset, kegg_prior, encode_prior = generate_synthetic_data_with_priors(NUM_SAMPLES, NUM_GENES, NUM_CONTEXTS)


###############################################################################
# 2. EMBEDDING LAYERS
###############################################################################
class MLPTimeEmbedding(tf.keras.layers.Layer):
    """
    A simple MLP-based time embedding.
    Inputs: t of shape (batch_size, 1).
    Outputs: embedded time of shape (batch_size, emb_dim).
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(emb_dim, activation='relu'),
            tf.keras.layers.Dense(emb_dim)  # final linear layer
        ])

    def call(self, t):
        # t is expected to have shape (batch_size, 1)
        return self.net(t)


class ContextEmbedding(tf.keras.layers.Layer):
    """
    An embedding layer for discrete context IDs.
    Inputs: context_ids of shape (batch_size,).
    Outputs: embedded context of shape (batch_size, emb_dim).
    """

    def __init__(self, num_contexts, emb_dim):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=num_contexts,
            output_dim=emb_dim
        )

    def call(self, context_ids):
        return self.embedding(context_ids)


###############################################################################
# 3. LEARNABLE ADJACENCY LAYER
###############################################################################
class LearnableAdjacency(tf.keras.layers.Layer):
    """
    A layer that maintains a learnable adjacency matrix A (num_genes x num_genes).
    """

    def __init__(self, num_genes):
        super().__init__()
        self.A = self.add_weight(
            shape=(num_genes, num_genes),
            initializer="glorot_uniform",
            trainable=True,
            name="learnable_adjacency"
        )

    def build(self, input_shape):
        pass  # Not strictly necessary; no additional build steps required.

    def call(self):
        return tf.cast(self.A, tf.float32)


###############################################################################
# 4. GRN DIFFUSION MODEL
###############################################################################
class GRNDiffusionModel(tf.keras.Model):
    """
    Main diffusion model that:
      - Learns an adjacency matrix for gene-gene interactions.
      - Embeds time (t) and context (context_ids).
      - Merges expression x, adjacency-transformed x, time embedding, context embedding.
      - Predicts noise (or score) via a final MLP.
    """

    def __init__(self,
                 num_genes,
                 num_contexts,
                 hidden_dim=256,
                 time_emb_dim=32,
                 context_emb_dim=64):
        super().__init__()

        # Learnable adjacency
        self.adjacency = LearnableAdjacency(num_genes)

        # Time embedding
        self.time_embed = MLPTimeEmbedding(time_emb_dim)

        # Context embedding
        self.context_embed = ContextEmbedding(num_contexts, context_emb_dim)

        # Final MLP that merges expression, adjacency output, time emb, context emb
        self.final_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='gelu'),
            tf.keras.layers.Dense(hidden_dim, activation='gelu'),
            tf.keras.layers.Dense(num_genes)  # predict noise for each gene
        ])

    def call(self, x, t, context_ids):
        """
        x:            (batch_size, num_genes)
        t:            (batch_size, 1) for time
        context_ids:  (batch_size,) for discrete contexts
        Returns: predicted noise (batch_size, num_genes)
        """
        # Convert types
        x = tf.cast(x, tf.float32)
        t = tf.cast(t, tf.float32)
        context_ids = tf.cast(context_ids, tf.int32)

        # Adjacency matrix
        A = self.adjacency.A

        # Gene–gene interaction term
        x_adj = tf.matmul(x, A)  # (batch_size, num_genes)

        # Embeddings
        t_emb = self.time_embed(t)  # (batch_size, time_emb_dim)
        c_emb = self.context_embed(context_ids)  # (batch_size, context_emb_dim)

        # Concatenate expression, adjacency-transformed expression, and embeddings
        merged = tf.concat([x, x_adj, t_emb, c_emb],
                           axis=-1)  # (batch_size, num_genes + num_genes + time_emb_dim + context_emb_dim)

        # Predict noise
        pred_noise = self.final_mlp(merged)
        return pred_noise

    def get_adjacency_matrix(self):
        """
        Returns the learned adjacency matrix (num_genes x num_genes).
        """
        return self.adjacency.A


###############################################################################
# 5. VP-SDE FORWARD NOISING
###############################################################################
def vp_sde_forward_noising(x0, t, beta=5.0):
    """
    Simple variance-preserving SDE forward process.
    x0: (batch_size, num_genes)
    t:  (batch_size, 1), values in [0, T_MAX]
    beta: float, controlling noise growth rate

    x_t = alpha(t)*x_0 + sigma(t)*eps
    where alpha(t) = exp(-0.5 * beta * t),
          sigma(t) = sqrt(1 - alpha(t)^2),
          eps ~ N(0, I).
    Returns:
      x_t: noised expression
      noise: the actual noise eps
    """
    alpha_t = tf.exp(-0.5 * beta * t)  # (batch_size, 1)
    sigma_t = tf.sqrt(1.0 - alpha_t ** 2)
    noise = tf.random.normal(shape=tf.shape(x0), dtype=x0.dtype)
    x_t = alpha_t * x0 + sigma_t * noise
    return x_t, noise


###############################################################################
# 6. TRAIN STEP WITH BIOLOGICAL PRIORS & L1 SPARSITY
###############################################################################
@tf.function
def train_step(model, optimizer, x0, context_ids, kegg_prior, encode_prior, beta=5.0):
    """
    model:        GRNDiffusionModel
    optimizer:    tf.keras Optimizer
    x0:           (batch_size, num_genes)
    context_ids:  (batch_size,)
    kegg_prior:   (num_genes, num_genes)
    encode_prior: (num_genes, num_genes)
    beta:         float, for forward noising in vp_sde_forward_noising()

    Returns:
      mse_loss: the MSE of noise prediction
      reg_loss: the combined regularization loss (kegg, encode, L1)
    """
    batch_size = tf.shape(x0)[0]

    # Sample random t from [0, T_MAX]
    t_random = tf.random.uniform((batch_size, 1), minval=0.0, maxval=T_MAX, dtype=tf.float32)

    # Forward noising process
    x_t, true_noise = vp_sde_forward_noising(x0, t_random, beta)

    with tf.GradientTape() as tape:
        # Predict noise
        noise_pred = model(x_t, t_random, context_ids)

        # MSE loss (noise prediction vs actual noise)
        mse_loss = tf.reduce_mean(tf.square(noise_pred - true_noise))

        # Regularization from adjacency matrix
        A = model.get_adjacency_matrix()
        kegg_loss = tf.reduce_mean(tf.square(A - kegg_prior))
        encode_loss = tf.reduce_mean(tf.square(A - encode_prior))
        l1_sparsity = tf.reduce_sum(tf.abs(A))

        reg_loss = (LAMBDA_KEGG * kegg_loss +
                    LAMBDA_ENCODE * encode_loss +
                    LAMBDA_SPARSE * l1_sparsity)

        total_loss = mse_loss + reg_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return mse_loss, reg_loss


###############################################################################
# 7. TRAINING LOOP
###############################################################################
def main():
    # Instantiate model and optimizer
    model = GRNDiffusionModel(
        num_genes=NUM_GENES,
        num_contexts=NUM_CONTEXTS,
        hidden_dim=HIDDEN_DIM,
        time_emb_dim=TIME_EMB_DIM,
        context_emb_dim=CONTEXT_EMB_DIM
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Example biological priors (fake data here; in practice, use real priors)
    kegg_prior = tf.random.normal((NUM_GENES, NUM_GENES), dtype=tf.float32)
    encode_prior = tf.random.normal((NUM_GENES, NUM_GENES), dtype=tf.float32)

    # Example dataset
    # Each element is (x_batch, context_batch)
    # x_batch: (batch_size, NUM_GENES), context_batch: (batch_size,)
    dataset = [
        (tf.random.normal((16, NUM_GENES)),
         tf.random.uniform((16,), minval=0, maxval=NUM_CONTEXTS, dtype=tf.int32))
        for _ in range(100)  # 100 steps per epoch, for example
    ]

    # Training loop
    for epoch in range(EPOCHS):
        epoch_mse, epoch_reg = 0.0, 0.0
        steps = 0

        for (x_batch, c_batch) in dataset:
            mse_val, reg_val = train_step(model, optimizer, x_batch, c_batch, kegg_prior, encode_prior)
            epoch_mse += mse_val.numpy()
            epoch_reg += reg_val.numpy()
            steps += 1

        print(f"Epoch {epoch + 1}/{EPOCHS} "
              f"| MSE: {epoch_mse / steps:.4f} "
              f"| RegLoss: {epoch_reg / steps:.4f}")

    # After training, let's do a quick forward pass with some random input
    sampled_expr = model(
        tf.random.normal((5, NUM_GENES)),  # x
        tf.ones((5, 1)),  # t
        tf.ones((5,), dtype=tf.int32)  # context_ids
    )
    print("Sampled expression shape:", sampled_expr.shape)


if __name__ == "__main__":
    main()