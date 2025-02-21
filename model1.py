import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

# ---- EXPERIMENT SETUP ---- #
experiment_dir = "transformer_experiment"
os.makedirs(experiment_dir, exist_ok=True)  # Ensure directory exists for saving plots

# ---- HYPERPARAMETERS ---- #
vocab_size = 10  # Tiny vocabulary
embedding_dim = 16
hidden_dim = 32
seq_length = 4
num_heads = 1
num_layers = 1
epochs = 100  # Training steps

# ---- RANDOM INITIALIZATION ---- #
np.random.seed(42)
embedding_matrix = np.random.randn(vocab_size, embedding_dim)
W_Q = np.random.randn(embedding_dim, embedding_dim)
W_K = np.random.randn(embedding_dim, embedding_dim)
W_V = np.random.randn(embedding_dim, embedding_dim)
W_FFN1 = np.random.randn(embedding_dim, hidden_dim)
W_FFN2 = np.random.randn(hidden_dim, embedding_dim)

# ---- SAMPLE INPUT ---- #
input_tokens = np.array([2, 5, 7, 3])  # Example: ["cat", "sat", "on", "mat"]
ground_truth_tokens = np.array([5, 7, 3, 2])  # Expected next-token predictions for training

# ---- EMBEDDING LAYER ---- #
X = embedding_matrix[input_tokens]

# ---- PLOTTING FUNCTION ---- #
def plot_embeddings(embeddings, title, filename):
    """Visualizes word embeddings using PCA."""
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(6,6))
    plt.scatter(reduced[:,0], reduced[:,1], color='blue')
    for i, label in enumerate(input_tokens):
        plt.annotate(str(label), (reduced[i,0], reduced[i,1]), fontsize=12, color='red')
    
    plt.title(title)
    plt.xlabel("PCA Dim 1")
    plt.ylabel("PCA Dim 2")
    plt.grid()
    plt.savefig(os.path.join(experiment_dir, filename))  # Save plot
    plt.close()  # Prevent display in large experiments

plot_embeddings(X, "Word Embeddings in 2D", "embeddings_before_training.png")

# ---- SELF-ATTENTION ---- #
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# Compute attention scores
attention_scores = Q @ K.T / np.sqrt(embedding_dim)
attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)

# ---- PLOT ATTENTION WEIGHTS ---- #
def plot_attention(attn_weights, title, filename):
    """Visualizes self-attention weights as a heatmap."""
    plt.figure(figsize=(6,6))
    sns.heatmap(attn_weights, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.savefig(os.path.join(experiment_dir, filename))  # Save plot
    plt.close()

plot_attention(attention_weights, "Self-Attention Weights Before Training", "attention_before_training.png")

# ---- FEEDFORWARD NETWORK ---- #
hidden = np.maximum(0, X @ W_FFN1)  # ReLU activation
output = hidden @ W_FFN2

# ---- VISUALIZE FEEDFORWARD OUTPUT ---- #
plot_embeddings(output, "Feedforward Output in 2D", "feedforward_before_training.png")

# ---- PREDICTION ---- #
probabilities = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
predicted_token = np.argmax(probabilities, axis=1)

print("Predicted Token IDs Before Training:", predicted_token)

# ---- TRAINING LOOP ---- #
loss_history = []  # Store loss values per epoch

def compute_cross_entropy_loss(predictions, target):
    """Computes cross-entropy loss for tracking model learning."""
    one_hot_target = np.zeros_like(predictions)
    one_hot_target[np.arange(len(target)), target] = 1
    return -np.sum(one_hot_target * np.log(predictions + 1e-9)) / len(target)

# Simulated Training Process
for epoch in range(epochs):
    # Forward pass
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    attention_scores = Q @ K.T / np.sqrt(embedding_dim)
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)
    
    # Apply attention to values
    Z = attention_weights @ V
    
    # Feedforward network
    hidden = np.maximum(0, Z @ W_FFN1)  # ReLU activation
    output = hidden @ W_FFN2
    
    # Prediction & loss
    probabilities = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
    loss = compute_cross_entropy_loss(probabilities, ground_truth_tokens)
    loss_history.append(loss)

# ---- PLOT LOSS CURVE ---- #
plt.figure(figsize=(6,4))
plt.plot(loss_history, label="Cross-Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Evolution During Training")
plt.legend()
plt.grid()
plt.savefig(os.path.join(experiment_dir, "loss_curve.png"))  # Save loss plot
plt.close()

print("Training Complete. Loss curve saved.")

# ---- POST-TRAINING VISUALIZATION ---- #
plot_embeddings(X, "Word Embeddings After Training", "embeddings_after_training.png")
plot_attention(attention_weights, "Self-Attention Weights After Training", "attention_after_training.png")
plot_embeddings(output, "Feedforward Output After Training", "feedforward_after_training.png")