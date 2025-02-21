import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap
import os
from scipy.stats import entropy #Import Entropy Function

# ---- EXPERIMENT SETUP ---- #
experiment_dir = "transformer_experiment"
os.makedirs(experiment_dir, exist_ok=True)

# ---- HYPERPARAMETERS ---- #
vocab_size = 10
embedding_dim = 16
hidden_dim = 32
seq_length = 4
num_heads = 1
num_layers = 1
epochs = 100

# ---- RANDOM INITIALIZATION ---- #
np.random.seed(42)
embedding_matrix = np.random.randn(vocab_size, embedding_dim)
W_Q = np.random.randn(embedding_dim, embedding_dim)
W_K = np.random.randn(embedding_dim, embedding_dim)
W_V = np.random.randn(embedding_dim, embedding_dim)
W_FFN1 = np.random.randn(embedding_dim, hidden_dim)
W_FFN2 = np.random.randn(hidden_dim, embedding_dim)

# ---- SAMPLE INPUT ---- #
input_tokens = np.array([2, 5, 7, 3])
ground_truth_tokens = np.array([5, 7, 3, 2])

# ---- STORAGE FOR ANALYSIS ---- #
embedding_history = []
attention_history = []
ffn_singular_values = []
attention_entropy = [] 
mutual_info_history = []
output_entropy_history = []

# ---- ENTROPY FUNCTIONS ---- #
def compute_entropy(probabilities):
    return np.mean([entropy(prob) for prob in probabilities])

def compute_conditional_entropy(joint_probs):
    # Compute conditional entropy H(X|Y)
    marginal_y = np.sum(joint_probs, axis=0)
    conditional_probs = joint_probs / (marginal_y + 1e-10)
    return -np.sum(joint_probs * np.log(conditional_probs + 1e-10))

def compute_mutual_information(joint_probs):
    # Compute marginal probabilities
    marginal_x = np.sum(joint_probs, axis=1)
    marginal_y = np.sum(joint_probs, axis=0)
    
    # Compute entropies
    H_X = -np.sum(marginal_x * np.log(marginal_x + 1e-10))
    H_Y = -np.sum(marginal_y * np.log(marginal_y + 1e-10))
    H_XY = -np.sum(joint_probs * np.log(joint_probs + 1e-10))
    
    # Mutual information is I(X;Y) = H(X) + H(Y) - H(X,Y)
    return H_X + H_Y - H_XY

def compute_relative_mutual_information(I_XY, H_X):
    return I_XY / H_X if H_X != 0 else 0

# ---- PLOTTING FUNCTION ---- #
def plot_embeddings(embeddings, title, filename, method="PCA"):
    if method == "PCA":
        reducer = PCA(n_components=2)
    elif method == "UMAP":
        reducer = umap.UMAP(n_components=2)
    else:
        raise ValueError("Method must be 'PCA' or 'UMAP'")
    
    reduced = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(6,6))
    plt.scatter(reduced[:,0], reduced[:,1], color='blue')
    for i, label in enumerate(input_tokens):
        plt.annotate(str(label), (reduced[i,0], reduced[i,1]), fontsize=12, color='red')
    
    plt.title(f"{title} ({method})")
    plt.xlabel(f"{method} Dim 1")
    plt.ylabel(f"{method} Dim 2")
    plt.grid()
    plt.savefig(os.path.join(experiment_dir, filename))
    plt.close()

# ---- LOSS FUNCTION ---- #
loss_history = []

def compute_cross_entropy_loss(predictions, target):
    one_hot_target = np.zeros_like(predictions)
    one_hot_target[np.arange(len(target)), target] = 1
    return -np.sum(one_hot_target * np.log(predictions + 1e-9)) / len(target)

# ---- ADAMW PARAMETERS ---- #
learning_rate = 0.01 # Adjusted for OPENAI- scale models
beta1 = 0.9 # Standard in AdamW
beta2 = 0.95 # DeepSeek-style beta2
epsilon = 1e-8 # Helps numerical stability
weight_decay = 0.01

# Initialize AdamW moment estimates
m_W_Q = np.zeros_like(W_Q)
v_W_Q = np.zeros_like(W_Q)
m_W_K = np.zeros_like(W_K)
v_W_K = np.zeros_like(W_K)
m_W_V = np.zeros_like(W_V)
v_W_V = np.zeros_like(W_V)
m_W_FFN1 = np.zeros_like(W_FFN1)
v_W_FFN1 = np.zeros_like(W_FFN1)
m_W_FFN2 = np.zeros_like(W_FFN2)
v_W_FFN2 = np.zeros_like(W_FFN2)

# ---- TRAINING LOOP ---- #
for epoch in range(epochs):
    # ---- FORWARD PASS ---- #
    X = embedding_matrix[input_tokens]
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    attention_scores = Q @ K.T / np.sqrt(embedding_dim)
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)
    Z = attention_weights @ V
    hidden = np.maximum(0, Z @ W_FFN1)  # ReLU activation
    output = hidden @ W_FFN2
    probabilities = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)

     # Compute Entropy of Outputs
    H_Y = compute_entropy(probabilities)
    output_entropy_history.append(H_Y)
    
    # Compute joint probability matrix between input and output
    joint_probs = np.zeros((vocab_size, vocab_size))
    predictions = np.argmax(probabilities, axis=1)
    # Clip predictions to be within vocabulary range
    predictions = np.clip(predictions, 0, vocab_size - 1)
    for i, j in zip(input_tokens, predictions):
        joint_probs[i, j] += 1
    joint_probs /= len(input_tokens)
    
    # Compute mutual information
    I_XY = compute_mutual_information(joint_probs)
    H_X = -np.sum(np.sum(joint_probs, axis=1) * np.log(np.sum(joint_probs, axis=1) + 1e-10))
    mutual_info_history.append(compute_relative_mutual_information(I_XY, H_X))
    
    # Compute loss ( Loss tracking)
    loss = compute_cross_entropy_loss(probabilities, ground_truth_tokens)
    loss_history.append(loss)

    # Compute entropy of attention matrix
    attn_entropy = entropy(attention_weights.flatten())
    attention_entropy.append(attn_entropy)

    # Store analysis data
    embedding_history.append(X.copy())
    attention_history.append(attention_weights.copy())
    singular_values = np.linalg.svd(output, compute_uv=False)
    ffn_singular_values.append(singular_values)

    # ---- BACKPROPAGATION ---- #
    grad_output = probabilities
    grad_output[np.arange(len(ground_truth_tokens)), ground_truth_tokens] -= 1
    grad_output /= len(ground_truth_tokens)

    # Gradients for FFN
    grad_W_FFN2 = hidden.T @ grad_output
    grad_hidden = grad_output @ W_FFN2.T
    grad_hidden[hidden <= 0] = 0  # ReLU derivative
    grad_W_FFN1 = Z.T @ grad_hidden
    grad_Z = grad_hidden @ W_FFN1.T

    # Gradients for Attention
    grad_V = attention_weights.T @ grad_Z
    grad_attention_weights = grad_Z @ V.T
    grad_attention_scores = grad_attention_weights * attention_weights
    grad_Q = (grad_attention_scores @ K) / np.sqrt(embedding_dim)
    grad_K = (grad_attention_scores.T @ Q) / np.sqrt(embedding_dim)
    grad_W_K = X.T @ grad_K
    grad_W_Q = X.T @ grad_Q
    grad_W_V = X.T @ grad_V

    # ---- ADAMW UPDATE ---- #
    def adamw_update(W, grad_W, m_W, v_W):
        # Weight decay
        W = W * (1 - learning_rate * weight_decay)
        
        # Adam update
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        v_W = beta2 * v_W + (1 - beta2) * (grad_W ** 2)
        
        # Bias correction
        m_hat = m_W / (1 - beta1 ** (epoch + 1))
        v_hat = v_W / (1 - beta2 ** (epoch + 1))
        
        # Update weights
        W -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        return W, m_W, v_W

    W_FFN2, m_W_FFN2, v_W_FFN2 = adamw_update(W_FFN2, grad_W_FFN2, m_W_FFN2, v_W_FFN2)
    W_FFN1, m_W_FFN1, v_W_FFN1 = adamw_update(W_FFN1, grad_W_FFN1, m_W_FFN1, v_W_FFN1)
    W_Q, m_W_Q, v_W_Q = adamw_update(W_Q, grad_W_Q, m_W_Q, v_W_Q)
    W_K, m_W_K, v_W_K = adamw_update(W_K, grad_W_K, m_W_K, v_W_K)
    W_V, m_W_V, v_W_V = adamw_update(W_V, grad_W_V, m_W_V, v_W_V)

    # Store analysis data
    embedding_history.append(X.copy())
    attention_history.append(attention_weights.copy())
    
    # SVD analysis on FFN outputs
    singular_values = np.linalg.svd(output, compute_uv=False)
    ffn_singular_values.append(singular_values)

# ---- PLOT ATTENTION ENTROPY ---- #
plt.figure(figsize=(6,4))
plt.plot(attention_entropy, label="Attention Entropy")
plt.xlabel("Epoch")
plt.ylabel("Entropy")
plt.title("Entropy of Attention Matrix Over Training")
plt.legend()
plt.grid()
plt.savefig(os.path.join(experiment_dir, "attention_entropy.png"))
plt.close()

# ---- PLOT LOSS CURVE ---- #
plt.figure(figsize=(6,4))
plt.plot(loss_history, label="Cross-Entropy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Evolution During Training")
plt.legend()
plt.grid()
plt.savefig(os.path.join(experiment_dir, "loss_curve.png"))
plt.close()

# ---- PLOT ENTROPY OVER TIME ---- #
plt.figure(figsize=(6,4))
plt.plot(output_entropy_history, label="Output Entropy")
plt.xlabel("Epoch")
plt.ylabel("Entropy")
plt.title("Entropy Evolution During Training")
plt.legend()
plt.grid()
plt.savefig(os.path.join(experiment_dir, "output_entropy_curve.png"))
plt.close()

# ---- PLOT MUTUAL INFORMATION OVER TIME ---- #
plt.figure(figsize=(6,4))
plt.plot(mutual_info_history, label="Relative Mutual Information")
plt.xlabel("Epoch")
plt.ylabel("RMI")
plt.title("Relative Mutual Information During Training")
plt.legend()
plt.grid()
plt.savefig(os.path.join(experiment_dir, "mutual_info_curve.png"))
plt.close()

# ---- VISUALIZE EMBEDDING EVOLUTION ---- #
for i, embeddings in enumerate([embedding_history[0], embedding_history[epochs//2], embedding_history[-1]]):
    plot_embeddings(embeddings, f"Embeddings at Epoch {i * (epochs//2)}", f"embeddings_epoch_{i}.png", method="PCA")
    plot_embeddings(embeddings, f"Embeddings at Epoch {i * (epochs//2)}", f"umap_embeddings_epoch_{i}.png", method="UMAP")

# ---- VISUALIZE ATTENTION EVOLUTION ---- #
for i, attn in enumerate([attention_history[0], attention_history[epochs//2], attention_history[-1]]):
    plt.figure(figsize=(6,6))
    sns.heatmap(attn, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title(f"Self-Attention Weights at Epoch {i * (epochs//2)}")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.savefig(os.path.join(experiment_dir, f"attention_epoch_{i}.png"))
    plt.close()

# ---- VISUALIZE SINGULAR VALUE COLLAPSE ---- #
plt.figure(figsize=(6,4))
for i in range(len(ffn_singular_values[0])):  # Track each singular value separately
    plt.plot([sv[i] for sv in ffn_singular_values], label=f"Singular Value {i+1}")
plt.xlabel("Epoch")
plt.ylabel("Singular Values")
plt.title("Singular Value Collapse in FFN Outputs")
plt.legend()
plt.grid()
plt.savefig(os.path.join(experiment_dir, "ffn_singular_values.png"))
plt.close()

print("Experiment complete. Visualizations saved.")