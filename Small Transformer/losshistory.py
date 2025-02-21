import matplotlib.pyplot as plt

loss_history = []  # Store loss values per epoch

def compute_cross_entropy_loss(predictions, target):
    one_hot_target = np.zeros_like(predictions)
    one_hot_target[np.arange(len(target)), target] = 1
    return -np.sum(one_hot_target * np.log(predictions + 1e-9)) / len(target)

# Example during training loop
for epoch in range(epochs):
    predicted_probabilities = model(input_tokens)  # Simulated model prediction
    loss = compute_cross_entropy_loss(predicted_probabilities, ground_truth_tokens)
    loss_history.append(loss)

# Plot Loss Curve
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Loss Evolution During Training")
plt.show()