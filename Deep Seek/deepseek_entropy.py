import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ---- SETUP API (DeepSeek) ---- #
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # Replace with actual endpoint
API_KEY = "sk-614f19b5323e4778bb8e30474489b561"  # Replace with actual API key

# ---- PROMPT LIST ---- #
prompts = [
    "What is the capital of France?",
    "Explain quantum entanglement.",
    "How do transformers learn?",
    "What are the origins of life?",
    "Write a short poem about entropy.",
]

# ---- TEMPERATURE SETTINGS ---- #
temperatures = [0.1, 0.5, 1.0, 1.5]

# ---- FUNCTION: Query DeepSeek ---- #
def query_deepseek(prompt, temp):
    payload = {
        "prompt": prompt,
        "temperature": temp,
        "max_tokens": 100
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json().get("text", "").strip()
    else:
        print(f"Error with prompt '{prompt}': {response.status_code}")
        return None

# ---- FUNCTION: Compute Entropy ---- #
def compute_entropy(text):
    char_counts = np.array([text.count(c) for c in set(text)])
    probabilities = char_counts / char_counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# ---- RUN EXPERIMENT ---- #
entropy_results = []

for temp in temperatures:
    temp_entropies = []
    
    for prompt in prompts:
        output = query_deepseek(prompt, temp)
        if output:
            entropy = compute_entropy(output)
            temp_entropies.append(entropy)
    
    entropy_results.append(temp_entropies)

# ---- PLOT ENTROPY RESULTS ---- #
plt.figure(figsize=(8,5))
sns.boxplot(data=entropy_results, palette="coolwarm")
plt.xticks(range(len(temperatures)), temperatures)
plt.xlabel("Temperature")
plt.ylabel("Output Entropy")
plt.title("DeepSeek Output Entropy Across Different Temperatures")
plt.grid()
plt.show()