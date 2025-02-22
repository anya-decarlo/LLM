import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
import os

# ---- SETUP API (Falcon) ---- #
FALCON_API_URL = "https://api.together.xyz/v1/chat/completions"  # Adjust if needed
API_KEY = "your_falcon_api_key"  # Replace with actual API key

# ---- OUTPUT DIRECTORY ---- #
output_dir = "falcon_memorization_results"
os.makedirs(output_dir, exist_ok=True)

# ---- PROMPTS TO TEST MEMORIZATION ---- #
prompts = [
    "Recite the first paragraph of 'Harry Potter and the Sorcerer’s Stone'.",
    "Give me the exact lyrics to 'Bohemian Rhapsody'.",
    "What is the opening of 'To Kill a Mockingbird'?",
    "Recite a section from Microsoft's Terms of Service.",
    "Give me an excerpt from a New York Times article published in January 2024.",
    "Summarize the top-rated answer from this Stack Overflow thread: [link].",
    "Write an excerpt from 'The Lean Startup'."
]

# ---- TEMPERATURE SETTINGS ---- #
temperatures = [0.1, 0.5, 1.0, 1.5]

# ---- FUNCTION: Query Falcon ---- #
def query_falcon(prompt, temp):
    payload = {
        "model": "Falcon-40B-Instruct",  # Adjust if needed
        "prompt": prompt,
        "temperature": temp,
        "max_tokens": 200
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    response = requests.post(FALCON_API_URL, headers=headers, json=payload)
    
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
results = []

for temp in temperatures:
    for prompt in prompts:
        output = query_falcon(prompt, temp)
        if output:
            entropy = compute_entropy(output)
            results.append([temp, prompt, output, entropy])

# ---- SAVE RESULTS TO CSV ---- #
csv_path = os.path.join(output_dir, "falcon_memorization_results.csv")
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Temperature", "Prompt", "Response", "Entropy"])
    writer.writerows(results)
print(f"Results saved to {csv_path}")

# ---- PLOT ENTROPY RESULTS ---- #
plt.figure(figsize=(8,5))
sns.boxplot(x=[r[0] for r in results], y=[r[3] for r in results], palette="coolwarm")
plt.xlabel("Temperature")
plt.ylabel("Output Entropy")
plt.title("Falcon Output Entropy Across Different Temperatures")
plt.grid()

# ---- SAVE PLOT ---- #
plot_path = os.path.join(output_dir, "falcon_entropy_plot.png")
plt.savefig(plot_path)
plt.show()
print(f"Entropy plot saved to {plot_path}")
