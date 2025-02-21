import openai
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---- SETUP API (OpenAI) ---- #
OPENAI_API_KEY = "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"  # Replace with your actual API key
openai.api_key = OPENAI_API_KEY

# ---- SETUP OUTPUT DIRECTORY ---- #
output_dir = "entropy_results"
os.makedirs(output_dir, exist_ok=True)

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

# ---- FUNCTION: Query OpenAI API ---- #
def query_openai(prompt, temp):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with prompt '{prompt}': {e}")
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
    print(f"\nTesting temperature: {temp}")
    temp_entropies = []
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        output = query_openai(prompt, temp)
        if output:
            print(f"Response: {output[:50]}...")  # Show first 50 chars
            entropy = compute_entropy(output)
            print(f"Entropy: {entropy:.4f}")
            temp_entropies.append(entropy)
        else:
            print("No response received!")
    
    print(f"\nEntropies for temperature {temp}: {temp_entropies}")
    entropy_results.append(temp_entropies)

print("\nAll entropy results:")
for temp, results in zip(temperatures, entropy_results):
    print(f"Temperature {temp}: {results}")

# ---- PLOT & SAVE ENTROPY RESULTS ---- #
plt.figure(figsize=(8,5))
sns.boxplot(data=entropy_results, palette="coolwarm")
plt.xticks(range(len(temperatures)), temperatures)
plt.xlabel("Temperature")
plt.ylabel("Output Entropy")
plt.title("GPT-3.5 Turbo Output Entropy Across Different Temperatures")
plt.grid()

# Save results as PNG in current repo
entropy_plot_path = os.path.join(output_dir, "openai_entropy_results.png")
plt.savefig(entropy_plot_path)
plt.close()

print(f"Experiment complete! Results saved in: {output_dir}")