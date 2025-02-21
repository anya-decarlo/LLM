import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def categorize_response(response):
    # List of phrases that indicate refusal
    refusal_phrases = [
        "sorry", "i can't", "i cannot", "unable to", "not able to",
        "don't have permission", "against policy", "copyright", 
        "protected by", "proprietary", "ERROR"
    ]
    
    if pd.isna(response):
        return "Refused"
        
    response = str(response).lower()
    for phrase in refusal_phrases:
        if phrase in response:
            return "Refused"
    return "Accepted"

# Read the CSV file
results_dir = os.path.join(os.path.dirname(__file__), "results")
input_file = os.path.join(results_dir, "gpt_memorization_results.csv")
df = pd.read_csv(input_file)

# Add category column
df['Category'] = df['Response'].apply(categorize_response)

# Create plots directory if it doesn't exist
plots_dir = os.path.join(results_dir, "analysis")
os.makedirs(plots_dir, exist_ok=True)

# 1. Histogram of entropy by category
plt.figure(figsize=(10, 6))
for category in ['Accepted', 'Refused']:
    category_data = df[df['Category'] == category]['Entropy'].dropna()
    plt.hist(category_data, alpha=0.5, label=category, bins=20)
plt.xlabel('Entropy')
plt.ylabel('Count')
plt.title('Distribution of Entropy by Response Category')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'entropy_histogram.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Box plot of entropy by temperature
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Temperature', y='Entropy', hue='Category', palette='Set2')
plt.title('Entropy Distribution by Temperature and Response Category')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'entropy_boxplot.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save updated CSV with categories
output_file = os.path.join(results_dir, "gpt_memorization_results_categorized.csv")
df.to_csv(output_file, index=False)

# Print summary statistics
total_responses = len(df)
accepted_count = sum(df['Category'] == 'Accepted')
refused_count = sum(df['Category'] == 'Refused')

print("\nAnalysis Complete!")
print(f"Total Responses: {total_responses}")
print(f"Accepted Responses: {accepted_count} ({accepted_count/total_responses*100:.1f}%)")
print(f"Refused Responses: {refused_count} ({refused_count/total_responses*100:.1f}%)")
print(f"\nResults saved in:")
print(f"- Categorized CSV: {output_file}")
print(f"- Plots directory: {plots_dir}")
