import json

# Load results
with open("results/comparison_results.json", "r") as f:
    results = json.load(f)

# Print and visualize results
for model_name, metrics in results.items():
    print(f"{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
