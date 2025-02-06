import matplotlib.pyplot as plt

# Define categories (x-axis) and accuracy scores (y-values)
categories = ["Before", "After", "Equal", "Vague"]
scores = {
    "CoT": [70.7, 78.2, 33.5, 20.6], # CoT
    "Global": [64.0, 63.5, 44.2, 1.9], # Global
    "TimeLine": [69.7, 73.5, 49.8, 10.5], # TimeLine
    # 4: [74.2, 77.4, 62.8, 0.0],  # TimeLine + Const
    # 4: [81.8, 80.1, 63.9, 46.9] # Bayesian
}

# Define bar width and x positions
bar_width = 0.22
x = range(len(categories))

# Colors for different bars
colors = ['royalblue', 'darkorange', 'seagreen', 'crimson', 'purple']

# Apply style
plt.style.use('ggplot')

# Create figure
plt.figure(figsize=(8, 6))

# Plot bars for each score group
for i, (key, values) in enumerate(scores.items()):
    plt.bar([pos + i * bar_width for pos in x], values, width=bar_width,
            color=colors[i], edgecolor='black', label=f"{key}")

# Labels and title
# plt.xlabel("Relation Type", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=16)
plt.title("OmniTemp", fontsize=18, fontweight='bold')

# Ensure all categories are shown on x-axis
plt.xticks([pos + (len(scores) / 2 - 0.5) * bar_width for pos in x], categories, fontsize=12)

# Show legend
plt.legend()

# Show grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.show()
