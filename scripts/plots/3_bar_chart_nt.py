import matplotlib.pyplot as plt
from collections import Counter

# Define the three data sets
counter1 = Counter({1: 796, 2: 627, 0: 537, 3: 533, 4: 454, 6: 451, 5: 422, 7: 336, 8: 215, 9: 156, 10: 132, 11: 83, 12: 70, 17: 62, 14: 45, 13: 43, 15: 40, 16: 33, 18: 24, 20: 24, 19: 13})
counter2 = Counter({1: 796, 0: 537})
counter3 = Counter({1: 796, 0: 537, 2: 144, 3: 6})

# Get all unique keys sorted
all_keys = sorted(set(counter1.keys()).union(set(counter2.keys())).union(set(counter3.keys())))

# Extract values, setting missing keys to 0
values1 = [counter1.get(k, 0) for k in all_keys]
values2 = [counter2.get(k, 0) for k in all_keys]
values3 = [counter3.get(k, 0) for k in all_keys]

# Define bar width
bar_width = 0.3
x = range(len(all_keys))

# Apply style
plt.style.use('ggplot')

# Create the bar chart
plt.figure(figsize=(12, 6))
plt.bar(x, values1, width=bar_width, color='royalblue', edgecolor='black', label="All Relations", align='center')
plt.bar([i + bar_width for i in x], values2, width=bar_width, color='darkorange', edgecolor='black', label="Only in Consecutive Sentences", align='center')
plt.bar([i + 2 * bar_width for i in x], values3, width=bar_width, color='seagreen', edgecolor='black', label="Consecutive Sentences + Transitive", align='center')

# Labels and title
plt.xlabel("Sentence Distance", fontsize=16)
plt.ylabel("Number Of Non-Vague Relations", fontsize=16)
# plt.title("NarrativeTime Relations Distance", fontsize=14)

# Ensure all keys are shown on x-axis
plt.xticks([i + bar_width for i in x], all_keys, fontsize=10, rotation=45)

# Show legend
plt.legend()

# Show grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.show()
