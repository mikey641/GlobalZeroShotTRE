import matplotlib.pyplot as plt
from collections import Counter

# Define the two data sets
counter1 = Counter({1: 845, 2: 634, 0: 552, 3: 546, 4: 500, 6: 476, 5: 459, 7: 358, 8: 224, 9: 159,
                    10: 130, 11: 84, 17: 62, 12: 57, 14: 50, 13: 46, 15: 41, 16: 35, 18: 24, 20: 20, 19: 13})
counter2 = Counter({1: 845, 0: 552, 2: 157, 3: 30, 4: 9})

# Get all unique keys sorted
all_keys = sorted(set(counter1.keys()).union(set(counter2.keys())))

# Extract values, setting missing keys to 0
values1 = [counter1.get(k, 0) for k in all_keys]
values2 = [counter2.get(k, 0) for k in all_keys]

# Define bar width
bar_width = 0.4
x = range(len(all_keys))

# Apply style
plt.style.use('ggplot')

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(x, values1, width=bar_width, color='royalblue', edgecolor='black', label="Counter 1", align='center')
plt.bar([i + bar_width for i in x], values2, width=bar_width, color='darkorange', edgecolor='black', label="Counter 2", align='center')

# Labels and title
plt.xlabel("Key", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.title("Comparison of Two Counters", fontsize=14)

# Ensure all keys are shown on x-axis
plt.xticks([i + bar_width / 2 for i in x], all_keys, fontsize=10, rotation=45)

# Show legend
plt.legend()

# Show grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.show()
