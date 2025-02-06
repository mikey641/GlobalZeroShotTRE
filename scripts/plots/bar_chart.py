import matplotlib.pyplot as plt
from collections import Counter

# Data
data = Counter({1: 845, 2: 634, 0: 552, 3: 546, 4: 500, 6: 476, 5: 459, 7: 358, 8: 224, 9: 159, 10: 130, 11: 84, 17: 62, 12: 57, 14: 50, 13: 46, 15: 41, 16: 35, 18: 24, 20: 20, 19: 13})

# Extract keys and values
keys = list(data.keys())
values = list(data.values())

# Define plot style
plt.style.use('ggplot')

# Create bar chart
plt.figure(figsize=(8, 6))
plt.bar(keys, values, color='royalblue', edgecolor='black')

# Labels and title
plt.xlabel("Sent Distance", fontsize=12)
plt.ylabel("Number Of Pairs", fontsize=12)
plt.title("Bar Chart of Counter Data", fontsize=14)

# Ensure all keys are shown
plt.xticks(keys, fontsize=10)
plt.yticks(fontsize=10)

# Show grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.show()
