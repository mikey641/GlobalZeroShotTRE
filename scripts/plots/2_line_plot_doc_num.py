import matplotlib.pyplot as plt

# Data
buckets_line1 = ["1", "2", "3", "4", "5", "6", "7", "8"]
values_line1 = [69.9, 72.4, 72.9, 72.4, 73.6, 72.4, 73.2, 73.1]

buckets_line2 = ["1", "2", "3", "4", "5", "6", "7", "8"]
values_line2 = [53.7, 54.9, 57.6, 57.6, 58.4]

# Plot
plt.figure(figsize=(12, 10))
plt.plot(buckets_line1, values_line1, marker='o', linestyle='-', label="OmniTemp")
plt.plot(buckets_line2, values_line2, marker='s', linestyle='--', label="NT-6")

# Labels and title
plt.xlabel("Num Of Generations", fontsize=16, fontweight='bold')
plt.ylabel("F1 (%)", fontsize=16, fontweight='bold')
# plt.title("Number of Events in Documents", fontsize=18, fontweight='bold')
plt.legend(fontsize=16)
# plt.xticks(rotation=45, fontsize=11)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)

# Show plot
plt.show()
