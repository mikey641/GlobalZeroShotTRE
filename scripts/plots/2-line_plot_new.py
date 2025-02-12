import matplotlib.pyplot as plt

# Data
buckets_line1 = ["n<=10", "n<=15", "n<=20", "n<=25", "n<=30", "n<=50"]
values_line1 = [72.6, 67.1, 66.7, 62.2, 63.0, 63.3]

buckets_line2 = ["n<=20", "n<=25", "n<=30", "n<=50", "n<=60", "n>60"]
values_line2 = [66.2, 45.0, 50.9, 48.8, 43.3, 42.8]

# Plot
plt.figure(figsize=(12, 10))
plt.plot(buckets_line1, values_line1, marker='o', linestyle='-', label="MATRES")
plt.plot(buckets_line2, values_line2, marker='s', linestyle='--', label="TimeBank-Dense")

# Labels and title
# plt.xlabel("Threshold", fontsize=16)
plt.ylabel("F1 (%)", fontsize=16, fontweight='bold')
# plt.title("Number of Events in Documents", fontsize=18, fontweight='bold')
plt.legend(fontsize=16)
# plt.xticks(rotation=45, fontsize=11)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)

# Show plot
plt.show()
