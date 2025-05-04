import matplotlib.pyplot as plt

# Data
buckets_line1 = ["n<=10", "n<=15", "n<=20", "n<=25", "n<=30", "n<=50"]
values_line1 = [72.6, 67.1, 66.7, 62.2, 63.0, 63.3]

buckets_line2 = ["n<=10", "n<=15", "n<=20", "n<=25", "n<=30", "n<=50"]
values_line2 = [64.8, 68.7, 64.7, 66.9, 65.0, 66.7]

buckets_line3 = ["n<=20", "n<=25", "n<=30", "n<=50", "n<=60", "n>60"]
values_line3 = [66.2, 45.0, 50.9, 48.8, 43.3, 42.8]

buckets_line4 = ["n<=20", "n<=25", "n<=30", "n<=50", "n<=60", "n>60"]
values_line4 = [66.7, 52.6, 56.1, 56.6, 47.2, 48.9]

marker_size = 8
# Plot
plt.figure(figsize=(12, 10))
plt.plot(buckets_line1, values_line1, marker='o', markersize=marker_size, linestyle='-', label="MATRES (GPT-4o)")
plt.plot(buckets_line2, values_line2, marker='s', markersize=marker_size, linestyle='-', label="MATRES (DeepSeek-R1)")
plt.plot(buckets_line3, values_line3, marker='^', markersize=marker_size, linestyle='--', label="TimeBank-Dense (GPT-4o)")
plt.plot(buckets_line4, values_line4, marker='D', markersize=marker_size, linestyle='--', label="TimeBank-Dense (DeepSeek-R1)")

# Labels and title
# plt.xlabel("Threshold", fontsize=16)
plt.ylabel("F1 (%)", fontsize=20, fontweight='bold')
# plt.title("Number of Events in Documents", fontsize=18, fontweight='bold')
plt.legend(fontsize=18)
plt.xticks(rotation=15, fontsize=18)
# plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)

# Show plot
plt.show()
