import matplotlib.pyplot as plt

# Data
buckets_line1 = ["n<=10", "n<=15", "n<=20", "n<=25", "n<=30", "n<=50"]
values_line1 = [0.636693564, 0.609674099, 0.600135209, 0.557392836, 0.554644165, 0.561648057]

buckets_line2 = ["n<=20", "n<=25", "n<=30", "n<=50", "n<=60", "n>60"]
values_line2 = [0.566102412, 0.40395301, 0.464876875, 0.431357841, 0.386787784, 0.391036787]

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
