import matplotlib.pyplot as plt
import numpy as np



def draw_plot(categories, scores, min_val, max_val):
    # Define bar width and x positions
    bar_width = 0.05  # Keep bar width fixed
    spacing_factor = 0.3  # Adjust this to control spacing between categories

    # Compute x positions based on spacing factor
    x = np.arange(len(categories))  # Original evenly spaced positions
    adjusted_x = x * spacing_factor  # Reduce category spacing

    # Colors for different bars
    colors = ['royalblue', 'darkorange', 'seagreen', 'crimson', 'purple']

    # Apply style
    plt.style.use('ggplot')

    # Create figure
    plt.figure(figsize=(14, 7))

    # Plot bars for each score group
    for i, (key, values) in enumerate(scores.items()):
        plt.bar(adjusted_x + i * bar_width, values, width=bar_width,
                color=colors[i], edgecolor='black', label=f"{key}")

    # Labels and title
    # plt.xlabel("Relation Type", fontsize=12)
    plt.ylabel("F1 (%)", fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16)
    # plt.title(title, fontsize=20, fontweight='bold')

    plt.ylim(min_val, max_val)

    # Ensure all categories are shown on x-axis
    plt.xticks(adjusted_x + (len(scores) / 2 - 0.5) * bar_width, categories, fontsize=16)

    # Show legend
    plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    # Show grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.show()


def plot_nt():
    # Define categories (x-axis) and accuracy scores (y-values)
    # NarrativeTime
    categories = ["CoT", "ZSL-Global", "ZSL-TimeLine", "SelfConsistency", "GlobalConsistency"]
    scores = {
        "All": [57.9, 57.0, 59.4, 62.1, 64.1],
        "Consecutive Sentences": [64.2, 58.4, 62.8, 64.5, 67.7],
        "Non-Consecutive Sentences": [54.7, 56.3, 57.7, 61.4, 62.2],
        # 4: [81.8, 80.1, 63.9, 46.9, 0, 0] # Bayesian
    }
    draw_plot(categories, scores, 45, 70)


def plot_eventfull():
    # OmniTemp
    categories = ["CoT", "ZSL-Global", "ZSL-TimeLine", "SelfConsistency", "GlobalConsistency"]
    scores = {
        "All": [78.4, 70.5, 74.6, 78.7, 79.2],
        "Consecutive Sentences": [78.6, 72.0, 77.2, 81.3, 79.9],
        "Non-Consecutive Sentences": [78.4, 68.8, 73.0, 78.1, 78.6],
        # 4: [81.8, 80.1, 63.9, 46.9, 0, 0] # Bayesian
    }
    draw_plot(categories, scores, 60, 85)


if __name__ == '__main__':
    plot_eventfull()
    plot_nt()
