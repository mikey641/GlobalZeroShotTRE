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
        "All": [49.3, 48.4, 52.2, 55.6, 58.4],
        "Consecutive Sentences": [54.5, 49.8, 52.8, 56.1, 59.1],
        "Non-Consecutive Sentences": [46.7, 47.6, 51.9, 55.0, 58.2],
        # 4: [81.8, 80.1, 63.9, 46.9, 0, 0] # Bayesian
    }
    draw_plot(categories, scores, 45, 60)


def plot_eventfull():
    # OmniTemp
    categories = ["CoT", "ZSL-Global", "ZSL-TimeLine", "SelfConsistency", "GlobalConsistency"]
    scores = {
        "All": [67.2, 62.3, 68.5, 71.0, 73.6],
        "Consecutive Sentences": [67.7, 69.7, 71.1, 73.1, 78.4],
        "Non-Consecutive Sentences": [67.1, 60.0, 67.4, 70.4, 70.4],
        # 4: [81.8, 80.1, 63.9, 46.9, 0, 0] # Bayesian
    }
    draw_plot(categories, scores, 57.5, 80)


if __name__ == '__main__':
    plot_eventfull()
    plot_nt()
