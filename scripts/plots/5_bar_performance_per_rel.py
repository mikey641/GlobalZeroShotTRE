import matplotlib.pyplot as plt
import numpy as np


def draw_plot(categories, scores):
    # Define bar width and spacing factor
    bar_width = 0.20  # Keep bar width fixed
    spacing_factor = 0.7  # Adjust this to control spacing between categories

    # Compute x positions based on spacing factor
    x = np.arange(len(categories))  # Original evenly spaced positions
    adjusted_x = x * spacing_factor  # Reduce category spacing

    # Colors for different bars
    colors = ['royalblue', 'darkorange', 'seagreen', 'crimson', 'purple']

    # Apply style
    plt.style.use('ggplot')

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot bars for each score group
    for i, (key, values) in enumerate(scores.items()):
        plt.bar(adjusted_x + i * bar_width, values, width=bar_width,
                color=colors[i], edgecolor='black', label=f"{key}")

    # Labels and title
    plt.ylabel("F1 (%)", fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16)
    # plt.title("Global-Timeline Performance Per Relation", fontsize=18, fontweight='bold')

    # Adjust x-ticks to match the reduced category spacing
    plt.xticks(adjusted_x + (len(scores) / 2 - 0.5) * bar_width, categories, fontsize=16)

    # Show legend
    plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    # Show grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.show()


def plot_matres_omni():
    categories = ["B", "A", "E", "V"]
    scores = {
        "OmniTemp": [69.7, 73.5, 49.8, 10.5],  # Global
        "MATRES": [69.7, 50.2, 8.7, 1.7],  # TimeLine
        # 4: [74.2, 77.4, 62.8, 0.0],  # TimeLine + Const
        # 4: [81.8, 80.1, 63.9, 46.9] # Bayesian
    }
    draw_plot(categories, scores)


def plot_nt_tbd():
    # Define categories (x-axis) and accuracy scores (y-values)
    _categories = ["B", "A", "I", "II", "E", "V"]
    _scores = {
        "NarrativeTime": [62.4, 63.3, 8.81, 12.6, 31.4, 22.2],  # Global
        "TimeBank-Dense": [47.4, 34.0, 13.3, 12.5, 18.2, 1.6],
        # 4: [74.2, 77.4, 62.8, 0.0],  # TimeLine + Const
        # 4: [81.8, 80.1, 63.9, 46.9] # Bayesian
    }

    draw_plot(_categories, _scores)

if __name__ == "__main__":
    plot_nt_tbd()
    plot_matres_omni()
