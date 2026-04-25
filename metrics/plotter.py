import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List
from arss.monitor.signals import IterationMetrics

STRATEGY_COLORS = {
    "Ring": "#2196F3",  # Blue
    "PS":   "#FF9800",  # Orange
}

class MetricsPlotter:
    @staticmethod
    def plot_experiment_results(metrics: List[IterationMetrics], save_path: str = "result.png", title: str = "Experiment Results"):
        if not metrics:
            print("No metrics to plot.")
            return

        metrics = sorted(metrics, key=lambda m: m.iteration)
        iterations = [m.iteration for m in metrics]
        max_times = [m.max_time for m in metrics]
        mean_times = [m.mean_time for m in metrics]
        strategies = [m.strategy_used for m in metrics]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

        # --- Top plot: worker timing ---
        ax1.plot(iterations, max_times, label="Max Worker Time", color='red', linewidth=2, zorder=3)
        ax1.plot(iterations, mean_times, label="Mean Worker Time", color='blue', linewidth=2, zorder=3)

        y_min = 0
        y_max = max(max_times) * 1.15 if max_times else 1.0

        # Draw colored background spans per strategy region
        seen_strategies = set()
        i = 0
        while i < len(iterations):
            current_strat = strategies[i]
            j = i + 1
            while j < len(iterations) and strategies[j] == current_strat:
                j += 1
            x_start = iterations[i] - 0.5
            x_end = (iterations[j - 1] + 0.5)
            color = STRATEGY_COLORS.get(current_strat, "#AAAAAA")
            ax1.axvspan(x_start, x_end, alpha=0.15, color=color, zorder=1)
            # Label the region in the middle
            mid_x = (x_start + x_end) / 2
            ax1.text(mid_x, y_max * 0.97, current_strat, ha='center', va='top',
                     fontsize=9, fontweight='bold', color=color)
            seen_strategies.add(current_strat)
            # Draw a vertical switch line at the boundary
            if i > 0:
                ax1.axvline(x=iterations[i], color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
            i = j

        ax1.set_ylabel("Time (seconds)")
        ax1.set_title(title)
        ax1.legend(loc='upper left')
        ax1.set_ylim(y_min, y_max)
        ax1.grid(True, alpha=0.4)

        # Legend patches for strategies
        patches = [mpatches.Patch(color=STRATEGY_COLORS.get(s, "#AAAAAA"), alpha=0.4, label=s)
                   for s in sorted(seen_strategies)]
        ax1.legend(handles=[
            plt.Line2D([0], [0], color='red', linewidth=2, label='Max Worker Time'),
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Mean Worker Time'),
        ] + patches, loc='upper left')

        # --- Bottom plot: strategy as a step function ---
        strategy_indices = {s: idx for idx, s in enumerate(sorted(set(strategies)))}
        strategy_values = [strategy_indices[s] for s in strategies]
        y_labels = sorted(set(strategies))

        ax2.step(iterations, strategy_values, where='post', color='purple', linewidth=2)
        ax2.fill_between(iterations, strategy_values, step='post', alpha=0.2, color='purple')
        ax2.set_yticks(range(len(y_labels)))
        ax2.set_yticklabels(y_labels)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Strategy")
        ax2.grid(True, alpha=0.4)

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"[Plotter] Saved plot to {save_path}")
