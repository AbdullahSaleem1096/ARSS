import matplotlib.pyplot as plt
from typing import List
from arss.monitor.signals import IterationMetrics

class MetricsPlotter:
    @staticmethod
    def plot_experiment_results(metrics: List[IterationMetrics], save_path: str = "result.png", title: str = "Experiment Results"):
        if not metrics:
            print("No metrics to plot.")
            return
            
        iterations = [m.iteration for m in metrics]
        max_times = [m.max_time for m in metrics]
        mean_times = [m.mean_time for m in metrics]
        strategies = [m.strategy_used for m in metrics]
        
        plt.figure(figsize=(10, 6))
        
        # Plot max time and mean time
        plt.plot(iterations, max_times, label="Max Worker Time", color='red')
        plt.plot(iterations, mean_times, label="Mean Worker Time", color='blue')
        
        # Highlight strategy switches
        for i in range(1, len(strategies)):
            if strategies[i] != strategies[i-1]:
                plt.axvline(x=iterations[i], color='gray', linestyle='--', alpha=0.5)
                plt.text(iterations[i], max(max_times), strategies[i], rotation=90, verticalalignment='top')
                
        if len(iterations) > 0:
            plt.text(iterations[0], max(max_times), strategies[0], rotation=90, verticalalignment='top')
            
        plt.xlabel('Iteration')
        plt.ylabel('Time (seconds)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
