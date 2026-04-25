import argparse
from arss.experiments.runner import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bandwidth experiment with dynamic values.")
    parser.add_argument("--bu-values", type=float, nargs="+", default=[0.1, 0.5, 0.9], help="List of bandwidth utilization values to simulate.")
    parser.add_argument("--workers", type=int, nargs="+", default=[4], help="List of number of workers to test.")
    parser.add_argument("--iterations", type=int, default=10, help="Maximum iterations per run.")
    
    args = parser.parse_args()
    
    for bu in args.bu_values:
        for n in args.workers:
            run_experiment(
                exp_name=f"exp4_bandwidth_bu{bu}_n{n}",
                num_workers=n,
                max_iterations=args.iterations,
                simulated_bu=bu
            )
