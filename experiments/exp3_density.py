import argparse
from arss.experiments.runner import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run density experiment with dynamic values.")
    parser.add_argument("--densities", type=float, nargs="+", default=[0.1, 0.5, 0.9], help="List of gradient densities to simulate.")
    parser.add_argument("--workers", type=int, nargs="+", default=[4], help="List of number of workers to test.")
    parser.add_argument("--iterations", type=int, default=10, help="Maximum iterations per run.")
    
    args = parser.parse_args()
    
    for gd in args.densities:
        for n in args.workers:
            run_experiment(
                exp_name=f"exp3_density_gd{gd}_n{n}",
                num_workers=n,
                max_iterations=args.iterations,
                simulated_gd=gd
            )
