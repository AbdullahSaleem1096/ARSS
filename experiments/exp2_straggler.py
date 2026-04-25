import argparse
from arss.experiments.runner import run_experiment
from arss.workers.straggler import StragglerInjector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run straggler experiment with dynamic values.")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay in seconds for the straggler.")
    parser.add_argument("--start-iter", type=int, default=5, help="Iteration to start injecting delay.")
    parser.add_argument("--target-worker", type=int, default=0, help="ID of the worker to slow down.")
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8, 16], help="List of number of workers to test.")
    parser.add_argument("--iterations", type=int, default=20, help="Maximum iterations per run.")
    
    args = parser.parse_args()
    
    # Straggler configuration
    straggler = StragglerInjector(target_worker_id=args.target_worker, delay_sec=args.delay, start_iter=args.start_iter)
    
    for n in args.workers:
        run_experiment(
            exp_name=f"exp2_straggler_n{n}_d{args.delay}",
            num_workers=n,
            max_iterations=args.iterations,
            straggler=straggler
        )
