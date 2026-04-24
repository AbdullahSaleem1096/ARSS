from arss.experiments.runner import run_experiment
from arss.workers.straggler import StragglerInjector

if __name__ == "__main__":
    # Straggler at worker 1 starting from iteration 8
    straggler = StragglerInjector(target_worker_id=1, delay_sec=1.0, start_iter=8)
    run_experiment(
        exp_name="exp5_e2e",
        num_workers=4,
        max_iterations=30,
        straggler=straggler,
        simulated_bu=0.0
    )
