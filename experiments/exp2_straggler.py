from arss.experiments.runner import run_experiment
from arss.workers.straggler import StragglerInjector

if __name__ == "__main__":
    # Straggler at worker 0 starting from iteration 5
    straggler = StragglerInjector(target_worker_id=0, delay_sec=1.5, start_iter=5)
    run_experiment(
        exp_name="exp2_straggler",
        num_workers=4,
        max_iterations=20,
        straggler=straggler
    )
