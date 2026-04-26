from arss.experiments.runner import run_experiment
from arss.workers.straggler import StragglerInjector

if __name__ == "__main__":
    # Straggler at worker 0 starting from iteration 6
    straggler = StragglerInjector(target_worker_id=0, delay_sec=2.0, start_iter=6)
    
    # Schedule:
    # - Start (Iter 0): Low BU (0.2), Low GD (0.1) -> Ring (Default)
    # - Iter 6: Straggler joins -> (SR > 1.5, GD < 0.5, BU < 0.8) -> Switch to PS
    # - Iter 16: Bandwidth Spike (BU 0.9) -> (BU >= 0.8) -> Switch back to Ring
    
    schedule = {
        0: {'bu': 0.2, 'gd': 0.1},
        6: {'bu': 0.2, 'gd': 0.7},
        16: {'bu': 0.3, 'gd': 0.1}
    }
    
    run_experiment(
        exp_name="exp5_real_world_e2e",
        num_workers=4,
        max_iterations=30,
        straggler=straggler,
        simulation_schedule=schedule
    )
