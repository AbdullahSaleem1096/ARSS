from arss.experiments.runner import run_experiment

if __name__ == "__main__":
    # Simulate high bandwidth utilization
    run_experiment(
        exp_name="exp4_bandwidth",
        num_workers=4,
        max_iterations=10,
        simulated_bu=0.9 # Forces Ring
    )
