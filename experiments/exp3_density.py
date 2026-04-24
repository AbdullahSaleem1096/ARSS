from arss.experiments.runner import run_experiment

if __name__ == "__main__":
    run_experiment(
        exp_name="exp3_density",
        num_workers=4,
        max_iterations=10
    )
