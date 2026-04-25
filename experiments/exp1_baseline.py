from arss.experiments.runner import run_experiment

if __name__ == "__main__":
    for n in [2, 4, 8, 16]:
        run_experiment(
            exp_name=f"exp1_n{n}",
            num_workers=n,
            max_iterations=10
        )
