import torch
import multiprocessing as mp
from arss.model.cnn import SimpleCNN

from arss.monitor.monitor import ARSSMonitor
from arss.sync.ps_backend import PSBackend
from arss.sync.ring_backend import RingBackend
from arss.workers.worker import worker_loop
from arss.workers.straggler import StragglerInjector
from arss.metrics.plotter import MetricsPlotter

def run_experiment(
    exp_name: str,
    num_workers: int = 4,
    epochs: int = 1,
    max_iterations: int = 20,
    straggler: StragglerInjector = None,
    simulated_bu: float = 0.0,
    simulated_gd: float = None,
    simulation_schedule: dict = None,
    learning_rate: float = 0.01
):
    print(f"Starting {exp_name} with {num_workers} workers.")
    # Use spawn for Windows compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    model = SimpleCNN()
    initial_params = {name: param.detach().clone() for name, param in model.named_parameters()}
    
    monitor = ARSSMonitor(num_workers=num_workers, k=3)
    monitor.set_simulated_bu(simulated_bu)
    if simulated_gd is not None:
        monitor.set_simulated_gd(simulated_gd)
    if simulation_schedule is not None:
        monitor.set_simulation_schedule(simulation_schedule)
    
    ps_backend = PSBackend(initial_params, learning_rate)
    ring_backend = RingBackend(num_workers, initial_params, learning_rate)
    
    worker_processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker_loop,
            args=(
                i,
                initial_params,
                num_workers,
                max_iterations,
                monitor.get_signal_queue(),
                monitor.get_worker_conn(i),
                ps_backend,
                ring_backend,
                straggler,
                learning_rate
            )
        )
        worker_processes.append(p)
        p.start()
        
    try:
        monitor.run(max_iterations)
    except KeyboardInterrupt:
        monitor.stop()
        
    for p in worker_processes:
        p.join()
        
    metrics = monitor.metrics_collector.get_metrics()
    MetricsPlotter.plot_experiment_results(metrics, save_path=f"{exp_name}.png", title=exp_name)
    print(f"Finished {exp_name}. Results saved to {exp_name}.png")
