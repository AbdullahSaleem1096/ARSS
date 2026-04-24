import torch
import torch.nn as nn
import time
import multiprocessing as mp
from typing import Optional
from arss.monitor.signals import WorkerSignal
from arss.sync.backend import SyncBackend
from arss.workers.straggler import StragglerInjector

def worker_loop(
    worker_id: int,
    initial_params: dict,
    num_workers: int,
    max_iterations: int,
    signal_queue: mp.Queue,
    strategy_conn,
    ps_backend: SyncBackend,
    ring_backend: SyncBackend,
    hybrid_backend: SyncBackend,
    straggler_injector: Optional[StragglerInjector] = None,
    learning_rate: float = 0.01
):
    from arss.model.cnn import SimpleCNN
    from arss.model.loader import get_worker_dataloader
    
    model = SimpleCNN()
    model.load_state_dict(initial_params)
    
    dataloader = get_worker_dataloader(worker_id, num_workers)
    
    criterion = nn.CrossEntropyLoss()
    
    current_iteration = 0
    
    # We iterate over the dataloader until we hit max_iterations
    while current_iteration < max_iterations:
        for batch_idx, (data, target) in enumerate(dataloader):
            if current_iteration >= max_iterations:
                return
                
            iter_start_time = time.perf_counter()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Extract gradients
            gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.detach().clone()
                    
            # Inject straggler delay
            if straggler_injector:
                straggler_injector.inject(worker_id, current_iteration)
                
            iter_end_time = time.perf_counter()
            duration = iter_end_time - iter_start_time
            
            # Compute gradient density
            total_elements = sum(g.numel() for g in gradients.values())
            non_zero_elements = sum((g != 0).sum().item() for g in gradients.values())
            density = non_zero_elements / total_elements if total_elements > 0 else 0.0
            
            # Estimate data volume (bytes)
            data_volume = sum(g.numel() * g.element_size() for g in gradients.values())
            
            # Report signal to Monitor
            signal = WorkerSignal(
                worker_id=worker_id,
                iteration=current_iteration,
                timestamp=duration,
                gradient_density=density,
                outbound_data_volume=data_volume
            )
            signal_queue.put(signal)
            
            # Wait for Strategy Token
            strategy = strategy_conn.recv()
            
            # Synchronization
            if strategy == "PS":
                updated_params = ps_backend.sync(worker_id, gradients)
            elif strategy == "Ring":
                updated_params = ring_backend.sync(worker_id, gradients)
            elif strategy == "Hybrid":
                updated_params = hybrid_backend.sync(worker_id, gradients)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
            # Apply updated params
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in updated_params:
                        param.copy_(updated_params[name])
                        
            current_iteration += 1
