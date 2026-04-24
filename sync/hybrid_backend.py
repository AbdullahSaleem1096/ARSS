import torch
from typing import Dict
from arss.sync.backend import SyncBackend

class HybridBackend(SyncBackend):
    def __init__(self, ps_backend: SyncBackend, ring_backend: SyncBackend):
        self.ps_backend = ps_backend
        self.ring_backend = ring_backend
        
    def sync(self, worker_id: int, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ps_grads = {}
        ring_grads = {}
        
        # We classify based on density to match the design doc.
        # In a real system, the Monitor would broadcast the routing map to ensure 
        # all workers agree on the routing to avoid Ring AllReduce deadlocks.
        # Here we assume artificial sparsity makes them agree.
        for name, grad in gradients.items():
            density = (grad != 0).float().mean().item()
            if density < 0.5:
                ps_grads[name] = grad
            else:
                ring_grads[name] = grad
                
        updated_params = {}
        if ps_grads:
            updated_ps = self.ps_backend.sync(worker_id, ps_grads)
            updated_params.update(updated_ps)
            
        if ring_grads:
            updated_ring = self.ring_backend.sync(worker_id, ring_grads)
            updated_params.update(updated_ring)
            
        return updated_params

    def flush(self) -> None:
        self.ps_backend.flush()
        self.ring_backend.flush()
