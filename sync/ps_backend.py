import torch
import multiprocessing as mp
from typing import Dict
from arss.sync.backend import SyncBackend

class PSBackend(SyncBackend):
    def __init__(self, initial_params: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        self.manager = mp.Manager()
        self.params = self.manager.dict()
        self.lock = self.manager.Lock()
        self.lr = learning_rate
        
        # Initialize
        for name, tensor in initial_params.items():
            self.params[name] = tensor.cpu().clone()
            
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'manager' in state:
            del state['manager']
        return state
        
    def sync(self, worker_id: int, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        updated_params = {}
        with self.lock:
            # Async PS: Apply gradients immediately and return updated weights
            for name, grad in gradients.items():
                current_w = self.params[name]
                new_w = current_w - self.lr * grad.cpu()
                self.params[name] = new_w
                updated_params[name] = new_w.clone()
        return updated_params
        
    def flush(self) -> None:
        # For async PS, flush is a no-op as gradients are applied immediately under lock.
        pass
