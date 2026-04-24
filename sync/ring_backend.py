import torch
import multiprocessing as mp
from typing import Dict
from arss.sync.backend import SyncBackend

class RingBackend(SyncBackend):
    def __init__(self, num_workers: int, initial_params: Dict[str, torch.Tensor], learning_rate: float = 0.01):
        self.num_workers = num_workers
        self.lr = learning_rate
        
        self.recv_conns = {}
        self.send_conns = {}
        
        for i in range(num_workers):
            recv_conn, send_conn = mp.Pipe(duplex=False)
            self.send_conns[i] = send_conn
            self.recv_conns[(i + 1) % num_workers] = recv_conn
            
        self.params = {k: v.cpu().clone() for k, v in initial_params.items()}

    def sync(self, worker_id: int, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.num_workers <= 1:
            avg_grads = gradients
        else:
            avg_grads = self._ring_allreduce(worker_id, gradients)
            
        updated_params = {}
        for name, grad in avg_grads.items():
            self.params[name] = self.params[name] - self.lr * grad
            updated_params[name] = self.params[name].clone()
            
        return updated_params
        
    def _ring_allreduce(self, worker_id: int, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # flatten gradients
        names = sorted(gradients.keys())
        shapes = {name: gradients[name].shape for name in names}
        flat_grads = torch.cat([gradients[name].cpu().flatten() for name in names])
        
        N = self.num_workers
        chunk_size = (flat_grads.numel() + N - 1) // N
        
        # pad to make it divisible by N
        pad_size = chunk_size * N - flat_grads.numel()
        if pad_size > 0:
            flat_grads = torch.nn.functional.pad(flat_grads, (0, pad_size))
            
        chunks = list(torch.split(flat_grads, chunk_size))
        
        send_conn = self.send_conns[worker_id]
        recv_conn = self.recv_conns[worker_id]
        
        # Scatter-Reduce
        for step in range(N - 1):
            send_chunk_idx = (worker_id - step) % N
            recv_chunk_idx = (worker_id - step - 1) % N
            
            # Send chunk to right neighbor
            send_conn.send(chunks[send_chunk_idx])
            # Receive chunk from left neighbor
            recv_chunk = recv_conn.recv()
            
            # Accumulate
            chunks[recv_chunk_idx] = chunks[recv_chunk_idx] + recv_chunk
            
        # All-Gather
        for step in range(N - 1):
            send_chunk_idx = (worker_id - step + 1) % N
            recv_chunk_idx = (worker_id - step) % N
            
            # Send chunk to right neighbor
            send_conn.send(chunks[send_chunk_idx])
            # Receive chunk from left neighbor
            recv_chunk = recv_conn.recv()
            
            # Update
            chunks[recv_chunk_idx] = recv_chunk
            
        # Average
        flat_grads = torch.cat(chunks) / N
        if pad_size > 0:
            flat_grads = flat_grads[:-pad_size]
            
        # Unflatten
        avg_grads = {}
        offset = 0
        for name in names:
            numel = gradients[name].numel()
            avg_grads[name] = flat_grads[offset:offset+numel].view(shapes[name])
            offset += numel
            
        return avg_grads

    def flush(self) -> None:
        pass
