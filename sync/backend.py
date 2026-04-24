from abc import ABC, abstractmethod
import torch

class SyncBackend(ABC):
    @abstractmethod
    def sync(self, worker_id: int, gradients: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Input:  per-worker gradient tensors, keyed by param name
        Output: globally averaged parameter tensors
        Contract: must complete before next iteration begins
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """
        Flush all in-flight updates.
        Called by Monitor before any strategy switch.
        """
        pass
