import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class WorkerSignal:
    worker_id: int
    iteration: int
    timestamp: float
    gradient_density: float
    outbound_data_volume: float

@dataclass
class IterationMetrics:
    iteration: int
    timestamps: Dict[int, float] = field(default_factory=dict)
    gradient_densities: Dict[int, float] = field(default_factory=dict)
    data_volumes: Dict[int, float] = field(default_factory=dict)
    strategy_used: str = "PS"
    
    @property
    def max_time(self) -> float:
        if not self.timestamps:
            return 0.0
        return max(self.timestamps.values())
        
    @property
    def mean_time(self) -> float:
        if not self.timestamps:
            return 0.0
        return sum(self.timestamps.values()) / len(self.timestamps)
        
    @property
    def straggler_ratio(self) -> float:
        if self.mean_time == 0:
            return 1.0
        return self.max_time / self.mean_time

    @property
    def avg_gradient_density(self) -> float:
        if not self.gradient_densities:
            return 0.0
        return sum(self.gradient_densities.values()) / len(self.gradient_densities)
        
    @property
    def total_data_volume(self) -> float:
        return sum(self.data_volumes.values())
