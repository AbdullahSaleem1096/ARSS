from typing import List
from arss.monitor.signals import IterationMetrics

class DecisionEngine:
    def __init__(self, k: int = 3):
        self.k = k
        self.current_strategy = "Ring" # Default homogeneous baseline
        self.history: List[str] = []
        # External override for simulated Bandwidth Utilization
        self.simulated_bu = 0.0
        # External override for simulated Gradient Density
        self.simulated_gd = None
        
    def decide(self, metrics: IterationMetrics) -> str:
        sr = metrics.straggler_ratio
        gd = self.simulated_gd if self.simulated_gd is not None else metrics.avg_gradient_density
        bu = self.simulated_bu
        
        target = "Ring"
        
        # Decision Logic from Document:
        # - SR > 1.5 -> PS
        # - BU > 0.8 -> Ring
        # - Mixed density (e.g. GD ~ 0.5) -> Hybrid
        
        if sr > 1.5:
            target = "PS"
            
        # BU saturation overrides PS if server is completely choked
        if bu > 0.8:
            target = "Ring"
        elif sr <= 1.5 and 0.3 < gd < 0.7:
            target = "Hybrid"
            
        self.history.append(target)
        
        # Apply hysteresis (k=3 consecutive)
        if len(self.history) >= self.k:
            recent = self.history[-self.k:]
            if all(t == target for t in recent):
                self.current_strategy = target
                
        return self.current_strategy
