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
        
        # New Logic:
        # 1. Default to Ring.
        # 2. Only switch to PS if:
        #    - Straggler detected (SR > 1.5)
        #    - AND Density is low (GD < 0.5)
        #    - AND Bandwidth is not high (BU < 0.8)
        
        vote = "Ring"
        
        if sr > 1.5 and gd < 0.5 and bu < 0.8:
            vote = "PS"
        else:
            vote = "Ring"
            
        self.history.append(vote)
        
        # Apply hysteresis (k=3 consecutive)
        if len(self.history) >= self.k:
            recent = self.history[-self.k:]
            if all(t == vote for t in recent):
                if self.current_strategy != vote:
                    print(f"[Decision] Iter {metrics.iteration}: *** SWITCHING {self.current_strategy} -> {vote} (after {self.k} consecutive votes) ***")
                    self.current_strategy = vote
        
        print(f"[Decision] Iter {metrics.iteration}: SR={sr:.2f}, GD={gd:.2f}, BU={bu:.2f} | Vote: {vote} | Active: {self.current_strategy}")
                
        return self.current_strategy
