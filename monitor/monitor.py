import multiprocessing as mp
import time
import queue
from typing import Dict, List
from arss.monitor.signals import WorkerSignal, IterationMetrics
from arss.monitor.decision import DecisionEngine
from arss.metrics.collector import MetricsCollector

class ARSSMonitor:
    def __init__(self, num_workers: int, k: int = 3):
        self.num_workers = num_workers
        
        self.signal_queue = mp.Queue()
        self.strategy_pipes = [] 
        self.worker_strategy_conns = []
        
        for i in range(num_workers):
            mon_conn, work_conn = mp.Pipe()
            self.strategy_pipes.append(mon_conn)
            self.worker_strategy_conns.append(work_conn)
            
        self.decision_engine = DecisionEngine(k=k)
        self.metrics_collector = MetricsCollector()
        self.is_running = mp.Value('b', True)
        
    def get_worker_conn(self, worker_id: int):
        return self.worker_strategy_conns[worker_id]
        
    def get_signal_queue(self):
        return self.signal_queue
        
    def set_simulated_bu(self, bu: float):
        self.decision_engine.simulated_bu = bu

    def set_simulated_gd(self, gd: float):
        self.decision_engine.simulated_gd = gd

    def run(self, max_iterations: int):
        current_strategy = self.decision_engine.current_strategy
        
        metrics_buffer: Dict[int, IterationMetrics] = {}
        strategy_log: Dict[int, str] = {}
        
        while self.is_running.value:
            try:
                # Wait for a signal from any worker
                signal: WorkerSignal = self.signal_queue.get(timeout=1.0)
                
                iter_idx = signal.iteration
                if iter_idx not in metrics_buffer:
                    metrics_buffer[iter_idx] = IterationMetrics(iteration=iter_idx)
                    
                metrics = metrics_buffer[iter_idx]
                metrics.timestamps[signal.worker_id] = signal.timestamp
                metrics.gradient_densities[signal.worker_id] = signal.gradient_density
                metrics.data_volumes[signal.worker_id] = signal.outbound_data_volume
                
                # Assign the strategy for this iteration if not already assigned
                if iter_idx not in strategy_log:
                    strategy_log[iter_idx] = current_strategy
                    
                strategy_to_send = strategy_log[iter_idx]
                
                # Immediately reply to the worker with the strategy to use
                self.strategy_pipes[signal.worker_id].send(strategy_to_send)
                
                # If we have all signals for this iteration, run decision engine
                if len(metrics.timestamps) == self.num_workers:
                    new_strategy = self.decision_engine.decide(metrics)
                    
                    if new_strategy != current_strategy:
                        # Flush barrier is automatically handled: all workers will eventually sync
                        # on the same iteration when the strategy switches to a blocking one (Ring)
                        current_strategy = new_strategy
                        
                    metrics.strategy_used = strategy_to_send
                    self.metrics_collector.record(metrics)
                    
                    # Clean up old metrics
                    if iter_idx - 5 in metrics_buffer:
                        del metrics_buffer[iter_idx - 5]
                    if iter_idx - 5 in strategy_log:
                        del strategy_log[iter_idx - 5]
                
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Monitor error: {e}")
                
            # Exit condition
            if len(metrics_buffer) > 0:
                completed = [m for k, m in metrics_buffer.items() if len(m.timestamps) == self.num_workers]
                if completed and max(m.iteration for m in completed) >= max_iterations - 1:
                    break

    def stop(self):
        self.is_running.value = False
