import time

class StragglerInjector:
    def __init__(self, target_worker_id: int, delay_sec: float, start_iter: int = 0):
        self.target_worker_id = target_worker_id
        self.delay_sec = delay_sec
        self.start_iter = start_iter
        
    def inject(self, worker_id: int, iteration: int):
        if worker_id == self.target_worker_id and iteration >= self.start_iter:
            time.sleep(self.delay_sec)
