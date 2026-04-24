class MetricsCollector:
    def __init__(self):
        self.metrics = []
        
    def record(self, metric):
        self.metrics.append(metric)
        
    def get_metrics(self):
        return self.metrics
