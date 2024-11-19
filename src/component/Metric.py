import matplotlib.pyplot as plt

class Metric:
    def __init__(self):
        self.metrics={}
        self.current_metrics={}

    def update(self, batch_metrics, preds=None, targets=None):
        for key,value in batch_metrics.items():
            if key not in self.current_metrics.keys():
                self.current_metrics[key]=[]

            self.current_metrics[key].append(value)

    def finalize(self):
        for key, value in self.current_metrics.items():
            if key not in self.metrics.keys():
                self.metrics[key]=[]
            self.metrics[key].append(sum(value)/len(value))

        self.current_metrics={}

    def plot(self):
        plt.figure(figsize=(10, 5))
        for key, val in self.metrics.items():
            plt.plot(val, label=key)

        plt.title("Metrics per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()