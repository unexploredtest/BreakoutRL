import os
from datetime import datetime

import matplotlib.pyplot as plt


class LivePlot:
    def __init__(self):
        self.figure, self.ax = plt.subplots()
        self.ax.set_xlabel("Epoch x 10")
        self.ax.set_ylabel("Returns ")
        self.ax.set_title("Returns over Epochs")

        self.data = None
        self.eps_data = None

        self.epochs = 0

    def update_plot(self, stats):
        self.data = stats['AvgReturns']
        self.eps_data = stats['EpsilonCheckpoint']

        self.epochs = len(self.data)

        self.ax.clear()
        self.ax.set_xlim(0, self.epochs)

        self.ax.plot(self.data, "b-", label="Returns")
        self.ax.plot(self.eps_data, "r-", label="Epsilon")
        self.ax.legend(loc='upper left')

        # Ensure the plots directory exists
        if not os.path.exists("plots"):
            os.mkdir("plots")

        current_date = datetime.now().strftime("%Y-%m-%d")


        self.figure.savefig(f"plots/plot_{current_date}.png")
        
