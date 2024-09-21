import matplotlib.pyplot as plt
import numpy as np
import argparse

def loss_plot(data:str='seq',
              start_cluster:int=4):
    
    seq_losses = []

    for i in range(start_cluster, 10 + 1, 2):
        seq_losses.append((i, np.load(rf"saved\losses\{data}_data_{i:02d}_clusters_loss.npy")))

    x = np.arange(0, len(seq_losses[0][1]), 1)

    for clusters, seq_loss in seq_losses:
        plt.plot(x, seq_loss, label=f"{clusters:02d} clusters")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data", type=str, default='seq')
    parser.add_argument("--start-cluster", type=int, default=4)
    
    args = parser.parse_args()
    
    loss_plot(args.data, args.start_cluster)