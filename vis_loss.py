import matplotlib.pyplot as plt
import numpy as np
import argparse

def loss_plot(data:str='image'):
    
    losses = []

    for i in [3, 4, 5, 6, 7, 8, 9, 10]:
        losses.append((i, np.load(rf"saved\losses\{data}_data_{i:02d}_clusters_loss.npy")))

    x = np.arange(0, len(losses[0][1]), 1)

    for clusters, loss in losses:
        plt.plot(x, loss, label=f"{clusters:02d} clusters")
        
    data_title = {
        'seq': 'Sequence',
        'mask': 'Mask',
        'image': 'Image'
    }
    
    plt.title(f"{data_title[data]} Clustering Loss")
    plt.legend()
    # plt.show()
    plt.savefig(f'result/_losses/{data}_expr_losses.jpg', dpi=500)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data", type=str, default='seq')
    
    args = parser.parse_args()
    
    loss_plot(args.data)