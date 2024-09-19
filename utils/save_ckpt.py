import torch
import os
import numpy as np

def save_model_ckpt(model, dataset, n_clusters, epochs, save_weights_dir):
    ckpt = {}
    ckpt['model'] = model.state_dict()
    ckpt['epochs'] = epochs
    
    save_name = f"{dataset}_data_{n_clusters:02d}_clusters_{epochs:03d}_epochs.pth"

    try:
        torch.save(ckpt, os.path.join(save_weights_dir, save_name))
        print(f"Save Model @epoch: {epochs}")
    except:
        print(f"Can\'t Save Model @epoch: {epochs}")
        
def save_loss_ckpt(train_loss, dataset, n_clusters, save_losses_dir ):
    save_name = f"{dataset}_data_{n_clusters:02d}_clusters_loss.npy"
    
    try:
        np.save(os.path.join(save_losses_dir, save_name), np.array(train_loss))
        print('Save Train Loss')
    except:
        print('Can\'t Save Train Loss') 