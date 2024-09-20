import torch
import numpy as np

import sys

def train_one_epoch(model, dataloader, optimizer, loss_fns, scheduler, device):
    model.train()
    
    total_loss = []
    total_instance_loss = []
    total_cluster_loss = []
    
    for batch_idx, ((x_i, x_j), _) in enumerate(dataloader, start=1):
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        
        loss_instance = loss_fns['instance'](z_i, z_j)
        loss_cluster = loss_fns['cluster'](c_i, c_j)
        
        loss = loss_instance + loss_cluster
        
        total_loss.append(loss.item())
        total_instance_loss.append(loss_instance.item())
        total_cluster_loss.append(loss_cluster.item())
        
        loss.backward()
        
        optimizer.step()
        
        print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, Instance Loss: {sum(total_instance_loss)/len(total_instance_loss):.6f}, Cluster Loss: {sum(total_cluster_loss)/len(total_cluster_loss):.6f}, Loss: {sum(total_loss)/len(total_loss):.6f}, LR: {scheduler.get_last_lr()[0]:.8f}", end="")
    print()
    scheduler.step(sum(total_loss)/len(total_loss))

    return sum(total_loss)/len(total_loss) # One Epoch Mean Loss

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    
    cluster_assignments = []
    instance_vectors = []
    data_paths = []
    
    for batch_idx, (x, data_path) in enumerate(dataloader, start=1):
        x = x.to(device)
        
        c, z = model.forward_evaluate(x)
        
        cluster_assignments.extend(c.detach().cpu().tolist())
        instance_vectors.extend(z.detach().cpu().tolist())
        data_paths.extend(data_path)
        
        if batch_idx == 10: break
        
        print(f"\rEvaluate: {100*batch_idx/len(dataloader):.2f}%", end="")
    print()
    
    cluster_assignments = np.array(cluster_assignments)
    instance_vectors = np.array(instance_vectors)
    
    return cluster_assignments, instance_vectors, data_paths