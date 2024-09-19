import torch

def train_one_epoch(model, dataloader, optimizer, loss_fns, scheduler, device):
    model.train()
    
    total_loss = []
    
    for batch_idx, ((x_i, x_j), _) in enumerate(dataloader, start=1):
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        
        loss_instance = loss_fns['instance'](z_i, z_j)
        loss_cluster = loss_fns['cluster'](c_i, c_j)
        
        loss = loss_instance + loss_cluster
        
        total_loss.append(loss.item())
        
        loss.backward()
        
        optimizer.step()
        
        print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, Loss: {sum(total_loss)/len(total_loss):.6f}, LR: {scheduler.get_last_lr()[0]:.8f}", end="")
    print()

    return sum(total_loss)/len(total_loss) # One Epoch Mean Loss

@torch.no_grad()
def evaluate():
    pass