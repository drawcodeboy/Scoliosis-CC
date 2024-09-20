import torch

def collate_fn(batch):
    longest_size = max([sample[0][0].shape[0] for sample in batch])
    
    paths = [sample[1] for sample in batch]
        
    x_i_batch, x_j_batch = [], []
        
    for sample in batch: 
        if longest_size != sample[0][0].shape[0]:
            zero_temp = torch.zeros(longest_size - sample[0][0].shape[0], 1)
        
            x_i_batch.append(torch.cat([zero_temp, sample[0][0]], dim=0))
            x_j_batch.append(torch.cat([zero_temp, sample[0][1]], dim=0))
        else:
            x_i_batch.append(sample[0][0])
            x_j_batch.append(sample[0][1])
    
    x_i_batch = torch.stack(x_i_batch, dim=0)
    x_j_batch = torch.stack(x_j_batch, dim=0)
    
    return (x_i_batch, x_j_batch), paths