from torch.utils.data import Dataset, DataLoader

import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from .transform import SequenceTransforms

class SequenceDataset(Dataset):
    def __init__(self, 
                 data_dir:str,
                 mode:str='train',
                 transform=SequenceTransforms):
        self.data_dir = data_dir
        self.data_li = [] # {image_path}
        
        self.mode = mode
        self.transform = transform()
        
        self._check()
    
    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        seq = self.get_seq(self.data_li[idx])
        
        seq = torch.tensor(seq, dtype=torch.float32).view(len(seq), 1)
        
        seq = (seq-torch.mean(seq))/(torch.std(seq))
        
        return self.transform(seq, self.mode), self.data_li[idx]
    
    def get_seq(self, seq_path):
        sequence = []
        with open(seq_path) as f:
            for step in f:
                sequence.append(float(step.split()[0])) # only X coordinate
        
        return sequence
    
    def _check(self):
        dirs = ['train', 'test', 'valid']
        
        file_cnt = 0
        
        for dir in dirs:
            sequences_path = rf"{self.data_dir}/{dir}/sequences"
            for filename in os.listdir(sequences_path):
                sequence_path = rf"{sequences_path}/{filename}"
                
                try:
                    with open(sequence_path) as f:
                        temp = f.read()
                    self.data_li.append(sequence_path)
                    file_cnt += 1
                except:
                    print("Can\'t Open this file")
                                
                print(f"\rLoad Data {file_cnt:04d} samples", end="")
        print()
            
if __name__ == '__main__':
    ds = SequenceDataset(
        data_dir = rf"data/AIS.v1i.yolov8"
    )
    
    input_t = ds[0]
    print(input_t)
    
    y = input_t.numpy()
    x = np.array([i for i in range(0, len(input_t))])
    
    plt.plot(x, y)
    plt.show()