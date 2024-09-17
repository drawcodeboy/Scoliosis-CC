from torch.utils.data import Dataset, DataLoader

import torch
from PIL import Image
import numpy as np
import cv2
import os

class MaskDataset(Dataset):
    def __init__(self, data_dir,
                 width:int=640,
                 height:int=640):
        self.data_dir = data_dir
        self.data_li = [] # {image_path}
        self.width, self.height = width, height
        self._check()
    
    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        mask = self.get_mask(self.data_li[idx])
        
        # mask To Tensor
        mask /= 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) # for channel
        
        return mask
    
    def get_mask(self, label_path):
        with open(label_path) as f:
            pts = f.read().split()
        
        polygon = []
    
        for x, y in zip(pts[1::2], pts[2::2]):
            x, y = float(x) * self.width, float(y) * self.height
            polygon.append([int(x), int(y)])
        
        polygon = np.array(polygon)
        
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        mask_value = 255
        cv2.fillPoly(mask, [polygon], mask_value)
        
        return mask
    
    def _check(self):
        dirs = ['train', 'test', 'valid']
        
        file_cnt = 0
        
        for dir in dirs:
            labels_path = rf"{self.data_dir}/{dir}/labels"
            for filename in os.listdir(labels_path):
                label_path = rf"{labels_path}/{filename}"
                
                try:
                    with open(label_path) as f:
                        pts = f.read().split()
                    self.data_li.append(label_path)
                    file_cnt += 1
                except:
                    print("Can\'t Open this file")
                                
                print(f"\rLoad Data {file_cnt:04d} samples", end="")
        print()
            
if __name__ == '__main__':
    ds = MaskDataset(
        data_dir = rf"data/AIS.v1i.yolov8"
    )
    
    input_t = ds[0]
    print(input_t.shape)