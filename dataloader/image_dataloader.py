from torch.utils.data import Dataset, DataLoader

import torch
from PIL import Image
import numpy as np
import cv2
import os

from .transform import ImageTransforms

class ImageDataset(Dataset):
    def __init__(self, 
                 data_dir:str,
                 transform=ImageTransforms,
                 mode:str='train'):
        
        self.data_dir = data_dir
        self.data_li = [] # {image_path}
        
        self.transform = transform(size=244) # Resize
        self.mode = mode
        
        self._check()
    
    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        image, height, width = self.get_image(self.data_li[idx])
        
        # Image To Tensor (여기 Transform에서 해결하는 거 고려)
        image /= 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) # for channel
        
        return self.transform(image, self.mode)
        
    def get_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        height, width = image.shape
        return image, height, width
    
    def _check(self):
        dirs = ['train', 'test', 'valid']
        
        file_cnt = 0
        
        for dir in dirs:
            images_path = rf"{self.data_dir}/{dir}/images"
            for filename in os.listdir(images_path):
                image_path = rf"{images_path}/{filename}"
                
                try:
                    image = Image.open(image_path)
                    self.data_li.append(image_path)
                    file_cnt += 1
                except:
                    print("Can\'t Open this file")
                                
                print(f"\rLoad Data {file_cnt:04d} samples", end="")
        print()
            
if __name__ == '__main__':
    ds = ImageDataset(
        data_dir = rf"data/AIS.v1i.yolov8"
    )
    
    input_t = ds[0]
    print(input_t.shape)