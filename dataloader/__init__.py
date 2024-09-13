from .scoliosis_dataloader import ScoliosisDataset

from typing import

def load_dataset(dataset:str="image",
                 data_dir:str='data/AIS.v1i.yolov8', 
                 mode):
    if dataset == "image":
        # Scoliosis Dataset
        return ScoliosisDataset(data_dir, mode, 'U-Net')