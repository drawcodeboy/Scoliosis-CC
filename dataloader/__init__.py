from .image_dataloader import ImageDataset
from .mask_dataloader import MaskDataset
from .seq_dataloader import SeqDataset

from typing import

def load_dataset(dataset:str="image",
                 data_dir:str='data/AIS.v1i.yolov8'):
    
    if dataset == "image":
        # Original Image
        return ImageDataset(data_dir)
    
    if dataset == "mask":
        # Binary Mask
        return MaskDataset(data_dir)
    
    if dataset == "seq":
        # Spine Sequence
        return SeqDataset(data_dir)