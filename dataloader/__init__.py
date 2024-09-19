from .image_dataloader import ImageDataset
from .mask_dataloader import MaskDataset
from .seq_dataloader import SequenceDataset

def load_dataset(dataset:str="image",
                 data_dir:str='data/AIS.v1i.yolov8',
                 mode:str='train'):
    
    if dataset == "image":
        # Original Image
        return ImageDataset(data_dir=data_dir, 
                            mode=mode)
    
    if dataset == "mask":
        # Binary Mask
        return MaskDataset(data_dir=data_dir, 
                           mode=mode)
    
    if dataset == "seq":
        # Spine Sequence
        return SequenceDataset(data_dir=data_dir, 
                               mode=mode)