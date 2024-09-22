import torch
import torchvision
import cv2
import numpy as np

import random
import copy

__all__ = ["ImageTransforms", "MaskTransforms", "SequenceTransforms"]

class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample

class GaussianNoise:
    def __init__(self, keep_prob=0.8):
        self.keep_prob = 1-keep_prob
    
    def __call__(self, sample):
        C, H, W = sample.size()
        
        noise = torch.empty((C, H, W), 
                            dtype=sample.dtype, 
                            device=sample.device)
        noise.bernoulli_(self.keep_prob)
        
        noise = 1 - noise
        return sample * noise
    
class MaskedSignal:
    def __init__(self, masked_range=10):
        self.masked_range = masked_range
    
    def __call__(self, sample):
        seq_len, dim = sample.size()
        start_step = self.masked_range-1
        end_step = seq_len-1
        
        random_step = random.randint(start_step, end_step)
        
        sample[(random_step-self.masked_range):random_step,:] = 0.
        
        return sample

class ImageTransforms:
    def __init__(self, size, s=1.0, blur=False):
        # Train Transform
        self.train_transform = [
            torchvision.transforms.Resize(size=(size, size)),
            # torchvision.transforms.RandomResizedCrop(size=size),
            # torchvision.transforms.RandomHorizontalFlip(),
            GaussianNoise(),
            # torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
            #                                    p=0.8),
            # torchvision.transforms.RandomGrayscale(p=0.2),
        ]
        if blur:
            self.train_transform.append(GaussianBlur(kernel_size=23))
        
        # Test Transform
        self.test_transform = [
            torchvision.transforms.Resize(size=(size, size)),
        ]

        self.train_transform = torchvision.transforms.Compose(self.train_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x, mode):
        if mode == 'train':
            return self.train_transform(x), self.train_transform(x)
        elif mode == 'test':
            return self.test_transform(x)

class MaskTransforms:
    def __init__(self, size):
        
        self.train_transform = [
            torchvision.transforms.Resize(size=(size, size)),
            GaussianNoise(),
            torchvision.transforms.RandomAffine(degrees=(-10, 10))
        ]
        
        self.test_transform = [
            torchvision.transforms.Resize(size=(size, size)),
        ]
        
        self.train_transform = torchvision.transforms.Compose(self.train_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)
        
    def __call__(self, x, mode):
        if mode == 'train':
            return self.train_transform(x), self.train_transform(x)
        elif mode == 'test':
            return self.test_transform(x)
        
class SequenceTransforms:
    def __init__(self):
        self.train_transform = [
            MaskedSignal(),
        ]
        
        self.test_transform = [
            
        ]
        
        self.train_transform = torchvision.transforms.Compose(self.train_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)
        
    def __call__(self, x, mode):
        if mode == 'train':
            x_cp = copy.deepcopy(x)
            return self.train_transform(x), self.train_transform(x_cp)
        elif mode == 'test':
            return self.test_transform(x)