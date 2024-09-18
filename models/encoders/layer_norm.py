import torch
from torch import nn

class LayerNorm2d(nn.Module):
    def __init__(self, eps: float=0.00001, dim:int = -3):
        super().__init__()
        
        self.gamma = nn.Parameter(torch.tensor([1.]))
        self.bias = nn.Parameter(torch.tensor([0.]))
        self.eps = eps
        self.dim = dim
    
    def forward(self, x):
        mean = torch.mean(x, dim=self.dim, keepdim=True)
        var = torch.square(x - mean).mean(dim=self.dim, keepdim=True)
            
        return ((x - mean) / torch.sqrt(var + self.eps)) * self.gamma + self.bias