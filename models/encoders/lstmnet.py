import torch
from torch import nn
from typing import Tuple

from .layer_norm import LayerNorm2d

class LSTMNet(nn.Module):
    def __init__(self, input_size:int=1):
        super().__init__()
        
        self.input_size = input_size
        
        self.layer1 = self._make_layer(self.input_size, 256)
        self.layer2 = self._make_layer(256, 512)
        self.layer3 = self._make_layer(512, 1024)
        self.layer4 = self._make_layer(1024, 2048)
        
        # Layer Normalization Parameter
        self.gamma = nn.Parameter(torch.tensor([1.]))
        self.bias = nn.Parameter(torch.tensor([0.]))
        
    def _make_layer(self, input_size, hidden_size):
        layers = []
        
        layers.append(nn.LSTM(input_size, hidden_size, batch_first=True))
        layers.append(LayerNorm2d())
        layers.append(nn.ReLU())
        
        return nn.ModuleList(layers)
    
    def _forward_layer(self, layer, x):
        x, _ = layer[0](x)
        x = layer[1](x)
        x = layer[2](x)
        
        return x
    
    def forward(self, x):
        x = self._forward_layer(self.layer1, x)
        x = self._forward_layer(self.layer2, x)
        x = self._forward_layer(self.layer3, x)
        x = self._forward_layer(self.layer4, x)
        
        return x[:, -1, :]

if __name__ == '__main__':
    input_tensor = torch.randn(1, 100, 1)
    
    model = LSTMNet()
    
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
    
    p_sum = 0
    for p in model.parameters():
        p_sum += p.numel()
    print(p_sum)