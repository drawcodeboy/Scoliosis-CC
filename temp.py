from models import load_encoder
import torch

if __name__ == '__main__':
    model = load_encoder('ResNet')
    
    temp_tensor = torch.randn(3, 1, 224, 224)
    output = model(temp_tensor)
    
    print(output.shape)