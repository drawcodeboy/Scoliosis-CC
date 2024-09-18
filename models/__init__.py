from .contrastive.contrastive_loss import *
from .contrastive.contrastive_network import *

from .encoders.lstmnet import LSTMNet
from .encoders.resnet import ResNet
from torchvision.models.resnet import Bottleneck

def load_encoder(encoder:str="ResNet"):
    if encoder == 'ResNet':
        # ResNet50
        return ResNet(block=Bottleneck, layers=[3, 4, 6, 3])

    elif encoder == 'LSTMNet':
        return LSTMNet()