import torch.nn as nn
import torch
from torch.nn.functional import normalize

__all__ = ["ContrastiveNetwork"]

class ContrastiveNetwork(nn.Module):
    def __init__(self, encoder, feature_dim, class_num):
        super(Network, self).__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        
        self.instance_projector = nn.Sequential(
            nn.Linear(self.encoder.rep_dim, self.encoder.rep_dim),
            nn.ReLU(),
            nn.Linear(self.encoder.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.encoder.rep_dim, self.encoder.rep_dim),
            nn.ReLU(),
            nn.Linear(self.encoder.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        # L2 Normalization
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.encoder(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c