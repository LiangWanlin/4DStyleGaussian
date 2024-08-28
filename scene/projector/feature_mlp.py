import torch
import torch.nn as nn

class FeatureMLP(nn.Module):
    def __init__(self, in_dim=32, out_dim=32):
        super(FeatureMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
            )
    def forward(self, x):
        feature = self.mlp(x)
        feature = torch.sigmoid(feature)
        return feature      
    