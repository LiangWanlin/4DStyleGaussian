import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorMLP(nn.Module):
    def __init__(self, in_dim=32, out_dim=3):
        super(ColorMLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, out_dim)
        )


    def forward(self, x):
        rgb = self.mlp(x)
        rgb = torch.sigmoid(rgb)
        return rgb


    
