import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(inplace=True)
        )
        self.v = nn.Linear(64, 1)

        self.a = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pre(x)
        v = self.v(x)
        a = self.a(x)
        return v + (a - a.mean(dim=1, keepdim=True))