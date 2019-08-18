import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.selu(self.fc1(x), inplace=True)
        x = F.selu(self.fc2(x), inplace=True)
        return F.softmax(self.fc3(x), dim=1)