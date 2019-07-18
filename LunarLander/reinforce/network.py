import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.selu(self.fc1(x), inplace=True)
        x = F.selu(self.fc2(x), inplace=True)
        x = F.selu(self.fc3(x), inplace=True)
        return F.softmax(self.fc4(x), dim=1)


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.selu(self.fc1(x), inplace=True)
        x = F.selu(self.fc2(x), inplace=True)
        x = F.selu(self.fc3(x), inplace=True)
        return self.fc4(x)