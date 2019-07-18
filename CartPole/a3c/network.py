import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.bn1 = nn.BatchNorm1d()
        self.fc1 = nn.Linear(4, 16)
        self.bn2 = nn.BatchNorm1d()
        self.fc2 = nn.Linear(16, 16)
        self.bn3 = nn.BatchNorm1d()
        self.fc3 = nn.Linear(16, 16)
        self.bn4 = nn.BatchNorm1d()
        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.selu(self.fc1(self.bn1(x)), inplace=True)
        x = F.selu(self.fc2(self.bn2(x)), inplace=True)
        x = F.selu(self.fc3(self.bn3(x)), inplace=True)
        return F.softmax(self.fc4(self.bn4(x)), dim=1)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.bn1 = nn.BatchNorm1d()
        self.fc1 = nn.Linear(4, 16)
        self.bn2 = nn.BatchNorm1d()
        self.fc2 = nn.Linear(16, 16)
        self.bn3 = nn.BatchNorm1d()
        self.fc3 = nn.Linear(16, 16)
        self.bn4 = nn.BatchNorm1d()
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.selu(self.fc1(self.bn1(x)), inplace=True)
        x = F.selu(self.fc2(self.bn2(x)), inplace=True)
        x = F.selu(self.fc3(self.bn3(x)), inplace=True)
        return self.fc4(self.bn4(x))