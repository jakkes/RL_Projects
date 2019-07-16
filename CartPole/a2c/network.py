import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.selu(self.fc1(x), inplace=True)
        return F.softmax(self.fc4(x), dim=1)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.selu(self.fc1(x), inplace=True)
        return self.fc4(x)