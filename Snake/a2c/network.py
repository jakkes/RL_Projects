import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5, padding=2)
        self.conv2 = nn.Conv2d(128, 64, 3)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.max_pool2d(F.selu(self.conv1(x), inplace=True), 2)
        x = F.max_pool2d(F.selu(self.conv2(x), inplace=True), 2)
        x = F.selu(self.conv3(x), inplace=True)
        x = F.selu(self.fc1(x.view(-1, 128)), inplace=True)
        x = F.selu(self.fc2(x), inplace=True)
        return F.softmax(self.fc3(x), dim=1)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5, padding=2)
        self.conv2 = nn.Conv2d(128, 64, 3)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.max_pool2d(F.selu(self.conv1(x), inplace=True), 2)
        x = F.max_pool2d(F.selu(self.conv2(x), inplace=True), 2)
        x = F.selu(self.conv3(x), inplace=True)
        x = F.selu(self.fc1(x.view(-1, 128)), inplace=True)
        x = F.selu(self.fc2(x), inplace=True)
        return self.fc3(x)