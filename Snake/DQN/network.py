import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.conv4 = nn.Conv2d(32, 16, 2)
        self.fc1 = nn.Linear(16 * 2 * 2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.selu(self.conv1(x), inplace=True)
        x = F.max_pool2d(F.selu(self.conv2(x), inplace=True), 2)
        x = F.max_pool2d(F.selu(self.conv3(x), inplace=True), 2)
        x = F.selu(self.conv4(x), inplace=True)
        x = F.selu(self.fc1(x.view(x.shape[0], -1)), inplace=True)
        x = F.selu(self.fc2(x), inplace=True)
        return self.fc3(x)