import torch
from torch import nn

class Net(nn.Module):
    def __init__(self, z):
        super().__init__()

        self.z = z

        self.net = nn.Sequential(
            nn.Linear(8, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 4 * self.z.shape[0])
        )

    def get_Q_from_p(self, p):
        return torch.sum(self.z.view(1, 1, -1) * p, dim=2)

    def getQ(self, x):
        return torch.sum(self.z.view(1, 1, -1) * self(x), dim=2)

    def forward(self, x):
        return self.net(x).view(-1, 4, self.z.shape[0]).softmax(dim=2)
