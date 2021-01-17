from torch import nn


class ConnectFourNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (2, 3)),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Linear(64, 1)
        self.policy = nn.Linear(64, 7)

    def forward(self, states, action_masks):
        N = states.shape[0]
        player = states[:, -1].view(N, 1, 1, 1)
        states = states[:, :-1].view(N, 1, 6, 7) * player
        x = self.body(states).view(N, -1)
        v = self.value(x)
        p = self.policy(x)
        p[~action_masks] = -float("inf")
        return p, v
