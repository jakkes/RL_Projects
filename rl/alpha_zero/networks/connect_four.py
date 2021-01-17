from torch import nn


class ConnectFourNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 256, 4, padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (3, 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 1),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Conv2d(64, 1, (1, 7))
        self.policy = nn.Conv2d(64, 1, 1)

    def forward(self, states, action_masks):
        N = states.shape[0]
        player = states[:, -1].view(N, 1, 1, 1)
        states = states[:, :-1].view(N, 1, 6, 7) * player
        x = self.body(states)
        v = self.value(x).view(N, 1)
        p = self.policy(x).view(N, -1)
        p[~action_masks] = -float("inf")
        return p, v
