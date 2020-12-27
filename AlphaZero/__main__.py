import numpy as np
import torch
from torch import nn, optim, Tensor

from simulators import TicTacToe, Simulator
from AlphaZero import mcts, train_step
from .config import AlphaZeroConfig


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Linear(64, 1)
        self.policy = nn.Linear(64, 9)

    def forward(self, states, action_masks):
        x = self.body(states)
        v = self.value(x)
        p = self.policy(x)
        p[~action_masks] = -float("inf")
        return p, v


def run_episode(simulator: Simulator, network: Network, config: AlphaZeroConfig):
    states, action_masks, action_policies = [], [], []
    terminal = False
    reward = 0
    state, action_mask = simulator.reset()

    root = None

    while not terminal:
        root = mcts(state, action_mask, simulator,
                    network, config, root_node=root)
        action_policy = root.action_policy

        states.append(state)
        action_masks.append(action_mask)
        action_policies.append(torch.as_tensor(action_policy, dtype=torch.float))

        action = np.random.choice(action_mask.shape[0], p=action_policy)
        state, action_mask, reward, terminal, _ = simulator.step(state, action)
        root = root.children[action]

    states = torch.as_tensor(np.stack(states), dtype=torch.float)
    action_masks = torch.stack(action_masks)
    z = torch.ones(states.shape[0])
    i = torch.arange(1, states.shape[0]+1, 2)
    j = torch.arange(2, states.shape[0]+1, 2)
    z[-i] *= reward
    z[-j] *= -reward

    return torch.stack(states), torch.stack(action_masks), torch.stack(action_policies), z


def train_step(network: nn.Module, optimizer: optim.Optimizer, states: Tensor,
               action_masks: Tensor, action_policies: Tensor, z: Tensor):
    p, v = network(states, action_masks)
    loggedp = torch.log_softmax(p, dim=1)
    loggedp[action_policies == 0] = 0

    loss = (z - v).square().mean() - \
        (action_policies * loggedp).sum(dim=1).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    network = Network()
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    config = AlphaZeroConfig()
    run_episode(TicTacToe, network, config)


if __name__ == "__main__":
    main()
