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
    action_masks = torch.as_tensor(np.stack(action_masks))
    action_policies = torch.as_tensor(np.stack(action_policies), dtype=torch.float)
    z = torch.ones(states.shape[0])
    i = torch.arange(1, states.shape[0]+1, 2)
    j = torch.arange(2, states.shape[0]+1, 2)
    z[-i] *= reward
    z[-j] *= -reward

    return states, action_masks, action_policies, z


def main():
    network = Network()
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    config = AlphaZeroConfig()

    for _ in range(100):
        batch_states, batch_action_masks, batch_action_policies, batch_z = [], [], [], []
        for _ in range(8):
            states, action_masks, action_policies, z = run_episode(TicTacToe, network, config)
            indices = torch.randint(0, states.shape[0], (4, ))
            batch_states.append(states[indices])
            batch_action_masks.append(action_masks[indices])
            batch_action_policies.append(action_policies[indices])
            batch_z.append(z[indices])

        batch_states = torch.cat(batch_states)
        batch_action_masks = torch.cat(batch_action_masks)
        batch_action_policies = torch.cat(batch_action_policies)
        batch_z = torch.cat(batch_z)

        train_step(network, optimizer, batch_states, batch_action_masks, batch_action_policies, batch_z)


if __name__ == "__main__":
    main()
