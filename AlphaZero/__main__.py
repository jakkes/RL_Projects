import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter

from simulators import TicTacToe, Simulator
from AlphaZero import mcts, train_step
from .config import AlphaZeroConfig


_SUMMARY_WRITER = None


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(2, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Linear(64, 1)
        self.policy = nn.Linear(64, 9)

    def forward(self, states, action_masks):
        N = states.shape[0]
        reshaped_states = torch.zeros(N, 2, 3, 3)
        reshaped_states[:, 0, :, :] = states[:, :-1].view(N, 3, 3)
        reshaped_states[:, 1, :, :] = states[:, -1].view(N, 1, 1)
        x = self.body(reshaped_states).view(N, -1)
        v = self.value(x)
        p = self.policy(x)
        p[~action_masks] = -float("inf")
        return p, v


def run_episode(episode: int, simulator: Simulator, network: Network, config: AlphaZeroConfig):
    states, action_masks, action_policies = [], [], []
    terminal = False
    reward = 0
    state, action_mask = simulator.reset()

    with torch.no_grad():
        start_prior, start_value = network(
            torch.as_tensor(state, dtype=torch.float).unsqueeze_(0),
            torch.as_tensor(action_mask).unsqueeze_(0)
        )
    start_value = start_value.item()
    start_prior = start_prior.softmax(dim=1).squeeze_(0)
    first_action_policy = None
    first_action = None

    root = None
    while not terminal:
        root = mcts(state, action_mask, simulator,
                    network, config, root_node=root)

        action_policy = root.action_policy
        if first_action_policy is None:
            first_action_policy = action_policy

        states.append(state)
        action_masks.append(action_mask)
        action_policies.append(torch.as_tensor(
            action_policy, dtype=torch.float))

        action = np.random.choice(action_mask.shape[0], p=action_policy)
        if first_action is None:
            first_action = action

        state, action_mask, reward, terminal, _ = simulator.step(state, action)
        root = root.children[action]

    if _SUMMARY_WRITER is not None:
        _SUMMARY_WRITER.add_scalar(
            "Episode/Reward", abs(reward), global_step=episode)
        _SUMMARY_WRITER.add_scalar(
            "Episode/Start value", start_value, global_step=episode)
        _SUMMARY_WRITER.add_scalar("Episode/Start KL-div", -(first_action_policy *
                                                             torch.log_softmax(start_prior, dim=0).numpy()).sum(), global_step=episode)
        _SUMMARY_WRITER.add_scalar(
            "Episode/First action", first_action, global_step=episode)

    states = torch.as_tensor(np.stack(states), dtype=torch.float)
    action_masks = torch.as_tensor(np.stack(action_masks))
    action_policies = torch.as_tensor(
        np.stack(action_policies), dtype=torch.float)
    z = torch.ones(states.shape[0])
    i = torch.arange(1, states.shape[0]+1, 2)
    j = torch.arange(2, states.shape[0]+1, 2)
    z[-i] *= reward
    z[-j] *= -reward

    return states, action_masks, action_policies, z


def main():
    network = Network()
    optimizer = optim.RMSprop(network.parameters(), lr=1e-3)
    config = AlphaZeroConfig()

    for i in range(10000):
        batch_states, batch_action_masks, batch_action_policies, batch_z = [], [], [], []
        for j in range(4):
            states, action_masks, action_policies, z = run_episode(
                4*i+j, TicTacToe, network, config)
            batch_states.append(states)
            batch_action_masks.append(action_masks)
            batch_action_policies.append(action_policies)
            batch_z.append(z)

        batch_states = torch.cat(batch_states)
        batch_action_masks = torch.cat(batch_action_masks)
        batch_action_policies = torch.cat(batch_action_policies)
        batch_z = torch.cat(batch_z)

        loss = train_step(network, optimizer, batch_states,
                          batch_action_masks, batch_action_policies, batch_z)

        if _SUMMARY_WRITER is not None:
            _SUMMARY_WRITER.add_scalar("Training/Loss", loss, global_step=i)
            _SUMMARY_WRITER.add_scalar(
                "Training/Batch size", batch_states.shape[0], global_step=i)


if __name__ == "__main__":
    _SUMMARY_WRITER = SummaryWriter()
    main()
    _SUMMARY_WRITER.close()
