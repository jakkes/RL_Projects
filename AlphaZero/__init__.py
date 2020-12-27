import torch
from torch import nn, optim
import numpy as np

from simulators import Simulator
from .config import AlphaZeroConfig
from .node import Node


def mcts(state: np.ndarray, action_mask: np.ndarray, simulator: Simulator, network: nn.Module, config: AlphaZeroConfig, simulations: int = 100, root_node: Node=None):
    root = Node(state, action_mask, simulator, network, config=config) if root_node is None else root_node

    for _ in range(simulations):
        node = root

        while not node.is_leaf:
            node = node.select()

        if not node.is_terminal:
            node.expand()
            node = node.select()

        node.rollout_and_backpropagate()
    return root


def train_step(network: nn.Module, optimizer: optim.Optimizer, target_policies: torch.Tensor,
               target_values: torch.Tensor, states: torch.Tensor, action_masks: torch.Tensor):
    optimizer.zero_grad()
    p, v = network(states, action_masks)
    plog = torch.log_softmax(p[action_masks])
    target_policies = target_policies[action_masks]

    loss = (target_policies * plog).sum(dim=1).mean() + (v - target_values).square().mean()
    loss.backward()
    optimizer.step()
