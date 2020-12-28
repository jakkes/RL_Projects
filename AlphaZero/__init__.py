import torch
from torch import nn, optim, Tensor
import numpy as np

from simulators import Simulator
from .config import AlphaZeroConfig
from .node import Node


def mcts(state: np.ndarray, action_mask: np.ndarray, simulator: Simulator, network: nn.Module, config: AlphaZeroConfig, simulations: int = 20, root_node: Node=None):
    root = Node(state, action_mask, simulator, network, config=config) if root_node is None else root_node

    for _ in range(simulations):
        node = root

        while not node.is_leaf:
            node = node.select()

        if not node.is_terminal:
            node.expand()

        node.rollout_and_backpropagate()
    return root


def train_step(network: nn.Module, optimizer: optim.Optimizer, states: Tensor,
               action_masks: Tensor, action_policies: Tensor, z: Tensor):
    p, v = network(states, action_masks)
    loggedp = torch.where(torch.isinf(p), torch.zeros_like(p), torch.log_softmax(p, dim=1))

    loss = (z - v).square().mean() - \
        (action_policies * loggedp).sum(dim=1).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(loss)
    print(z.abs().mean())
