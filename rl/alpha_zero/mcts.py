from torch import nn
import numpy as np

from rl.simulators import Simulator

from .node import Node
from .config import AlphaZeroConfig


def mcts(state: np.ndarray, action_mask: np.ndarray, simulator: Simulator, network: nn.Module, config: AlphaZeroConfig, simulations: int = 50, root_node: Node = None) -> Node:
    root = Node(state, action_mask, simulator, network,
                config=config) if root_node is None else root_node
    root.rootify()

    for _ in range(simulations):
        node = root
        while not node.is_leaf:
            node = node.select()

        node.expand()
        node.backup()
    return root
