from typing import Dict, List, Tuple

import torch
from torch import nn, Tensor, LongTensor, Size

from . import MuZeroAgent, MuZeroConfig

class Node:
    def __init__(self, state: Tensor, prior: Tensor, _action: int=None, _parent: Node=None):
        self._action: int = _action
        self._state: Tensor = state
        self._parent: Node = _parent

        action_dim = prior.shape[-1]

        self.P: Tensor = prior
        self.Q: Tensor = torch.zeros(action_dim)
        self.N: LongTensor = torch.zeros(action_dim, dtype=torch.long)
        self.S: Tensor = torch.empty((action_dim, ) + state.shape)
        self.R: Tensor = torch.zeros(action_dim)
        self.children: List[Node] = [None] * action_dim
    
    def expanded(self, action: int):
        return self.children[action] is not None

    def expand(self, action: int, dynamics_net: nn.Module, prediction_net: nn.Module) -> Tuple[Node, float]:
        with torch.no_grad():
            state, reward = dynamics_net(self._state, action)
            prior, value = prediction_net(state)
        self.R[action] = reward; self.S[action] = state
        self.children[action] = Node(state, prior, _action=action, _parent=self)
        return self.children[action], value

    def select_action(self, c1, c2) -> LongTensor:
        coeff = (((self.N.sum() + c2 + 1) / c2).log_() + c1) * self.N.sum().sqrt_()
        return torch.argmax(
            self.Q + self.P / (1 + self.N) * coeff
        )

    def get_child(self, action: int) -> Node:
        return self.children[action]

    def backup(self, G: float, discount: float):
        if self._parent is None:
            return

        self._parent.Q[self._action] = (self._parent.N[self._action] * self._parent.Q[self._action] + G) / (self._parent.N[self._action] + 1)
        self._parent.N[self._action] += 1

        self._parent.backup(self._parent.R[self._action] + discount * G, discount)
        

def run_mcts(state: Tensor, agent: MuZeroAgent) -> Tensor:
    with torch.no_grad():
        root_state = agent.representation_net(state)
        prior, value = agent.prediction_net(root_state)
    root_node = Node(root_state, prior)

    for _ in range(agent.config.simulations):
        node = root_node
        action = node.select_action(agent.config.c1, agent.config.c2).item()
        
        while node.expanded(action):
            node = node.get_child(action)

        node, value = node.expand(action, agent.dynamics_net, agent.prediction_net)
        node.backup(value, agent.config.discount)

class Simulation:
    def __init__(self, root: Node, representation_net: nn.Module, prediction_net: nn.Module):
        self.history: List[Node] = [root]
        self.node: Node = root
