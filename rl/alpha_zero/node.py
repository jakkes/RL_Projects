from __future__ import annotations
from typing import List

import torch
from torch import nn

import numpy as np
from rl.simulators import Simulator
from rl.utils.random import choice

from .config import AlphaZeroConfig


class Node:
    def __init__(self, state: np.ndarray, action_mask: np.ndarray, simulator: Simulator, network: nn.Module,
                 parent: Node = None, config: AlphaZeroConfig = None, action: int = None,
                 reward: float = None, terminal: bool = None):
        self._state = state
        self._action_mask = action_mask
        self._action = action
        self._reward = reward
        self._terminal = terminal
        self._simulator = simulator
        self._network = network
        self._config = config
        self._parent = parent
        self._P = None
        self._N = None
        self._W = None
        self._V = None
        self._children: List[Node] = None
        self._expanded: bool = False

        if np.all(self._action_mask == False) and not self._terminal:
            print("Stop")

    @property
    def is_leaf(self):
        return self._children is None

    @property
    def is_root(self):
        return self._parent is None

    @property
    def is_terminal(self):
        return self._terminal

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action

    @property
    def reward(self):
        return self._reward

    @property
    def action_mask(self):
        return self._action_mask

    @property
    def action_policy(self):
        distribution = np.power(self._N, 1 / self._config.T)
        return distribution / np.sum(distribution)

    def select(self) -> Node:
        if self.is_terminal:
            raise ValueError("Cannot select action from terminal state.")

        Q = np.zeros_like(self._N)
        mask = self._N > 0
        Q[mask] = self._W[mask] / self._N[mask]
        if np.any(self._N > 0):
            U = self._P * np.sqrt(np.sum(self._N)) / (1 + self._N)
        else:
            U = self._P
        QU = Q + self._config.c * U
        QU[~self._action_mask] = -np.inf
        return self._children[np.argmax(QU)]

    def expand(self):

        if self._expanded:
            return

        self._init_pv()

        if not self.is_terminal:
            actions = np.arange(self._action_mask.shape[0])[self._action_mask]
            states = np.expand_dims(self._state, 0)
            states = np.repeat(states, actions.shape[0], axis=0)
            next_states, next_masks, rewards, terminals, _ = self._simulator.step_bulk(
                states, actions)

            self._children = [None] * self._action_mask.shape[0]
            for next_state, next_mask, reward, terminal, action in zip(next_states, next_masks, rewards, terminals, actions):
                self._children[action] = Node(next_state, next_mask,
                                            self._simulator, self._network, parent=self,
                                            config=self._config, action=action, reward=reward, terminal=terminal)

            self._N = np.zeros(self._action_mask.shape[0])
            self._W = np.zeros(self._action_mask.shape[0])

        self._expanded = True

    def backup(self):
        if self.is_root:
            return

        if self.is_terminal:
            self.parent._backpropagate(self._action, self._reward)
        else:
            self.parent._backpropagate(self._action, -self._V)

    def _backpropagate(self, action: int, value: float):
        self._N[action] += 1
        self._W[action] += value

        if self.parent is not None:
            self.parent._backpropagate(self.action, -value)

    def _init_pv(self):
        with torch.no_grad():
            p, v = self._network(
                torch.as_tensor(self._state, dtype=torch.float).unsqueeze_(0),
                torch.as_tensor(self._action_mask,
                                dtype=torch.bool).unsqueeze_(0)
            )
        self._P = torch.softmax(p, dim=1).squeeze_(0).numpy()
        self._V = v[0, 0].numpy()

        if self.is_root:
            self.add_noise()

    def add_noise(self):
        d = np.random.dirichlet(self._config.alpha * np.ones(self._action_mask.shape[0])[self._action_mask])
        self._P[self._action_mask] = (1 - self._config.epsilon) * self._P[self._action_mask] + self._config.epsilon * d

    def rootify(self):

        if self.is_terminal:
            raise ValueError("Cannot rootify a terminal state.")

        self._parent = None
        self._action = None
        self._reward = None
        self._terminal = False

        if self._P is not None:
            self.add_noise()
