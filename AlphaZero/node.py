from __future__ import annotations
from typing import List

import torch
from torch import nn

import numpy as np
from simulators import Simulator
from utils.random import choice

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
        self._children: List[Node] = None

    @property
    def is_leaf(self):
        return self._P is None

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
        distribution = np.power(self._N, self._config.T)
        return distribution / np.sum(distribution)

    def select(self) -> Node:
        if self.is_terminal:
            return self

        Q = np.zeros_like(self._N)
        mask = self._N > 0
        Q[mask] = self._W[mask] / self._N[mask]
        U = self._P * np.sqrt(np.sum(self._N)) / (1 + self._N)
        QU = Q + self._config.c * U
        QU[~self._action_mask] = -np.inf
        return self._children[np.argmax(QU)]

    def expand(self):
        if self.is_terminal:
            return

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

        with torch.no_grad():
            P, _ = self._network(
                torch.as_tensor(self._state, dtype=torch.float).unsqueeze_(0),
                torch.as_tensor(self._action_mask,
                                dtype=torch.bool).unsqueeze_(0)
            )
        self._P = torch.softmax(P, dim=1).squeeze_(0).numpy()
        self._N = np.zeros(self._action_mask.shape[0])
        self._W = np.zeros(self._action_mask.shape[0])

    def rollout_and_backpropagate(self):
        state = self.state
        terminal = self.is_terminal
        reward = - self.reward
        rewardflip = 1
        action_mask = self.action_mask

        first_action = None

        while not terminal:
            with torch.no_grad():
                p, _ = self._network(
                    torch.as_tensor(state, dtype=torch.float).unsqueeze_(0),
                    torch.as_tensor(
                        action_mask, dtype=torch.bool).unsqueeze_(0)
                )
            p = torch.softmax(p, dim=1)
            action = choice(p).item()
            if first_action is None:
                first_action = action
            state, action_mask, reward, terminal, _ = self._simulator.step(
                state, action)
            rewardflip *= -1

        reward = rewardflip * reward
        self._backpropagate(first_action, reward)

    def _backpropagate(self, action: int, reward: float):
        self._N[action] += 1
        self._W[action] += reward

        if self.parent is not None:
            self.parent._backpropagate(self.action, -reward)
