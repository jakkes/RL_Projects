from math import log2
from typing import Union, Tuple
from abc import abstractmethod

import torch
from torch import Tensor, Size

import numpy as np

class Replay:

    @abstractmethod
    def get_all(self):
        pass

    @abstractmethod
    def get_size(self):
        pass

    @abstractmethod
    def sample(self, n):
        pass

    @abstractmethod
    def update_weights(self, weights):
        pass


class UniformReplayBuffer(Replay):
    def __init__(self, state_shape: Union[Size, Tuple], capacity: int, device: str="cpu"):
        self.capacity: int = capacity
        self.device: str = device

        self._pos = 0
        self._full = False
        
        self._states = torch.zeros((self.capacity, ) + state_shape, device=device)
        self._rewards = torch.zeros(self.capacity, device=device)
        self._actions = torch.zeros(self.capacity, dtype=torch.long, device=device)
        self._not_dones = torch.zeros(self.capacity, dtype=torch.bool, device=device)
        self._next_states = torch.zeros((self.capacity, ) + state_shape, device=device)

    def get_all(self):
        return (
            self._states[:self.get_size()],
            self._actions[:self.get_size()],
            self._rewards[:self.get_size()],
            self._not_dones[:self.get_size()],
            self._next_states[:self.get_size()]
        )

    def get_size(self):
        return self.capacity if self._full else self._pos

    def add(self, state, action, reward, not_done, next_state):
        self._states[self._pos] = state.to(self.device)
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._not_dones[self._pos] = not_done
        self._next_states[self._pos] = next_state.to(self.device)

        self._pos += 1 
        if self._pos >= self.capacity:
            self._full = True
            self._pos = 0

    def update_weights(self, weights):
        return

    def sample(self, n: int):
        self._indices = torch.randint(self.get_size(), (n, ))
        return (
            self._states[self._indices],
            self._actions[self._indices],
            self._rewards[self._indices],
            self._not_dones[self._indices],
            self._next_states[self._indices],
            torch.tensor([1.0 / self.get_size()]).expand(self._indices.shape[0])
        )


class UniformSequenceBuffer(Replay):
    def __init__(self, capacity: int, subsequences: int):
        self._data = np.empty((capacity, subsequences), dtype=object)
        self._pos = 0
        self._filled = False
        self._capacity = capacity

    def get_all(self):
        return self._data[:self.get_size()]

    def get_size(self):
        return self._capacity if self._filled else self._pos

    def sample(self, n):
        indices = np.random.randint(self.get_size(), size=n)
        return self._data[indices], 1.0 / self.get_size() * np.ones_like(indices)

    def add(self, sequence):
        self._data[self._pos, :] = sequence
        self._pos += 1
        if self._pos >= self._capacity:
            self._pos = 0
            self._filled = True