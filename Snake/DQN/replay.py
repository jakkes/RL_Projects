import torch
from typing import List, Tuple
from random import choices

class Replay:
    def __init__(self, capacity: int, state_shape: tuple, device='cpu'):
        
        self.capacity = int(capacity)

        self._states = torch.empty((self.capacity, ) + state_shape, device=device)
        self._next_state = torch.empty((self.capacity, ) + state_shape, device=device)
        self._actions = torch.empty(self.capacity, dtype=torch.long, device=device)
        self._rewards = torch.empty(self.capacity, device=device)
        self._dones = torch.empty(self.capacity, device=device)
        self.position = 0
        self.full_loop = False

    def add(self, state, next_state, action, reward, done):
        self._states[self.position] = state
        self._next_state[self.position] = next_state
        self._actions[self.position] = action
        self._rewards[self.position] = reward
        self._dones[self.position] = 0.0 if done else 1.0
        self.position += 1
        if self.position >= self.capacity:
            if not self.full_loop:
                self.full_loop = True
            self.position = 0

    def get_random(self, n: int) -> Tuple:
        population = range(self.capacity) if self.full_loop else range(self.position)
        indices = choices(population, k=n)
        return (
            self._states[indices],
            self._next_state[indices],
            self._actions[indices],
            self._rewards[indices],
            self._dones[indices]
        )

    def count(self):
        return self.capacity if self.full_loop else self.position