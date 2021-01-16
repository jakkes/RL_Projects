from math import log2
from typing import Union
from abc import abstractmethod

import rl.rainbow_dqn as rainbow

import torch
from torch import Tensor

class Replay:
    
    @abstractmethod
    def __init__(self, config):
        pass

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
    def __init__(self, config: 'rainbow.RainbowConfig'):
        self.config = config
        self.capacity: int = config.replay_capacity

        self._pos = 0
        self._full = False
        
        self._states = torch.zeros(self.capacity, config.state_dim, device=config.device)
        self._rewards = torch.zeros(self.capacity, device=config.device)
        self._actions = torch.zeros(self.capacity, dtype=torch.long, device=config.device)
        self._not_dones = torch.zeros(self.capacity, dtype=torch.bool, device=config.device)
        self._next_states = torch.zeros(self.capacity, config.state_dim, device=config.device)

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
        self._states[self._pos] = state.to(self.config.device)
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._not_dones[self._pos] = not_done
        self._next_states[self._pos] = next_state.to(self.config.device)

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
            None
        )


class PrioritizedReplayBuffer(Replay):
    def __init__(self, config: 'rainbow.RainbowConfig'):

        self.config = config

        if log2(config.replay_capacity) % 1 != 0:
            raise ValueError("Replay capacity must be power of 2")
        
        self.capacity: int = config.replay_capacity
        
        self._pos = 0
        self._full = False
        self._states = torch.zeros(self.capacity, config.state_dim, device=config.device)
        self._rewards = torch.zeros(self.capacity, device=config.device)
        self._actions = torch.zeros(self.capacity, dtype=torch.long, device=config.device)
        self._not_dones = torch.zeros(self.capacity, dtype=torch.bool, device=config.device)
        self._next_states = torch.zeros(self.capacity, config.state_dim, device=config.device)

        self.weights = torch.zeros(self.capacity * 2 - 1)
        self._wi = self.capacity - 1
        self._depth = int(log2(self.capacity))

        self._indices = None
        self._max = 1.0

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
        self._states[self._pos] = state.to(self.config.device)
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._not_dones[self._pos] = not_done
        self._next_states[self._pos] = next_state.to(self.config.device)
        self._set_weight(self._max, self._pos)

        self._pos += 1 
        if self._pos >= self.capacity:
            self._full = True
            self._pos = 0

    def _set_weight(self, weight, i):
        i = torch.tensor([i])
        weight = torch.tensor(weight)
        self._set_weights(weight, i)

    def _set_weights(self, weights: Union[Tensor, float], i: Tensor):
        
        ######################################
        ###### Remove duplicate entries ######
        
        weights = torch.as_tensor(weights)
        _, argi = torch.unique(i, return_inverse=True)
        i = i[argi]
        if len(weights.shape) > 0:
            weights = weights[argi]

        ###### Done removing duplicate entries ######
        #############################################
        
        n = i.shape[0]
        wi = self._wi + i
        dw = weights - self.weights[wi]

        w_update_i = torch.empty((n, self._depth + 1), dtype=torch.long)
        w_update_i[:, 0] = wi
        for d in range(1, self._depth + 1):
            w_update_i[:, d] = (w_update_i[:, d-1] - 1) // 2
        
        for j in range(n):
            self.weights[w_update_i[j]] += dw[j]
        self._max = max(torch.max(weights).item(), self._max)

    def update_weights(self, weights):
        self._set_weights(weights, self._indices)
        self._indices = None
        if self._pos % 25 == 0:
            self._update_max()

    def _update_max(self):
        weights = self.weights[self._wi:]
        non_max = weights[weights < self._max]
        
        if non_max.shape[0] == 0:
            return

        new_max = non_max.max().item()
        if new_max == 0:
            return
        self._max = new_max

        update_indices = (weights >= self._max).nonzero(as_tuple=True)[0]

        if update_indices.shape[0] == 0:
            return
        
        self._set_weights(self._max, update_indices)

    def sample(self, n: int):
        if self._indices is not None:
            raise ValueError("Must update weights of last sampled batch before moving on!")
        
        #################################################
        #### Sample in bins for more stable sampling ####

        bin_starts = torch.linspace(0, self.weights[0].item(), steps=n+1)[:-1]     # Exclude endpoint
        bin_spacing = bin_starts[1] - bin_starts[0]
        random_samples = torch.rand(n) * bin_spacing
        w = random_samples + bin_starts

        #### End bin sampling ####
        ##########################

        self._indices = self.retrieve_indices(w)
        return (
            self._states[self._indices],
            self._actions[self._indices],
            self._rewards[self._indices],
            self._not_dones[self._indices],
            self._next_states[self._indices],
            self.weights[self._indices + self._wi] / self.weights[0]
        )

    def retrieve_indices(self, w: Tensor, w_inplace=True):
        if not w_inplace:
            w = w.clone()
        
        i = torch.zeros_like(w, dtype=torch.long)
        while i[0]*2+1 < self.weights.shape[0]:
            left = 2*i + 1
            right = 2*i + 2

            go_right = w > self.weights[left]
            w[go_right] -= self.weights[left[go_right]]

            i[go_right] = right[go_right]
            i[~go_right] = left[~go_right]
        return i - self._wi

    @staticmethod
    def get_right(i):
        return 2*i + 2

    @staticmethod
    def get_left(i):
        return 2*i + 1