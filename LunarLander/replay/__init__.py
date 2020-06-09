from abc import abstractmethod
from math import log2
from typing import Tuple, Any, Union

import numpy as np

class Replay:

    @abstractmethod
    def add(self, obj: Any):
        pass

    @abstractmethod
    def add_multiple(self, objs: np.ndarray):
        pass

    @abstractmethod
    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_size(self) -> int:
        pass

    @abstractmethod
    def get_all(self) -> np.ndarray:
        pass

    
class ExperienceReplay(Replay):
    def __init__(self, capacity):
        super().__init__()

        self.capacity = capacity
        self.data = np.empty(capacity, dtype=object)
        self.pos = 0
        self.filled = False

    def add(self, obj):
        self.data[self.pos] = obj
        self.pos += 1
        if self.pos >= self.capacity:
            self.pos = 0
            self.filled = True

    def add_multiple(self, objs: np.ndarray):
        size = objs.shape[0]
        indices = np.mod(np.arange(self.pos, self.pos + size), self.capacity)
        self.data[indices] = objs
        self.pos = (self.pos + size) % self.capacity
        if self.pos < indices[0]:
            self.filled = True
        

    def get_size(self):
        return self.capacity if self.filled else self.pos

    def sample(self, n):
        return self.data[np.random.randint(self.get_size(), size=n)], None

    def get_all(self):
        return self.data[:self.get_size()]


class PrioritizedReplayBuffer(Replay):
    def __init__(self, capacity: int):
        
        super().__init__()

        if log2(capacity) % 1 != 0:
            raise ValueError("Capacity must be power of 2")
        
        self.capacity: int = capacity
        
        self._pos = 0
        self._full = False
        self._data = np.empty((capacity, 5), dtype=object)
        self.weights = np.zeros(capacity * 2 - 1)
        self._wi = capacity - 1
        self._depth = int(log2(capacity))

        self._indices = None
        self._indices_mask = None
        self._max = 0.0001

    def get_all(self):
        return self._data[:self.get_size()]

    def get_size(self):
        return self.capacity if self._full else self._pos

    def add(self, obj):
        self._data[self._pos] = obj
        self._set_weight(self._max, self._pos)

        self._pos += 1 
        if self._pos >= self.capacity:
            self._full = True
            self._pos = 0

    def add_multiple(self, objs: np.ndarray, weights: np.ndarray=None):
        size = objs.shape[0]
        indices = np.mod(np.arange(self._pos, self._pos + size), self.capacity)
        self._data[indices] = objs
        if weights is None:
            self._set_weights(self._max, indices)
        else:
            self._set_weights(weights, indices)
        self._pos = (self._pos + size) % self.capacity
        if self._pos < indices[0]:
            self._full = True
        
        if self._indices is not None:
            _, index_intersect, _ = np.intersect1d(self._indices, indices, return_indices=True)
            self._indices_mask[index_intersect] = False

    def _set_weight(self, weight, i):
        i = np.asarray(i).reshape(1)
        weight = np.asarray(weight).reshape(1)
        self._set_weights(weight, i)

    def _set_weights(self, weights: Union[np.ndarray, float], i: np.ndarray):
        
        ######################################
        ###### Remove duplicate entries ######
        
        _, argi = np.unique(i, return_index=True)
        i = i[argi]
        if type(weights) == np.ndarray:
            weights = weights[argi]

        ###### Done removing duplicate entries ######
        #############################################
        
        n = i.shape[0]
        wi = self._wi + i
        dw = weights - self.weights[wi]

        w_update_i = np.empty((n, self._depth + 1), dtype=np.int)
        w_update_i[:, 0] = wi
        for d in range(1, self._depth + 1):
            w_update_i[:, d] = (w_update_i[:, d-1] - 1) / 2
        
        for j in range(n):
            self.weights[w_update_i[j]] += dw[j]
        self._max = max(np.max(weights), self._max)

    def update_weights(self, weights):
        if self._indices_mask.sum() > 0:
            self._set_weights(weights[self._indices_mask], self._indices[self._indices_mask])
        self._indices = None
        self._indices_mask = None

    def update_max(self):
        self._max = np.max(self.weights[self._wi:])

    def sample(self, n: int):
        if self._indices is not None:
            raise ValueError("Must update weights of last sampled batch before moving on!")
        w = np.random.random(n) * self.weights[0]
        self._indices = self.retrieve_indices(w)
        self._indices_mask = np.ones_like(self._indices).astype(bool)
        return self._data[self._indices], self.weights[self._indices + self._wi] / self.weights[0]

    def retrieve_indices(self, w: np.ndarray, w_inplace=True):
        if not w_inplace:
            w = w.copy()
        
        i = np.zeros_like(w, dtype=np.int)
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