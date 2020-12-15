from abc import abstractclassmethod
from typing import Dict, Iterable, List, Tuple
import numpy as np


class Simulator:

    @classmethod
    def step(cls, state: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        next_states, rewards, terminals, infos = cls.step_bulk(np.expand_dims(state, 0), np.array([action]))
        return next_states[0], rewards[0], terminals[0], infos[0]

    @abstractclassmethod
    def step_bulk(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        raise NotImplementedError

    @abstractclassmethod
    def reset(self, n: int) -> np.ndarray:
        raise NotImplementedError
