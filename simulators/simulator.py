from abc import abstractclassmethod
from typing import Dict, Iterable, List, Tuple

import numpy as np


class Simulator:


    @abstractclassmethod
    def action_masks(cls, states) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def action_mask(cls, state) -> np.ndarray:
        return cls.action_masks(np.expand_dims(state, 0))[0]

    @classmethod
    def step(cls, state: np.ndarray, action: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        next_states, next_action_masks, rewards, terminals, infos = cls.step_bulk(
            np.expand_dims(state, 0), np.array([action]))
        return next_states[0], next_action_masks[0], rewards[0], terminals[0], infos[0]

    @abstractclassmethod
    def step_bulk(cls, states: np.ndarray, actions: np.ndarray, terminal_masks: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        raise NotImplementedError

    @abstractclassmethod
    def reset_bulk(cls, n: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @classmethod
    def reset(cls) -> Tuple[np.ndarray, np.ndarray]:
        a, b = cls.reset_bulk(1)
        return a[0], b[0]

    @abstractclassmethod
    def render(cls, state):
        pass
