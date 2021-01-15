import numpy as np

from .simulator import Simulator


class ConnectFour(Simulator):

    @classmethod
    def reset_bulk(self, n: int) -> np.ndarray:
        states = np.zeros((n, 7*6 + 1))
        states[:, -1] = 1.0
        return states, np.ones((n, 7), dtype=np.bool_)

    @classmethod
    def step_bulk(self, states: np.ndarray, actions: np.ndarray, terminal_masks: np.ndarray):
        pass
