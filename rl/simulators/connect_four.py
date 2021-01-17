import numpy as np
from scipy.signal import convolve

from rl.utils.np_utils import cross_diag
from .simulator import Simulator


class ConnectFour(Simulator):

    @classmethod
    def reset_bulk(self, n: int) -> np.ndarray:
        states = np.zeros((n, 7*6 + 1))
        states[:, -1] = 1.0
        return states, np.ones((n, 7), dtype=np.bool_)

    @classmethod
    def compute_rewards(cls, states: np.ndarray, players: np.ndarray):

        players = 4 * players.reshape((-1, 1, 1))

        win_horisontal = convolve(states, np.ones((1, 1, 4)), mode="valid") == players
        win_horisontal = np.any(np.any(win_horisontal, axis=2), axis=1)

        win_vertical = convolve(states, np.ones((1, 4, 1)), mode="valid") == players
        win_vertical = np.any(np.any(win_vertical, axis=2), axis=1)

        win_diagonal = convolve(states, np.diag(np.ones(4)).reshape((1, 4, 4)), mode="valid") == players
        win_diagonal = np.any(np.any(win_diagonal, axis=2), axis=1)

        win_xdiagonal = convolve(states, cross_diag(np.ones(4)).reshape((1, 4, 4)), mode="valid") == players
        win_xdiagonal = np.any(np.any(win_xdiagonal, axis=2), axis=1)

        win = win_horisontal | win_vertical | win_diagonal | win_xdiagonal

        loss_horisontal = convolve(states, np.ones((1, 1, 4)), mode="valid") == -players
        loss_horisontal = np.any(np.any(loss_horisontal, axis=2), axis=1)

        loss_vertical = convolve(states, np.ones((1, 4, 1)), mode="valid") == -players
        loss_vertical = np.any(np.any(loss_vertical, axis=2), axis=1)

        loss_diagonal = convolve(states, np.diag(np.ones(4)).reshape((1, 4, 4)), mode="valid") == -players
        loss_diagonal = np.any(np.any(loss_diagonal, axis=2), axis=1)

        loss_xdiagonal = convolve(states, cross_diag(np.ones(4)).reshape((1, 4, 4)), mode="valid") == -players
        loss_xdiagonal = np.any(np.any(loss_xdiagonal, axis=2), axis=1)

        loss = loss_horisontal | loss_vertical | loss_diagonal | loss_xdiagonal

        rewards = np.zeros(states.shape[0])
        rewards[win] = 1.0
        rewards[loss] = -1.0

        return rewards

    @classmethod
    def step_bulk(cls, states: np.ndarray, actions: np.ndarray):

        next_states = states.copy()
        batchvec = np.arange(next_states.shape[0])
        players = next_states[:, -1]

        next_states = next_states[:, :-1].reshape((-1, 6, 7))
        heights = (next_states != 0).sum(axis=1)

        if np.any(heights[batchvec, actions] >= 6):
            raise ValueError("Cannot place a piece in an already full column")

        next_states[batchvec, 5 - heights[batchvec, actions], actions] = players[batchvec]
        heights[batchvec, actions] += 1
        rewards = cls.compute_rewards(next_states, players)
        terminals = (rewards != 0) | np.all(heights == 6, axis=1)

        next_masks = heights < 6
        next_states = next_states.reshape((-1, 6*7))
        next_states = np.concatenate((next_states, -players.reshape((-1, 1))), axis=1)

        return next_states, next_masks, rewards, terminals, [{} for _ in range(next_states.shape[0])]

    @classmethod
    def render(cls, state: np.ndarray):
        
        def tile(value) -> str:
            if value == 0:
                return " "
            elif value == 1:
                return "X"
            elif value == -1:
                return "O"
            else:
                raise ValueError(f"Unexpected value {value}")

        def print_line(i: int):
            print(" | ".join(tile(state[7*i + j]) for j in range(7)))

        print(" | ".join(str(x) for x in range(7)))
        for i in range(6):
            print_line(i)
