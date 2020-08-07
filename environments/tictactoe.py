from gym import Env
from gym.spaces import Discrete
from gym.spaces import Box

import torch

@torch.jit.script
def _check_winning_position(board: torch.Tensor):
    
    tv = torch.arange(3)                # triple vector
    rtv = torch.arange(2, -1, -1)       # reversed triple vector

    dims = len(board.shape)
    for i in range(dims):
        if any(board.sum(i).abs_() == 3):
            return True

    for i in range(dims):
        for j in range(dims):
            other_dims = 


class TicTacToe(Env):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.board_shape = tuple([3] * dim)
        
        self.observation_space = Box(-1, 1, shape=self.board_shape)
        self.action_space = Discrete(3 ** dim)

        self._board = None
        self._should_reset = True

        self._to_play = 1

    def reset(self):
        self._board = torch.zeros(*self.board_shape)
        self._should_reset = False
        return self._board

    def step(self, action):
        assert self.action_space.contains(action), "invalid action!"

        self._board.view(-1)[action] = self._to_play
        self._to_play = - self._to_play


