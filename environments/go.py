import torch
from torch import Tensor, LongTensor, BoolTensor
from torch.multiprocessing import Pool

from gym import Env
from gym.spaces import Discrete, Box


# @torch.jit.script
def _next_board(board, player, actions):
    n = actions.shape[0]
    next_boards = board.unsqueeze(0).expand(n, -1, -1, -1).clone()
    rows = actions // self._size
    cols = actions % self._size
    next_boards[torch.arange(n), player, rows, cols] = 1.0
    return next_boards

# @torch.jit.script
def _distance_to_empty(boards, player):
    
    if len(boards.shape) < 4:
        boards = boards.unsqueeze()
    
    N_BOARDS = boards.shape[0]
    b = torch.arange(N_BOARDS)

    SIZE = boards.shape[-1]
    EMPTY = (boards == 0).all(1).view(N_BOARDS, -1)
    ENEMY = (boards[b, 1-player] == 1).view(N_BOARDS, -1)

    DIRECTIONS = [-SIZE, 1, SIZE, -1]
    START_INDICES = [
        (SIZE**2-1-SIZE) + torch.arange(SIZE),
        SIZE * torch.arange(SIZE),
        torch.arange(SIZE),
        SIZE - 1 + SIZE * torch.arange(SIZE)
    ]

    INF = 4 * SIZE
    closest = torch.empty(N_BOARDS, SIZE, SIZE).fill_(INF).view(N_BOARDS, -1)

    updated = True
    while updated:
        updated = False

        for direction, start_index in zip(DIRECTIONS, START_INDICES):
            distance = torch.empty(N_BOARDS, SIZE).fill_(INF)
            index = start_index.clone()

            for _ in range(SIZE):
                reset_mask = ENEMY[b, index]
                distance[reset_mask].fill_(INF)
                
                empty_mask = EMPTY[b, index]
                distance[empty_mask].fill_(0)

                increment_mask = ~(reset_mask or empty_mask)
                distance[increment_mask] += 1

                closer_mask = distance < closest[:, index]
                closest[closer_mask] = distance[closer_mask]

                index += direction

                if not updated:
                    updated = closer_mask.sum() > 0

    closest[closest == INF].fill_(-1)
    return closest.view(N_BOARDS, SIZE, SIZE)


def _get_suicidemask(board, player):

    SIZE = board.shape[-1]
    EMPTY = (board == 0).all(0).view(-1)
    
    empty_actions = torch.arange(SIZE * SIZE)[empty]
    next_boards = _next_board(board, player, empty_actions)
    distance_to_empty = _distance_to_empty(next_boards, player)


class Go(Env):
    def __init__(self, board_size: int, pool_processes: int=4):
        super().__init__()

        self._pool_processes = pool_processes
        self._pool = Pool(pool_processes)

        self._size = board_size
        self._board: Tensor = None
        self._prev_board: Tensor = None
        self._next_is_black = True
        self._player = 0
        self._action_mask: BoolTensor = None
        self._prev_action: int = None

        self._black_captures: int = None
        self._white_captures: int = None

        self.observation_space = Box(0, 1, shape=(2, board_size, board_size))
        self.action_space = Discrete(board_size ** 2 + 1)

        self._i = torch.arange(2)
        self._ri = reversed(self._i)
        self._all_actions = torch.arange(board_size ** 2 + 1)
        self._true_tensor = torch.tensor(True)

    def get_state(self) -> torch.Tensor:
        return self._board if self._next_is_black else self._board[self.ri]

    def reset(self, state: Tensor=None, prev_state: Tensor=None, prev_action: int=None, black_to_play: bool=None, black_captures: int=None, white_captures: int=None):

        if state is not None:
            self._board = state if black_to_play else state[self._ri]
            self._prev_board = prev_state[self._ri] if black_to_play else prev_state
            self._prev_action = prev_action
            self._next_is_black = black_to_play
            self._player = 0 if black_to_play else 1
            self._black_captures = black_captures
            self._white_captures = white_captures
        else:
            self._board = torch.zeros(2, self._size, self._size)
            self._prev_board = torch.zeros(2, self._size, self._size)
            self._prev_action = None
            self._next_is_black = True
            self._player = 0
            self._black_captures = 0
            self._white_captures = 0

        self._update_action_mask()
        return self.get_state()

    def _update_action_mask(self):
        next_boards = _next_board(self._board, self._player, self._all_actions)
        maskprevboard = (next_boards != self._prev_board.unsqueeze(0)).any(-1).any(-1).any(-1)
        maskcurrboard = (next_boards != self._board.unsqueeze(0)).any(-1).any(-1).any(-1)
        maskfreeplace = self._board[self._player].view(-1) == 0
        self._action_mask = maskcurrboard and maskprevboard and maskfreeplace
        self._action_mask = torch.cat((self._action_mask, self._true_tensor))

    def _compute_reward(self) -> float:
        raise NotImplementedError

    def action_mask(self) -> BoolTensor:
        return self._action_mask

    def _place_stone(self, row, col):
        self._board[self._player, row, col] = 1.0


    def step(self, action: int):
        assert action in self.action_space, "Invalid action"
        assert self._action_mask[action], "Invalid action in current state"

        self._prev_board = self._board.clone()
        
        if action == self._board ** 2:  # action is pass    
            if self._prev_action == action:     # double pass, game over
                reward = self._compute_reward()
                return None, reward, True
        else:
            row = action // self._size; col = action % self._size
            captures = self._place_stone(self._board, row, col)

        self._prev_action = action
        self._next_is_black = not self._next_is_black
        self._player = 1 - self._player

        return self.get_state(), 0, False
