import torch
from torch import Tensor, LongTensor, BoolTensor

from gym import Env
from gym.spaces import Discrete, Box


@torch.jit.script
def _next_board(board, player: int, actions):
    SIZE = board.shape[-1]
    ONE = torch.tensor(1.0, dtype=torch.float)
    n = actions.shape[0]
    next_boards = board.unsqueeze(0).expand(n, -1, -1, -1).clone()
    rows = actions // SIZE
    cols = actions % SIZE
    next_boards[torch.arange(n), player, rows, cols] = ONE
    return next_boards


@torch.jit.script
def _distance_to_stone(boards, player: int):

    if len(boards.shape) < 4:
        boards = boards.unsqueeze(0)
    
    N_BOARDS = boards.shape[0]
    b = torch.arange(N_BOARDS)

    SIZE = boards.shape[-1]
    STONE = (boards[b, player] == 1).view(N_BOARDS, -1)
    OTHER_STONE = (boards[b, 1-player] == 1).view(N_BOARDS, -1)

    DIRECTIONS = [-SIZE, 1, SIZE, -1]
    START_INDICES = [
        ((SIZE**2-SIZE) + torch.arange(SIZE, dtype=torch.long)).long(),
        (SIZE * torch.arange(SIZE, dtype=torch.long)).long(),
        (torch.arange(SIZE, dtype=torch.long)).long(),
        (SIZE - 1 + SIZE * torch.arange(SIZE, dtype=torch.long))
    ]

    INF = torch.tensor(4.0 * SIZE, dtype=torch.float)
    ZERO = torch.tensor(0.0, dtype=torch.float)
    FALSE = torch.tensor(False)
    closest = torch.empty(N_BOARDS, SIZE, SIZE).fill_(INF).view(N_BOARDS, -1)

    updated = torch.tensor(True)
    while updated:
        updated = FALSE

        for direction, start_index in zip(DIRECTIONS, START_INDICES):
            distance = torch.empty(N_BOARDS, SIZE).fill_(INF)
            index = start_index.clone()

            for _ in range(SIZE):
                reset_mask = OTHER_STONE[:, index]
                distance[reset_mask] = INF
                
                empty_mask = STONE[:, index]
                distance[empty_mask] = ZERO
                
                increment_mask = ~torch.logical_or(reset_mask, empty_mask)
                distance[increment_mask] += 1

                closer_mask = distance < closest[:, index]
                further_mask = ~closer_mask
                a = closest[:, index]
                a[closer_mask] = distance[closer_mask]
                closest[:, index] = a
                distance[further_mask] = closest[:, index][further_mask]

                index += direction

                if not updated:
                    updated = closer_mask.sum() > 0

    closest[closest == INF] = torch.tensor(-1.0, dtype=torch.float)
    return closest.view(N_BOARDS, SIZE, SIZE)


@torch.jit.script
def _distance_to_empty(boards, player: int):
    
    if len(boards.shape) < 4:
        boards = boards.unsqueeze(0)
    
    N_BOARDS = boards.shape[0]
    b = torch.arange(N_BOARDS)

    SIZE = boards.shape[-1]
    EMPTY = (boards == 0).all(1).view(N_BOARDS, -1)
    ENEMY = (boards[b, 1-player] == 1).view(N_BOARDS, -1)

    DIRECTIONS = [-SIZE, 1, SIZE, -1]
    START_INDICES = [
        ((SIZE**2-SIZE) + torch.arange(SIZE, dtype=torch.long)).long(),
        (SIZE * torch.arange(SIZE, dtype=torch.long)).long(),
        (torch.arange(SIZE, dtype=torch.long)).long(),
        (SIZE - 1 + SIZE * torch.arange(SIZE, dtype=torch.long))
    ]

    INF = torch.tensor(4.0 * SIZE, dtype=torch.float)
    ZERO = torch.tensor(0.0, dtype=torch.float)
    FALSE = torch.tensor(False)
    closest = torch.empty(N_BOARDS, SIZE, SIZE).fill_(INF).view(N_BOARDS, -1)

    updated = torch.tensor(True)
    while updated:
        updated = FALSE

        for direction, start_index in zip(DIRECTIONS, START_INDICES):
            distance = torch.empty(N_BOARDS, SIZE).fill_(INF)
            index = start_index.clone()

            for _ in range(SIZE):
                reset_mask = ENEMY[:, index]
                distance[reset_mask] = INF
                
                empty_mask = EMPTY[:, index]
                distance[empty_mask] = ZERO
                
                increment_mask = ~torch.logical_or(reset_mask, empty_mask)
                distance[increment_mask] += 1

                closer_mask = distance < closest[:, index]
                further_mask = ~closer_mask
                a = closest[:, index]
                a[closer_mask] = distance[closer_mask]
                closest[:, index] = a
                distance[further_mask] = closest[:, index][further_mask]

                index += direction

                if not updated:
                    updated = closer_mask.sum() > 0

    closest[closest == INF] = torch.tensor(-1.0, dtype=torch.float)
    return closest.view(N_BOARDS, SIZE, SIZE)


@torch.jit.script
def _get_suicidemask(board, player: int):

    SIZE = board.shape[-1]
    EMPTY = (board == 0).all(0).view(-1)
    ENEMY = board[1-player] == 1.0
    mask = torch.zeros(SIZE * SIZE, dtype=torch.bool)

    empty_actions = torch.arange(SIZE * SIZE)[EMPTY]
    n_actions = empty_actions.shape[0]

    next_boards = _next_board(board, player, empty_actions)
    distance_to_empty = _distance_to_empty(next_boards, player)
    suicidal_actions = distance_to_empty.view(n_actions, -1)[torch.arange(n_actions), empty_actions] < 0
    
    enemy_captured = (_distance_to_empty(next_boards, 1-player)[:, ENEMY] < 0).any(1)
    suicidal_actions.logical_and_(~enemy_captured)

    mask[empty_actions[suicidal_actions]] = torch.tensor(True)
    return mask.view(SIZE, SIZE)


class Go(Env):
    def __init__(self, board_size: int, white_base_score: float=0.0):
        super().__init__()

        self._size = board_size
        self._board: Tensor = None
        self._prev_board: Tensor = None
        self._next_is_black = True
        self._player = 0
        self._action_mask: BoolTensor = None
        self._prev_action: int = None

        self._black_captures: int = None
        self._white_captures: int = None
        self._white_base_score: float = white_base_score

        self.observation_space = Box(0, 1, shape=(2, board_size, board_size))
        self.action_space = Discrete(board_size ** 2 + 1)

        self._i = torch.arange(2)
        self._ri = reversed(self._i)
        self._all_actions = torch.arange(board_size ** 2 + 1)
        self._true_tensor = torch.tensor([True])

    def get_state(self) -> torch.Tensor:
        return self._board if self._next_is_black else self._board[self._ri]

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
        next_boards = _next_board(self._board, self._player, self._all_actions[:-1])
        maskprevboard: BoolTensor = (next_boards != self._prev_board.unsqueeze(0)).any(-1).any(-1).any(-1)
        maskcurrboard: BoolTensor = (next_boards != self._board.unsqueeze(0)).any(-1).any(-1).any(-1)
        maskfreeplace: BoolTensor = self._board[self._player].view(-1) == 0
        masksuicide: BoolTensor = _get_suicidemask(self._board, self._player).view(-1)
        self._action_mask = maskcurrboard.logical_and_(maskprevboard).logical_and_(maskfreeplace).logical_and_(~masksuicide).view(-1)
        self._action_mask = torch.cat((self._action_mask, self._true_tensor))

    def _compute_reward(self) -> float:
        empty_space = (self._board == 0).all(0)
        white_space = (_distance_to_stone(self._board, 0) < 0).logical_and_(empty_space).sum()
        black_space = (_distance_to_stone(self._board, 1) < 0).logical_and_(empty_space).sum()
        black_win = black_space + self._black_captures > white_space + self._white_captures + self._white_base_score
        if self._next_is_black:
            return 1 if black_win else -1
        else:
            return -1 if black_space else 1

    def action_mask(self) -> BoolTensor:
        return self._action_mask

    def _place_stone(self, row, col):
        self._board[self._player, row, col] = 1.0
        enemy_space = self._board[1-self._player] == 1.0
        captures = (_distance_to_empty(self._board, 1-self._player).squeeze_() < 0).logical_and_(enemy_space)
        self._board[1-self._player, captures] = 0.0

        if self._next_is_black:
            self._black_captures += captures.sum()
        else:
            self._white_captures += captures.sum()

    def step(self, action: int):
        assert action in self.action_space, "Invalid action"
        assert self._action_mask[action], "Invalid action in current state"

        self._prev_board = self._board.clone()

        if action == self._size ** 2:  # action is pass    
            if self._prev_action == action:     # double pass, game over
                reward = self._compute_reward()
                return None, reward, True
        else:
            row = action // self._size; col = action % self._size
            captures = self._place_stone(row, col)

        self._prev_action = action
        self._next_is_black = not self._next_is_black
        self._player = 1 - self._player

        self._update_action_mask()

        return self.get_state(), 0, False
