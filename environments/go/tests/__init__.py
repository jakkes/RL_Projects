import torch

from .. import _next_board, _distance_to_empty, _get_suicidemask
from .. import Go


def test_next_board():
    board = torch.zeros(2, 3, 3)

    next_boards1 = _next_board(board, 0, torch.arange(9))
    next_boards2 = _next_board(board, 1, torch.arange(9))

    true_next_boards1 = torch.zeros(9, 2, 3, 3).view(9, 2, 9)
    true_next_boards1[torch.arange(9), 0, torch.arange(9)] = 1.0
    true_next_boards1 = true_next_boards1.view(9, 2, 3, 3)

    true_next_boards2 = torch.zeros(9, 2, 3, 3).view(9, 2, 9)
    true_next_boards2[torch.arange(9), 1, torch.arange(9)] = 1.0
    true_next_boards2 = true_next_boards2.view(9, 2, 3, 3)

    assert (next_boards1 == true_next_boards1).all()
    assert (next_boards2 == true_next_boards2).all()


def test_distance_to_empty():
    boards = torch.tensor([
        [[[0.0, 0.0, 1.0],
         [1.0, 1.0, 0.0],
         [0.0, 0.0, 0.0]],

        [[1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0]]],


        [[[1.0, 1.0, 0.0],
         [1.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]],

        [[0.0, 0.0, 1.0],
         [0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0]]]
    ])

    distances = _distance_to_empty(boards, 0)
    true_distances = torch.tensor([
        [[-1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, -1.0]],

        [[3.0, 2.0, -1.0],
        [2.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0]]
    ])

    assert (true_distances == distances).all()


def test_suicide_mask():
    board1 = torch.tensor([
        [[0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]],

        [[0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]]
    ])

    mask1 = _get_suicidemask(board1, 0)
    true1 = torch.tensor(
        [[True, False, False],
        [False, False, False],
        [False, False, False]]
    )

    assert (mask1 == true1).all()


def test_go():
    env = Go(5)
    state = env.reset()
    


def run_tests():
    test_next_board()
    test_distance_to_empty()
    test_suicide_mask()
    test_go()